"""
Minimal 2D Swin Transformer Implementation in PyTorch
A general-purpose backbone for 2D Computer Vision

This implementation demonstrates the core components of Swin Transformer:
- 2D Patch Embedding
- Windowed Multi-Head Self-Attention (W-MSA)
- Shifted Window Multi-Head Self-Attention (SW-MSA)
- Relative Position Bias
- Patch Merging for hierarchical features

Reference: Liu et al., "Swin Transformer", ICCV 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import data
from skimage.transform import resize
from typing import Optional

def window_partition(x, window_size: int):
    """
    Partition 2D feature maps into non-overlapping windows.

    Args:
        x: (B, H, W, C)
        window_size: Window size (e.g., 7)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: (B, H//ws, W//ws, ws, ws, C)
    # contiguous: make sure memory is contiguous for view
    # view: (B * num_windows, ws, ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Reverse window partition back to feature maps.

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H, W: Spatial dimensions of original feature map

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: (B, H//ws, W//ws, ws, ws, C)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: (B, H//ws, ws, W//ws, ws, C)
    # contiguous: make sure memory is contiguous for view
    # view: (B, H, W, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def create_mask(H, W, window_size, shift_size):
    """
    Create an attention mask for Shifted Window Multi-Head Self-Attention (SW-MSA).
    
    This prevents attention between patches that are not neighbors
    (i.e., patches that were wrapped around).

    When a cyclic shift is applied to the feature map, patches that wrap around (e.g., from 
    the bottom edge to the top edge) are not true neighbors but end up in the same attention 
    window. This mask prevents attention between these non-adjacent, wrapped-around patches.

    The core idea is to divide the original feature map into nine distinct regions (based on 
    the window size and shift size), assign a unique integer ID to each region, and then 
    partition this ID map into windows. If two patches within a window have different IDs, 
    it means they originated from different, non-contiguous regions of the original image 
    and must be masked out.

    Args:
        H (int): Height of the feature map.
        W (int): Width of the feature map.
        window_size (int): Size of the local window (e.g., 7).
        shift_size (int): The amount of cyclic shift (e.g., window_size // 2).

    Returns:
        torch.Tensor: An attention mask of shape (num_windows, window_size*window_size, 
                    window_size*window_size). It contains:
                    - 0.0: Where patches within a window have the same ID (valid attention).
                    - -100.0: Where patches have different IDs (invalid attention).
                                This large negative value ensures the attention weight is 
                                zero after the softmax operation.
    """
    # Create a "map" of window IDs
    img_mask = torch.zeros(1, H, W, 1)  # (1, H, W, 1)
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    # Partition the ID map into windows
    mask_windows = window_partition(img_mask, window_size)  # (num_windows, ws, ws, 1)
    mask_windows = mask_windows.view(-1, window_size * window_size) # (num_windows, ws*ws)
    
    # Calculate the mask
    # (num_windows, 1, ws*ws) - (num_windows, ws*ws, 1)
    # This broadcasts to (num_windows, ws*ws, ws*ws)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    
    # Fill with -100 (for softmax) where IDs are different (i.e., not neighbors)
    # Fill with 0 where IDs are the same (i.e., valid neighbors)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    
    return attn_mask


class WindowAttention(nn.Module):
    """ 2D Window-based Multi-head Self-Attention with Relative Position Bias """
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Define a learnable parameter table for relative position bias (RPB)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        
        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # (2, ws, ws)
        coords_flatten = torch.flatten(coords, 1)  # (2, ws*ws)
        
        # (2, ws*ws, 1) - (2, 1, ws*ws) -> (2, ws*ws, ws*ws)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (ws*ws, ws*ws, 2)
        
        # Shift to non-negative indices
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        
        # This is the 1D index into the bias table
        relative_position_index = relative_coords.sum(-1)  # (ws*ws, ws*ws)
        
        # Register as a buffer so it's part of the model state but not a parameter
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (num_windows*B, N, C) where N = window_size * window_size
            mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        
        # qkv: (B_, N, 3*C) -> (B_, N, 3, num_heads, head_dim) -> (3, B_, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention: (B_, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Get relative position bias: (ws*ws, ws*ws, num_heads)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)
        
        # Add RPB to attention scores
        attn = attn + relative_position_bias.unsqueeze(0) # (B_, num_heads, N, N)

        if mask is not None:
            # mask: (num_windows, N, N)
            # Add mask to attention
            nW = mask.shape[0] # number of windows
            # (B_, num_heads, N, N) + (1, nW, 1, N, N) -> (B_, num_heads, N, N)
            # We assume B_ = B * nW
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        # Output: (B_, num_heads, N, head_dim) -> (B_, N, num_heads, head_dim) -> (B_, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerLayer(nn.Module):
    """ Swin Transformer Layer with W-MSA or SW-MSA """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x, H, W):
        """
        Args:
            x: (B, N, C) where N = H * W
            H, W: Spatial dimensions
        """
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C) # retrieve the height and width by unpacking N

        # Create mask if we are shifting
        if self.shift_size > 0:
            shift_mask = create_mask(H, W, self.window_size, self.shift_size)
            shift_mask = shift_mask.to(x.device)
            # Cyclic shift
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shift_mask = None
        
        # Partition into windows
        x_windows = window_partition(shifted_x, self.window_size) # (B*nW, ws, ws, C)
        # now flatten each window as attention mechanism needs a flattened array
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) # (B*nW, ws*ws, C)
        
        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=shift_mask) # (B*nW, ws*ws, C)
        
        # Reverse window partition
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C) # (B*nW, ws, ws, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W) # (B, H, W, C)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C) # to (B, N, C) needed for shortcut connection and MLP 

        # Residual connection + MLP
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer for 2D """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim # just note that dim = C
        # Linear layer to halve the channels (from 4*C to 2*C)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        """
        Args:
            x: (B, N, C) where N = H * W
            H, W: Spatial dimensions
        """
        B, N, C = x.shape
        x = x.view(B, H, W, C)

        # Sample 4 neighboring patches
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # (B, H/2, W/2, C)
        
        # Concatenate along channel dimension
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4*C)
        
        # Update H, W
        H, W = H // 2, W // 2
        x = x.view(B, -1, 4 * C)  # (B, H/2, W/2, 4*C) to (B, N/4, 4*C)

        x = self.norm(x)
        x = self.reduction(x) # (B, N/4, 4*C) to (B, n/4, 2*C)
        
        return x, H, W


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, N, embed_dim)
            H, W: New spatial dimensions
        """
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)
        B, C, H, W = x.shape
        x = x.flatten(2)  # (B, C, N) such that N = H * W
        x = x.transpose(1, 2) # (B, N, C)
        x = self.norm(x)
        return x, H, W


class BasicBlock(nn.Module):
    """ A basic Swin Transformer block for one stage with as many layers as depth. """
    def __init__(self, dim, depth, num_heads, window_size, downsample=None):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            layer = SwinTransformerLayer(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2
            )
            self.layers.append(layer)

        self.downsample = downsample

    def forward(self, x, H, W):
        for layer in self.layers:
            x = layer(x, H, W)

        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W) # one time downsampling at the end of the block

        return x, H, W


class SwinTransformer(nn.Module):
    """
    Swin Transformer "Tiny" (Swin-T) model.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7):
        super().__init__()
        
        self.num_blocks = len(depths)
        
        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        # 2. Build Swin Transformer Stages
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            # specify the feature_dim (or number of channels C) for this block
            this_dim = int(embed_dim * 2 ** i)
            # determine if we need/can do downsampling
            this_downsampling = PatchMerging(this_dim) if (i < self.num_blocks - 1) else None
            # initialize the block
            block = BasicBlock(
                dim=this_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                downsample=this_downsampling,
            )
            self.blocks.append(block)

        # 3. Final Classification Head
        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_blocks - 1)))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(int(embed_dim * 2 ** (self.num_blocks - 1)), num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        x, H, W = self.patch_embed(x) # output x has shape (B, N, C)
        
        # Pass through all stages
        for block in self.blocks:
            x, H, W = block(x, H, W)
            
        # Final layers for classification
        x = self.norm(x)  # (B, N, C)
        x = x.transpose(1, 2)  # (B, C, N)
        x = self.avgpool(x)  # (B, C, 1)
        x = torch.flatten(x, 1) # (B, C)
        x = self.head(x)
        
        return x


def run_test():
    """
    Test function to verify Swin Transformer implementation
    with real images from skimage.
    """
    print("Testing Swin Transformer (Swin-T)...")
    
    # 1. Load and preprocess 2 popular images
    # Chelsea (cat) is 3-channel (RGB)
    img1 = data.chelsea() # (300, 451, 3)
    # Camera (cameraman) is 1-channel (Grayscale)
    img2 = data.camera()  # (512, 512)
    
    # Define target size
    IMG_SIZE = 224
    
    # Process img1 (RGB)
    img1_resized = resize(img1, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    
    # Process img2 (Grayscale) - convert to 3-channel by repeating
    img2_resized = resize(img2, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    img2_rgb = np.stack([img2_resized] * 3, axis=-1)

    # 2. Stack into a batch
    # (2, 224, 224, 3) -> (2, 3, 224, 224)
    batch_np = np.stack([img1_resized, img2_rgb], axis=0).astype(np.float32)
    batch_torch = torch.from_numpy(batch_np).permute(0, 3, 1, 2)
    
    print(f"Input batch shape: {batch_torch.shape}")
    
    # 3. Create Swin-T model (Swin-Tiny configuration)
    # img_size=224, patch_size=4 -> H=W=56
    # Stage 1: dim=96, H=W=56
    # Stage 2: dim=192, H=W=28
    # Stage 3: dim=384, H=W=14
    # Stage 4: dim=768, H=W=7
    # window_size=7 fits perfectly into 56, 28, 14, 7
    model = SwinTransformer(
        img_size=IMG_SIZE,
        patch_size=4,
        in_chans=3,
        num_classes=10, # Dummy number of classes
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7
    )

    # Parameter count
    total_params = sum(param.numel() for param in model.parameters())
    print("-" * 50)
    print(f"Total parameters: {total_params}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    
    # 4. Forward pass
    with torch.no_grad():
        output = model(batch_torch)

    print("-" * 50)    
    print(f"Output shape: {output.shape}")
    
    # 5. Verify
    assert output.shape == (2, 10), "Output shape is incorrect!"
    print("-" * 50)
    print("Swin Transformer test passed successfully!")
    print(f"Output logits (first image): {output[0, :5].numpy()}...")
    print(f"Output logits (second image): {output[1, :5].numpy()}...")
    print("-" * 50)

if __name__ == "__main__":
    run_test()