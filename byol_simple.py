# -*- coding: utf-8 -*-
"""
BYOL (Bootstrap Your Own Latent) Implementation
Self-supervised learning with Vision Transformer backbone on CIFAR-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import copy

# ============================================================================
# REUSED COMPONENTS FROM MAE
# ============================================================================

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )

    def forward(self, x):
        return self.projection(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Non-learnable positional encoding from the original Transformer paper.
    """
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        pe = torch.zeros(num_patches, embed_dim)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        current_num_patches = x.shape[1]
        return x + self.pe[:, :current_num_patches, :]


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# BYOL-SPECIFIC COMPONENTS
# ============================================================================

class VisionTransformerBackbone(nn.Module):
    """
    Vision Transformer backbone for BYOL.
    Outputs a [CLS] token representation for the entire image.
    """
    def __init__(self, image_size=32, patch_size=4, in_channels=3,
                 embed_dim=192, depth=6, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Positional encoding (num_patches + 1 for CLS token)
        self.pos_embed = SinusoidalPositionalEncoding(num_patches + 1, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)

        # Add positional encoding
        x = self.pos_embed(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Return only the CLS token representation
        return x[:, 0]  # (B, embed_dim)


class ProjectionHead(nn.Module):
    """
    MLP projection head for BYOL.
    Projects the backbone representation to a lower-dimensional space.
    """
    def __init__(self, input_dim=192, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class PredictionHead(nn.Module):
    """
    MLP prediction head for BYOL online network.
    Predicts the target network's projection from the online network's projection.
    """
    def __init__(self, input_dim=128, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    """
    BYOL: Bootstrap Your Own Latent

    Architecture:
    - Online network: backbone + projector + predictor
    - Target network: backbone + projector (EMA of online network)

    The online network is trained to predict the target network's representation
    of the same image under different augmentations.
    """
    def __init__(self, backbone_config, projection_dim=128, hidden_dim=512,
                 ema_decay=0.996):
        super().__init__()
        self.ema_decay = ema_decay

        # Online network
        self.online_backbone = VisionTransformerBackbone(**backbone_config)
        embed_dim = backbone_config['embed_dim']
        self.online_projector = ProjectionHead(embed_dim, hidden_dim, projection_dim)
        self.predictor = PredictionHead(projection_dim, hidden_dim, projection_dim)

        # Target network (no gradients)
        self.target_backbone = VisionTransformerBackbone(**backbone_config)
        self.target_projector = ProjectionHead(embed_dim, hidden_dim, projection_dim)

        # Initialize target network with online network weights
        self._initialize_target_network()

        # Target network should not have gradients
        for param in self.target_backbone.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def _initialize_target_network(self):
        """Copy online network weights to target network"""
        for online_params, target_params in zip(
            self.online_backbone.parameters(),
            self.target_backbone.parameters()
        ):
            target_params.data.copy_(online_params.data)

        for online_params, target_params in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters()
        ):
            target_params.data.copy_(online_params.data)

    @torch.no_grad()
    def update_target_network(self):
        """
        Update target network using exponential moving average (EMA).
        target = tau * target + (1 - tau) * online
        """
        for online_params, target_params in zip(
            self.online_backbone.parameters(),
            self.target_backbone.parameters()
        ):
            target_params.data = (
                self.ema_decay * target_params.data +
                (1 - self.ema_decay) * online_params.data
            )

        for online_params, target_params in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters()
        ):
            target_params.data = (
                self.ema_decay * target_params.data +
                (1 - self.ema_decay) * online_params.data
            )

    def forward(self, view1, view2):
        """
        Forward pass with two augmented views.

        Args:
            view1, view2: Two different augmented views of the same batch

        Returns:
            loss: BYOL loss (negative cosine similarity)
        """
        # Online network forward pass
        online_repr1 = self.online_backbone(view1)
        online_proj1 = self.online_projector(online_repr1)
        online_pred1 = self.predictor(online_proj1)

        online_repr2 = self.online_backbone(view2)
        online_proj2 = self.online_projector(online_repr2)
        online_pred2 = self.predictor(online_proj2)

        # Target network forward pass (no gradients)
        with torch.no_grad():
            target_repr1 = self.target_backbone(view1)
            target_proj1 = self.target_projector(target_repr1)

            target_repr2 = self.target_backbone(view2)
            target_proj2 = self.target_projector(target_repr2)

        # Compute loss: predict view2 from view1 and vice versa
        loss1 = self.byol_loss(online_pred1, target_proj2)
        loss2 = self.byol_loss(online_pred2, target_proj1)

        # Total loss is the mean of both directions
        loss = (loss1 + loss2) / 2.0

        return loss

    def byol_loss(self, pred, target):
        """
        BYOL loss: negative cosine similarity (mean squared error in normalized space)

        Loss = 2 - 2 * cosine_similarity(pred, target)
        """
        pred = F.normalize(pred, dim=-1, p=2)
        target = F.normalize(target, dim=-1, p=2)
        return 2 - 2 * (pred * target).sum(dim=-1).mean()


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class BYOLAugmentation:
    """
    BYOL-style augmentation pipeline with 5 key transformations:
    1. RandomResizedCrop (30-100% of original)
    2. RandomErasing (masking 20-60%)
    3. GaussianBlur
    4. ColorJitter
    5. RandomRotation
    """
    def __init__(self, image_size=32):
        self.transform = transforms.Compose([
            # 1. Random resized crop (30-100% of original size)
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.3, 1.0),
                ratio=(0.75, 1.33),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),

            # 4. Color jitter (brightness, contrast, saturation, hue)
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1
                )
            ], p=0.8),

            # 5. Random rotation
            transforms.RandomRotation(degrees=15),

            # 3. Gaussian blur
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.5),

            # Convert to tensor
            transforms.ToTensor(),

            # 2. Random erasing (masking 20-60% of patches)
            transforms.RandomErasing(
                p=0.5,
                scale=(0.2, 0.6),
                ratio=(0.3, 3.3),
                value='random'
            ),

            # Normalize (CIFAR-10 statistics)
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])

    def __call__(self, x):
        return self.transform(x)


class BYOLTransform:
    """Wrapper that applies two different augmentations to create two views"""
    def __init__(self, image_size=32):
        self.transform = BYOLAugmentation(image_size)

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


# ============================================================================
# TRAINING
# ============================================================================

def train_byol():
    """Train BYOL on a subset of CIFAR-10"""

    # Configuration
    IMAGE_SIZE = 32
    PATCH_SIZE = 4

    # Training hyperparameters
    NUM_EPOCHS = 100
    BATCH_SIZE = 256
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-6
    EMA_DECAY_START = 0.996
    EMA_DECAY_END = 1.0

    # Use a subset of CIFAR-10 for faster training
    DATASET_SUBSET_SIZE = 10000  # Use 10k images instead of full 50k

    # Model configuration (smaller ViT for CIFAR-10)
    BACKBONE_CONFIG = {
        'image_size': IMAGE_SIZE,
        'patch_size': PATCH_SIZE,
        'in_channels': 3,
        'embed_dim': 192,
        'depth': 6,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'dropout': 0.1
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loading
    transform = BYOLTransform(image_size=IMAGE_SIZE)

    # Load full dataset first
    full_dataset = datasets.CIFAR10(
        './data',
        train=True,
        download=True,
        transform=transform
    )

    # Create subset
    indices = torch.randperm(len(full_dataset))[:DATASET_SUBSET_SIZE]
    train_dataset = Subset(full_dataset, indices)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )

    print(f"Training on {len(train_dataset)} images from CIFAR-10")

    # Model initialization
    model = BYOL(
        backbone_config=BACKBONE_CONFIG,
        projection_dim=128,
        hidden_dim=512,
        ema_decay=EMA_DECAY_START
    ).to(device)

    # Optimizer (only for online network)
    optimizer = optim.AdamW(
        [
            {'params': model.online_backbone.parameters()},
            {'params': model.online_projector.parameters()},
            {'params': model.predictor.parameters()}
        ],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6
    )

    print(f"\nModel initialized:")
    print(f"  - Patches per image: {(IMAGE_SIZE // PATCH_SIZE) ** 2}")
    print(f"  - Embedding dimension: {BACKBONE_CONFIG['embed_dim']}")
    print(f"  - Transformer depth: {BACKBONE_CONFIG['depth']}")
    print(f"  - Projection dimension: 128")

    # Training loop
    model.train()
    loss_history = []

    for epoch in range(NUM_EPOCHS):
        total_loss = 0

        # Update EMA decay (cosine schedule from start to end)
        progress = epoch / NUM_EPOCHS
        model.ema_decay = EMA_DECAY_START + (EMA_DECAY_END - EMA_DECAY_START) * (
            0.5 * (1 + math.cos(math.pi * progress))
        )

        for batch_idx, (views, _) in enumerate(train_dataloader):
            # views is a tuple of (view1, view2)
            view1, view2 = views
            view1 = view1.to(device)
            view2 = view2.to(device)

            optimizer.zero_grad()

            # Forward pass
            loss = model(view1, view2)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update target network
            model.update_target_network()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                      f"Batch {batch_idx}/{len(train_dataloader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"EMA: {model.ema_decay:.4f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        avg_loss = total_loss / len(train_dataloader)
        loss_history.append(avg_loss)

        print(f"--- Epoch {epoch+1} Finished | Average Loss: {avg_loss:.4f} ---\n")

        scheduler.step()

    print("Training complete!")

    return model, loss_history, BACKBONE_CONFIG


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_training_progress(loss_history):
    """Plot the training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2)
    plt.title('BYOL Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('byol_training_loss.png', dpi=150)
    print("Training loss plot saved to 'byol_training_loss.png'")
    plt.show()


def visualize_augmentations(num_samples=5):
    """Visualize the augmentation pipeline"""
    # Load one image
    dataset = datasets.CIFAR10('./data', train=True, download=True)

    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    plt.suptitle("BYOL Augmentation Examples", y=1.02, fontsize=14, fontweight='bold')

    augmentation = BYOLTransform(image_size=32)

    for i in range(num_samples):
        img, _ = dataset[np.random.randint(len(dataset))]

        # Original
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original {i+1}")
        axes[i, 0].axis('off')

        # Two augmented views
        view1, view2 = augmentation(img)

        # Denormalize for visualization
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

        view1_denorm = view1 * std + mean
        view2_denorm = view2 * std + mean

        view1_denorm = torch.clamp(view1_denorm, 0, 1)
        view2_denorm = torch.clamp(view2_denorm, 0, 1)

        axes[i, 1].imshow(view1_denorm.permute(1, 2, 0).numpy())
        axes[i, 1].set_title("Augmented View 1")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(view2_denorm.permute(1, 2, 0).numpy())
        axes[i, 2].set_title("Augmented View 2")
        axes[i, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('byol_augmentations.png', dpi=150)
    print("Augmentation examples saved to 'byol_augmentations.png'")
    plt.show()


def visualize_learned_representations(model, num_samples=100):
    """
    Visualize learned representations using t-SNE or PCA
    Colored by CIFAR-10 class labels
    """
    device = next(model.parameters()).device
    model.eval()

    # Load test data without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])

    test_dataset = datasets.CIFAR10(
        './data',
        train=False,
        download=True,
        transform=test_transform
    )

    # Sample random images
    indices = torch.randperm(len(test_dataset))[:num_samples]

    representations = []
    labels = []

    with torch.no_grad():
        for idx in indices:
            img, label = test_dataset[idx]
            img = img.unsqueeze(0).to(device)

            # Get representation from online backbone
            repr = model.online_backbone(img)
            representations.append(repr.cpu())
            labels.append(label)

    representations = torch.cat(representations, dim=0).numpy()
    labels = np.array(labels)

    # Use PCA for dimensionality reduction
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    repr_2d = pca.fit_transform(representations)

    # Plot
    plt.figure(figsize=(10, 8))

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    scatter = plt.scatter(
        repr_2d[:, 0],
        repr_2d[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.7,
        s=50
    )

    plt.colorbar(scatter, ticks=range(10), label='Class')
    plt.clim(-0.5, 9.5)

    plt.title('BYOL Learned Representations (PCA)', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=plt.cm.tab10(i/10), markersize=8,
                         label=class_names[i]) for i in range(10)]
    plt.legend(handles=handles, loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('byol_representations.png', dpi=150)
    print("Learned representations plot saved to 'byol_representations.png'")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("BYOL (Bootstrap Your Own Latent) Training on CIFAR-10")
    print("="*70)
    print()

    # First, visualize the augmentation pipeline
    print("Step 1: Visualizing augmentation pipeline...")
    visualize_augmentations(num_samples=3)
    print()

    # Train the model
    print("Step 2: Training BYOL model...")
    trained_model, loss_history, config = train_byol()
    print()

    # Visualize training progress
    print("Step 3: Visualizing training progress...")
    visualize_training_progress(loss_history)
    print()

    # Visualize learned representations
    print("Step 4: Visualizing learned representations...")
    visualize_learned_representations(trained_model, num_samples=200)
    print()

    print("="*70)
    print("BYOL training and visualization complete!")
    print("="*70)
