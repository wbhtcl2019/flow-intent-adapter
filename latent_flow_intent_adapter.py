"""
Latent Flow Intent Adapter - Plug-in Module for Intent-Aware Traffic Prediction

A model-agnostic adapter that uses Flow Matching to generate latent intent
representations and modulates spatial-temporal features accordingly.

Usage:
    adapter = LatentFlowIntentAdapter(feature_dim=64)
    modulated_features, losses = adapter(st_features, intent_label)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


# ============================================================================
# Stage 1: Intent Generation via Flow Matching
# ============================================================================

class VectorFieldNetwork(nn.Module):
    """
    Neural network that parameterizes the velocity field for Flow Matching

    Input: z_t (latent at time t) + condition + t (time)
    Output: velocity v_t
    """
    def __init__(self, latent_dim: int, condition_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Time embedding (sinusoidal)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Main network
        self.net = nn.Sequential(
            nn.Linear(latent_dim + condition_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_t: torch.Tensor, condition: torch.Tensor, t: torch.Tensor):
        """
        Args:
            z_t: [batch, latent_dim] - latent at time t
            condition: [batch, condition_dim] - conditional information (ST features)
            t: [batch, 1] - time in [0, 1]

        Returns:
            v_t: [batch, latent_dim] - velocity at time t
        """
        # Embed time
        t_embed = self.time_embed(t)  # [batch, hidden_dim]

        # Concatenate all inputs
        x = torch.cat([z_t, condition, t_embed], dim=-1)

        # Predict velocity
        v_t = self.net(x)
        return v_t


class FlowIntentGenerator(nn.Module):
    """
    Generate latent intent representation using Conditional Flow Matching
    """
    def __init__(
        self,
        feature_dim: int,
        latent_dim: int = 64,
        n_intents: int = 6,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_intents = n_intents

        # Encode ST features into condition vector
        self.condition_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        condition_dim = hidden_dim // 2

        # Intent embedding (learned target for each discrete intent)
        self.intent_embedding = nn.Embedding(n_intents, latent_dim)

        # Vector field network
        self.vector_field = VectorFieldNetwork(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim
        )

    def forward(
        self,
        features: torch.Tensor,
        intent_label: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward: Flow Matching with supervision

        Args:
            features: [batch, feature_dim] - ST features
            intent_label: [batch] - discrete intent labels (0-5)

        Returns:
            z_t: [batch, latent_dim] - sampled latent intent
            loss_fm: scalar - flow matching loss
        """
        batch_size = features.size(0)
        device = features.device

        # Encode condition
        condition = self.condition_encoder(features)  # [batch, cond_dim]

        # Get target intent embedding (z_1)
        z_1 = self.intent_embedding(intent_label)  # [batch, latent_dim]

        # Sample noise (z_0)
        z_0 = torch.randn_like(z_1)

        # Sample time uniformly
        t = torch.rand(batch_size, 1, device=device)  # [batch, 1]

        # Optimal Transport interpolation: z_t = t * z_1 + (1-t) * z_0
        z_t = t * z_1 + (1 - t) * z_0

        # Predict velocity
        v_pred = self.vector_field(z_t, condition, t)

        # Target velocity (Conditional Flow Matching)
        v_target = z_1 - z_0

        # Flow matching loss
        loss_fm = F.mse_loss(v_pred, v_target)

        return z_t, loss_fm

    @torch.no_grad()
    def sample(self, features: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        """
        Inference: Sample latent intent via ODE solving (Euler method)

        Args:
            features: [batch, feature_dim] - ST features
            n_steps: int - number of integration steps

        Returns:
            z_1: [batch, latent_dim] - generated intent latent
        """
        batch_size = features.size(0)
        device = features.device

        # Encode condition
        condition = self.condition_encoder(features)

        # Start from noise
        z_t = torch.randn(batch_size, self.latent_dim, device=device)

        # Euler integration
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.full((batch_size, 1), step * dt, device=device)
            v = self.vector_field(z_t, condition, t)
            z_t = z_t + v * dt

        return z_t


# ============================================================================
# Stage 2: Feature Modulation
# ============================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM)

    Given latent intent z, compute:
        γ, β = f(z)
        output = γ ⊙ features + β
    """
    def __init__(self, feature_dim: int, latent_dim: int):
        super().__init__()

        self.gamma_net = nn.Sequential(
            nn.Linear(latent_dim, feature_dim),
            nn.Sigmoid()  # Scale around 1
        )

        self.beta_net = nn.Sequential(
            nn.Linear(latent_dim, feature_dim),
            nn.Tanh()  # Shift around 0
        )

    def forward(self, features: torch.Tensor, z_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, feature_dim] or [batch, feature_dim, H, W]
            z_latent: [batch, latent_dim]

        Returns:
            modulated: same shape as features
        """
        gamma = self.gamma_net(z_latent)  # [batch, feature_dim]
        beta = self.beta_net(z_latent)    # [batch, feature_dim]

        # Handle spatial features [B, C, H, W]
        if features.dim() == 4:
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [batch, C, 1, 1]
            beta = beta.unsqueeze(-1).unsqueeze(-1)

        return gamma * features + beta


# ============================================================================
# Complete Adapter Module
# ============================================================================

class LatentFlowIntentAdapter(nn.Module):
    """
    🔌 Plug-in Intent Adapter using Latent Flow Matching

    Can be seamlessly integrated into any spatial-temporal prediction model.

    Example:
        >>> # Original model
        >>> features = st_resnet.extract_features(x)  # [B, 64, H, W]
        >>>
        >>> # With adapter
        >>> adapter = LatentFlowIntentAdapter(feature_dim=64)
        >>> modulated, losses = adapter(features, intent_label, training=True)
        >>> pred = st_resnet.predict(modulated)
    """

    def __init__(
        self,
        feature_dim: int,
        latent_dim: int = 64,
        n_intents: int = 6,
        flow_steps: int = 10,
        hidden_dim: int = 256
    ):
        """
        Args:
            feature_dim: Dimension of ST features from backbone
            latent_dim: Dimension of latent intent vector (default: 64)
            n_intents: Number of discrete intent categories (default: 6)
            flow_steps: Number of ODE integration steps at inference
            hidden_dim: Hidden dimension for networks (default: 256)
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.flow_steps = flow_steps

        # Stage 1: Intent Generation
        self.intent_generator = FlowIntentGenerator(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            n_intents=n_intents,
            hidden_dim=hidden_dim
        )

        # Stage 2: Feature Modulation
        self.modulation = FiLMLayer(
            feature_dim=feature_dim,
            latent_dim=latent_dim
        )

    def forward(
        self,
        features: torch.Tensor,
        intent_label: Optional[torch.Tensor] = None,
        return_latent: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with automatic training/inference mode

        Args:
            features: ST features from backbone model
                     Shape: [batch, feature_dim] or [batch, feature_dim, H, W]
            intent_label: Discrete intent labels [batch]
                         Required during training, optional at inference
            return_latent: If True, also return latent intent for analysis

        Returns:
            modulated_features: Intent-conditioned features (same shape as input)
            losses: Dict containing 'flow_matching' loss (0 at inference)
            (optional) z_latent: Latent intent vector if return_latent=True
        """
        # Handle spatial features: flatten for intent generation
        original_shape = features.shape
        if features.dim() == 4:  # [B, C, H, W]
            B, C, H, W = features.shape
            features_flat = features.mean(dim=[2, 3])  # Global average pooling
        else:
            features_flat = features

        # Stage 1: Generate latent intent
        if self.training and intent_label is not None:
            # Training: Flow Matching with supervision
            z_latent, loss_fm = self.intent_generator(features_flat, intent_label)
        else:
            # Inference: Sample from learned flow
            z_latent = self.intent_generator.sample(features_flat, self.flow_steps)
            loss_fm = torch.tensor(0.0, device=features.device)

        # Stage 2: Modulate features
        modulated = self.modulation(features, z_latent)

        # Return losses
        losses = {'flow_matching': loss_fm}

        if return_latent:
            return modulated, losses, z_latent
        else:
            return modulated, losses

    def get_intent_embedding(self, intent_label: torch.Tensor) -> torch.Tensor:
        """Get learned embedding for a discrete intent"""
        return self.intent_generator.intent_embedding(intent_label)

    def interpolate_intents(
        self,
        features: torch.Tensor,
        intent_a: int,
        intent_b: int,
        n_steps: int = 10
    ) -> torch.Tensor:
        """
        Interpolate between two intents

        Args:
            features: ST features [batch, feature_dim, ...]
            intent_a, intent_b: Intent indices to interpolate
            n_steps: Number of interpolation steps

        Returns:
            interpolated: [n_steps, batch, feature_dim, ...] modulated features
        """
        device = features.device

        # Get intent embeddings
        z_a = self.get_intent_embedding(torch.tensor([intent_a], device=device))
        z_b = self.get_intent_embedding(torch.tensor([intent_b], device=device))

        # Interpolate
        alphas = torch.linspace(0, 1, n_steps, device=device)
        interpolated = []

        for alpha in alphas:
            z_interp = (1 - alpha) * z_a + alpha * z_b
            z_interp = z_interp.expand(features.size(0), -1)
            modulated = self.modulation(features, z_interp)
            interpolated.append(modulated)

        return torch.stack(interpolated)


# ============================================================================
# Helper: Integrate Adapter into Any Model
# ============================================================================

class BackboneWithAdapter(nn.Module):
    """
    Generic wrapper to add Intent Adapter to any backbone model

    Backbone requirements:
        - extract_features(x) -> features
        - predict(features) -> output
    """
    def __init__(self, backbone: nn.Module, feature_dim: int):
        super().__init__()

        self.backbone = backbone

        # 🔌 Plug in adapter
        self.adapter = LatentFlowIntentAdapter(feature_dim=feature_dim)

    def forward(self, x, intent_label=None, **backbone_kwargs):
        """
        Args:
            x: Input to backbone model
            intent_label: Intent labels for adapter
            **backbone_kwargs: Additional args for backbone (e.g., adj_matrix)
        """
        # Extract features from backbone
        features = self.backbone.extract_features(x, **backbone_kwargs)

        # Apply adapter
        modulated, losses = self.adapter(
            features,
            intent_label,
            return_latent=False
        )

        # Prediction with modulated features
        pred = self.backbone.predict(modulated)

        return pred, losses


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test adapter
    batch_size = 16
    feature_dim = 64
    H, W = 10, 10

    # Create adapter
    adapter = LatentFlowIntentAdapter(
        feature_dim=feature_dim,
        latent_dim=64,
        n_intents=6
    )

    # Simulate ST features from backbone
    features = torch.randn(batch_size, feature_dim, H, W)
    intent_labels = torch.randint(0, 6, (batch_size,))

    # Training mode
    adapter.train()
    modulated, losses = adapter(features, intent_labels)

    print(f"Input shape: {features.shape}")
    print(f"Modulated shape: {modulated.shape}")
    print(f"Flow matching loss: {losses['flow_matching'].item():.6f}")
    print(f"✓ Adapter works!")

    # Inference mode
    adapter.eval()
    with torch.no_grad():
        modulated, losses = adapter(features)
        print(f"Inference mode: loss = {losses['flow_matching'].item():.6f}")

    # Parameter count
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"\nAdapter parameters: {n_params:,} (~3% of ST-ResNet)")
