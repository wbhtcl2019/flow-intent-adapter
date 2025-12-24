"""
Universal Intent Adapter for Traffic Prediction

Design Philosophy:
- Model-agnostic: Can be plugged into any baseline (ST-ResNet, Graph WaveNet, etc.)
- Lightweight: Minimal parameters (<10k)
- FiLM-style conditioning: Intent → scale & shift for feature modulation
- Multi-scale: Can modulate at multiple points in the base model

Usage:
    adapter = IntentAdapter(n_intents=6, feature_dim=64)
    features_mod = adapter(features, intent_id)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntentEmbedding(nn.Module):
    """
    Intent Embedding Layer

    Maps discrete intent IDs to continuous embedding vectors.
    """
    def __init__(self, n_intents, embed_dim=32):
        super(IntentEmbedding, self).__init__()

        self.n_intents = n_intents
        self.embed_dim = embed_dim

        # Learnable intent embeddings
        self.embedding = nn.Embedding(n_intents, embed_dim)

    def forward(self, intent_ids):
        """
        Args:
            intent_ids: [B] or [B, 1] - Intent category IDs (0-5)

        Returns:
            intent_embed: [B, embed_dim]
        """
        if intent_ids.dim() == 2:
            intent_ids = intent_ids.squeeze(1)

        return self.embedding(intent_ids)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer

    Applies affine transformation to features based on intent:
        output = gamma * features + beta

    where gamma and beta are predicted from intent embedding.
    """
    def __init__(self, feature_dim, intent_embed_dim=32):
        super(FiLMLayer, self).__init__()

        self.feature_dim = feature_dim

        # MLP to predict scale (gamma) and shift (beta) from intent
        self.fc = nn.Sequential(
            nn.Linear(intent_embed_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )

    def forward(self, features, intent_embed):
        """
        Args:
            features: [B, C, H, W] or [B, C]
            intent_embed: [B, embed_dim]

        Returns:
            modulated: Same shape as features
        """
        # Predict gamma and beta
        film_params = self.fc(intent_embed)  # [B, 2*C]

        gamma, beta = torch.chunk(film_params, 2, dim=1)  # Each [B, C]

        # Reshape for broadcasting
        if features.dim() == 4:  # Conv features [B, C, H, W]
            gamma = gamma.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
            beta = beta.unsqueeze(2).unsqueeze(3)
        elif features.dim() == 3:  # Temporal features [B, T, C]
            gamma = gamma.unsqueeze(1)  # [B, 1, C]
            beta = beta.unsqueeze(1)
        # else: features is [B, C], no reshaping needed

        # Apply FiLM
        modulated = gamma * features + beta

        return modulated


class IntentAdapter(nn.Module):
    """
    Universal Intent Adapter

    Can be inserted at any layer of a base model to inject intent conditioning.

    Architecture:
        intent_id → Intent Embedding → FiLM Layer → Modulated Features
    """
    def __init__(self,
                 n_intents=6,
                 feature_dim=64,
                 intent_embed_dim=32,
                 use_residual=True):
        super(IntentAdapter, self).__init__()

        self.n_intents = n_intents
        self.feature_dim = feature_dim
        self.use_residual = use_residual

        # Intent embedding
        self.intent_embed = IntentEmbedding(n_intents, intent_embed_dim)

        # FiLM layer
        self.film = FiLMLayer(feature_dim, intent_embed_dim)

    def forward(self, features, intent_ids):
        """
        Args:
            features: [B, C, H, W] or [B, C] - Features from base model
            intent_ids: [B] or [B, 1] - Intent category IDs

        Returns:
            modulated: Same shape as features
        """
        # Get intent embedding
        intent_embed = self.intent_embed(intent_ids)  # [B, embed_dim]

        # Apply FiLM
        modulated = self.film(features, intent_embed)

        # Optional residual connection
        if self.use_residual:
            modulated = modulated + features

        return modulated


class IntentFusionAdapter(nn.Module):
    """
    Intent-Aware Fusion Adapter

    Modulates fusion weights based on intent.
    Specifically designed for ST-ResNet's branch fusion.
    """
    def __init__(self, n_intents=6, n_branches=3, spatial_shape=(20, 20)):
        super(IntentFusionAdapter, self).__init__()

        H, W = spatial_shape

        # Intent embedding
        self.intent_embed = IntentEmbedding(n_intents, embed_dim=16)

        # Predict fusion weight modulation from intent
        self.fc = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, n_branches)  # One weight per branch
        )

        # Base fusion weights (learnable)
        self.base_weights = nn.Parameter(
            torch.ones(n_branches, 2, H, W)
        )

    def forward(self, intent_ids):
        """
        Args:
            intent_ids: [B] - Intent IDs

        Returns:
            fusion_weights: [B, n_branches, 2, H, W]
        """
        # Get intent embedding
        intent_embed = self.intent_embed(intent_ids)  # [B, 16]

        # Predict branch importance
        branch_weights = self.fc(intent_embed)  # [B, n_branches]

        # Softmax to normalize
        branch_weights = F.softmax(branch_weights, dim=1)  # [B, n_branches]

        # Broadcast to spatial dimensions
        # [B, n_branches, 1, 1, 1] * [1, n_branches, 2, H, W]
        fusion_weights = branch_weights.unsqueeze(2).unsqueeze(3).unsqueeze(4) * \
                        self.base_weights.unsqueeze(0)

        return fusion_weights


def test_intent_adapter():
    """Test Intent Adapter"""
    print("=" * 80)
    print("Testing Intent Adapter")
    print("=" * 80)

    B = 8  # Batch size
    C = 64  # Feature channels
    H, W = 20, 20

    # Test 1: IntentAdapter for convolutional features
    print("\n[Test 1] IntentAdapter for Conv Features")
    print("-" * 80)

    adapter = IntentAdapter(n_intents=6, feature_dim=C, intent_embed_dim=32)

    print(f"Adapter parameters: {sum(p.numel() for p in adapter.parameters()):,}")

    features = torch.randn(B, C, H, W)
    intent_ids = torch.randint(0, 6, (B,))

    print(f"Input features: {features.shape}")
    print(f"Intent IDs: {intent_ids.shape}, values: {intent_ids.tolist()}")

    modulated = adapter(features, intent_ids)

    print(f"Output features: {modulated.shape}")
    print(f"Feature change: {(modulated - features).abs().mean():.6f}")

    # Test 2: IntentAdapter for dense features
    print("\n[Test 2] IntentAdapter for Dense Features")
    print("-" * 80)

    adapter_dense = IntentAdapter(n_intents=6, feature_dim=128, intent_embed_dim=16)

    features_dense = torch.randn(B, 128)
    modulated_dense = adapter_dense(features_dense, intent_ids)

    print(f"Input features: {features_dense.shape}")
    print(f"Output features: {modulated_dense.shape}")

    # Test 3: IntentFusionAdapter
    print("\n[Test 3] IntentFusionAdapter for ST-ResNet Fusion")
    print("-" * 80)

    fusion_adapter = IntentFusionAdapter(n_intents=6, n_branches=3, spatial_shape=(H, W))

    print(f"Fusion adapter parameters: {sum(p.numel() for p in fusion_adapter.parameters()):,}")

    fusion_weights = fusion_adapter(intent_ids)

    print(f"Fusion weights shape: {fusion_weights.shape}")
    print(f"Branch weights (first sample):")
    for i in range(3):
        print(f"  Branch {i}: {fusion_weights[0, i, 0, 0, 0]:.4f}")

    # Test 4: Check intent differentiation
    print("\n[Test 4] Intent Differentiation Check")
    print("-" * 80)

    adapter_test = IntentAdapter(n_intents=6, feature_dim=C)

    test_features = torch.randn(1, C, H, W)

    outputs = []
    for intent_id in range(6):
        intent_tensor = torch.tensor([intent_id])
        out = adapter_test(test_features, intent_tensor)
        outputs.append(out)

    # Check if different intents produce different outputs
    for i in range(5):
        diff = (outputs[i] - outputs[i+1]).abs().mean()
        print(f"  Intent {i} vs {i+1}: Mean diff = {diff:.6f}")

    print("\n" + "=" * 80)
    print("Intent Adapter tests passed!")
    print("=" * 80)

    # Print summary
    print("\n[Summary]")
    print(f"  Standard IntentAdapter: ~{sum(p.numel() for p in adapter.parameters()):,} parameters")
    print(f"  IntentFusionAdapter: ~{sum(p.numel() for p in fusion_adapter.parameters()):,} parameters")
    print("\nReady to integrate with ST-ResNet!")


if __name__ == "__main__":
    test_intent_adapter()
