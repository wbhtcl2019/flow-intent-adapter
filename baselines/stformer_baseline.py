"""
STFormer (Spatio-Temporal Transformer) Baseline

Reference:
  Xu et al. "Spatial-Temporal Transformer Networks for Traffic Flow Forecasting"
  (AAAI 2022, arXiv:2001.02908)

Architecture: Pure Transformer with spatial-temporal positional encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for Transformer
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class SpatialPositionalEncoding(nn.Module):
    """
    Learnable spatial positional encoding for grid locations
    """
    def __init__(self, num_nodes, d_model):
        super().__init__()
        self.spatial_embed = nn.Parameter(torch.randn(num_nodes, d_model))

    def forward(self, x):
        """
        Args:
            x: [batch, num_nodes, d_model]
        Returns:
            x with spatial positional encoding added
        """
        return x + self.spatial_embed.unsqueeze(0)


class MultiHeadSpatialAttention(nn.Module):
    """
    Multi-Head Attention for spatial dimension

    Computes attention across different spatial locations
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: [batch, num_nodes, d_model]
        Returns:
            output: [batch, num_nodes, d_model]
        """
        batch_size, num_nodes, d_model = x.size()

        # Residual connection
        residual = x

        # Linear projections
        Q = self.W_q(x).view(batch_size, num_nodes, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, num_nodes, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, num_nodes, self.num_heads, self.d_k).transpose(1, 2)
        # Q, K, V: [batch, num_heads, num_nodes, d_k]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        # attn_output: [batch, num_heads, num_nodes, d_k]

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_nodes, d_model)

        # Output projection
        output = self.W_o(attn_output)

        # Residual + Layer norm
        output = self.layer_norm(residual + self.dropout(output))

        return output


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-Head Attention for temporal dimension

    Computes attention across different time steps
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()

        residual = x

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        output = self.W_o(attn_output)
        output = self.layer_norm(residual + self.dropout(output))

        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: [batch, *, d_model]
        """
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.layer_norm(residual + self.dropout(x))
        return x


class STTransformerBlock(nn.Module):
    """
    Spatio-Temporal Transformer Block

    Architecture:
      1. Spatial Attention (across locations)
      2. Temporal Attention (across time)
      3. Feed-Forward
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.spatial_attn = MultiHeadSpatialAttention(d_model, num_heads, dropout)
        self.temporal_attn = MultiHeadTemporalAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, num_nodes, d_model]
        Returns:
            output: [batch, seq_len, num_nodes, d_model]
        """
        batch_size, seq_len, num_nodes, d_model = x.size()

        # Spatial attention (across nodes at each time step)
        x_spatial = x.view(batch_size * seq_len, num_nodes, d_model)
        x_spatial = self.spatial_attn(x_spatial)
        x = x_spatial.view(batch_size, seq_len, num_nodes, d_model)

        # Temporal attention (across time for each node)
        x_temporal = x.permute(0, 2, 1, 3).contiguous()  # [batch, num_nodes, seq_len, d_model]
        x_temporal = x_temporal.view(batch_size * num_nodes, seq_len, d_model)
        x_temporal = self.temporal_attn(x_temporal)
        x = x_temporal.view(batch_size, num_nodes, seq_len, d_model).permute(0, 2, 1, 3)

        # Feed-forward
        x = x.view(batch_size * seq_len * num_nodes, d_model)
        x = self.feed_forward(x)
        x = x.view(batch_size, seq_len, num_nodes, d_model)

        return x


class STFormer(nn.Module):
    """
    Spatio-Temporal Transformer

    Args:
        num_nodes: Number of spatial locations
        in_channels: Input feature dimension (e.g., 2 for inflow/outflow)
        out_channels: Output feature dimension
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        d_ff: Feed-forward hidden dimension
        seq_len: Input sequence length
        horizon: Prediction horizon
        dropout: Dropout rate
    """
    def __init__(self, num_nodes, in_channels=2, out_channels=2,
                 d_model=64, num_heads=4, num_layers=3, d_ff=256,
                 seq_len=12, horizon=1, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.horizon = horizon

        # Input embedding
        self.input_embed = nn.Linear(in_channels, d_model)

        # Positional encodings
        self.temporal_pe = PositionalEncoding(d_model, max_len=seq_len)
        self.spatial_pe = SpatialPositionalEncoding(num_nodes, d_model)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            STTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, out_channels * horizon)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, num_nodes, in_channels]
        Returns:
            output: [batch, horizon, num_nodes, out_channels]
        """
        batch_size, seq_len, num_nodes, in_channels = x.size()

        # Input embedding
        x = self.input_embed(x)  # [batch, seq_len, num_nodes, d_model]

        # Add positional encodings
        # Temporal PE (add to each time step)
        x_reshaped = x.view(batch_size * num_nodes, seq_len, self.d_model)
        x_reshaped = self.temporal_pe(x_reshaped)
        x = x_reshaped.view(batch_size, seq_len, num_nodes, self.d_model)

        # Spatial PE (add to each location)
        x_reshaped = x.view(batch_size * seq_len, num_nodes, self.d_model)
        x_reshaped = self.spatial_pe(x_reshaped)
        x = x_reshaped.view(batch_size, seq_len, num_nodes, self.d_model)

        x = self.dropout(x)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Use last time step for prediction
        x_last = x[:, -1, :, :]  # [batch, num_nodes, d_model]

        # Output projection
        output = self.output_proj(x_last)  # [batch, num_nodes, out_channels * horizon]
        output = output.view(batch_size, num_nodes, self.horizon, -1)
        output = output.permute(0, 2, 1, 3)  # [batch, horizon, num_nodes, out_channels]

        return output


class STFormerWithFlowAdapter(nn.Module):
    """
    STFormer + Flow Intent Adapter

    Integration Strategy:
      Apply adapter to the output of the last Transformer block
    """
    def __init__(self, num_nodes, in_channels=2, out_channels=2,
                 d_model=64, num_heads=4, num_layers=3, d_ff=256,
                 seq_len=12, horizon=1, dropout=0.1,
                 use_adapter=True, latent_dim=64, n_intents=6):
        super().__init__()
        self.use_adapter = use_adapter
        self.num_nodes = num_nodes

        # Base STFormer
        self.stformer = STFormer(
            num_nodes, in_channels, out_channels,
            d_model, num_heads, num_layers, d_ff,
            seq_len, horizon, dropout
        )

        # Flow Intent Adapter
        if use_adapter:
            from latent_flow_intent_adapter import LatentFlowIntentAdapter

            # Apply adapter to last time step features
            self.adapter = LatentFlowIntentAdapter(
                feature_dim=d_model,
                latent_dim=latent_dim,
                n_intents=n_intents
            )

    def forward(self, x, intent_label=None):
        """
        Args:
            x: [batch, seq_len, num_nodes, in_channels]
            intent_label: [batch]
        Returns:
            output: [batch, horizon, num_nodes, out_channels]
            losses: dict
        """
        losses = {}

        batch_size, seq_len, num_nodes, in_channels = x.size()

        # Input embedding
        x_embed = self.stformer.input_embed(x)

        # Add positional encodings
        x_embed_t = x_embed.view(batch_size * num_nodes, seq_len, self.stformer.d_model)
        x_embed_t = self.stformer.temporal_pe(x_embed_t)
        x_embed = x_embed_t.view(batch_size, seq_len, num_nodes, self.stformer.d_model)

        x_embed_s = x_embed.view(batch_size * seq_len, num_nodes, self.stformer.d_model)
        x_embed_s = self.stformer.spatial_pe(x_embed_s)
        x_embed = x_embed_s.view(batch_size, seq_len, num_nodes, self.stformer.d_model)

        x_embed = self.stformer.dropout(x_embed)

        # Transformer blocks
        for block in self.stformer.transformer_blocks:
            x_embed = block(x_embed)

        # Last time step
        x_last = x_embed[:, -1, :, :]  # [batch, num_nodes, d_model]

        # Apply adapter
        if self.use_adapter and intent_label is not None:
            # Flatten for adapter
            x_flat = x_last.reshape(batch_size, -1)  # [batch, num_nodes * d_model]

            x_mod, adapter_losses = self.adapter(
                x_flat,
                intent_label,
                training=self.training
            )

            # Reshape back
            x_last = x_mod.reshape(batch_size, num_nodes, self.stformer.d_model)

            if 'flow_loss' in adapter_losses:
                losses['flow_loss'] = adapter_losses['flow_loss']

        # Output projection
        output = self.stformer.output_proj(x_last)
        output = output.view(batch_size, num_nodes, self.stformer.horizon, -1)
        output = output.permute(0, 2, 1, 3)

        return output, losses


def test_stformer():
    """Quick test"""
    batch_size = 4
    seq_len = 12
    horizon = 1
    num_nodes = 1024  # 32×32 grid
    in_channels = 2
    out_channels = 2

    # Create model
    model = STFormerWithFlowAdapter(
        num_nodes=num_nodes,
        in_channels=in_channels,
        out_channels=out_channels,
        d_model=64,
        num_heads=4,
        num_layers=3,
        d_ff=256,
        seq_len=seq_len,
        horizon=horizon,
        use_adapter=True
    )

    # Dummy data
    x = torch.randn(batch_size, seq_len, num_nodes, in_channels)
    intent_labels = torch.randint(0, 6, (batch_size,))

    # Forward
    y_pred, losses = model(x, intent_labels)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_pred.shape}")
    print(f"Losses: {losses}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_stformer()
