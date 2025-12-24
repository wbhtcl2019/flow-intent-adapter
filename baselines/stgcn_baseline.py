"""
STGCN (Spatio-Temporal Graph Convolutional Network) Baseline

Reference:
  Yu et al. "Spatio-Temporal Graph Convolutional Networks: A Deep Learning
  Framework for Traffic Forecasting" (IJCAI 2018)

Architecture: (Temporal Conv → Spatial Graph Conv → Temporal Conv) × N
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalConvLayer(nn.Module):
    """
    Temporal Convolution Layer (GLU activation)

    Uses gated linear units: output = (X * W1 + b1) ⊙ σ(X * W2 + b2)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        # Two convolutions for GLU
        self.conv_gate = nn.Conv2d(
            in_channels, 2 * out_channels, (1, kernel_size),
            padding=(0, (kernel_size - 1) // 2)
        )

    def forward(self, X):
        """
        Args:
            X: [batch, in_channels, num_nodes, time_steps]

        Returns:
            output: [batch, out_channels, num_nodes, time_steps]
        """
        # Apply convolution
        X_conv = self.conv_gate(X)

        # Split for GLU
        P, Q = torch.split(X_conv, self.out_channels, dim=1)

        # GLU: P ⊙ σ(Q)
        output = P * torch.sigmoid(Q)

        return output


class SpatialGraphConvLayer(nn.Module):
    """
    Spatial Graph Convolution (Chebyshev polynomial approximation)

    Implements: g_θ ⋆ x = Σ_{k=0}^{K-1} θ_k T_k(L̃) x

    where T_k is k-th Chebyshev polynomial, L̃ is normalized Laplacian
    """
    def __init__(self, in_channels, out_channels, K=3):
        super().__init__()
        self.K = K
        self.out_channels = out_channels

        # Learnable parameters for each polynomial order
        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X, L_tilde):
        """
        Args:
            X: [batch, in_channels, num_nodes, time_steps]
            L_tilde: [num_nodes, num_nodes] - scaled Laplacian

        Returns:
            output: [batch, out_channels, num_nodes, time_steps]
        """
        batch, in_channels, num_nodes, time_steps = X.shape

        # Reshape for matrix multiplication
        X_reshaped = X.permute(0, 3, 2, 1).contiguous()  # [batch, time, nodes, channels]
        X_reshaped = X_reshaped.view(batch * time_steps, num_nodes, in_channels)

        # Compute Chebyshev polynomials
        # T_0(L̃) = I
        # T_1(L̃) = L̃
        # T_k(L̃) = 2L̃T_{k-1}(L̃) - T_{k-2}(L̃)
        chebyshev_polys = []
        T_0 = X_reshaped  # [batch * time, nodes, channels]
        chebyshev_polys.append(T_0)

        if self.K > 1:
            T_1 = torch.matmul(L_tilde, T_0)  # [batch * time, nodes, channels]
            chebyshev_polys.append(T_1)

        for k in range(2, self.K):
            T_k = 2 * torch.matmul(L_tilde, chebyshev_polys[k-1]) - chebyshev_polys[k-2]
            chebyshev_polys.append(T_k)

        # Apply weights and sum
        output = torch.zeros(batch * time_steps, num_nodes, self.out_channels).to(X.device)
        for k in range(self.K):
            # chebyshev_polys[k]: [batch * time, nodes, in_channels]
            # weight[k]: [in_channels, out_channels]
            output += torch.matmul(chebyshev_polys[k], self.weight[k])

        # Reshape back
        output = output.view(batch, time_steps, num_nodes, self.out_channels)
        output = output.permute(0, 3, 2, 1).contiguous()  # [batch, out_channels, nodes, time]

        return output


class STConvBlock(nn.Module):
    """
    Spatio-Temporal Convolutional Block

    Architecture: Temporal Conv → Graph Conv → Temporal Conv → Layer Norm
    """
    def __init__(self, in_channels, out_channels, num_nodes, temporal_kernel=3, K=3):
        super().__init__()
        self.num_nodes = num_nodes

        # Two temporal convs + one spatial conv
        self.temporal_conv1 = TemporalConvLayer(in_channels, out_channels, temporal_kernel)
        self.spatial_conv = SpatialGraphConvLayer(out_channels, out_channels, K)
        self.temporal_conv2 = TemporalConvLayer(out_channels, out_channels, temporal_kernel)

        # Layer normalization
        self.layer_norm = nn.LayerNorm([num_nodes, out_channels])

        # Residual connection (if dimensions match)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, (1, 1)) if in_channels != out_channels else None

    def forward(self, X, L_tilde):
        """
        Args:
            X: [batch, in_channels, num_nodes, time_steps]
            L_tilde: [num_nodes, num_nodes]

        Returns:
            output: [batch, out_channels, num_nodes, time_steps]
        """
        # Temporal
        X_t1 = self.temporal_conv1(X)

        # Spatial
        X_s = self.spatial_conv(X_t1, L_tilde)

        # Temporal
        X_t2 = self.temporal_conv2(X_s)

        # Residual
        if self.residual_conv is not None:
            X_res = self.residual_conv(X)
        else:
            X_res = X

        output = X_t2 + X_res

        # Layer norm: [batch, channels, nodes, time] → [batch, time, nodes, channels]
        output = output.permute(0, 3, 2, 1)
        output = self.layer_norm(output)
        output = output.permute(0, 3, 2, 1)  # Back to [batch, channels, nodes, time]

        return output


class STGCN(nn.Module):
    """
    Complete STGCN Model

    Args:
        num_nodes: Number of spatial locations (e.g., 1024 for 32×32 grid)
        in_channels: Input feature dimension (e.g., 2 for inflow/outflow)
        out_channels: Output feature dimension
        temporal_len: Input temporal length
        horizon: Prediction horizon
        hidden_channels: Hidden layer dimensions
        num_blocks: Number of ST-Conv blocks
        K: Order of Chebyshev polynomials
    """
    def __init__(self, num_nodes, in_channels=2, out_channels=2,
                 temporal_len=12, horizon=1, hidden_channels=64,
                 num_blocks=2, K=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.temporal_len = temporal_len
        self.horizon = horizon

        # ST-Conv blocks
        self.st_blocks = nn.ModuleList()
        channels = [in_channels] + [hidden_channels] * num_blocks

        for i in range(num_blocks):
            self.st_blocks.append(
                STConvBlock(channels[i], channels[i+1], num_nodes, K=K)
            )

        # Output layer (fully connected on last time step)
        self.output_conv = nn.Conv2d(
            hidden_channels, out_channels * horizon,
            kernel_size=(1, temporal_len)
        )

    def forward(self, X, L_tilde):
        """
        Args:
            X: [batch, temporal_len, num_nodes, in_channels]
            L_tilde: [num_nodes, num_nodes] - scaled Laplacian

        Returns:
            Y_pred: [batch, horizon, num_nodes, out_channels]
        """
        # Reshape to [batch, channels, nodes, time]
        X = X.permute(0, 3, 2, 1)

        # Pass through ST-Conv blocks
        for block in self.st_blocks:
            X = block(X, L_tilde)

        # Output projection
        X_out = self.output_conv(X)  # [batch, out_channels * horizon, nodes, 1]

        # Reshape to [batch, horizon, nodes, out_channels]
        X_out = X_out.squeeze(-1)  # [batch, out_channels * horizon, nodes]
        X_out = X_out.view(-1, self.horizon, self.out_channels, self.num_nodes)
        X_out = X_out.permute(0, 1, 3, 2)  # [batch, horizon, nodes, out_channels]

        return X_out


class STGCNWithFlowAdapter(nn.Module):
    """
    STGCN + Flow Intent Adapter

    Integration Strategy:
      Apply adapter to intermediate features in ST-Conv blocks
    """
    def __init__(self, num_nodes, in_channels=2, out_channels=2,
                 temporal_len=12, horizon=1, hidden_channels=64,
                 num_blocks=2, K=3, use_adapter=True,
                 latent_dim=64, n_intents=6):
        super().__init__()
        self.use_adapter = use_adapter
        self.num_blocks = num_blocks

        # Base STGCN
        self.stgcn = STGCN(num_nodes, in_channels, out_channels,
                          temporal_len, horizon, hidden_channels,
                          num_blocks, K)

        # Flow Intent Adapter
        if use_adapter:
            from latent_flow_intent_adapter import LatentFlowIntentAdapter

            # Apply adapter after first ST block
            self.adapter = LatentFlowIntentAdapter(
                feature_dim=hidden_channels,
                latent_dim=latent_dim,
                n_intents=n_intents
            )

    def forward(self, X, L_tilde, intent_label=None):
        """
        Args:
            X: [batch, temporal_len, num_nodes, in_channels]
            L_tilde: [num_nodes, num_nodes]
            intent_label: [batch]

        Returns:
            Y_pred: [batch, horizon, num_nodes, out_channels]
            losses: dict
        """
        losses = {}

        # Reshape
        X_conv = X.permute(0, 3, 2, 1)  # [batch, channels, nodes, time]

        # Pass through ST blocks
        for i, block in enumerate(self.stgcn.st_blocks):
            X_conv = block(X_conv, L_tilde)

            # Apply adapter after first block
            if self.use_adapter and i == 0 and intent_label is not None:
                batch, channels, num_nodes, time_steps = X_conv.shape

                # Reshape for adapter (aggregate temporal)
                X_agg = X_conv.mean(dim=-1)  # [batch, channels, nodes]
                X_flat = X_agg.permute(0, 2, 1).reshape(batch, -1)  # [batch, nodes * channels]

                # Apply adapter
                X_mod_flat, adapter_losses = self.adapter(
                    X_flat,
                    intent_label,
                    training=self.training
                )

                # Reshape back
                X_mod = X_mod_flat.reshape(batch, num_nodes, channels).permute(0, 2, 1)  # [batch, channels, nodes]

                # Broadcast to temporal dimension
                X_conv = X_mod.unsqueeze(-1).expand(-1, -1, -1, time_steps)

                if 'flow_loss' in adapter_losses:
                    losses['flow_loss'] = adapter_losses['flow_loss']

        # Output projection
        Y_pred = self.stgcn.output_conv(X_conv)
        Y_pred = Y_pred.squeeze(-1)
        Y_pred = Y_pred.view(-1, self.stgcn.horizon, self.stgcn.out_channels, self.stgcn.num_nodes)
        Y_pred = Y_pred.permute(0, 1, 3, 2)

        return Y_pred, losses


# ============================================================================
# Utility Functions
# ============================================================================

def compute_laplacian(adj_matrix, normalized=True):
    """
    Compute graph Laplacian matrix

    Args:
        adj_matrix: [num_nodes, num_nodes] adjacency matrix
        normalized: if True, use normalized Laplacian

    Returns:
        L: Laplacian matrix (unnormalized or normalized)
    """
    # Degree matrix
    D = torch.diag(adj_matrix.sum(dim=1))

    if normalized:
        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diag() + 1e-6))
        L = torch.eye(adj_matrix.size(0)) - torch.mm(torch.mm(D_inv_sqrt, adj_matrix), D_inv_sqrt)
    else:
        # Unnormalized Laplacian: L = D - A
        L = D - adj_matrix

    return L


def scale_laplacian(L, lambda_max=2):
    """
    Scale Laplacian to [-1, 1] for Chebyshev polynomial

    L_tilde = (2 / λ_max) * L - I
    """
    num_nodes = L.size(0)
    L_scaled = (2.0 / lambda_max) * L - torch.eye(num_nodes)
    return L_scaled


def construct_graph_from_grid(grid_size=32):
    """
    Construct adjacency matrix and Laplacian for grid network
    """
    num_nodes = grid_size * grid_size

    # Adjacency (4-neighbor)
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(grid_size):
        for j in range(grid_size):
            node = i * grid_size + j

            # Neighbors
            neighbors = []
            if j < grid_size - 1:
                neighbors.append(i * grid_size + (j + 1))  # Right
            if i < grid_size - 1:
                neighbors.append((i + 1) * grid_size + j)  # Down
            if j > 0:
                neighbors.append(i * grid_size + (j - 1))  # Left
            if i > 0:
                neighbors.append((i - 1) * grid_size + j)  # Up

            for neighbor in neighbors:
                adj[node, neighbor] = 1

    adj = torch.FloatTensor(adj)

    # Compute Laplacian
    L = compute_laplacian(adj, normalized=True)

    # Scale for Chebyshev
    L_tilde = scale_laplacian(L, lambda_max=2.0)

    return adj, L_tilde


def test_stgcn():
    """Quick test"""
    batch_size = 4
    temporal_len = 12
    horizon = 1
    num_nodes = 1024  # 32×32 grid
    in_channels = 2
    out_channels = 2

    # Create model
    model = STGCNWithFlowAdapter(
        num_nodes=num_nodes,
        in_channels=in_channels,
        out_channels=out_channels,
        temporal_len=temporal_len,
        horizon=horizon,
        hidden_channels=64,
        num_blocks=2,
        use_adapter=True
    )

    # Dummy data
    X = torch.randn(batch_size, temporal_len, num_nodes, in_channels)
    intent_labels = torch.randint(0, 6, (batch_size,))

    # Construct graph
    _, L_tilde = construct_graph_from_grid(grid_size=32)

    # Forward
    Y_pred, losses = model(X, L_tilde, intent_labels)

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {Y_pred.shape}")
    print(f"Losses: {losses}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_stgcn()
