"""
DCRNN (Diffusion Convolutional Recurrent Neural Network) Baseline

Reference:
  Li et al. "Diffusion Convolutional Recurrent Neural Network: Data-Driven
  Traffic Forecasting" (KDD 2018)

This implementation can work with/without Flow Intent Adapter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiffusionGraphConv(nn.Module):
    """
    Diffusion Graph Convolution Layer

    Implements K-hop diffusion:
      DO = sum_{k=0}^{K-1} (D_O)^k / k!
    """
    def __init__(self, input_dim, output_dim, num_nodes, K=2, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.K = K

        # Learnable weights for each diffusion step
        self.weights = nn.Parameter(torch.Tensor(K, input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, X, adj_matrix):
        """
        Args:
            X: [batch, num_nodes, input_dim]
            adj_matrix: [num_nodes, num_nodes] (transition matrix)

        Returns:
            output: [batch, num_nodes, output_dim]
        """
        batch_size = X.size(0)

        # Random walk diffusion
        supports = []
        support = X  # k=0
        supports.append(support)

        for k in range(1, self.K):
            # D^k X = D * (D^{k-1} X)
            support = torch.einsum('ij,bjd->bid', adj_matrix, support)
            supports.append(support)

        # Weighted sum
        output = torch.zeros(batch_size, self.num_nodes, self.output_dim).to(X.device)
        for k in range(self.K):
            output += torch.einsum('bid,do->bio', supports[k], self.weights[k])

        if self.bias is not None:
            output += self.bias

        return output


class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional GRU Cell
    """
    def __init__(self, input_dim, hidden_dim, num_nodes, K=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        # Gates: update gate (z) and reset gate (r)
        self.graph_conv_z = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim, num_nodes, K)
        self.graph_conv_r = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim, num_nodes, K)

        # Candidate hidden state
        self.graph_conv_h = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim, num_nodes, K)

    def forward(self, X, H, adj_matrix):
        """
        Args:
            X: [batch, num_nodes, input_dim]
            H: [batch, num_nodes, hidden_dim]
            adj_matrix: [num_nodes, num_nodes]

        Returns:
            H_new: [batch, num_nodes, hidden_dim]
        """
        # Concatenate input and hidden state
        combined = torch.cat([X, H], dim=-1)  # [batch, num_nodes, input_dim + hidden_dim]

        # Update gate
        z = torch.sigmoid(self.graph_conv_z(combined, adj_matrix))

        # Reset gate
        r = torch.sigmoid(self.graph_conv_r(combined, adj_matrix))

        # Candidate hidden state
        combined_r = torch.cat([X, r * H], dim=-1)
        h_tilde = torch.tanh(self.graph_conv_h(combined_r, adj_matrix))

        # New hidden state
        H_new = z * H + (1 - z) * h_tilde

        return H_new


class DCRNNEncoder(nn.Module):
    """
    DCRNN Encoder: Seq2Seq encoder
    """
    def __init__(self, input_dim, hidden_dim, num_nodes, num_layers=2, K=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers

        # Stack of DCGRU cells
        self.dcgru_cells = nn.ModuleList([
            DCGRUCell(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                num_nodes,
                K
            ) for i in range(num_layers)
        ])

    def forward(self, X, adj_matrix):
        """
        Args:
            X: [batch, seq_len, num_nodes, input_dim]
            adj_matrix: [num_nodes, num_nodes]

        Returns:
            H: [num_layers, batch, num_nodes, hidden_dim]
        """
        batch_size, seq_len, num_nodes, _ = X.size()

        # Initialize hidden states
        H = [torch.zeros(batch_size, num_nodes, self.hidden_dim).to(X.device)
             for _ in range(self.num_layers)]

        # Encode sequence
        for t in range(seq_len):
            X_t = X[:, t, :, :]  # [batch, num_nodes, input_dim]

            for layer in range(self.num_layers):
                H[layer] = self.dcgru_cells[layer](X_t, H[layer], adj_matrix)
                X_t = H[layer]  # Input to next layer

        return torch.stack(H, dim=0)  # [num_layers, batch, num_nodes, hidden_dim]


class DCRNNDecoder(nn.Module):
    """
    DCRNN Decoder: Seq2Seq decoder with scheduled sampling
    """
    def __init__(self, output_dim, hidden_dim, num_nodes, num_layers=2, K=2):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers

        # Stack of DCGRU cells
        # First layer takes output_dim, rest take hidden_dim
        self.dcgru_cells = nn.ModuleList([
            DCGRUCell(
                output_dim if i == 0 else hidden_dim,
                hidden_dim,
                num_nodes,
                K
            ) for i in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, H_enc, Y_true, adj_matrix, teacher_forcing_ratio=0.5):
        """
        Args:
            H_enc: [num_layers, batch, num_nodes, hidden_dim]
            Y_true: [batch, horizon, num_nodes, output_dim] (for teacher forcing)
            adj_matrix: [num_nodes, num_nodes]
            teacher_forcing_ratio: probability of using teacher forcing

        Returns:
            outputs: [batch, horizon, num_nodes, output_dim]
        """
        batch_size = H_enc.size(1)
        horizon = Y_true.size(1) if Y_true is not None else 1

        # Initialize hidden states from encoder
        H = [H_enc[i] for i in range(self.num_layers)]

        # First input (zeros or last observation)
        decoder_input = torch.zeros(batch_size, self.num_nodes, self.output_dim).to(H_enc.device)

        outputs = []
        for t in range(horizon):
            # Pass through DCGRU layers
            X_t = decoder_input
            for layer in range(self.num_layers):
                H[layer] = self.dcgru_cells[layer](X_t, H[layer], adj_matrix)
                X_t = H[layer]

            # Project to output
            output = self.output_proj(H[-1])  # [batch, num_nodes, output_dim]
            outputs.append(output)

            # Teacher forcing
            if Y_true is not None and np.random.random() < teacher_forcing_ratio:
                decoder_input = Y_true[:, t, :, :]
            else:
                decoder_input = output

        return torch.stack(outputs, dim=1)  # [batch, horizon, num_nodes, output_dim]


class DCRNN(nn.Module):
    """
    Complete DCRNN Model

    Args:
        num_nodes: Number of spatial locations
        input_dim: Input feature dimension (e.g., 2 for inflow/outflow)
        output_dim: Output feature dimension
        hidden_dim: Hidden state dimension
        num_layers: Number of RNN layers
        horizon: Prediction horizon
        K: Diffusion steps
    """
    def __init__(self, num_nodes, input_dim=2, output_dim=2, hidden_dim=64,
                 num_layers=2, horizon=1, K=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon

        self.encoder = DCRNNEncoder(input_dim, hidden_dim, num_nodes, num_layers, K)
        self.decoder = DCRNNDecoder(output_dim, hidden_dim, num_nodes, num_layers, K)

    def forward(self, X, adj_matrix, Y_true=None, teacher_forcing_ratio=0.5):
        """
        Args:
            X: [batch, seq_len, num_nodes, input_dim]
            adj_matrix: [num_nodes, num_nodes]
            Y_true: [batch, horizon, num_nodes, output_dim] (optional, for training)

        Returns:
            Y_pred: [batch, horizon, num_nodes, output_dim]
        """
        # Encode
        H_enc = self.encoder(X, adj_matrix)

        # Decode
        Y_pred = self.decoder(H_enc, Y_true, adj_matrix, teacher_forcing_ratio)

        return Y_pred


class DCRNNWithFlowAdapter(nn.Module):
    """
    DCRNN + Flow Intent Adapter

    Integration Strategy:
      1. Flow Generator produces intent representation z_intent
      2. Use FiLM to modulate DCGRU hidden states
      3. Apply modulation at each layer
    """
    def __init__(self, num_nodes, input_dim=2, output_dim=2, hidden_dim=64,
                 num_layers=2, horizon=1, K=2, use_adapter=True,
                 latent_dim=64, n_intents=6):
        super().__init__()
        self.use_adapter = use_adapter
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Base DCRNN
        self.dcrnn = DCRNN(num_nodes, input_dim, output_dim, hidden_dim,
                          num_layers, horizon, K)

        # Flow Intent Adapter (if enabled)
        if use_adapter:
            from latent_flow_intent_adapter import LatentFlowIntentAdapter

            # Adapter for hidden state modulation
            self.adapter = LatentFlowIntentAdapter(
                feature_dim=hidden_dim,  # Modulate hidden states
                latent_dim=latent_dim,
                n_intents=n_intents
            )

    def forward(self, X, adj_matrix, intent_label=None, Y_true=None,
                teacher_forcing_ratio=0.5):
        """
        Args:
            X: [batch, seq_len, num_nodes, input_dim]
            adj_matrix: [num_nodes, num_nodes]
            intent_label: [batch] - discrete intent labels (0-5)
            Y_true: [batch, horizon, num_nodes, output_dim]

        Returns:
            Y_pred: [batch, horizon, num_nodes, output_dim]
            losses: dict with flow_loss (if training with adapter)
        """
        losses = {}

        if self.use_adapter and intent_label is not None:
            # Encode with base DCRNN
            H_enc = self.dcrnn.encoder(X, adj_matrix)
            # H_enc: [num_layers, batch, num_nodes, hidden_dim]

            # Modulate each layer's hidden state
            H_modulated = []
            for layer_idx in range(self.num_layers):
                H_layer = H_enc[layer_idx]  # [batch, num_nodes, hidden_dim]

                # Apply adapter per-node (preserve spatial heterogeneity)
                batch_size, num_nodes, hidden_dim = H_layer.shape

                # Reshape to [batch * num_nodes, hidden_dim] for per-node modulation
                H_flat = H_layer.reshape(batch_size * num_nodes, hidden_dim)

                # Expand intent labels to match each node
                intent_label_expanded = intent_label.unsqueeze(1).expand(-1, num_nodes).reshape(-1)  # [batch * num_nodes]

                # Apply adapter (each node gets personalized modulation based on its hidden state)
                H_mod_flat, adapter_losses = self.adapter(
                    H_flat,
                    intent_label_expanded
                )

                # Reshape back to [batch, num_nodes, hidden_dim]
                H_mod = H_mod_flat.reshape(batch_size, num_nodes, hidden_dim)
                H_modulated.append(H_mod)

                # Accumulate losses (only from first layer to avoid redundancy)
                if layer_idx == 0 and 'flow_matching' in adapter_losses:
                    losses['flow_loss'] = adapter_losses['flow_matching']

            H_enc = torch.stack(H_modulated, dim=0)

            # Decode with modulated hidden states
            Y_pred = self.dcrnn.decoder(H_enc, Y_true, adj_matrix, teacher_forcing_ratio)

        else:
            # Standard DCRNN without adapter
            Y_pred = self.dcrnn(X, adj_matrix, Y_true, teacher_forcing_ratio)

        return Y_pred, losses


# ============================================================================
# Utility Functions
# ============================================================================

def construct_adj_matrix(num_nodes, grid_size=32):
    """
    Construct adjacency matrix for grid-based traffic network

    Strategy: 4-neighbor connectivity (up, down, left, right)
    """
    adj = np.zeros((num_nodes, num_nodes))

    for i in range(grid_size):
        for j in range(grid_size):
            node = i * grid_size + j

            # Right neighbor
            if j < grid_size - 1:
                neighbor = i * grid_size + (j + 1)
                adj[node, neighbor] = 1

            # Down neighbor
            if i < grid_size - 1:
                neighbor = (i + 1) * grid_size + j
                adj[node, neighbor] = 1

            # Left neighbor
            if j > 0:
                neighbor = i * grid_size + (j - 1)
                adj[node, neighbor] = 1

            # Up neighbor
            if i > 0:
                neighbor = (i - 1) * grid_size + j
                adj[node, neighbor] = 1

    # Normalize (row-wise): D^{-1}A
    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1  # Avoid division by zero
    adj = adj / row_sum

    return torch.FloatTensor(adj)


def test_dcrnn():
    """Quick test"""
    batch_size = 4
    seq_len = 12
    horizon = 1
    num_nodes = 1024  # 32×32 grid
    input_dim = 2
    output_dim = 2

    # Create model
    model = DCRNNWithFlowAdapter(
        num_nodes=num_nodes,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=64,
        num_layers=2,
        horizon=horizon,
        K=2,
        use_adapter=True
    )

    # Dummy data
    X = torch.randn(batch_size, seq_len, num_nodes, input_dim)
    Y_true = torch.randn(batch_size, horizon, num_nodes, output_dim)
    intent_labels = torch.randint(0, 6, (batch_size,))
    adj_matrix = construct_adj_matrix(num_nodes)

    # Forward
    Y_pred, losses = model(X, adj_matrix, intent_labels, Y_true)

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {Y_pred.shape}")
    print(f"Losses: {losses}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_dcrnn()
