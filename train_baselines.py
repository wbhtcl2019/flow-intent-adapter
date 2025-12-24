"""
Unified Training Script for All Baselines

Supports:
  - ST-ResNet (baseline + Flow Adapter)
  - DCRNN (baseline + Flow Adapter)
  - STGCN (baseline + Flow Adapter)
  - STFormer (baseline + Flow Adapter)

Usage:
    python train_baselines.py --model stresnet --use_flow --data_path nyc_96k_with_intents.parquet
    python train_baselines.py --model dcrnn --use_flow --data_path nyc_2M_with_intents.parquet
    python train_baselines.py --model stgcn --data_path nyc_96k_with_intents.parquet
    python train_baselines.py --model stformer --use_flow --data_path nyc_96k_with_intents.parquet
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys

# Import dataset
sys.path.append(os.path.dirname(__file__))
from train_flow_adapter_96k import FlowDataset

# Import models
from baselines.dcrnn_baseline import DCRNNWithFlowAdapter, construct_adj_matrix
from baselines.stgcn_baseline import STGCNWithFlowAdapter, construct_graph_from_grid
from baselines.stformer_baseline import STFormerWithFlowAdapter
from st_resnet_baseline import STResNet
from train_flow_adapter_96k import STResNetWithFlowAdapter


def get_model(args, device):
    """
    Create model based on args.model

    Returns:
        model, requires_graph (bool)
    """
    num_nodes = args.n_tiles * args.n_tiles  # Grid size

    if args.model == 'stresnet':
        if args.use_flow:
            model = STResNetWithFlowAdapter(
                n_tiles=args.n_tiles,
                closeness_len=args.closeness_len,
                nb_residual_unit=4,
                latent_dim=args.latent_dim,
                use_adapter=True
            )
        else:
            model = STResNet(
                n_tiles=args.n_tiles,
                closeness_len=args.closeness_len,
                nb_residual_unit=4
            )
        requires_graph = False

    elif args.model == 'dcrnn':
        model = DCRNNWithFlowAdapter(
            num_nodes=num_nodes,
            input_dim=2,  # inflow + outflow
            output_dim=2,
            hidden_dim=args.hidden_dim,
            num_layers=2,
            horizon=1,
            K=2,
            use_adapter=args.use_flow,
            latent_dim=args.latent_dim,
            n_intents=6
        )
        requires_graph = True

    elif args.model == 'stformer':
        model = STFormerWithFlowAdapter(
            num_nodes=num_nodes,
            in_channels=2,
            out_channels=2,
            d_model=args.hidden_dim,
            num_heads=4,
            num_layers=3,
            d_ff=args.hidden_dim * 4,
            seq_len=args.closeness_len,
            horizon=1,
            dropout=0.1,
            use_adapter=args.use_flow,
            latent_dim=args.latent_dim,
            n_intents=6
        )
        requires_graph = False

    elif args.model == 'stgcn':
        model = STGCNWithFlowAdapter(
            num_nodes=num_nodes,
            in_channels=2,
            out_channels=2,
            temporal_len=args.closeness_len,
            horizon=1,
            hidden_channels=args.hidden_dim,
            num_blocks=2,
            K=3,
            use_adapter=args.use_flow,
            latent_dim=args.latent_dim,
            n_intents=6
        )
        requires_graph = True

    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    return model, requires_graph


def prepare_graph_structures(args, device):
    """
    Prepare graph adjacency/Laplacian matrices

    Returns:
        adj_matrix or L_tilde (depending on model)
    """
    if args.model == 'dcrnn':
        # DCRNN needs adjacency matrix
        adj_matrix = construct_adj_matrix(args.n_tiles * args.n_tiles, args.n_tiles)
        return adj_matrix.to(device)

    elif args.model == 'stgcn':
        # STGCN needs scaled Laplacian
        _, L_tilde = construct_graph_from_grid(args.n_tiles)
        return L_tilde.to(device)

    else:
        return None


def forward_pass(model, X_closeness, intent_label, Y_true, graph_structure, args):
    """
    Unified forward pass for all models

    Returns:
        Y_pred, losses
    """
    losses = {}

    if args.model == 'stresnet':
        if args.use_flow:
            Y_pred = model(X_closeness, intent_label)
            if hasattr(model, 'adapter') and model.training:
                # Compute flow loss separately (already computed in forward)
                pass  # Loss handled in training loop
        else:
            Y_pred = model(X_closeness)

    elif args.model == 'stformer':
        # STFormer expects [batch, seq_len, num_nodes, channels]
        batch_size = X_closeness.size(0)
        seq_len = X_closeness.size(1)
        h, w, c = X_closeness.size(2), X_closeness.size(3), X_closeness.size(4)

        X_stformer = X_closeness.view(batch_size, seq_len, h * w, c)

        Y_pred_flat, losses = model(
            X_stformer,
            intent_label if args.use_flow else None
        )

        # Reshape back: [batch, horizon=1, num_nodes, channels] → [batch, h, w, c]
        Y_pred = Y_pred_flat.squeeze(1).view(batch_size, h, w, c)

    elif args.model == 'dcrnn':
        # DCRNN expects [batch, seq_len, num_nodes, channels]
        batch_size = X_closeness.size(0)
        seq_len = X_closeness.size(1)
        h, w, c = X_closeness.size(2), X_closeness.size(3), X_closeness.size(4)

        X_dcrnn = X_closeness.view(batch_size, seq_len, h * w, c)
        Y_dcrnn_true = Y_true.view(batch_size, 1, h * w, c) if Y_true is not None else None

        Y_pred_flat, losses = model(
            X_dcrnn,
            graph_structure,  # adj_matrix
            intent_label if args.use_flow else None,
            Y_dcrnn_true,
            teacher_forcing_ratio=0.5 if model.training else 0.0
        )

        # Reshape back
        Y_pred = Y_pred_flat.view(batch_size, h, w, c)

    elif args.model == 'stgcn':
        # STGCN expects [batch, time, num_nodes, channels]
        batch_size = X_closeness.size(0)
        seq_len = X_closeness.size(1)
        h, w, c = X_closeness.size(2), X_closeness.size(3), X_closeness.size(4)

        X_stgcn = X_closeness.view(batch_size, seq_len, h * w, c)

        Y_pred_flat, losses = model(
            X_stgcn,
            graph_structure,  # L_tilde
            intent_label if args.use_flow else None
        )

        # Reshape back: [batch, horizon=1, num_nodes, channels] → [batch, h, w, c]
        Y_pred = Y_pred_flat.squeeze(1).view(batch_size, h, w, c)

    return Y_pred, losses


def train_epoch(model, train_loader, optimizer, criterion, device, args, graph_structure=None):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    total_pred_loss = 0
    total_flow_loss = 0
    n_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # FlowDataset returns: 'closeness', 'target', 'intent_dist', 'intent_label'
        X_closeness = batch['closeness'].to(device)  # [B, 2, T, H, W]
        Y_true = batch['target'].to(device)  # [B, 2, H, W]
        intent_label = batch['intent_label'].to(device) if args.use_flow else None

        # Reshape X_closeness: [B, 2, T, H, W] -> [B, T, H, W, 2]
        B, C, T, H, W = X_closeness.shape
        X_closeness = X_closeness.permute(0, 2, 3, 4, 1)  # [B, T, H, W, 2]

        # Reshape Y_true: [B, 2, H, W] -> [B, H, W, 2]
        Y_true = Y_true.permute(0, 2, 3, 1)  # [B, H, W, 2]

        optimizer.zero_grad()

        # Forward
        Y_pred, losses = forward_pass(model, X_closeness, intent_label, Y_true, graph_structure, args)

        # Prediction loss
        loss_pred = criterion(Y_pred, Y_true)

        # Flow loss (if applicable)
        loss_flow = losses.get('flow_loss', torch.tensor(0.0).to(device))

        # Total loss
        if args.use_flow:
            loss = loss_pred + args.alpha * loss_flow
        else:
            loss = loss_pred

        # Backward
        loss.backward()
        optimizer.step()

        # Accumulate
        total_loss += loss.item()
        total_pred_loss += loss_pred.item()
        if args.use_flow:
            total_flow_loss += loss_flow.item()
        n_batches += 1

        # Update progress
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'pred': f'{loss_pred.item():.6f}',
            'flow': f'{loss_flow.item():.6f}' if args.use_flow else 'N/A'
        })

    avg_loss = total_loss / n_batches
    avg_pred_loss = total_pred_loss / n_batches
    avg_flow_loss = total_flow_loss / n_batches if args.use_flow else 0.0

    return avg_loss, avg_pred_loss, avg_flow_loss


def validate_epoch(model, val_loader, criterion, device, args, graph_structure=None):
    """
    Validate for one epoch
    """
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            # FlowDataset returns: 'closeness', 'target', 'intent_dist', 'intent_label'
            X_closeness = batch['closeness'].to(device)
            Y_true = batch['target'].to(device)
            intent_label = batch['intent_label'].to(device) if args.use_flow else None

            # Reshape
            B, C, T, H, W = X_closeness.shape
            X_closeness = X_closeness.permute(0, 2, 3, 4, 1)  # [B, T, H, W, 2]
            Y_true = Y_true.permute(0, 2, 3, 1)  # [B, H, W, 2]

            # Forward
            Y_pred, _ = forward_pass(model, X_closeness, intent_label, Y_true, graph_structure, args)

            # Loss
            loss = criterion(Y_pred, Y_true)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def evaluate(model, val_loader, device, args, graph_structure=None):
    """
    Evaluate model and compute MAE

    Args:
        model: The model to evaluate
        val_loader: Validation data loader
        device: Device to run on
        args: Arguments containing model type
        graph_structure: Optional graph structure for graph-based models

    Returns:
        avg_mae: Average MAE across all batches
    """
    model.eval()
    total_mae = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            X_closeness = batch['closeness'].to(device)
            Y_true = batch['target'].to(device)

            # Reshape
            B, C, T, H, W = X_closeness.shape
            X_closeness = X_closeness.permute(0, 2, 3, 4, 1)  # [B, T, H, W, 2]
            Y_true = Y_true.permute(0, 2, 3, 1)  # [B, H, W, 2]

            # Forward pass (no intent at evaluation)
            Y_pred, _ = forward_pass(model, X_closeness, None, Y_true, graph_structure, args)

            # Compute MAE
            mae = torch.abs(Y_pred - Y_true).mean().item()
            total_mae += mae
            n_batches += 1

    avg_mae = total_mae / n_batches
    return avg_mae


def main():
    parser = argparse.ArgumentParser()

    # Model selection
    parser.add_argument('--model', type=str, default='stresnet',
                       choices=['stresnet', 'dcrnn', 'stgcn', 'stformer'],
                       help='Model architecture')
    parser.add_argument('--use_flow', action='store_true',
                       help='Use Flow Intent Adapter')

    # Data
    parser.add_argument('--data_path', type=str, default='nyc_100k_with_intents.parquet')
    parser.add_argument('--n_tiles', type=int, default=32)
    parser.add_argument('--closeness_len', type=int, default=12)

    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension (for DCRNN, STGCN)')
    parser.add_argument('--latent_dim', type=int, default=64,
                       help='Latent dimension for Flow Adapter')

    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=0.02,
                       help='Flow loss weight')

    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    print(f"Total trips: {len(df):,}")

    # Split data: 70% train, 15% val, 15% test
    n_samples = len(df)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    df_train = df.iloc[:train_size]
    df_val = df.iloc[train_size:train_size+val_size]

    print(f"Train trips: {len(df_train):,}")
    print(f"Val trips: {len(df_val):,}")

    # Create datasets
    print("\nBuilding datasets...")
    train_dataset = FlowDataset(df_train, n_tiles=args.n_tiles,
                                closeness_len=args.closeness_len)
    val_dataset = FlowDataset(df_val, n_tiles=args.n_tiles,
                             closeness_len=args.closeness_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    model, requires_graph = get_model(args, device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Prepare graph structures
    graph_structure = None
    if requires_graph:
        print(f"Preparing graph structures for {args.model}...")
        graph_structure = prepare_graph_structures(args, device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_val_mae = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_pred_loss, train_flow_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, args, graph_structure
        )

        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device, args, graph_structure)

        # Evaluate (MAE)
        val_mae = evaluate(model, val_loader, device, args, graph_structure)

        # Log
        print(f"  Train Loss: {train_loss:.6f} (Pred: {train_pred_loss:.6f}, Flow: {train_flow_loss:.6f})")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val MAE: {val_mae:.6f}")

        # Save best
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            print(f"  ✓ New best MAE: {best_val_mae:.6f}")

            # Save checkpoint
            model_name = f"{args.model}_{'flow' if args.use_flow else 'baseline'}"
            checkpoint_path = f"{model_name}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mae': best_val_mae,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  ✓ Model saved: {checkpoint_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Val MAE: {best_val_mae:.6f}")


if __name__ == "__main__":
    main()
