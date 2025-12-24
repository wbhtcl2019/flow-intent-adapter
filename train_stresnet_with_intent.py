"""
Train ST-ResNet with Intent Adapter on 96k Dataset

Colab-ready training script for baseline comparison:
- Baseline: ST-ResNet (closeness-only for speed)
- Proposed: ST-ResNet + IntentAdapter

Usage in Colab:
    !python train_stresnet_with_intent.py --use_adapter --epochs 50
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm

from st_resnet_baseline import STResNet
from intent_adapter import IntentAdapter


# ============================================================================
# Dataset
# ============================================================================

class FlowDataset(Dataset):
    """
    Dataset for traffic flow prediction with intents

    Input: OD flow tensor [T, N, N, C]
    Output: Node-level flow [T, N, C] + Intent labels
    """
    def __init__(self, df, n_tiles=300, n_intents=6, closeness_len=6):
        """
        Args:
            df: DataFrame with columns [pickup_grid, dropoff_grid, intent_category, ...]
            n_tiles: Number of spatial tiles
            n_intents: Number of intent categories
            closeness_len: Number of recent timesteps to use
        """
        self.n_tiles = n_tiles
        self.n_intents = n_intents
        self.closeness_len = closeness_len

        # Convert to grid indices
        print("Building spatial grid...")
        self.grid_to_idx = self._build_grid_mapping(df)

        # Aggregate to time bins (30-min bins)
        print("Aggregating to time bins...")
        self.time_bins, self.flow_tensor, self.intent_dist = self._aggregate_flows(df)

        print(f"  Time bins: {len(self.time_bins)}")
        print(f"  Flow tensor shape: {self.flow_tensor.shape}")

    def _build_grid_mapping(self, df):
        """Map grid cells to indices"""
        all_grids = set(df['pickup_grid'].unique()) | set(df['dropoff_grid'].unique())
        all_grids = sorted(list(all_grids))[:self.n_tiles]
        return {grid: idx for idx, grid in enumerate(all_grids)}

    def _aggregate_flows(self, df):
        """Aggregate trips to time bins and build OD flow tensor"""
        df = df.copy()
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

        # 30-min bins
        df['time_bin'] = df['pickup_datetime'].dt.floor('30min')

        time_bins = sorted(df['time_bin'].unique())
        T = len(time_bins)

        # Initialize flow tensor [T, N, N, 2] (inflow/outflow)
        flow_tensor = np.zeros((T, self.n_tiles, self.n_tiles, 2), dtype=np.float32)

        # Intent distribution per time bin [T, N_intents]
        intent_dist = np.zeros((T, self.n_intents), dtype=np.float32)

        for t, time_bin in enumerate(tqdm(time_bins, desc="Building tensor")):
            bin_df = df[df['time_bin'] == time_bin]

            for _, row in bin_df.iterrows():
                o_grid = row['pickup_grid']
                d_grid = row['dropoff_grid']
                intent = int(row['intent_category'])

                if o_grid in self.grid_to_idx and d_grid in self.grid_to_idx:
                    o_idx = self.grid_to_idx[o_grid]
                    d_idx = self.grid_to_idx[d_grid]

                    # Outflow from origin
                    flow_tensor[t, o_idx, d_idx, 0] += 1
                    # Inflow to destination
                    flow_tensor[t, o_idx, d_idx, 1] += 1

                    # Intent count
                    intent_dist[t, intent] += 1

        return time_bins, flow_tensor, intent_dist

    def __len__(self):
        return len(self.time_bins) - self.closeness_len

    def __getitem__(self, idx):
        """
        Returns:
            closeness: [T_c, H, W, 2] - Recent timesteps
            target: [H, W, 2] - Next timestep flow
            intent: Dominant intent for this time bin
        """
        # Get closeness inputs
        closeness = self.flow_tensor[idx:idx+self.closeness_len]  # [T_c, N, N, 2]

        # Target is next timestep
        target = self.flow_tensor[idx + self.closeness_len]  # [N, N, 2]

        # Get dominant intent for target timestep
        intent = self.intent_dist[idx + self.closeness_len].argmax()

        return (
            torch.FloatTensor(closeness),
            torch.FloatTensor(target),
            torch.LongTensor([intent])
        )


# ============================================================================
# Model Wrapper
# ============================================================================

class STResNetWithIntent(nn.Module):
    """ST-ResNet + Intent Adapter"""
    def __init__(self, n_tiles=300, n_intents=6, closeness_len=6,
                 hidden_channels=64, use_adapter=True):
        super(STResNetWithIntent, self).__init__()

        self.use_adapter = use_adapter

        # Base ST-ResNet (closeness only)
        self.st_resnet = STResNet(
            closeness_len=closeness_len,
            period_len=0,  # Disabled for speed
            trend_len=0,
            hidden_channels=hidden_channels,
            n_resunits=3,
            grid_height=n_tiles,
            grid_width=n_tiles
        )

        # Intent adapter (optional)
        if use_adapter:
            self.adapter = IntentAdapter(
                n_intents=n_intents,
                feature_dim=hidden_channels,
                intent_embed_dim=32
            )

    def forward(self, closeness, intent_ids):
        """
        Args:
            closeness: [B, T_c, H, W, 2]
            intent_ids: [B, 1]

        Returns:
            pred: [B, H, W, 2]
        """
        if self.use_adapter:
            # Get intermediate features from ST-ResNet
            B, T, H, W, C = closeness.shape
            x = closeness.permute(0, 1, 4, 2, 3).reshape(B, T*C, H, W)

            # Extract features after first conv
            features = self.st_resnet.closeness_branch.conv_in(x)
            features = self.st_resnet.closeness_branch.bn_in(features)
            features = self.st_resnet.closeness_branch.relu(features)

            # Apply intent adapter
            features = self.adapter(features, intent_ids.squeeze(1))

            # Continue through ResUnits
            for resunit in self.st_resnet.closeness_branch.resunits:
                features = resunit(features)

            # Output projection
            out = self.st_resnet.closeness_branch.conv_out(features)
            pred = out.permute(0, 2, 3, 1)
        else:
            # Standard ST-ResNet
            pred = self.st_resnet(closeness=closeness)

        return pred


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for closeness, target, intent in tqdm(dataloader, desc="Training"):
        closeness = closeness.to(device)
        target = target.to(device)
        intent = intent.to(device)

        optimizer.zero_grad()

        pred = model(closeness, intent)
        loss = criterion(pred, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for closeness, target, intent in dataloader:
            closeness = closeness.to(device)
            target = target.to(device)
            intent = intent.to(device)

            pred = model(closeness, intent)

            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    mae = (preds - targets).abs().mean().item()
    rmse = ((preds - targets)**2).mean().sqrt().item()

    return mae, rmse


def main(args):
    print("=" * 80)
    print("Training ST-ResNet with Intent Adapter")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data
    print("\n[Loading Data]")
    print("-" * 80)
    df = pd.read_parquet(args.data_path)
    print(f"  Total trips: {len(df):,}")

    # Split train/val/test
    n_samples = len(df)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    df_train = df.iloc[:train_size]
    df_val = df.iloc[train_size:train_size+val_size]
    df_test = df.iloc[train_size+val_size:]

    print(f"  Train: {len(df_train):,}")
    print(f"  Val: {len(df_val):,}")
    print(f"  Test: {len(df_test):,}")

    # Create datasets
    print("\n[Creating Datasets]")
    print("-" * 80)

    train_dataset = FlowDataset(df_train, n_tiles=args.n_tiles,
                                closeness_len=args.closeness_len)
    val_dataset = FlowDataset(df_val, n_tiles=args.n_tiles,
                             closeness_len=args.closeness_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    # Create model
    print("\n[Creating Model]")
    print("-" * 80)

    model = STResNetWithIntent(
        n_tiles=args.n_tiles,
        n_intents=args.n_intents,
        closeness_len=args.closeness_len,
        hidden_channels=args.hidden_channels,
        use_adapter=args.use_adapter
    ).to(device)

    print(f"  Model: ST-ResNet {'+ IntentAdapter' if args.use_adapter else '(baseline)'}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    print("\n[Training]")
    print("-" * 80)

    best_val_mae = float('inf')
    results = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_mae, val_rmse = evaluate(model, val_loader, device)

        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}")

        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_mae': val_mae,
            'val_rmse': val_rmse
        })

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            model_name = 'stresnet_with_intent.pt' if args.use_adapter else 'stresnet_baseline.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_rmse': val_rmse
            }, model_name)
            print(f"  [OK] Saved best model: {model_name}")

    # Save results
    result_name = 'results_with_intent.json' if args.use_adapter else 'results_baseline.json'
    with open(result_name, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best Val MAE: {best_val_mae:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', type=str,
                       default='nyc_100k_with_intents.parquet')
    parser.add_argument('--n_tiles', type=int, default=100,
                       help='Number of spatial tiles (reduce for speed)')
    parser.add_argument('--n_intents', type=int, default=6)
    parser.add_argument('--closeness_len', type=int, default=6)

    # Model
    parser.add_argument('--use_adapter', action='store_true',
                       help='Use intent adapter (default: baseline)')
    parser.add_argument('--hidden_channels', type=int, default=64)

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    main(args)
