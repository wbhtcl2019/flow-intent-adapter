"""
Quick validation: ST-ResNet + Latent Flow Intent Adapter on 96k data

Compare:
  1. Baseline (no intent)
  2. Discrete Intent (old FiLM)
  3. Flow Intent (new Flow Matching) ← This one!

Usage in Colab:
    !python train_flow_adapter_96k.py --use_flow --epochs 30
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import os

from st_resnet_baseline import STResNet
from latent_flow_intent_adapter import LatentFlowIntentAdapter


# ============================================================================
# Dataset (same as before)
# ============================================================================

class FlowDataset(Dataset):
    def __init__(self, df, n_tiles=100, n_intents=6, closeness_len=6):
        self.n_tiles = n_tiles
        self.n_intents = n_intents
        self.closeness_len = closeness_len

        print("Building spatial grid...")
        self.grid_to_idx = self._build_grid_mapping(df)

        print("Aggregating to time bins...")
        self.time_bins, self.flow_tensor, self.intent_dist = self._aggregate_flows(df)

        print(f"  Time bins: {len(self.time_bins)}")
        print(f"  Flow tensor shape: {self.flow_tensor.shape}")

    def _build_grid_mapping(self, df):
        # Compute grid if not exist
        if 'pickup_grid' not in df.columns:
            df = self._compute_grids(df)

        # Get all unique grids
        all_grids = set(df['pickup_grid'].unique()) | set(df['dropoff_grid'].unique())
        all_grids = sorted(list(all_grids))

        # If we have more grids than n_tiles, select top-k by frequency
        if len(all_grids) > self.n_tiles * self.n_tiles:
            # Count frequency of each grid
            pickup_counts = df['pickup_grid'].value_counts()
            dropoff_counts = df['dropoff_grid'].value_counts()
            total_counts = pickup_counts.add(dropoff_counts, fill_value=0)

            # Select top n_tiles^2 most frequent grids
            top_grids = total_counts.nlargest(self.n_tiles * self.n_tiles).index.tolist()
            all_grids = sorted(top_grids)

        return {grid: idx for idx, grid in enumerate(all_grids)}

    def _compute_grids(self, df):
        """Compute grid cells from lat/lon"""
        grid_size = 0.01
        lat_min, lon_min = 40.5, -74.5

        df['pickup_grid_x'] = ((df['pickup_longitude'] - lon_min) / grid_size).astype(int)
        df['pickup_grid_y'] = ((df['pickup_latitude'] - lat_min) / grid_size).astype(int)
        df['dropoff_grid_x'] = ((df['dropoff_longitude'] - lon_min) / grid_size).astype(int)
        df['dropoff_grid_y'] = ((df['dropoff_latitude'] - lat_min) / grid_size).astype(int)

        df['pickup_grid'] = df['pickup_grid_y'] * 1000 + df['pickup_grid_x']
        df['dropoff_grid'] = df['dropoff_grid_y'] * 1000 + df['dropoff_grid_x']

        return df

    def _aggregate_flows(self, df):
        df = df.copy()
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['time_bin'] = df['pickup_datetime'].dt.floor('30min')

        time_bins = sorted(df['time_bin'].unique())
        T = len(time_bins)

        flow_tensor = np.zeros((T, self.n_tiles, self.n_tiles, 2), dtype=np.float32)
        intent_dist = np.zeros((T, self.n_intents), dtype=np.float32)

        # Create reverse mapping: idx -> (row, col) in n_tiles x n_tiles grid
        idx_to_grid = {v: k for k, v in self.grid_to_idx.items()}
        n_grids = len(idx_to_grid)

        # Map linear indices to 2D grid positions
        # We'll map indices 0..n_grids-1 to a sqrt(n_grids) x sqrt(n_grids) grid
        grid_dim = int(np.ceil(np.sqrt(n_grids)))

        for t, time_bin in enumerate(tqdm(time_bins, desc="Building tensor")):
            bin_df = df[df['time_bin'] == time_bin]

            for _, row in bin_df.iterrows():
                o_grid = row['pickup_grid']
                d_grid = row['dropoff_grid']
                intent = int(row['intent_category'])

                if o_grid in self.grid_to_idx and d_grid in self.grid_to_idx:
                    o_idx = self.grid_to_idx[o_grid]
                    d_idx = self.grid_to_idx[d_grid]

                    # Map to 2D positions
                    o_row, o_col = o_idx // self.n_tiles, o_idx % self.n_tiles
                    d_row, d_col = d_idx // self.n_tiles, d_idx % self.n_tiles

                    # Accumulate flows (outflow from origin, inflow to destination)
                    if o_row < self.n_tiles and o_col < self.n_tiles:
                        flow_tensor[t, o_row, o_col, 0] += 1  # outflow
                    if d_row < self.n_tiles and d_col < self.n_tiles:
                        flow_tensor[t, d_row, d_col, 1] += 1  # inflow

                intent_dist[t, intent] += 1

        # Normalize intent distribution
        row_sums = intent_dist.sum(axis=1, keepdims=True)
        intent_dist = np.divide(intent_dist, row_sums, where=row_sums > 0)

        return time_bins, flow_tensor, intent_dist

    def __len__(self):
        return len(self.time_bins) - self.closeness_len

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.closeness_len

        closeness = self.flow_tensor[start_idx:end_idx]
        target = self.flow_tensor[end_idx]
        intent = self.intent_dist[end_idx]

        closeness = torch.FloatTensor(closeness).permute(3, 0, 1, 2)
        target = torch.FloatTensor(target).permute(2, 0, 1)
        intent = torch.FloatTensor(intent)

        # Get dominant intent as label
        intent_label = torch.argmax(intent)

        return {
            'closeness': closeness,
            'target': target,
            'intent_dist': intent,
            'intent_label': intent_label
        }


# ============================================================================
# Model with Flow Adapter
# ============================================================================

class STResNetWithFlowAdapter(nn.Module):
    """ST-ResNet + Latent Flow Intent Adapter"""

    def __init__(
        self,
        n_tiles=100,
        closeness_len=6,
        nb_residual_unit=4,
        latent_dim=64,
        use_adapter=True
    ):
        super().__init__()

        self.use_adapter = use_adapter

        # ST-ResNet backbone (corrected parameter names)
        self.stresnet = STResNet(
            closeness_len=closeness_len,
            period_len=0,  # Only use closeness for speed
            trend_len=0,
            hidden_channels=64,
            n_resunits=nb_residual_unit,
            grid_height=n_tiles,
            grid_width=n_tiles
        )

        if use_adapter:
            # 🔌 Latent Flow Intent Adapter
            # Feature dim = 64 (ST-ResNet output channels)
            self.adapter = LatentFlowIntentAdapter(
                feature_dim=64,
                latent_dim=latent_dim,
                n_intents=6,
                flow_steps=10
            )

    def forward(self, closeness, intent_label=None):
        """
        Args:
            closeness: [B, 2, T, H, W]
            intent_label: [B] discrete intent labels

        Returns:
            pred: [B, 2, H, W]
            losses: dict with 'flow_matching' loss
        """
        # Reshape for ST-ResNet: [B, 2, T, H, W] -> [B, T, H, W, 2]
        B, C, T, H, W = closeness.shape
        closeness_reshaped = closeness.permute(0, 2, 3, 4, 1)  # [B, T, H, W, 2]

        # Extract ST features
        features = self.stresnet.extract_features(closeness=closeness_reshaped)  # [B, 64, H, W]

        losses = {}

        if self.use_adapter:
            # Apply Flow Adapter
            modulated, adapter_losses = self.adapter(
                features,
                intent_label,
                return_latent=False
            )
            losses.update(adapter_losses)
        else:
            modulated = features

        # Prediction head
        pred = self.stresnet.predict(modulated)  # [B, 2, H, W]

        return pred, losses


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, loader, optimizer, device, alpha=0.1):
    model.train()
    total_loss = 0
    total_flow_loss = 0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        closeness = batch['closeness'].to(device)
        target = batch['target'].to(device)
        intent_label = batch['intent_label'].to(device)

        pred, losses = model(closeness, intent_label)

        # Task loss (prediction)
        loss_task = nn.functional.mse_loss(pred, target)

        # Flow matching loss (if using adapter)
        loss_flow = losses.get('flow_matching', torch.tensor(0.0, device=device))

        # Total loss
        loss_total = loss_task + alpha * loss_flow

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        total_loss += loss_total.item()
        total_flow_loss += loss_flow.item() if isinstance(loss_flow, torch.Tensor) else loss_flow

        pbar.set_postfix({
            'loss': f'{loss_total.item():.6f}',
            'flow': f'{loss_flow.item() if isinstance(loss_flow, torch.Tensor) else loss_flow:.6f}'
        })

    avg_loss = total_loss / len(loader)
    avg_flow = total_flow_loss / len(loader)

    return avg_loss, avg_flow


def evaluate(model, loader, device):
    model.eval()
    total_mae = 0
    total_rmse = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            closeness = batch['closeness'].to(device)
            target = batch['target'].to(device)

            pred, _ = model(closeness, None)  # No intent at inference

            mae = torch.abs(pred - target).mean().item()
            rmse = torch.sqrt(torch.pow(pred - target, 2).mean()).item()

            total_mae += mae
            total_rmse += rmse

    avg_mae = total_mae / len(loader)
    avg_rmse = total_rmse / len(loader)

    return avg_mae, avg_rmse


# ============================================================================
# Main
# ============================================================================

def main(args):
    print("=" * 80)
    print("ST-ResNet + Latent Flow Intent Adapter (96k Validation)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data
    print("\n[Loading Data]")
    print("-" * 80)
    df = pd.read_parquet(args.data_path)
    print(f"  Total trips: {len(df):,}")

    # Split
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

    train_dataset = FlowDataset(df_train, n_tiles=args.n_tiles, closeness_len=args.closeness_len)
    val_dataset = FlowDataset(df_val, n_tiles=args.n_tiles, closeness_len=args.closeness_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model
    print("\n[Creating Model]")
    print("-" * 80)

    model = STResNetWithFlowAdapter(
        n_tiles=args.n_tiles,
        closeness_len=args.closeness_len,
        latent_dim=args.latent_dim,
        use_adapter=args.use_flow
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: ST-ResNet + {'Flow Adapter' if args.use_flow else 'Baseline'}")
    print(f"  Total parameters: {n_params:,}")

    if args.use_flow:
        adapter_params = sum(p.numel() for p in model.adapter.parameters())
        print(f"  Adapter parameters: {adapter_params:,} ({adapter_params/n_params*100:.1f}%)")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    print("\n[Training]")
    print("-" * 80)

    results = []
    best_mae = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_flow = train_epoch(model, train_loader, optimizer, device, alpha=args.alpha)
        val_mae, val_rmse = evaluate(model, val_loader, device)

        print(f"  Train Loss: {train_loss:.6f} (Flow: {train_flow:.6f})")
        print(f"  Val MAE:    {val_mae:.6f}")
        print(f"  Val RMSE:   {val_rmse:.6f}")

        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_flow_loss': train_flow,
            'val_mae': val_mae,
            'val_rmse': val_rmse
        })

        if val_mae < best_mae:
            best_mae = val_mae
            print(f"  ✓ New best MAE: {best_mae:.6f}")

            # Save best model checkpoint
            checkpoint_path = f"model_{'flow' if args.use_flow else 'baseline'}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mae': best_mae,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  ✓ Model checkpoint saved: {checkpoint_path}")

    # Save results
    output_name = f"results_flow_adapter_96k.json" if args.use_flow else "results_baseline_96k.json"
    with open(output_name, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Val MAE: {best_mae:.6f}")
    print(f"Results saved to: {output_name}")

    # Auto-visualize intent embeddings for Flow Adapter
    if args.use_flow:
        print("\n" + "=" * 80)
        print("VISUALIZING INTENT EMBEDDINGS")
        print("=" * 80)

        try:
            import visualize_intent_embeddings
            from importlib import reload
            reload(visualize_intent_embeddings)
            from visualize_intent_embeddings import visualize_embeddings

            vis_results = visualize_embeddings(
                model=model,
                save_dir='.',
                use_tsne=False
            )
            print("\n✓ Intent embedding visualization complete!")

        except Exception as e:
            print(f"\n⚠️  Visualization skipped: {e}")
            print("   (This is optional, training results are still valid)")

    return best_mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='nyc_100k_with_intents.parquet')
    parser.add_argument('--n_tiles', type=int, default=100)
    parser.add_argument('--closeness_len', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=0.1, help='Weight for flow matching loss')
    parser.add_argument('--use_flow', action='store_true', help='Use Flow Adapter (vs baseline)')

    args = parser.parse_args()
    main(args)
