"""
ST-ResNet Baseline Implementation

Paper: Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction
https://arxiv.org/abs/1610.00081

Architecture:
- 3 branches: Closeness, Period, Trend
- Each branch: Conv2d + ResUnits
- Fusion: Learnable weighted combination
- Output: Predicted flow grid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResUnit(nn.Module):
    """
    Residual Unit for ST-ResNet

    Architecture:
        x -> Conv2d -> ReLU -> Conv2d -> (+) -> ReLU
         |_________________________________|
    """
    def __init__(self, channels, kernel_size=3):
        super(ResUnit, self).__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class STResNetBranch(nn.Module):
    """
    Single branch (Closeness/Period/Trend) of ST-ResNet

    Architecture:
        Input -> Conv2d -> [ResUnit x N] -> Conv2d -> Output
    """
    def __init__(self, input_channels, hidden_channels=64, n_resunits=4):
        super(STResNetBranch, self).__init__()

        # Initial convolution to expand channels
        self.conv_in = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual units
        self.resunits = nn.ModuleList([
            ResUnit(hidden_channels) for _ in range(n_resunits)
        ])

        # Output convolution (project back to 2 channels: inflow/outflow)
        self.conv_out = nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Args:
            x: [B, T*C, H, W] where C=2 (inflow/outflow)
        Returns:
            out: [B, 2, H, W]
        """
        out = self.conv_in(x)
        out = self.bn_in(out)
        out = self.relu(out)

        for resunit in self.resunits:
            out = resunit(out)

        out = self.conv_out(out)

        return out


class STResNet(nn.Module):
    """
    ST-ResNet: Deep Spatio-Temporal Residual Networks

    Components:
    - Closeness: Recent time steps (e.g., last 3 hours)
    - Period: Same hour in previous days (e.g., 1-3 days ago)
    - Trend: Same day of week in previous weeks (e.g., 1-2 weeks ago)

    Each component is a STResNetBranch, and outputs are fused with learnable weights.
    """
    def __init__(self,
                 closeness_len=6,      # Number of closeness steps
                 period_len=3,         # Number of period steps
                 trend_len=2,          # Number of trend steps
                 hidden_channels=64,   # Hidden dimension
                 n_resunits=4,         # Number of ResUnits per branch
                 grid_height=20,       # Grid height (for fusion weights)
                 grid_width=20):       # Grid width (for fusion weights)
        super(STResNet, self).__init__()

        self.closeness_len = closeness_len
        self.period_len = period_len
        self.trend_len = trend_len

        # Three branches
        if closeness_len > 0:
            self.closeness_branch = STResNetBranch(
                input_channels=closeness_len * 2,  # 2 channels per timestep
                hidden_channels=hidden_channels,
                n_resunits=n_resunits
            )

        if period_len > 0:
            self.period_branch = STResNetBranch(
                input_channels=period_len * 2,
                hidden_channels=hidden_channels,
                n_resunits=n_resunits
            )

        if trend_len > 0:
            self.trend_branch = STResNetBranch(
                input_channels=trend_len * 2,
                hidden_channels=hidden_channels,
                n_resunits=n_resunits
            )

        # Learnable fusion weights (one per branch, per location)
        n_branches = (closeness_len > 0) + (period_len > 0) + (trend_len > 0)
        self.fusion_weights = nn.Parameter(
            torch.ones(n_branches, 2, grid_height, grid_width)
        )

        # Store dimensions for adapter integration
        self.hidden_channels = hidden_channels
        self.n_branches = n_branches

    def forward(self, closeness=None, period=None, trend=None):
        """
        Args:
            closeness: [B, T_c, H, W, 2] - Recent timesteps
            period: [B, T_p, H, W, 2] - Same hour previous days
            trend: [B, T_t, H, W, 2] - Same weekday previous weeks

        Returns:
            pred: [B, H, W, 2] - Predicted flow (inflow/outflow)
        """
        outputs = []

        # Closeness branch
        if self.closeness_len > 0 and closeness is not None:
            B, T, H, W, C = closeness.shape
            x_c = closeness.permute(0, 1, 4, 2, 3).reshape(B, T*C, H, W)  # [B, T*2, H, W]
            out_c = self.closeness_branch(x_c)  # [B, 2, H, W]
            outputs.append(out_c)

        # Period branch
        if self.period_len > 0 and period is not None:
            B, T, H, W, C = period.shape
            x_p = period.permute(0, 1, 4, 2, 3).reshape(B, T*C, H, W)
            out_p = self.period_branch(x_p)
            outputs.append(out_p)

        # Trend branch
        if self.trend_len > 0 and trend is not None:
            B, T, H, W, C = trend.shape
            x_t = trend.permute(0, 1, 4, 2, 3).reshape(B, T*C, H, W)
            out_t = self.trend_branch(x_t)
            outputs.append(out_t)

        # Fusion
        if len(outputs) == 0:
            raise ValueError("At least one branch must be provided")

        # Stack outputs: [N_branches, B, 2, H, W]
        outputs = torch.stack(outputs, dim=0)

        # Apply fusion weights: [N_branches, 2, H, W]
        weights = F.softmax(self.fusion_weights, dim=0)  # Normalize across branches

        # Weighted sum: [B, 2, H, W]
        fused = (outputs * weights.unsqueeze(1)).sum(dim=0)

        # Permute back to [B, H, W, 2]
        pred = fused.permute(0, 2, 3, 1)

        return pred

    def extract_features(self, closeness=None, period=None, trend=None):
        """
        Extract intermediate features for adapter integration

        Returns:
            features: [B, hidden_channels, H, W] - features before final fusion
        """
        features_list = []

        # Closeness branch features
        if self.closeness_len > 0 and closeness is not None:
            B, T, H, W, C = closeness.shape
            x_c = closeness.permute(0, 1, 4, 2, 3).reshape(B, T*C, H, W)

            # Extract features (before final conv_out)
            out = self.closeness_branch.conv_in(x_c)
            out = self.closeness_branch.bn_in(out)
            out = self.closeness_branch.relu(out)

            for resunit in self.closeness_branch.resunits:
                out = resunit(out)

            features_list.append(out)  # [B, hidden_channels, H, W]

        # Period branch features
        if self.period_len > 0 and period is not None:
            B, T, H, W, C = period.shape
            x_p = period.permute(0, 1, 4, 2, 3).reshape(B, T*C, H, W)

            out = self.period_branch.conv_in(x_p)
            out = self.period_branch.bn_in(out)
            out = self.period_branch.relu(out)

            for resunit in self.period_branch.resunits:
                out = resunit(out)

            features_list.append(out)

        # Trend branch features
        if self.trend_len > 0 and trend is not None:
            B, T, H, W, C = trend.shape
            x_t = trend.permute(0, 1, 4, 2, 3).reshape(B, T*C, H, W)

            out = self.trend_branch.conv_in(x_t)
            out = self.trend_branch.bn_in(out)
            out = self.trend_branch.relu(out)

            for resunit in self.trend_branch.resunits:
                out = resunit(out)

            features_list.append(out)

        # Average features from all branches
        if len(features_list) > 0:
            features = torch.mean(torch.stack(features_list), dim=0)
        else:
            raise ValueError("At least one branch must be provided")

        return features  # [B, hidden_channels, H, W]

    def predict(self, features):
        """
        Predict from modulated features

        Args:
            features: [B, hidden_channels, H, W] - features from adapter

        Returns:
            pred: [B, 2, H, W] - predicted flow
        """
        # Use closeness branch's output layer (they all have same structure)
        if self.closeness_len > 0:
            pred = self.closeness_branch.conv_out(features)
        elif self.period_len > 0:
            pred = self.period_branch.conv_out(features)
        else:
            pred = self.trend_branch.conv_out(features)

        return pred  # [B, 2, H, W]


def test_st_resnet():
    """Quick test of ST-ResNet"""
    print("=" * 80)
    print("Testing ST-ResNet")
    print("=" * 80)

    B = 4  # Batch size
    H, W = 20, 20  # Grid size

    # Create model
    model = STResNet(
        closeness_len=6,
        period_len=3,
        trend_len=2,
        hidden_channels=64,
        n_resunits=4,
        grid_height=H,
        grid_width=W
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create sample inputs
    closeness = torch.randn(B, 6, H, W, 2)  # Last 6 timesteps
    period = torch.randn(B, 3, H, W, 2)     # Same hour, 1-3 days ago
    trend = torch.randn(B, 2, H, W, 2)      # Same weekday, 1-2 weeks ago

    print(f"\nInput shapes:")
    print(f"  Closeness: {closeness.shape}")
    print(f"  Period: {period.shape}")
    print(f"  Trend: {trend.shape}")

    # Forward pass
    with torch.no_grad():
        pred = model(closeness=closeness, period=period, trend=trend)

    print(f"\nOutput shape: {pred.shape}")
    print(f"Output range: [{pred.min():.4f}, {pred.max():.4f}]")

    # Test with only closeness
    print("\n" + "-" * 80)
    print("Test with only closeness branch:")
    model_simple = STResNet(
        closeness_len=6,
        period_len=0,
        trend_len=0,
        hidden_channels=32,
        n_resunits=2,
        grid_height=H,
        grid_width=W
    )

    print(f"Model parameters: {sum(p.numel() for p in model_simple.parameters()):,}")

    with torch.no_grad():
        pred_simple = model_simple(closeness=closeness)

    print(f"Output shape: {pred_simple.shape}")

    print("\n" + "=" * 80)
    print("ST-ResNet test passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_st_resnet()
