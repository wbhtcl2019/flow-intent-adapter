# 🐛 数据全0问题 - 调试指南

## 问题现象

```
X range: [0.00, 0.00]  ← 全是0
Y range: [0.00, 0.00]  ← 全是0
```

## 已尝试的修复

1. ✅ 修复grid映射逻辑（选择top-k频繁grids）
2. ✅ 修复2D坐标映射
3. ❌ 数据仍然全0

## 可能的根本原因

### 原因1: grid_to_idx 映射仍然有问题

**验证代码**（在Colab中运行）：
```python
import pandas as pd
from train_flow_adapter_96k import FlowDataset

df = pd.read_parquet('data/nyc_2m_jan_feb_with_intents.parquet')
df_small = df.iloc[:10000]

# 手动检查grid_to_idx
dataset = FlowDataset(df_small, n_tiles=32, closeness_len=12)

print("=" * 60)
print("Grid mapping 诊断")
print("=" * 60)
print(f"Total unique grids in grid_to_idx: {len(dataset.grid_to_idx)}")
print(f"Grid indices range: [0, {max(dataset.grid_to_idx.values())}]")
print(f"\nFirst 10 mappings:")
for k, v in list(dataset.grid_to_idx.items())[:10]:
    print(f"  Grid {k} -> Index {v}")

# 检查数据中的实际grids
if 'pickup_grid' in df_small.columns:
    actual_grids = set(df_small['pickup_grid'].unique())
else:
    # Compute grids
    grid_size = 0.01
    lat_min, lon_min = 40.5, -74.5
    df_small['pickup_grid'] = (
        ((df_small['pickup_latitude'] - lat_min) / grid_size).astype(int) * 1000 +
        ((df_small['pickup_longitude'] - lon_min) / grid_size).astype(int)
    )
    actual_grids = set(df_small['pickup_grid'].unique())

print(f"\nActual grids in data: {len(actual_grids)}")
print(f"Sample actual grids: {list(actual_grids)[:10]}")

# 检查overlap
mapped_grids = set(dataset.grid_to_idx.keys())
overlap = actual_grids & mapped_grids
print(f"\nOverlap: {len(overlap)} grids")
print(f"Coverage: {len(overlap) / len(actual_grids) * 100:.1f}%")

if len(overlap) == 0:
    print("\n🚨 问题确认：grid_to_idx 和实际数据没有任何overlap！")
```

### 原因2: _compute_grids 没有被正确调用

在 `_build_grid_mapping` 中，`df = self._compute_grids(df)` 这一行可能没有生效（pandas的copy问题）。

**修复**：
```python
def _build_grid_mapping(self, df):
    # 强制copy
    df = df.copy()

    # Compute grid if not exist
    if 'pickup_grid' not in df.columns:
        df = self._compute_grids(df)

    # ... 后续代码
```

### 原因3: SettingWithCopyWarning 导致修改没生效

那些警告不是无害的！可能grid计算根本没有写入df。

**修复**：
```python
def _compute_grids(self, df):
    """Compute grid cells from lat/lon"""
    # 强制copy
    df = df.copy()

    grid_size = 0.01
    lat_min, lon_min = 40.5, -74.5

    df['pickup_grid_x'] = ((df['pickup_longitude'] - lon_min) / grid_size).astype(int)
    df['pickup_grid_y'] = ((df['pickup_latitude'] - lat_min) / grid_size).astype(int)
    df['dropoff_grid_x'] = ((df['dropoff_longitude'] - lon_min) / grid_size).astype(int)
    df['dropoff_grid_y'] = ((df['dropoff_latitude'] - lat_min) / grid_size).astype(int)

    df['pickup_grid'] = df['pickup_grid_y'] * 1000 + df['pickup_grid_x']
    df['dropoff_grid'] = df['dropoff_grid_y'] * 1000 + df['dropoff_grid_x']

    return df
```

## 明天的行动计划

### 步骤1: 运行上面的诊断代码
找出grid_to_idx和实际数据的overlap是否为0

### 步骤2: 根据结果修复
- 如果overlap=0 → 修复_compute_grids（加.copy()）
- 如果overlap>0但数据还是0 → 检查_aggregate_flows的逻辑

### 步骤3: 最简单的临时方案
如果还是不行，直接用预计算好的grid：

```python
# 在读取数据时就预先计算grid
df = pd.read_parquet('data/nyc_2m_jan_feb_with_intents.parquet')

# 预计算grid（不依赖FlowDataset）
grid_size = 0.01
lat_min, lon_min = 40.5, -74.5

df['pickup_grid'] = (
    ((df['pickup_latitude'] - lat_min) / grid_size).astype(int) * 1000 +
    ((df['pickup_longitude'] - lon_min) / grid_size).astype(int)
)
df['dropoff_grid'] = (
    ((df['dropoff_latitude'] - lat_min) / grid_size).astype(int) * 1000 +
    ((df['dropoff_longitude'] - lon_min) / grid_size).astype(int)
)

# 然后再split和创建dataset
df_train = df.iloc[:train_size]
dataset = FlowDataset(df_train, n_tiles=32, closeness_len=12)
```

## 终极备用方案

如果grid方案一直有问题，可以考虑：

1. **暂时用96k的小数据**（`nyc_100k_with_intents.parquet`）验证模型逻辑
2. **或者直接用n_tiles=100**（匹配原始的grid数量）
3. **或者修改模型架构**，不用grid，用graph node表示

## 相关文件

- `train_flow_adapter_96k.py` - FlowDataset定义
- `train_baselines.py` - 主训练脚本
- `baselines/dcrnn_baseline.py` - DCRNN模型

---

**今天辛苦了！明天继续 💪**
