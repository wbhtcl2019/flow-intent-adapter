# 🚀 Colab 2M数据训练 - 快速开始

## 📋 完整的Colab训练代码（复制粘贴即可）

### Cell 1: 克隆仓库并检查GPU
```python
# 克隆仓库
!git clone https://github.com/wbhtcl2019/flow-intent-adapter.git
%cd flow-intent-adapter

# 检查GPU
import torch
import os

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_memory:.2f} GB")

    if 'T4' in gpu_name or 'V100' in gpu_name or 'A100' in gpu_name:
        print("✅ GPU足够强，可以跑2M数据")
    else:
        print("⚠️  GPU较弱，建议减小batch_size")
```

### Cell 2: 挂载Google Drive并复制2M数据
```python
from google.colab import drive
drive.mount('/content/drive')

# 复制2M数据（你的数据在Google Drive上）
# 修改路径为你的实际路径
!cp "/content/drive/MyDrive/nyc_2m_jan_feb_with_intents.parquet" data/

# 验证数据
!ls -lh data/
```

### Cell 3: 安装依赖并验证文件
```python
# 安装依赖
!pip install -r requirements.txt -q

# 验证所有文件
import os

files_to_check = [
    ('train_baselines.py', 10000),
    ('baselines/dcrnn_baseline.py', 10000),
    ('baselines/stformer_baseline.py', 10000),
    ('data/nyc_2m_jan_feb_with_intents.parquet', 50000000)  # 至少50MB
]

print("📁 文件检查:")
all_good = True
for fname, min_size in files_to_check:
    if os.path.exists(fname):
        size = os.path.getsize(fname)
        size_mb = size / (1024*1024)
        if size >= min_size:
            print(f"✅ {fname} ({size_mb:.2f} MB)")
        else:
            print(f"⚠️  {fname} ({size_mb:.2f} MB) - 可能太小")
            all_good = False
    else:
        print(f"❌ {fname} - 缺失!")
        all_good = False

if all_good:
    print("\n🎉 所有文件就绪！")
else:
    print("\n⚠️  请检查文件")
```

### Cell 4: 【实验1】DCRNN Baseline (2M)
```python
# 预计训练时间: 4-6小时
!python train_baselines.py \
    --model dcrnn \
    --data_path data/nyc_2m_jan_feb_with_intents.parquet \
    --epochs 100 \
    --lr 0.0001 \
    --batch_size 32 \
    --hidden_dim 64 \
    --n_tiles 32 \
    --closeness_len 12

# 训练完成后保存结果到Drive
!cp dcrnn_baseline_best.pth /content/drive/MyDrive/checkpoints/dcrnn_baseline_2M.pth
```

### Cell 5: 【实验2】DCRNN + Flow Adapter (2M)
```python
# 预计训练时间: 5-7小时
!python train_baselines.py \
    --model dcrnn \
    --use_flow \
    --data_path data/nyc_2m_jan_feb_with_intents.parquet \
    --epochs 100 \
    --lr 0.001 \
    --alpha 0.02 \
    --batch_size 32 \
    --hidden_dim 64 \
    --latent_dim 64 \
    --n_tiles 32 \
    --closeness_len 12

# 保存结果
!cp dcrnn_flow_best.pth /content/drive/MyDrive/checkpoints/dcrnn_flow_2M.pth
```

### Cell 6: 【实验3】STFormer Baseline (2M)
```python
# 预计训练时间: 3-5小时
!python train_baselines.py \
    --model stformer \
    --data_path data/nyc_2m_jan_feb_with_intents.parquet \
    --epochs 100 \
    --lr 0.0001 \
    --batch_size 32 \
    --hidden_dim 64 \
    --n_tiles 32 \
    --closeness_len 12

# 保存
!cp stformer_baseline_best.pth /content/drive/MyDrive/checkpoints/stformer_baseline_2M.pth
```

### Cell 7: 【实验4】STFormer + Flow Adapter (2M)
```python
# 预计训练时间: 4-6小时
!python train_baselines.py \
    --model stformer \
    --use_flow \
    --data_path data/nyc_2m_jan_feb_with_intents.parquet \
    --epochs 100 \
    --lr 0.001 \
    --alpha 0.02 \
    --batch_size 32 \
    --hidden_dim 64 \
    --latent_dim 64 \
    --n_tiles 32 \
    --closeness_len 12

# 保存
!cp stformer_flow_best.pth /content/drive/MyDrive/checkpoints/stformer_flow_2M.pth
```

### Cell 8: 【可选】STGCN实验
```python
# Baseline
!python train_baselines.py \
    --model stgcn \
    --data_path data/nyc_2m_jan_feb_with_intents.parquet \
    --epochs 100 \
    --lr 0.0001 \
    --batch_size 32

!cp stgcn_baseline_best.pth /content/drive/MyDrive/checkpoints/stgcn_baseline_2M.pth

# + Flow
!python train_baselines.py \
    --model stgcn \
    --use_flow \
    --data_path data/nyc_2m_jan_feb_with_intents.parquet \
    --epochs 100 \
    --lr 0.001 \
    --alpha 0.02 \
    --batch_size 32

!cp stgcn_flow_best.pth /content/drive/MyDrive/checkpoints/stgcn_flow_2M.pth
```

### Cell 9: 结果汇总
```python
import torch
import pandas as pd
import os

results = []

models = [
    ('dcrnn_baseline_best.pth', 'DCRNN Baseline'),
    ('dcrnn_flow_best.pth', 'DCRNN + Flow'),
    ('stformer_baseline_best.pth', 'STFormer Baseline'),
    ('stformer_flow_best.pth', 'STFormer + Flow'),
]

for fname, name in models:
    if os.path.exists(fname):
        ckpt = torch.load(fname, map_location='cpu')
        mae = ckpt.get('best_mae', 'N/A')
        epoch = ckpt.get('epoch', 'N/A')
        results.append({
            'Model': name,
            'Best MAE': f"{mae:.6f}" if isinstance(mae, float) else mae,
            'Epoch': epoch
        })

df = pd.DataFrame(results)
print("\n" + "="*60)
print("📊 2M数据实验结果汇总")
print("="*60)
print(df.to_string(index=False))
print("="*60)

# 计算提升
if len(results) >= 2:
    for i in range(0, len(results), 2):
        if i+1 < len(results):
            baseline_mae = float(results[i]['Best MAE'])
            flow_mae = float(results[i+1]['Best MAE'])
            improvement = (baseline_mae - flow_mae) / baseline_mae * 100
            print(f"\n{results[i]['Model']} → {results[i+1]['Model']}")
            print(f"提升: {improvement:.1f}%")
```

---

## 🎯 使用步骤

1. **打开Google Colab**: https://colab.research.google.com
2. **新建Notebook**
3. **复制上面的Cell 1-9**，依次运行
4. **等待结果**（预计每个模型4-6小时）

---

## 💡 重要提示

### 数据路径
确保你的2M数据在Google Drive上的路径正确：
```python
# 如果你的数据在不同位置，修改这行：
!cp "/content/drive/MyDrive/你的路径/nyc_2m_jan_feb_with_intents.parquet" data/
```

### Checkpoint保存
所有checkpoint会自动保存到Drive的 `MyDrive/checkpoints/` 文件夹

### 如果Colab断线
重新连接后，重新运行Cell 1-3，然后继续未完成的实验

### 如果遇到OOM
减小batch_size:
```bash
--batch_size 16  # 或 8
```

---

## 📊 预期结果

| Model | 2M Baseline | 2M +Flow | 提升幅度 |
|-------|-------------|----------|---------|
| DCRNN | ~0.0025 | ~0.0012 | 预期50%+ |
| STFormer | ~0.0022 | ~0.0010 | 预期55%+ |
| STGCN | ~0.0028 | ~0.0015 | 预期45%+ |

---

准备好了吗？复制Cell 1开始训练！ 🚀
