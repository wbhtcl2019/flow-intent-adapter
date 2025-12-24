# 🚀 2M数据实验指南

## 📊 实验目标

在2M NYC数据集上验证：
1. DCRNN baseline + Flow Adapter
2. STFormer baseline + Flow Adapter
3. STGCN baseline + Flow Adapter (可选)

预期结果：比96k提升幅度更大（参考ST-ResNet: 33% → 55%）

---

## 📁 准备2M数据集

### 方法1：如果你已有2M数据

将 `nyc_2m_jan_feb_with_intents.parquet` 上传到GitHub仓库的 `data/` 文件夹

**注意**：GitHub有100MB文件限制，需要使用Git LFS

```bash
# 在本地
cd "C:\coding\PhD-DS\didi-code\colab_upload"

# 安装Git LFS
git lfs install

# 追踪大文件
git lfs track "data/*.parquet"
git add .gitattributes

# 复制2M数据（从本地路径）
copy "D:\nyc-taxi-project\processed\nyc_2m_jan_feb_with_intents.parquet" data\

# 提交并推送
git add data/nyc_2m_jan_feb_with_intents.parquet
git commit -m "Add 2M dataset"
git push
```

### 方法2：上传到Google Drive（推荐）

**你的数据已在Google Drive上**，直接在Colab中使用：

```python
# 在Colab中
from google.colab import drive
drive.mount('/content/drive')

# 复制数据到工作目录（修改路径为你的实际路径）
!cp "/content/drive/MyDrive/nyc_2m_jan_feb_with_intents.parquet" data/

# 验证
!ls -lh data/*.parquet
```

### 方法3：从Kaggle下载（如果你上传到Kaggle）

```python
# 在Colab中
!pip install kaggle -q
!mkdir -p ~/.kaggle
!cp /path/to/kaggle.json ~/.kaggle/
!kaggle datasets download -d your-username/nyc-2m-intents
!unzip nyc-2m-intents.zip -d data/
```

---

## 🎯 完整的Colab训练脚本（2M数据）

复制这个到Colab，按顺序运行：

### Cell 1: 克隆仓库
```python
!git clone https://github.com/wbhtcl2019/flow-intent-adapter.git
%cd flow-intent-adapter
```

### Cell 2: 检查GPU和显存
```python
import torch
import os

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_memory:.2f} GB")

    # 检查是否是好GPU
    if 'T4' in gpu_name or 'V100' in gpu_name or 'A100' in gpu_name:
        print("✅ GPU足够强，可以跑2M数据")
    else:
        print("⚠️  GPU较弱，建议减小batch_size")
```

### Cell 3: 挂载Drive并复制2M数据（如果需要）
```python
from google.colab import drive
drive.mount('/content/drive')

# 复制2M数据（修改路径为你的实际路径）
!cp "/content/drive/MyDrive/nyc_2m_jan_feb_with_intents.parquet" data/

# 验证数据
!ls -lh data/
```

### Cell 4: 安装依赖
```python
!pip install -r requirements.txt -q
```

### Cell 5: 验证文件
```python
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

### Cell 6: 【实验1】DCRNN Baseline (2M)
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

# 训练完成后保存结果
!cp dcrnn_baseline_best.pth /content/drive/MyDrive/checkpoints/dcrnn_baseline_2M.pth
```

### Cell 7: 【实验2】DCRNN + Flow Adapter (2M)
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

### Cell 8: 【实验3】STFormer Baseline (2M)
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

### Cell 9: 【实验4】STFormer + Flow Adapter (2M)
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

### Cell 10: 【可选】STGCN实验
```python
# Baseline
!python train_baselines.py \
    --model stgcn \
    --data_path data/nyc_2m_jan_feb_with_intents.parquet \
    --epochs 100 \
    --lr 0.0001 \
    --batch_size 32

# + Flow
!python train_baselines.py \
    --model stgcn \
    --use_flow \
    --data_path data/nyc_2m_jan_feb_with_intents.parquet \
    --epochs 100 \
    --lr 0.001 \
    --alpha 0.02 \
    --batch_size 32
```

### Cell 11: 结果汇总
```python
import torch
import pandas as pd

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

## ⚙️ 参数调优建议

### 如果遇到OOM (Out of Memory)

```python
# 减小batch size
--batch_size 16  # 或 8

# 减小hidden dimension
--hidden_dim 32

# 减小序列长度
--closeness_len 6
```

### 如果想加速训练

```python
# 减少epochs
--epochs 50

# 使用更大的batch size（如果显存够）
--batch_size 64

# 使用混合精度训练（需要修改代码）
```

### 最佳配置（基于ST-ResNet经验）

```python
# DCRNN最佳配置
--lr 0.001 --alpha 0.02 --batch_size 32

# STFormer最佳配置
--lr 0.001 --alpha 0.02 --batch_size 32
```

---

## 📊 预期结果（基于ST-ResNet scaling）

| Model | 96k Baseline | 96k +Flow | 2M Baseline | 2M +Flow | 提升幅度 |
|-------|-------------|-----------|-------------|----------|---------|
| ST-ResNet | 0.00610 | 0.00408 | 0.00298 | 0.00134 | 33%→55% |
| DCRNN | ~0.0060 | ~0.0040 | ~0.0025 | ~0.0012 | 预期50%+ |
| STFormer | ~0.0055 | ~0.0038 | ~0.0022 | ~0.0010 | 预期55%+ |

---

## 🚀 并行训练策略

### 方案A：开4个Colab Notebook并行
```
Notebook 1: DCRNN baseline
Notebook 2: DCRNN + Flow
Notebook 3: STFormer baseline
Notebook 4: STFormer + Flow
```

### 方案B：顺序训练（节省资源）
```
Day 1: DCRNN baseline + Flow
Day 2: STFormer baseline + Flow
Day 3: 分析结果
```

---

## 💡 重要提示

1. **定期保存checkpoint到Drive**：防止Colab断线丢失结果
2. **监控训练曲线**：确保loss在下降
3. **记录超参数**：每个实验的配置都要记录
4. **对比96k结果**：看提升是否符合预期

---

## 🎯 下一步

1. **准备2M数据**（上传到Drive或GitHub LFS）
2. **打开Colab**：https://colab.research.google.com
3. **复制Cell 1-11**，开始训练
4. **等待结果**（预计每个模型4-6小时）

准备好了吗？开始2M数据的大规模实验！ 🚀
