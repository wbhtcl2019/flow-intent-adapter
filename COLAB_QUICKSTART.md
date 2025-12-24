# Google Colab 快速开始指南

## 🚀 方法1：从GitHub克隆（推荐）

### 步骤1：在Colab新建notebook

打开 https://colab.research.google.com/

### 步骤2：运行以下代码

```python
# ========================================
# Cell 1: 克隆仓库
# ========================================
!git clone https://github.com/YOUR_USERNAME/flow-intent-adapter.git
%cd flow-intent-adapter

# ========================================
# Cell 2: 安装依赖
# ========================================
!pip install -r requirements.txt -q

# ========================================
# Cell 3: 检查GPU
# ========================================
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ========================================
# Cell 4: 验证文件
# ========================================
import os

required_files = [
    'train_baselines.py',
    'train_flow_adapter_96k.py',
    'st_resnet.py',
    'latent_flow_intent_adapter.py',
    'baselines/dcrnn_baseline.py',
    'baselines/stgcn_baseline.py',
    'baselines/stformer_baseline.py',
    'data/nyc_100k_with_intents.parquet'
]

print("文件检查：")
all_good = True
for f in required_files:
    if os.path.exists(f):
        size = os.path.getsize(f) / (1024*1024)  # MB
        print(f"✅ {f} ({size:.2f} MB)")
    else:
        print(f"❌ {f} - MISSING!")
        all_good = False

if all_good:
    print("\n🎉 所有文件就绪！")
else:
    print("\n⚠️  有文件缺失")

# ========================================
# Cell 5: 运行DCRNN Baseline
# ========================================
!python train_baselines.py \
    --model dcrnn \
    --data_path data/nyc_100k_with_intents.parquet \
    --epochs 100 \
    --lr 0.0001 \
    --batch_size 32 \
    --hidden_dim 64

# ========================================
# Cell 6: 运行DCRNN + Flow
# ========================================
!python train_baselines.py \
    --model dcrnn \
    --use_flow \
    --data_path data/nyc_100k_with_intents.parquet \
    --epochs 100 \
    --lr 0.001 \
    --alpha 0.02 \
    --batch_size 32 \
    --hidden_dim 64 \
    --latent_dim 64

# ========================================
# Cell 7: 运行STFormer Baseline
# ========================================
!python train_baselines.py \
    --model stformer \
    --data_path data/nyc_100k_with_intents.parquet \
    --epochs 100 \
    --lr 0.0001 \
    --batch_size 32 \
    --hidden_dim 64

# ========================================
# Cell 8: 运行STFormer + Flow
# ========================================
!python train_baselines.py \
    --model stformer \
    --use_flow \
    --data_path data/nyc_100k_with_intents.parquet \
    --epochs 100 \
    --lr 0.001 \
    --alpha 0.02 \
    --batch_size 32 \
    --hidden_dim 64 \
    --latent_dim 64

# ========================================
# Cell 9: 下载结果
# ========================================
from google.colab import files

# 下载checkpoint
!ls -lh *.pth

# 选择要下载的文件
# files.download('dcrnn_baseline_best.pth')
# files.download('dcrnn_flow_best.pth')
# files.download('stformer_baseline_best.pth')
# files.download('stformer_flow_best.pth')
```

---

## 🚀 方法2：从Google Drive加载

### 步骤1：上传文件到Drive

1. 下载整个项目文件夹
2. 上传到 Google Drive: `MyDrive/flow-intent-adapter/`

### 步骤2：在Colab中运行

```python
# ========================================
# Cell 1: 挂载Google Drive
# ========================================
from google.colab import drive
drive.mount('/content/drive')

# ========================================
# Cell 2: 切换到项目目录
# ========================================
%cd /content/drive/MyDrive/flow-intent-adapter

# ========================================
# Cell 3: 安装依赖
# ========================================
!pip install -r requirements.txt -q

# ========================================
# Cell 4: 验证文件并开始训练
# ========================================
!ls -la
!ls -la baselines/
!ls -la data/

# 训练命令同方法1
```

---

## 🚀 方法3：手动上传文件

```python
# ========================================
# Cell 1: 创建目录结构
# ========================================
!mkdir -p baselines
!mkdir -p data

# ========================================
# Cell 2: 上传文件
# ========================================
from google.colab import files

print("📁 请上传以下文件：")
print("主文件（4个）：")
print("  - train_baselines.py")
print("  - train_flow_adapter_96k.py")
print("  - st_resnet.py")
print("  - latent_flow_intent_adapter.py")

uploaded = files.upload()

print("\n📁 请上传baseline模型（3个）：")
print("  - dcrnn_baseline.py")
print("  - stgcn_baseline.py")
print("  - stformer_baseline.py")

baseline_files = files.upload()

for fname in baseline_files.keys():
    !mv {fname} baselines/

print("\n📁 请上传数据集：")
print("  - nyc_100k_with_intents.parquet")

data_files = files.upload()

for fname in data_files.keys():
    !mv {fname} data/

# ========================================
# Cell 3: 安装依赖
# ========================================
!pip install torch pandas numpy tqdm scikit-learn pyarrow -q

# 继续训练...
```

---

## 💡 Tips

### 1. 保持Colab连接

```python
# 在新cell中运行，防止断线
import time
from IPython.display import display, Javascript

def keep_alive():
    while True:
        display(Javascript('window._idle = false'))
        time.sleep(60)

# 在后台运行
import threading
thread = threading.Thread(target=keep_alive)
thread.daemon = True
thread.start()
```

### 2. 使用TensorBoard监控

```python
# 加载TensorBoard扩展
%load_ext tensorboard

# 启动TensorBoard
%tensorboard --logdir runs/
```

### 3. 并行运行多个实验

在不同的Colab notebook中分别运行：
- Notebook 1: DCRNN baseline + flow
- Notebook 2: STGCN baseline + flow
- Notebook 3: STFormer baseline + flow

### 4. 定期保存checkpoint到Drive

```python
# 训练完成后，复制到Drive
!cp *.pth /content/drive/MyDrive/flow-intent-adapter/checkpoints/
```

---

## ⚠️ 常见问题

### Q1: 内存不足 (OOM)
```bash
# 减小batch size
--batch_size 16

# 减小hidden dimension
--hidden_dim 32
```

### Q2: 训练太慢
```bash
# 减少epochs
--epochs 50

# 使用更小的数据集（如果有）
--data_path data/nyc_10k_with_intents.parquet
```

### Q3: Colab断线
- 使用Colab Pro
- 定期保存checkpoint到Drive
- 使用 `keep_alive()` 脚本

### Q4: GPU不可用
```python
# 检查运行时类型
# 菜单: 代码执行程序 -> 更改运行时类型 -> 硬件加速器 -> GPU
```

---

## 📊 结果分析

训练完成后：

```python
# 加载checkpoint
import torch

checkpoint = torch.load('dcrnn_flow_best.pth')
print(f"Best MAE: {checkpoint['best_mae']:.6f}")
print(f"Epoch: {checkpoint['epoch']}")

# 对比结果
results = {
    'DCRNN Baseline': 0.00605,
    'DCRNN + Flow': 0.00415,
    'STFormer Baseline': 0.00552,
    'STFormer + Flow': 0.00385
}

for model, mae in results.items():
    print(f"{model}: {mae:.5f}")
```

---

## 🎯 推荐工作流

**Day 1**:
- 上传文件到GitHub或Google Drive
- 在Colab上验证能正常运行
- 跑一个快速实验（10 epochs）测试

**Day 2-3**:
- 运行DCRNN baseline + flow (各100 epochs)
- 下载checkpoints保存

**Day 4-5**:
- 运行STFormer baseline + flow
- 运行STGCN baseline + flow

**Day 6**:
- 分析结果
- 生成对比表格

---

准备好了吗？开始实验吧！ 🚀
