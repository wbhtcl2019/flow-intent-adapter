# 📦 文件上传指南

## ✅ 已整理好的文件清单

你的 `colab_upload/` 文件夹中包含以下文件：

```
colab_upload/
├── README.md                          # 项目说明
├── requirements.txt                   # 依赖列表
├── .gitignore                         # Git忽略文件
├── COLAB_QUICKSTART.md               # Colab快速开始
├── UPLOAD_GUIDE.md                   # 本文件
├── train_baselines.py                 # 主训练脚本 (15KB)
├── train_flow_adapter_96k.py          # 数据加载 (14KB)
├── st_resnet_baseline.py              # ST-ResNet模型 (12KB)
├── latent_flow_intent_adapter.py     # Flow Adapter (15KB)
├── baselines/
│   ├── dcrnn_baseline.py             # DCRNN (15KB)
│   ├── stgcn_baseline.py             # STGCN (14KB)
│   └── stformer_baseline.py          # STFormer (15KB)
└── data/
    └── nyc_100k_with_intents.parquet # 数据集 (4.7MB)

总大小：约 5MB
```

---

## 🚀 方案A：上传到GitHub（推荐）

### 步骤1：初始化Git仓库

在 `colab_upload/` 文件夹中打开终端：

```bash
cd C:\coding\PhD-DS\didi-code\colab_upload

# 初始化Git
git init

# 添加所有文件
git add .

# 提交
git commit -m "Initial commit: Flow Intent Adapter baseline experiments"
```

### 步骤2：创建GitHub仓库

1. 打开 https://github.com/new
2. 仓库名：`flow-intent-adapter` 或 `traffic-flow-prediction`
3. 选择 **Private**（如果论文还未发表）
4. **不要**勾选 "Add a README file"
5. 点击 "Create repository"

### 步骤3：推送到GitHub

GitHub会显示命令，复制并运行：

```bash
git remote add origin https://github.com/YOUR_USERNAME/flow-intent-adapter.git
git branch -M main
git push -u origin main
```

### 步骤4：在Colab中使用

```python
# 在Colab中运行
!git clone https://github.com/YOUR_USERNAME/flow-intent-adapter.git
%cd flow-intent-adapter
!pip install -r requirements.txt
```

**优点**：
- ✅ 版本控制
- ✅ 随时更新代码
- ✅ 可以在任何地方克隆
- ✅ 不占用Google Drive空间

---

## 🚀 方案B：上传到Google Drive

### 步骤1：打开Google Drive

访问 https://drive.google.com

### 步骤2：上传文件夹

1. 点击左上角 **"新建" → "文件夹上传"**
2. 选择 `C:\coding\PhD-DS\didi-code\colab_upload` 文件夹
3. 等待上传完成（约5MB，几分钟）

### 步骤3：在Colab中使用

```python
from google.colab import drive
drive.mount('/content/drive')

# 假设你上传到 MyDrive/colab_upload
%cd /content/drive/MyDrive/colab_upload
!pip install -r requirements.txt
```

**优点**：
- ✅ 简单直接
- ✅ 不需要GitHub账号
- ✅ 文件持久保存

**缺点**：
- ❌ 占用Drive空间
- ❌ 没有版本控制

---

## 🚀 方案C：压缩后上传

### 步骤1：压缩文件

在Windows中：
1. 右键 `colab_upload` 文件夹
2. 选择 "发送到 → 压缩(zipped)文件夹"
3. 得到 `colab_upload.zip`（约4MB）

### 步骤2：上传到Colab

```python
from google.colab import files
import zipfile
import os

# 上传zip文件
uploaded = files.upload()

# 解压
!unzip -q colab_upload.zip
%cd colab_upload

# 安装依赖
!pip install -r requirements.txt
```

**优点**：
- ✅ 上传快
- ✅ 一次上传所有文件

**缺点**：
- ❌ Colab重启后需要重新上传

---

## 📋 验证文件完整性

无论用哪种方案，上传后运行这个检查：

```python
import os

required_files = {
    'train_baselines.py': 15000,
    'train_flow_adapter_96k.py': 14000,
    'st_resnet_baseline.py': 12000,
    'latent_flow_intent_adapter.py': 15000,
    'baselines/dcrnn_baseline.py': 15000,
    'baselines/stgcn_baseline.py': 14000,
    'baselines/stformer_baseline.py': 15000,
    'data/nyc_100k_with_intents.parquet': 4700000
}

print("文件检查：")
all_good = True
for fname, min_size in required_files.items():
    if os.path.exists(fname):
        size = os.path.getsize(fname)
        if size >= min_size:
            print(f"✅ {fname} ({size/1024:.1f} KB)")
        else:
            print(f"⚠️  {fname} ({size/1024:.1f} KB) - 可能不完整")
            all_good = False
    else:
        print(f"❌ {fname} - 缺失!")
        all_good = False

if all_good:
    print("\n🎉 所有文件就绪！")
else:
    print("\n⚠️  请检查文件")
```

---

## 💡 我的推荐

### 如果你熟悉Git：
→ **方案A (GitHub)** ⭐⭐⭐⭐⭐

### 如果你不会Git：
→ **方案B (Google Drive)** ⭐⭐⭐⭐

### 如果你想快速测试：
→ **方案C (ZIP上传)** ⭐⭐⭐

---

## 🎯 下一步

文件上传后，参考 `COLAB_QUICKSTART.md` 开始训练！

主要命令：

```bash
# DCRNN baseline
python train_baselines.py --model dcrnn --epochs 100 --lr 0.0001

# DCRNN + Flow
python train_baselines.py --model dcrnn --use_flow --epochs 100 --lr 0.001 --alpha 0.02

# STFormer baseline
python train_baselines.py --model stformer --epochs 100 --lr 0.0001

# STFormer + Flow
python train_baselines.py --model stformer --use_flow --epochs 100 --lr 0.001 --alpha 0.02
```

---

## 📞 需要帮助？

如果遇到问题：
1. 检查文件是否完整（运行上面的验证脚本）
2. 确认Python包已安装（`!pip list`）
3. 检查GPU是否可用（`torch.cuda.is_available()`）
4. 查看错误日志

准备好了吗？开始上传吧！ 🚀
