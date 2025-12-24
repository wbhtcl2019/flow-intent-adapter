# 🚀 推送到GitHub详细步骤

## ✅ 已完成的准备工作

- ✅ Git仓库已初始化
- ✅ 所有文件已提交
- ✅ 远程仓库已配置: `https://github.com/wbhtcl2019/flow-intent-adapter.git`
- ✅ 分支已重命名为 `main`

---

## 📋 第一步：在GitHub上创建仓库

### 方法1：网页创建（推荐，简单）

1. **打开浏览器，访问**:
   ```
   https://github.com/new
   ```

2. **填写仓库信息**:
   - **Repository name**: `flow-intent-adapter`
   - **Description**: `Flow Intent Adapter for Traffic Flow Prediction (KDD 2025)`
   - **Public or Private**:
     - 选择 **Private** ✓ （论文发表前推荐）
     - 或 **Public** （如果你想公开）
   - **❌ 不要勾选** "Add a README file"
   - **❌ 不要勾选** "Add .gitignore"
   - **❌ 不要勾选** "Choose a license"

3. **点击**: `Create repository`

---

## 📋 第二步：推送代码

仓库创建后，在 `C:\coding\PhD-DS\didi-code\colab_upload` 文件夹中：

### 方法A：使用命令行（推荐）

打开终端（PowerShell或Git Bash），运行：

```bash
cd "C:\coding\PhD-DS\didi-code\colab_upload"
git push -u origin main
```

如果提示输入用户名和密码：
- **用户名**: `wbhtcl2019`
- **密码**: 使用 **Personal Access Token**（不是GitHub密码）

### 方法B：如果没有Token，创建一个

1. 访问: https://github.com/settings/tokens
2. 点击 `Generate new token` → `Generate new token (classic)`
3. 填写:
   - **Note**: `Colab Upload`
   - **Expiration**: `No expiration` 或 `90 days`
   - **Select scopes**: 勾选 `repo` (全部)
4. 点击 `Generate token`
5. **复制token**（只会显示一次！）
6. 在git push时，用这个token作为密码

### 方法C：使用GitHub Desktop（最简单）

如果你安装了GitHub Desktop:
1. 打开GitHub Desktop
2. File → Add Local Repository
3. 选择 `C:\coding\PhD-DS\didi-code\colab_upload`
4. 点击 `Publish repository`

---

## 📋 第三步：验证推送成功

访问:
```
https://github.com/wbhtcl2019/flow-intent-adapter
```

你应该能看到：
- ✅ README.md
- ✅ baselines/ 文件夹
- ✅ data/ 文件夹
- ✅ 所有Python文件

---

## 🎯 第四步：在Colab中使用

### 完整的Colab代码（复制粘贴）

```python
# ========================================
# Cell 1: 克隆仓库
# ========================================
!git clone https://github.com/wbhtcl2019/flow-intent-adapter.git
%cd flow-intent-adapter

# ========================================
# Cell 2: 检查GPU
# ========================================
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ========================================
# Cell 3: 安装依赖
# ========================================
!pip install -r requirements.txt -q

# ========================================
# Cell 4: 验证文件
# ========================================
import os

required_files = [
    'train_baselines.py',
    'train_flow_adapter_96k.py',
    'st_resnet_baseline.py',
    'latent_flow_intent_adapter.py',
    'baselines/dcrnn_baseline.py',
    'baselines/stgcn_baseline.py',
    'baselines/stformer_baseline.py',
    'data/nyc_100k_with_intents.parquet'
]

print("📁 文件检查:")
all_good = True
for f in required_files:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024
        print(f"✅ {f} ({size:.1f} KB)")
    else:
        print(f"❌ {f} - 缺失!")
        all_good = False

if all_good:
    print("\n🎉 所有文件就绪，可以开始训练！")
else:
    print("\n⚠️  有文件缺失，请检查")

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
# Cell 6: 运行DCRNN + Flow Adapter
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
# Cell 8: 运行STFormer + Flow Adapter
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

# 列出所有checkpoint
!ls -lh *.pth

# 下载checkpoint（取消注释需要的）
# files.download('dcrnn_baseline_best.pth')
# files.download('dcrnn_flow_best.pth')
# files.download('stformer_baseline_best.pth')
# files.download('stformer_flow_best.pth')
```

---

## 💡 Tips

### 如果推送失败

**问题1**: `remote: Repository not found`
- **原因**: 仓库还没创建
- **解决**: 先在GitHub网页上创建仓库（见第一步）

**问题2**: `Permission denied`
- **原因**: 没有访问权限或认证失败
- **解决**: 使用Personal Access Token作为密码

**问题3**: `fatal: unable to access`
- **原因**: 网络问题
- **解决**: 检查网络连接，或使用VPN

### 快速命令

```bash
# 检查远程仓库
git remote -v

# 查看状态
git status

# 查看提交历史
git log --oneline

# 推送到GitHub
git push -u origin main
```

---

## 🎯 完成后

GitHub仓库地址: `https://github.com/wbhtcl2019/flow-intent-adapter`

在Colab中使用:
```python
!git clone https://github.com/wbhtcl2019/flow-intent-adapter.git
%cd flow-intent-adapter
!pip install -r requirements.txt
```

---

## ❓ 需要帮助？

如果遇到问题，可以：
1. 检查GitHub仓库是否已创建
2. 确认Personal Access Token是否有效
3. 尝试使用GitHub Desktop（更简单）

准备好了吗？开始第一步：创建GitHub仓库！ 🚀
