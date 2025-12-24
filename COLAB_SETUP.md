# Colab实验完整指南

## 📦 需要上传的文件（总共5个）

### 1. 数据文件 (1个)
```
D:/nyc-taxi-project/processed/nyc_100k_with_intents.parquet  (4.7MB)
```

### 2. Python代码文件 (3个)
```
st_resnet_baseline.py          (8.3KB)
intent_adapter.py              (8.8KB)
train_stresnet_with_intent.py  (新创建)
```

### 3. Notebook (1个)
```
colab_experiment.ipynb  (可选，也可以直接在Colab新建)
```

**总大小：< 5MB** - 上传很快！

---

## 🚀 Colab操作步骤

### 方法A：使用准备好的Notebook

1. **打开Colab**
   - 访问 https://colab.research.google.com/
   - 上传 `colab_experiment.ipynb`

2. **上传文件**
   - 点击左侧文件夹图标
   - 上传以下4个文件：
     - `nyc_100k_with_intents.parquet`
     - `st_resnet_baseline.py`
     - `intent_adapter.py`
     - `train_stresnet_with_intent.py`

3. **运行实验**
   - 依次运行notebook中的每个cell
   - 先训练baseline (约20-30分钟)
   - 再训练with intent (约20-30分钟)
   - 查看对比结果

### 方法B：手动粘贴代码（更简单）

如果不想下载文件，可以直接在Colab中：

1. **新建Colab notebook**

2. **创建Python文件** (在cell中运行)：

```python
# Cell 1: 创建 st_resnet_baseline.py
%%writefile st_resnet_baseline.py
# [粘贴 st_resnet_baseline.py 的全部内容]
```

```python
# Cell 2: 创建 intent_adapter.py
%%writefile intent_adapter.py
# [粘贴 intent_adapter.py 的全部内容]
```

```python
# Cell 3: 创建 train_stresnet_with_intent.py
%%writefile train_stresnet_with_intent.py
# [粘贴 train_stresnet_with_intent.py 的全部内容]
```

3. **上传数据文件**
   - 只需要手动上传 `nyc_100k_with_intents.parquet` (4.7MB)

4. **运行训练**

```python
# Cell 4: 训练baseline
!python train_stresnet_with_intent.py \
    --data_path nyc_100k_with_intents.parquet \
    --n_tiles 100 \
    --epochs 30 \
    --batch_size 16
```

```python
# Cell 5: 训练with intent
!python train_stresnet_with_intent.py \
    --data_path nyc_100k_with_intents.parquet \
    --n_tiles 100 \
    --epochs 30 \
    --batch_size 16 \
    --use_adapter
```

---

## ⚙️ 训练参数说明

### 推荐配置（快速验证）
```bash
--n_tiles 100         # 使用100个tile（减少计算量）
--closeness_len 6     # 使用6个时间步
--epochs 30           # 30个epoch足够看到趋势
--batch_size 16       # Colab T4 GPU可以跑16
--lr 0.001            # 学习率
```

### 如果想跑完整实验
```bash
--n_tiles 300         # 使用全部300个tile
--epochs 50           # 更多epoch
--batch_size 8        # 减小batch size避免OOM
```

---

## 📊 预期结果

### 训练时间（T4 GPU）
- **Baseline**: 约20-30分钟 (30 epochs)
- **With Intent**: 约20-30分钟 (30 epochs)
- **总计**: 约1小时

### 预期性能提升
如果Intent Adapter有效，应该看到：
- MAE降低 5-15%
- 收敛更快
- 验证曲线更稳定

---

## 💾 文件位置参考

从本地复制文件：

```bash
# 数据文件
D:/nyc-taxi-project/processed/nyc_100k_with_intents.parquet

# 代码文件（在didi-code目录）
C:/coding/PhD-DS/didi-code/st_resnet_baseline.py
C:/coding/PhD-DS/didi-code/intent_adapter.py
C:/coding/PhD-DS/didi-code/train_stresnet_with_intent.py
C:/coding/PhD-DS/didi-code/colab_experiment.ipynb
```

---

## 🔧 故障排除

### 问题1: CUDA out of memory
**解决**:
```bash
--batch_size 8      # 减小batch size
--n_tiles 50        # 减少tile数量
```

### 问题2: 数据加载慢
**解决**:
- 第一次加载会慢（建立grid mapping）
- 后续epoch会快很多
- 可以先用小数据测试：只取df的前10000行

### 问题3: 训练太慢
**解决**:
```bash
--epochs 10         # 减少epoch
--n_tiles 50        # 减少tile
```

---

## 📝 快速测试命令

如果只是想验证代码能跑通：

```bash
# 5分钟快速测试
!python train_stresnet_with_intent.py \
    --n_tiles 50 \
    --epochs 5 \
    --batch_size 8
```

---

## ✅ 检查清单

上传文件前检查：
- [ ] `nyc_100k_with_intents.parquet` (4.7MB)
- [ ] `st_resnet_baseline.py`
- [ ] `intent_adapter.py`
- [ ] `train_stresnet_with_intent.py`

运行实验前检查：
- [ ] GPU已启用 (Runtime → Change runtime type → GPU)
- [ ] 所有文件已上传
- [ ] 依赖已安装 (`pip install pandas pyarrow tqdm`)

---

## 🎯 下一步

训练完成后：
1. 下载训练好的模型 (`stresnet_baseline.pt`, `stresnet_with_intent.pt`)
2. 下载结果JSON (`results_baseline.json`, `results_with_intent.json`)
3. 下载对比图 (`comparison_results.pdf`)

如果结果好：
- 写论文！
- 可以继续测试更多baselines (Graph WaveNet, ASTGCN等)
