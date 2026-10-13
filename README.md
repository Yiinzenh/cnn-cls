# 自定义卷积神经网络图像分类

基于 PyTorch 的视网膜 OCT 眼底图像分类项目。网络完全由手动堆叠网络层实现，未使用 AlexNet、VGG、ResNet 等现成经典网络，适合作为卷积神经网络基础组件的学习与练习代码。

## 功能

- 数据集统计：样本总量、类别数量、原始分辨率、通道数、各划分的类别分布
- 图像预处理：统一尺寸缩放、像素归一化，训练集附加随机水平翻转
- 自定义 CNN：卷积层、批归一化层、ReLU 激活层、最大池化层、Dropout 层、全连接层手动堆叠
- 训练流程：逐轮记录训练与验证损失、准确率，按验证准确率保存最优权重
- 控制变量对照实验：卷积核大小、网络深度、Dropout 概率、学习率、池化窗口、训练轮次
- 模型评估：测试集准确率、混淆矩阵、逐类精确率与召回率、错分样本收集
- 可视化输出：训练曲线、混淆矩阵热力图、类别分布、样本网格、对照实验对比柱状图

## 环境依赖

- Python 3.x
- PyTorch、torchvision
- NumPy、Pandas
- Matplotlib
- Pillow

安装依赖：

```bash
pip install torch torchvision numpy pandas matplotlib pillow
```

训练与评估会自动检测 GPU，CUDA 可用时在 GPU 上运行，否则自动退回 CPU。如需指定显卡，可设置环境变量 `CUDA_VISIBLE_DEVICES`。

## 目录结构

```
cnn-cls/
├── main.py                 入口，提供 stats / train / search / eval 四个子命令
├── config/
│   └── setting.py          路径、类别、图像尺寸、基准超参数与随机种子
├── data/
│   ├── transform.py        图像预处理与数据增强
│   └── dataset.py          数据集扫描、统计、子集抽样与 DataLoader 构建
├── model/
│   └── cnn.py              自定义 CNN 定义、参数统计与结构描述
├── engine/
│   ├── trainer.py          训练循环、最优权重与日志保存
│   └── evaluator.py        预测、混淆矩阵、逐类指标与错分样本收集
├── experiments/
│   └── search.py           控制变量法对照实验
├── utils/
│   └── plot.py             绘图工具
└── outputs/                运行后自动生成的产物目录
    ├── figures/            曲线图、混淆矩阵、对比图、样本网格
    ├── logs/               逐轮训练日志、数据集统计、评估结果
    └── weights/            模型权重
```

`outputs` 下三个子目录会在程序启动时自动创建，无需手动建立。

## 数据集准备

数据根目录为 `datasets/RetinalOCT_Dataset`，需按划分目录加类别目录的两级结构存放：

```
datasets/RetinalOCT_Dataset/
├── train/
│   ├── AMD/
│   ├── CNV/
│   ├── CSR/
│   ├── DME/
│   ├── DR/
│   ├── DRUSEN/
│   ├── MH/
│   └── NORMAL/
├── val/
│   └── （同上，八个类别目录）
└── test/
    └── （同上，八个类别目录）
```

类别名称由 `config/setting.py` 中的 `CLASSES` 决定，目录名必须与之完全一致。支持的图像扩展名为 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.tif`、`.tiff`。

若使用其他数据集，只需修改 `CLASSES` 与 `DATA_ROOT` 两项配置，并保持上述目录结构。

## 使用方式

所有命令在项目根目录下执行。

### 1. 数据集统计与预处理检查

```bash
python main.py stats
```

扫描数据集，在终端输出样本总量、各类别与各划分的样本数、原始分辨率范围与通道数，并把统计结果写入 `outputs/logs/dataset.json`，同时生成类别分布图 `class_distribution.png` 与八类样本网格图 `samples.png`。

### 2. 训练模型

```bash
python main.py train
```

按 `config/setting.py` 中的最优配置训练模型，训练完成后保存：

- `outputs/weights/<tag>.pt`：验证准确率最高轮次的权重
- `outputs/logs/<tag>.json`：超参数、最优准确率、最优轮次、逐轮指标
- `outputs/logs/<tag>.csv`：逐轮的损失、准确率与学习率
- `outputs/figures/curves_<tag>.png`：训练与验证曲线

`--tag` 用于指定本次运行的名称，默认值为 `baseline`：

```bash
python main.py train --tag my_run
```

### 3. 控制变量对照实验

```bash
python main.py search
```

依次运行全部六个实验组，每组只改动一个超参数，其余保持基准值不变。全部实验共用同一批固定子集，保证对比公平。结果汇总为：

- `outputs/logs/search.csv` 与 `search.json`：各组的参数量、训练准确率、验证准确率、泛化间隔、最优轮次与耗时
- `outputs/figures/compare_<group>.png`：该组各取值的准确率对比柱状图
- `outputs/figures/curves_<group>_<value>.png`：每次训练的训练与验证曲线

可用 `--groups` 只运行指定实验组，取值为 `kernel`、`depth`、`dropout`、`lr`、`pool`、`epoch`：

```bash
python main.py search --groups kernel depth
```

### 4. 测试集评估

```bash
python main.py eval --tag baseline
```

载入指定权重，在测试集上评估一次，输出：

- 终端打印整体准确率与逐类精确率、召回率、F1 值
- `outputs/logs/eval_<tag>.csv` 与 `eval_<tag>.json`
- `outputs/logs/misclassified_<tag>.csv`：错分样本清单
- `outputs/figures/confusion_<tag>.png`：混淆矩阵
- `outputs/figures/misclassified_<tag>.png`：按置信度排序的典型错分样本

`--tag` 需与训练时保存的权重名称一致。

## 配置说明

全部可调参数集中在 `config/setting.py`：

| 配置项 | 含义 |
| --- | --- |
| `DATA_ROOT` | 数据集根目录 |
| `CLASSES` | 类别名称列表，决定类别数与标签映射 |
| `IMAGE_SIZE` | 输入图像边长，默认 128 |
| `SEED` | 随机种子，默认 42 |
| `NUM_WORKERS` | DataLoader 工作进程数 |
| `BASELINE` | 基准超参数，对照实验在此基础上只改一项 |
| `BEST` | 训练命令使用的配置，覆盖 `BASELINE` 中的同名项 |
| `SEARCH_SUBSETS` | 对照实验每个划分每类抽取的样本数 |
| `SEARCH_EPOCHS` | 对照实验统一的训练轮次 |

`BASELINE` 包含以下可调超参数：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `base_channels` | 首个卷积模块的输出通道数，后续模块依次翻倍 | 32 |
| `kernel_size` | 卷积核大小，填充自动取核大小的一半 | 3 |
| `depth` | 卷积模块组数 | 4 |
| `dropout` | Dropout 丢弃概率 | 0.3 |
| `pool_size` | 最大池化窗口大小，步长与窗口相同 | 2 |
| `fc_hidden` | 全连接隐藏层神经元个数 | 128 |
| `batch_size` | 批大小 | 32 |
| `lr` | 初始学习率 | 1e-3 |
| `weight_decay` | Adam 权重衰减系数 | 1e-4 |
| `epochs` | 训练轮次 | 25 |

自定义设置可直接调用 `build_args()`：

```python
from config.setting import build_args

args = build_args(depth=6, dropout=0.2, lr=5e-4)
```

## 网络结构

网络由两部分组成，层数随 `depth` 变化：

- 特征提取部分：`depth` 组卷积模块串联，每组依次为 `Conv2d` → `BatchNorm2d` → `ReLU` → `MaxPool2d` → `Dropout2d`
- 分类部分：`Flatten` → `Linear` → `ReLU` → `Dropout` → `Linear`，输出维度等于类别数

卷积层使用 `padding = kernel_size // 2`、`stride = 1`，卷积前后特征图尺寸不变；池化层 `stride = pool_size`，每次池化特征图边长缩小为原来的 1 除以 `pool_size`。

展平维度不采用手工推算，而是在 `model/cnn.py` 中先用一张全零张量做一次无梯度前向计算，自动读取特征提取部分的输出元素个数，再据此构建全连接层，避免修改 `depth`、`pool_size`、`IMAGE_SIZE` 后出现维度不匹配错误。因此调整输入尺寸或网络深度时无需手动改动任何维度数值。

查看当前配置下每一层的输出维度与参数量：

```python
from config.setting import build_args
from model.cnn import build_model, count_params, describe

args = build_args()
model = build_model(args)

for line in describe(args, model):
    print(line)
print("参数量:", count_params(model))
```

## 输出文件命名规则

| 文件 | 命名 | 说明 |
| --- | --- | --- |
| 训练日志 | `<tag>.csv`、`<tag>.json` | `tag` 默认为 `baseline` |
| 模型权重 | `<tag>.pt` | 验证准确率最高轮次 |
| 训练曲线 | `curves_<tag>.png` | 每个模型一张 |
| 对照实验汇总 | `search.csv`、`search.json` | 全部实验组合并 |
| 对照实验对比图 | `compare_<group>.png` | 每个实验组一张 |
| 评估结果 | `eval_<tag>.csv`、`eval_<tag>.json` | 逐类指标 |
| 混淆矩阵 | `confusion_<tag>.png` | 每个评估模型一张 |
| 错分样本 | `misclassified_<tag>.png`、`misclassified_<tag>.csv` | 按预测置信度降序 |

训练与评估全程固定随机种子，相同配置重复运行可得到一致结果。
