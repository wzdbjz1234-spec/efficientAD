# EfficientAD + ORB ROI 工业缺陷检测工具链

## 目录结构

```
efficientAD/
├── roi_tool.py                        # ROI 模板创建 & ORB 裁剪 (独立脚本)
├── pipeline.py                        # 完整流水线: 匹配 → 识别 → 标注
├── templates/                         # ROI 模板存储目录
│   └── <template_name>/
│       ├── template.png               # 模板原图
│       └── roi.json                   # ROI 坐标 [x, y, w, h]
├── mydataset/                         # MVTec AD 格式数据集
│   └── my_product/
│       ├── train/good/                # 训练用正常样本
│       ├── test/good/                 # 测试用正常样本
│       └── test/broken/               # 测试用异常样本
├── EfficientAD-main/
│   ├── efficientad.py                 # EfficientAD 训练主程序
│   ├── inference.py                   # EfficientAD 推理 (不含ROI匹配)
│   ├── visualize_features.py          # Teacher/Student 特征图对比可视化
│   ├── common.py                      # PDN/Autoencoder 模型定义
│   ├── models/                        # 预训练 teacher 权重
│   └── output/1/trainings/            # 训练输出
│       └── mvtec_ad/my_product/
│           ├── teacher_final.pth
│           ├── student_final.pth
│           ├── autoencoder_final.pth
│           └── norm_params.json       # 归一化参数缓存
└── learn-efficientad/                 # 学习教程工作区
```

## 依赖安装

```bash
conda create -n llm python=3.10
conda activate llm

# 核心依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy tqdm scikit-learn tifffile matplotlib pillow

# 可视化依赖 (仅 visualize_features.py 需要)
pip install matplotlib
```

---

## 1. ROI 模板管理 (`roi_tool.py`)

基于 ORB 特征匹配的 ROI 自动定位与裁剪工具。

### 创建模板

打开一张产品图，用鼠标框选 ROI，保存为模板：

```bash
python roi_tool.py create my_product "path/to/template.jpg"
```

操作：鼠标拖拽框选区域 → 按 **Enter** 确认 / 按 **C** 取消。

### 列出所有模板

```bash
python roi_tool.py list
```

### 单张裁剪 (基于模板自动定位+透视矫正)

```bash
python roi_tool.py crop my_product "path/to/input.jpg"
# 输出: input_cropped.png

# 指定输出路径
python roi_tool.py crop my_product "path/to/input.jpg" -o output.png
```

### 批量裁剪整个目录

```bash
python roi_tool.py batch my_product ./raw_images/ ./cropped/
```

**匹配原理**: ORB 特征点检测 → BFMatcher + ratio test (0.75) → RANSAC 单应性 → 透视矫正裁剪。输入图和模板可以有旋转变换，比传统模板匹配更鲁棒。

---

## 2. EfficientAD 训练 (`efficientad.py`)

### 数据集准备

按 MVTec AD 格式组织数据（`train/good/` 放正常样本, `test/good/` + `test/<缺陷名>/` 放测试样本）。

### 开始训练

```bash
cd EfficientAD-main
python efficientad.py -d mvtec_ad -s my_product -a ../mydataset
```

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `-d` | 数据集类型 | mvtec_ad |
| `-s` | 产品名 (子目录名) | bottle |
| `-a` | 数据集根目录 | ./mvtec_anomaly_detection |
| `-o` | 输出目录 | output/1 |
| `-m` | 模型大小 | small (可选 medium) |
| `-t` | 训练步数 | 70000 |
| `-w` | teacher 预训练权重 | models/teacher_small.pth |
| `-i` | ImageNet 路径 (可选) | none (跳过 penalty) |

训练输出 `teacher_final.pth`, `student_final.pth`, `autoencoder_final.pth`。

---

## 3. EfficientAD 推理 (仅模型推理, 不含 ROI 匹配)

```bash
cd EfficientAD-main

# 分析单张图片 (需要原图已经裁剪好)
python inference.py --image ../mydataset/my_product/test/broken/img001.png -o anomaly_map.tiff

# 带阈值判定
python inference.py --image img.png --threshold 0.1
```

| 参数 | 说明 |
|------|------|
| `--image` | 输入图片路径 |
| `--output-map` | anomaly map 保存路径 (.tiff) |
| `--threshold` | 判定阈值, 超过 = 异常 |
| `--train-dir` | 训练集路径, 用于计算归一化参数 |
| `--norm-cache` | 归一化参数缓存文件 |

首次运行计算归一化参数并缓存, 后续复用。

---

## 4. 完整流水线 (`pipeline.py`)

一步到位: **模板匹配定位 ROI → 透视矫正裁剪 → EfficientAD 识别 → 标注输出**

### 单张图片检测

```bash
cd efficientAD
python pipeline.py --image "测试图.png" --template my_product

# 带阈值自动判定
python pipeline.py --image "测试图.png" --template my_product --threshold 0.1

# 保存结果
python pipeline.py --image "测试图.png" --template my_product -o result.png --threshold 0.1
```

### 批量检测整个文件夹

```bash
# 检测 test/good (正常样本)
python pipeline.py --template my_product --input-dir "mydataset\my_product\test\good" --output-dir "results\good"

# 检测 test/broken (异常样本)
python pipeline.py --template my_product --input-dir "mydataset\my_product\test\broken" --output-dir "results\broken" --threshold 0.1

# 不指定 output-dir 则自动输出到 <input-dir>/annotated/
python pipeline.py --template my_product --input-dir "mydataset\my_product\test\broken"
```

### 确定阈值

先不设 `--threshold` 跑几组正常和异常样本, 观察分数范围, 然后在正常样本最高分和异常样本最低分之间选一个阈值:

```bash
# 正常样本
python pipeline.py --template my_product --input-dir "mydataset\my_product\test\good" --output-dir "results\good"

# 异常样本  
python pipeline.py --template my_product --input-dir "mydataset\my_product\test\broken" --output-dir "results\broken"

# 然后设定阈值批量跑
python pipeline.py --template my_product --input-dir "mydataset\my_product\test\broken" --output-dir "results\broken_annotated" --threshold 0.15
```

**输出效果**: 每张原图上绘制 ROI 四边形框 + 绿色 `NORMAL` / 红色 `ANOMALY` 标注 + 异常分数。

---

## 5. 特征图对比可视化 (`visualize_features.py`)

直观展示 Teacher 和 Student 在哪些特征通道上产生了差异。

```bash
cd EfficientAD-main

# 生成对比图
python visualize_features.py --image "../mydataset/my_product/test/broken/img001.png"

# 指定输出文件 & 展示更多差异通道
python visualize_features.py --image img.png -o comparison.png --top-k 12
```

| 参数 | 说明 |
|------|------|
| `--image` | 输入图片 |
| `--output` | 输出 PNG 路径 (默认 feature_comparison.png) |
| `--top-k` | 展示差异最大的 K 个通道 (默认 8) |

**输出解读**:
- 第 1 行: 输入原图
- 第 2 行: Teacher 在差异最大通道上的特征图 — 这是"正确答案"
- 第 3 行: Student 对应通道的特征图 — 它在尝试模仿
- 第 4 行: 逐通道差值热力图 — 越亮 = Student 模仿得越差
- 第 5 行: 融合后的 Anomaly Map (叠在原图 + 纯热力图 + colorbar)

正常图上差异通道分布均匀且暗; 异常图上缺陷区域对应的通道会明显亮起。

---

## 常见问题

**Q: 首次运行 pipeline 很慢?**
A: 首次需要计算归一化参数 (在所有训练图上跑 Teacher/Student/Autoencoder), 结果缓存到 `norm_params.json`, 后续秒级。

**Q: ORB 匹配失败?**
A: 检查模板图是否包含在输入图中。如果产品摆放严重旋转/缩放, 可调低 `RANSAC_THRESH` (默认 5.0)。如果完全无法匹配, 需重新在相似角度下创建模板。

**Q: GPU 内存不足?**
A: pipeline.py 同时加载 Teacher + Student + Autoencoder 三个模型到显存。6GB 显存够用, 但要在 `compute_norm_params` 里加了 `torch.cuda.empty_cache()` 清理中间张量。如果还报 OOM, 给 reduce batch 或换 CPU 推理。

**Q: 阈值设为多少?**
A: 取决于你的数据。建议跑完正常和异常各一批, 用 `pipeline.py` 统计分数分布后选定。典型区间: 正常 0~0.10, 异常 0.15~3+。
