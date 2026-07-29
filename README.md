# EfficientAD 工业缺陷检测工具链

## 环境安装

```powershell
conda create -n llm python=3.10
conda activate llm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy tqdm scikit-learn tifffile matplotlib pillow
```

以下命令均在仓库根目录执行。

## 文件架构

```text
efficientAD/
├── EfficientAD-main/             # EfficientAD 源码及训练输出
│   └── output/<run_id>/
│       └── trainings/mvtec_ad/<product>/
│           ├── teacher_final.pth
│           ├── student_final.pth
│           ├── autoencoder_final.pth
│           └── norm_params.json
├── efficientad_tools/            # 推理、可视化、评测模块
├── model_tools.py                # 统一命令行入口（inspect / infer / visualize / evaluate）
├── pipeline.py                   # ORB ROI 匹配 + 推理流水线
├── roi_tool.py                   # ORB ROI 模板创建和裁剪
├── fixed_roi_crop.py             # 固定 ROI 裁剪与掩膜（固定机位场景）
├── roi_mask.py                   # ROI 掩膜公共函数
├── templates/                    # ORB ROI 模板
└── mydataset/<product>/          # MVTec AD 格式数据集
    ├── train/good/
    └── test/
        ├── good/
        └── <缺陷类型>/
```

`--model` 可以是 output 编号（如 `12`），也可以直接是包含 `*_final.pth` 的目录路径。

## 统一模型工具 (model_tools.py)

### 检查模型

```powershell
python model_tools.py inspect --model 12
python model_tools.py inspect --model 12 --product my_product
```

### 推理

输入图需为已裁剪好的 ROI 图片。

```powershell
# 单图
python model_tools.py infer --model 12 --input mydataset\my_product\test\broken\K8_0001.png --threshold 0.15

# 批量（输出热力图 + scores.csv）
python model_tools.py infer --model 12 --input mydataset\my_product\test\broken --output-dir results\model_12\broken --threshold 0.15

# 使用 CPU
python model_tools.py infer --model 12 --input path\to\image.png --device cpu

# 旧命令兼容入口
python batch_infer.py --model 12 --input-dir path\to\images --output-dir results
```

### 特征可视化

```powershell
# 单图
python model_tools.py visualize --model 12 --input mydataset\my_product\test\broken\K8_0001.png --output-dir vis\model_12 --top-k 8

# 批量（限制前 N 张）
python model_tools.py visualize --model 12 --input mydataset\my_product\test\broken --output-dir vis\model_12 --limit 5
```

### 评测

计算 AUROC、Youden 最优阈值、Accuracy、Precision、Recall、F1、混淆矩阵。误判图片自动生成诊断图。

```powershell
python model_tools.py evaluate --model 12 --data-dir mydataset\my_product --threshold 0.15 --output evaluation_model_12.json

# 自定义诊断图输出目录
python model_tools.py evaluate --model 12 --data-dir mydataset\my_product --threshold 0.15 --misclassified-dir results\errors --misclassified-top-k 8 --output eval.json

# 只计算指标，不生成诊断图
python model_tools.py evaluate --model 12 --data-dir mydataset\my_product --no-misclassified-visuals --output eval.json
```

## 固定 ROI 裁剪 (fixed_roi_crop.py)

适用于固定机位、所有原图位置一致的场景，不需要 ORB 特征匹配。

### 首次使用：GUI 框选 ROI 和掩膜，保存配置

```powershell
python fixed_roi_crop.py --reference path\to\ref.png --input-dir raw_images --output-dir mydataset\my_product\train\good --recursive --overwrite --save-roi roi_config.json
```

依次弹出两个框选窗口：① 拖拽框选 ROI（Enter/Space 确认）；② 在裁剪后的 ROI 中框选掩膜区域（Enter/Space 确认，C/Esc 跳过）。

### 后续使用：加载配置，批量处理

```powershell
# 训练集
python fixed_roi_crop.py --reference path\to\ref.png --input-dir raw_train --output-dir mydataset\my_product\train\good --load-roi roi_config.json --recursive --overwrite

# 测试集 good
python fixed_roi_crop.py --reference path\to\ref.png --input-dir raw_test_good --output-dir mydataset\my_product\test\good --load-roi roi_config.json --recursive --overwrite

# 测试集缺陷（保持子文件夹结构）
python fixed_roi_crop.py --reference path\to\ref.png --input-dir raw_test_defects --output-dir mydataset\my_product\test --load-roi roi_config.json --recursive --overwrite
```

### 无 GUI：命令行直接指定坐标

```powershell
python fixed_roi_crop.py --reference path\to\ref.png --input-dir raw_images --output-dir cropped --roi 100,50,400,300 --mask 20,10,80,60 --recursive
```

## ORB ROI 工具 (roi_tool.py)

适用于原图位置有偏移、需要 ORB 特征匹配对齐的场景。

```powershell
# 创建模板（GUI 框选 ROI 和掩膜）
python roi_tool.py create my_product path\to\template.jpg

# 列出所有模板
python roi_tool.py list

# 单图裁剪
python roi_tool.py crop my_product path\to\input.jpg -o cropped.png

# 批量裁剪
python roi_tool.py batch my_product raw_images cropped
```

模板保存在 `templates\my_product\`（template.png + roi.json）。

## ROI + 推理流水线 (pipeline.py)

整合 ORB ROI 裁剪和模型推理。

```powershell
# 单图
python pipeline.py --template my_product --model 12 --image path\to\raw.png --output result.png --threshold 0.15

# 批量
python pipeline.py --template my_product --model 12 --input-dir raw_images --output-dir results --threshold 0.15
```

## 训练

数据按 MVTec AD 格式组织后，在 `EfficientAD-main` 下执行：

```powershell
cd EfficientAD-main
python efficientad1.py -d mvtec_ad -s my_product -a ..\mydataset -o output\13 --mask-config ..\roi_config.json
```

`--mask-config` 可指向 `roi_config.json`、`templates\my_product\roi.json`，默认为 `auto` 自动查找，设为 `none` 关闭掩膜。

## 模型评测与权重分析

### 综合评测 (evaluate_model14.py)

报告单次推理延迟（CPU/GPU）、分类精度（最优阈值+混淆矩阵），并为所有误判样本生成 ST/STAE 特征热力图诊断。

内部硬编码模型路径和数据集目录，运行前需按实际调整 `MODEL_DIR`、`TEST_DIR` 等变量。

```powershell
# 完整评测（CPU + GPU 基准 + 精度 + 热力图）
python evaluate_model14.py

# 仅 GPU 基准
python evaluate_model14.py --gpu-only

# 仅 CPU 基准
python evaluate_model14.py --cpu-only

# 跳过热力图生成（加速评测）
python evaluate_model14.py --skip-heatmaps

# 指定设备
python evaluate_model14.py --device cpu
```

### ST-AE 权重扫描 — 召回率热力图 (sweep_weights.py)

在 ST/AE 双分支权重空间 `[0, 1] × [0, 1]` 做 21×21 网格扫描。对于每个 `(w_st, w_ae)` 组合，按 95% 特异度确定阈值，计算 broken 样本召回率，绘制多模型并排热力图。

```powershell
python sweep_weights.py
```

输出图片和 JSON 结果保存在 `weight_sweep_results/`。

### ST-AE 权重扫描 — FPR/FNR/Accuracy 热力图 (sweep_fpr_fnr.py)

41×41 网格扫描 ST/AE 权重组合。对每个组合搜索最优阈值，记录 FPR（假阳性率）、FNR（假阴性率）和 Accuracy，生成 3 模型 × 3 指标的九宫格热力图。

```powershell
python sweep_fpr_fnr.py
```

同样输出到 `weight_sweep_results/`。

### 模型体积与速度对比 (bench_tiny.py)

对比 tiny / small / medium 三种模型变体的参数量和 CPU 单张推理耗时（均值、中位数、FPS）。

```powershell
python bench_tiny.py
```

tiny 变体将 student 和 autoencoder 内部通道数减半，teacher 复用预训练权重不变。

## 自定义参数

模型权重文件、归一化缓存可通过以下参数覆盖默认路径：`--teacher`、`--student`、`--autoencoder`、`--norm-cache`。若模型目录缺少 `norm_params.json`，可通过 `--train-dir` 实时计算：

```powershell
python model_tools.py infer --model 12 --input path\to\image.png --train-dir mydataset\my_product\train
```
