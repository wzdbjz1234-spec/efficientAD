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

## 批量 ROI 推理 (`batch_detector.py`)

输入可以是单张图片或目录。目录模式默认递归处理并保持相对目录结构。
每张图片只执行一次模型前向，输出：

- `annotated/`：原图 ROI 框、判断类别、得分、阈值、推理时间、总处理时间和权重信息；
- `heatmaps/`：独立的 AE 差异热力图；
- `anomaly_heatmaps/`：仅包含判定为异常的零件热力图，便于集中复核；
- `results.csv` 与 `results.json`：图片路径、输出路径、类别、得分、阈值、ST/AE
  得分权重、模型权重路径、ROI、推理耗时、总耗时和错误信息。

```powershell
python batch_detector.py `
  --input path\to\raw_images `
  --output-dir path\to\batch_results `
  --model-dir EfficientAD-main\output\verytiny-batch=4 `
  --roi 1282,284,478,565 `
  --mask 0,0,80,120 `
  --mask 300,400,100,100 `
  --save-roi-config roi-with-masks.json `
  --threshold 0.014 `
  --ae-weight 0.025 `
  --device cuda
```

`--mask` 的坐标是相对于 ROI 左上角的 `x,y,width,height`，可以重复传入。
掩码区域不参与模型输入、差异得分和最大值判断；标注图中显示为黄色交叉框，
独立热力图中保存为黑色。`--save-roi-config` 会保存：

```json
{
  "roi": [1282, 284, 478, 565],
  "masks": [
    [0, 0, 80, 120],
    [300, 400, 100, 100]
  ]
}
```

也可以通过 JSON 传入 ROI，并分别覆盖模型产物：

```powershell
python batch_detector.py `
  --input path\to\raw_images `
  --output-dir path\to\batch_results `
  --roi-config roi-config.json `
  --student-weight path\to\student_final.pth `
  --autoencoder-weight path\to\autoencoder_final.pth `
  --norm-cache path\to\norm_params.json
```

`--threshold 0.014` 与 `--ae-weight 0.025` 是当前
`verytiny-batch=4` 权重扫描得到的一组配套尺度。更换模型或 AE 权重后应重新校准阈值。
若输入已经是裁剪好的 ROI 图片，可省略 `--roi` 和 `--roi-config`。

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

## 模型评测

统一使用 `model_tools.py evaluate`，避免旧实验脚本中硬编码的模型
14/15/16、数据集和输出目录。历史评测、权重扫描和 tiny/verytiny
对比脚本仅在本地 `legacy/` 目录保留，该目录不再纳入版本控制。

```powershell
python model_tools.py evaluate `
  --model 12 `
  --data-dir mydataset\my_product `
  --output evaluation_model_12.json
```

## 自定义参数

模型权重文件、归一化缓存可通过以下参数覆盖默认路径：`--teacher`、`--student`、`--autoencoder`、`--norm-cache`。若模型目录缺少 `norm_params.json`，可通过 `--train-dir` 实时计算：

```powershell
python model_tools.py infer --model 12 --input path\to\image.png --train-dir mydataset\my_product\train
```
