#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive experiment: ST×AE weight sweep for models 14/15/16.

For each model:
  1. Collect ST and AE scores for all good + broken test images
  2. Sweep ST×AE weights on a 41×41 grid — for each (w_st, w_ae):
       combined_score = w_st × ST + w_ae × AE
       Find optimal threshold that maximizes accuracy on ALL test images
       Record accuracy, FPR, FNR, threshold, confusion matrix
  3. Plot 3-models × 3-metrics heatmap grid
  4. At the best-accuracy weight combo, generate feature heatmaps for all
     misclassified images (FP + FN)
  5. Save summary JSON with all metrics

Output folder: experiment_weight_sweep/
  ├── heatmaps/weight_sweep_accuracy_heatmap.png
  ├── misclassified/<model_name>/*.png
  └── summary.json
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
EFFICIENTAD_DIR = PROJECT_ROOT / "EfficientAD-main"
TEST_DIR = PROJECT_ROOT / "mydataset" / "my_product" / "test"
TRAIN_DIR = PROJECT_ROOT / "mydataset" / "my_product" / "train"
OUTPUT_DIR = PROJECT_ROOT / "experiment_weight_sweep"

sys.path.insert(0, str(EFFICIENTAD_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
from efficientad_tools import EfficientADPredictor, ModelArtifacts
from efficientad_tools.visualization import save_feature_visualization


# ═══════════════════════════════════════════════════════════════════════
#  Score collection
# ═══════════════════════════════════════════════════════════════════════

def collect_all_scores(predictor):
    """Return (st_scores, ae_scores, labels, paths) for all test images."""
    st_all, ae_all, labels_all, paths_all = [], [], [], []

    for category, label in [("good", 0), ("broken", 1)]:
        cat_dir = TEST_DIR / category
        for p in sorted(cat_dir.glob("*.png")):
            img = cv2.imread(str(p))
            if img is None:
                continue
            pred = predictor.predict(img, include_features=False)
            st_all.append(pred.student_teacher_score)
            ae_all.append(pred.autoencoder_score)
            labels_all.append(label)
            paths_all.append(p)

    return (np.array(st_all), np.array(ae_all),
            np.array(labels_all), paths_all)


# ═══════════════════════════════════════════════════════════════════════
#  Weight sweep
# ═══════════════════════════════════════════════════════════════════════

def evaluate_at_weight(
    st_scores: np.ndarray,
    ae_scores: np.ndarray,
    labels: np.ndarray,
    w_st: float,
    w_ae: float,
) -> dict:
    """
    For given (w_st, w_ae), find threshold that maximizes accuracy.
    Returns best accuracy, FPR, FNR, threshold, and confusion matrix.
    """
    combined = w_st * st_scores + w_ae * ae_scores
    idx = np.argsort(combined)
    scores_sorted = combined[idx]
    labels_sorted = labels[idx]

    n_total = len(labels)
    n_good = int(np.sum(labels == 0))

    best = {
        "accuracy": 0.0,
        "fpr": 1.0,
        "fnr": 1.0,
        "threshold": 0.0,
        "tp": 0, "tn": 0, "fp": 0, "fn": 0,
    }

    for i in range(n_total - 1):
        thresh = (scores_sorted[i] + scores_sorted[i + 1]) / 2.0
        preds = (combined >= thresh).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        acc = (tp + tn) / n_total
        if acc > best["accuracy"]:
            best["accuracy"] = acc
            best["fpr"] = fp / max(fp + tn, 1)
            best["fnr"] = fn / max(fn + tp, 1)
            best["threshold"] = float(thresh)
            best["tp"], best["tn"] = tp, tn
            best["fp"], best["fn"] = fp, fn

    return best


def sweep_weights(st_scores, ae_scores, labels, n_steps=41):
    """Sweep ST×AE in [0, 1]; return ws and per-metric grids."""
    ws = np.linspace(0, 1, n_steps)
    acc_grid = np.full((n_steps, n_steps), np.nan)
    fpr_grid = np.full((n_steps, n_steps), np.nan)
    fnr_grid = np.full((n_steps, n_steps), np.nan)
    thresh_grid = np.full((n_steps, n_steps), np.nan)
    tp_grid = np.full((n_steps, n_steps), np.nan, dtype=int)
    tn_grid = np.full((n_steps, n_steps), np.nan, dtype=int)
    fp_grid = np.full((n_steps, n_steps), np.nan, dtype=int)
    fn_grid = np.full((n_steps, n_steps), np.nan, dtype=int)

    for i, w_st in enumerate(ws):
        for j, w_ae in enumerate(ws):
            if w_st == 0 and w_ae == 0:
                continue
            best = evaluate_at_weight(st_scores, ae_scores, labels, w_st, w_ae)
            acc_grid[j, i] = best["accuracy"]
            fpr_grid[j, i] = best["fpr"]
            fnr_grid[j, i] = best["fnr"]
            thresh_grid[j, i] = best["threshold"]
            tp_grid[j, i] = best["tp"]
            tn_grid[j, i] = best["tn"]
            fp_grid[j, i] = best["fp"]
            fn_grid[j, i] = best["fn"]

    return ws, acc_grid, fpr_grid, fnr_grid, thresh_grid, tp_grid, tn_grid, fp_grid, fn_grid


# ═══════════════════════════════════════════════════════════════════════
#  Heatmap plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_combined_heatmaps(models_data, ws, output_path):
    """3 models × 3 metrics grid."""
    metrics = [
        ("acc", "Accuracy", "RdYlGn"),
        ("fpr", "FPR (False Positive Rate)", "Reds"),
        ("fnr", "FNR (False Negative Rate)", "Blues"),
    ]
    n_models = len(models_data)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(
        n_models, n_metrics,
        figsize=(5.5 * n_metrics, 5.2 * n_models),
    )
    if n_models == 1:
        axes = axes.reshape(1, -1)

    for row, data in enumerate(models_data):
        for col, (key, title, _cmap_name) in enumerate(metrics):
            ax = axes[row, col]
            grid = data[key]
            masked = np.ma.masked_where(np.isnan(grid), grid)

            if key == "acc":
                vmin = float(np.nanmin(grid))
                vmax = float(np.nanmax(grid))
                cmap = LinearSegmentedColormap.from_list(
                    "acc_cmap", ["#D73027", "#FFFFBF", "#1A9850"], N=256)
            elif key == "fpr":
                vmin, vmax = 0.0, float(np.nanmax(grid))
                cmap = LinearSegmentedColormap.from_list(
                    "fpr_cmap", ["#F7FBFF", "#EF6C5E"], N=256)
            else:  # fnr
                vmin, vmax = 0.0, float(np.nanmax(grid))
                cmap = LinearSegmentedColormap.from_list(
                    "fnr_cmap", ["#F7FBFF", "#3A6FB5"], N=256)

            im = ax.pcolormesh(ws, ws, masked, cmap=cmap,
                               vmin=vmin, vmax=vmax,
                               shading="auto", rasterized=True)

            # Best point
            if key == "acc":
                best_idx = np.unravel_index(np.nanargmax(grid), grid.shape)
                best_marker, best_color = "*", "black"
                val_str = f"{grid[best_idx]*100:.2f}%"
            elif key == "fpr":
                best_idx = np.unravel_index(np.nanargmin(grid), grid.shape)
                best_marker, best_color = "v", "darkred"
                val_str = f"{grid[best_idx]*100:.2f}%"
            else:
                best_idx = np.unravel_index(np.nanargmin(grid), grid.shape)
                best_marker, best_color = "v", "darkblue"
                val_str = f"{grid[best_idx]*100:.2f}%"

            best_w_ae = ws[best_idx[0]]
            best_w_st = ws[best_idx[1]]
            ax.plot(best_w_st, best_w_ae, marker=best_marker, color=best_color,
                    markersize=16, markeredgecolor="white",
                    markeredgewidth=1.5, zorder=5)
            ax.annotate(
                f"best ({best_w_st:.2f},{best_w_ae:.2f})\n{val_str}",
                xy=(best_w_st, best_w_ae),
                xytext=(0.05, 0.92), textcoords="axes fraction",
                fontsize=7, fontweight="bold", color=best_color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          alpha=0.85, ec=best_color),
            )

            # Current config (ST=0, AE=1)
            cur_w_st, cur_w_ae = 0.0, 1.0
            cur_val = grid[(np.abs(ws - cur_w_ae)).argmin(),
                           (np.abs(ws - cur_w_st)).argmin()]
            ax.plot(cur_w_st, cur_w_ae, marker="D", color="magenta",
                    markersize=10, markeredgecolor="white",
                    markeredgewidth=1, zorder=5)
            ax.annotate(
                f"cur {cur_val*100:.1f}%",
                xy=(0.02, 0.82), textcoords="axes fraction",
                fontsize=7, color="magenta",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          alpha=0.85, ec="magenta"),
            )

            ax.set_xlabel("ST weight", fontsize=9)
            ax.set_ylabel("AE weight", fontsize=9)
            ax.set_title(f"{data['label']} — {title}", fontsize=10,
                         fontweight="bold")
            ax.set_xticks(np.arange(0, 1.1, 0.25))
            ax.set_yticks(np.arange(0, 1.1, 0.25))
            plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

    fig.suptitle(
        "ST × AE Weight Sweep — Accuracy, FPR & FNR at Optimal Threshold\n"
        "★ = best    ◆ = current (ST=0, AE=1)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Misclassified feature heatmaps
# ═══════════════════════════════════════════════════════════════════════

def generate_misclassified_heatmaps(
    predictor: EfficientADPredictor,
    st_scores: np.ndarray,
    ae_scores: np.ndarray,
    labels: np.ndarray,
    paths: list,
    w_st: float,
    w_ae: float,
    threshold: float,
    model_label: str,
    output_root: Path,
) -> list[dict]:
    """Generate feature-visualization heatmaps for all misclassified images."""
    mis_dir = output_root / "misclassified" / model_label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    mis_dir.mkdir(parents=True, exist_ok=True)

    combined = w_st * st_scores + w_ae * ae_scores
    preds = (combined >= threshold).astype(int)

    mis_list = []
    for i in range(len(labels)):
        if preds[i] == labels[i]:
            continue
        true_label = "broken" if labels[i] == 1 else "good"
        pred_label = "broken" if preds[i] == 1 else "good"
        outcome = "FP" if preds[i] == 1 else "FN"
        mis_list.append({
            "path": str(paths[i]),
            "filename": paths[i].name,
            "true_label": true_label,
            "pred_label": pred_label,
            "outcome": outcome,
            "st_score": float(st_scores[i]),
            "ae_score": float(ae_scores[i]),
            "combined_score": float(combined[i]),
        })

    print(f"  {model_label}: {len(mis_list)} misclassified "
          f"(FP={sum(1 for m in mis_list if m['outcome']=='FP')}, "
          f"FN={sum(1 for m in mis_list if m['outcome']=='FN')})")

    for idx, m in enumerate(mis_list):
        fname = Path(m["path"]).stem
        out_name = (
            f"{idx+1:03d}_{m['outcome']}_true-{m['true_label']}_"
            f"pred-{m['pred_label']}_combined-{m['combined_score']:.6f}_"
            f"ST-{m['st_score']:.5f}_AE-{m['ae_score']:.5f}_{fname}.png"
        )
        out_path = mis_dir / out_name

        img = cv2.imread(m["path"])
        pred = predictor.predict(img, include_features=True)
        title = (
            f"{m['outcome']} | true={m['true_label']} pred={m['pred_label']} | "
            f"combined={m['combined_score']:.5f} "
            f"ST={m['st_score']:.5f} AE={m['ae_score']:.5f}"
        )
        save_feature_visualization(
            m["path"], pred, str(out_path), top_k=8, title=title,
        )

    return mis_list


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    models = [
        {"name": "14", "bs": 4},
        {"name": "15", "bs": 6},
        {"name": "16", "bs": 1},
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Output dir: {OUTPUT_DIR}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    heatmap_dir = OUTPUT_DIR / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    models_data = []
    all_results = {}
    ws_ref = None  # store ws from last model (all use same grid)

    for m in models:
        label = f"Model {m['name']} (bs={m['bs']})"
        print(f"{'='*65}")
        print(f"  {label}")
        print(f"{'='*65}")

        # ── Load model ──
        model_dir = EFFICIENTAD_DIR / "output" / m["name"]
        artifacts = ModelArtifacts.from_output(
            str(model_dir), dataset="mvtec_ad", product="my_product",
            source_dir=str(EFFICIENTAD_DIR),
        )
        predictor = EfficientADPredictor.load(
            artifacts, device=device, train_dir=str(TRAIN_DIR),
        )

        # ── Collect scores ──
        print("  [1/4] Collecting ST and AE scores for all test images...")
        st_all, ae_all, labels_all, paths_all = collect_all_scores(predictor)
        n_good = int(np.sum(labels_all == 0))
        n_broken = int(np.sum(labels_all == 1))
        print(f"        Good: {n_good}  Broken: {n_broken}  Total: {len(labels_all)}")

        # ── Sweep weights ──
        print(f"  [2/4] Sweeping ST×AE weights (41×41 = 1681 combos)...")
        ws, acc, fpr, fnr, thresh, tp_g, tn_g, fp_g, fn_g = sweep_weights(
            st_all, ae_all, labels_all, n_steps=41,
        )
        ws_ref = ws

        # Best by accuracy
        best_idx = np.unravel_index(np.nanargmax(acc), acc.shape)
        best_w_st = float(ws[best_idx[1]])
        best_w_ae = float(ws[best_idx[0]])
        best_acc = float(acc[best_idx])
        best_fpr = float(fpr[best_idx])
        best_fnr = float(fnr[best_idx])
        best_thresh = float(thresh[best_idx])
        best_tp = int(tp_g[best_idx])
        best_tn = int(tn_g[best_idx])
        best_fp = int(fp_g[best_idx])
        best_fn = int(fn_g[best_idx])

        print(f"        Best accuracy: {best_acc*100:.2f}% "
              f"@ ST={best_w_st:.2f}, AE={best_w_ae:.2f}, thresh={best_thresh:.4f}")
        print(f"        FPR={best_fpr*100:.2f}%  FNR={best_fnr*100:.2f}%  "
              f"TP={best_tp} TN={best_tn} FP={best_fp} FN={best_fn}")

        # Current config
        cur_ae_idx = (np.abs(ws - 1.0)).argmin()
        cur_st_idx = (np.abs(ws - 0.0)).argmin()
        cur_acc = float(acc[cur_ae_idx, cur_st_idx])
        cur_fpr = float(fpr[cur_ae_idx, cur_st_idx])
        cur_fnr = float(fnr[cur_ae_idx, cur_st_idx])
        print(f"        Current (ST=0,AE=1): Acc={cur_acc*100:.2f}%  "
              f"FPR={cur_fpr*100:.2f}%  FNR={cur_fnr*100:.2f}%")
        if best_acc > cur_acc:
            print(f"        ▲ Improvement: +{(best_acc - cur_acc)*100:.2f}% accuracy")

        # ── Misclassified heatmaps ──
        print(f"  [3/4] Generating feature heatmaps for misclassified images...")
        misclassified = generate_misclassified_heatmaps(
            predictor, st_all, ae_all, labels_all, paths_all,
            best_w_st, best_w_ae, best_thresh, label, OUTPUT_DIR,
        )

        # ── Cleanup ──
        print(f"  [4/4] Cleaning up...")
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Store results ──
        models_data.append({
            "label": label, "bs": m["bs"],
            "acc": acc, "fpr": fpr, "fnr": fnr,
            "best_w_st": best_w_st, "best_w_ae": best_w_ae,
            "best_thresh": best_thresh,
            "best_acc": best_acc, "best_fpr": best_fpr, "best_fnr": best_fnr,
            "best_confusion": {"TP": best_tp, "TN": best_tn, "FP": best_fp, "FN": best_fn},
            "current_acc": cur_acc, "current_fpr": cur_fpr, "current_fnr": cur_fnr,
            "misclassified_count": len(misclassified),
        })

        all_results[m["name"]] = {
            "batch_size": m["bs"],
            "test_counts": {"good": n_good, "broken": n_broken, "total": len(labels_all)},
            "ws": ws.tolist(),
            "accuracy_grid": acc.tolist(),
            "fpr_grid": fpr.tolist(),
            "fnr_grid": fnr.tolist(),
            "threshold_grid": thresh.tolist(),
            "best": {
                "w_st": best_w_st,
                "w_ae": best_w_ae,
                "threshold": best_thresh,
                "accuracy": best_acc,
                "fpr": best_fpr,
                "fnr": best_fnr,
                "tp": best_tp,
                "tn": best_tn,
                "fp": best_fp,
                "fn": best_fn,
            },
            "current": {
                "w_st": 0.0,
                "w_ae": 1.0,
                "accuracy": cur_acc,
                "fpr": cur_fpr,
                "fnr": cur_fnr,
            },
            "misclassified": misclassified,
        }

    # ═════════════════════════════════════════════════════════════════
    #  Plot & save
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  Generating combined heatmap...")
    heatmap_path = heatmap_dir / "weight_sweep_accuracy_heatmap.png"
    plot_combined_heatmaps(models_data, ws_ref, heatmap_path)

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  Summary saved: {summary_path}")

    # ═════════════════════════════════════════════════════════════════
    #  Console summary
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'='*85}")
    print(f"  FINAL SUMMARY — ST×AE Weight Sweep Experiment")
    print(f"{'='*85}")
    print(f"  {'Model':<22} {'Best ST':>7} {'Best AE':>7} "
          f"{'Acc':>8} {'FPR':>8} {'FNR':>8} {'Thresh':>9} {'Mis':>5}")
    print(f"  {'-'*75}")
    for data in models_data:
        print(f"  {data['label']:<22} "
              f"{data['best_w_st']:>7.2f} {data['best_w_ae']:>7.2f} "
              f"{data['best_acc']*100:>7.2f}% {data['best_fpr']*100:>7.2f}% "
              f"{data['best_fnr']*100:>7.2f}% {data['best_thresh']:>9.4f} "
              f"{data['misclassified_count']:>5}")

    print(f"\n  Comparison: Best vs Current (ST=0, AE=1)")
    print(f"  {'Model':<22} {'Cur Acc':>8} {'Cur FPR':>8} {'Cur FNR':>8} "
          f"{'Best Acc':>8} {'Δ Acc':>8} {'Δ FPR':>8} {'Δ FNR':>8}")
    print(f"  {'-'*80}")
    for data in models_data:
        d_acc = data["best_acc"] - data["current_acc"]
        d_fpr = data["best_fpr"] - data["current_fpr"]
        d_fnr = data["best_fnr"] - data["current_fnr"]
        print(f"  {data['label']:<22} "
              f"{data['current_acc']*100:>7.2f}% {data['current_fpr']*100:>7.2f}% "
              f"{data['current_fnr']*100:>7.2f}% "
              f"{data['best_acc']*100:>7.2f}% "
              f"{d_acc*100:>+7.2f}% {d_fpr*100:>+7.2f}% {d_fnr*100:>+7.2f}%")

    print(f"\n  All outputs in: {OUTPUT_DIR}")
    print(f"    Heatmaps:       {heatmap_dir}")
    print(f"    Misclassified:  {OUTPUT_DIR / 'misclassified'}")
    print(f"    Summary JSON:   {summary_path}")
    print(f"{'='*85}")


if __name__ == "__main__":
    main()
