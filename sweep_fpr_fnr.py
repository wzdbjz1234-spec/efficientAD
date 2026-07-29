#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sweep ST×AE weights for models 14/15/16.
For each (w_st, w_ae): find optimal threshold → record FPR and FNR.
Plot heatmaps: 3 models × 2 metrics (FPR, FNR) in one figure.
"""
import os, sys, json
from pathlib import Path

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parent
EFFICIENTAD_DIR = PROJECT_ROOT / "EfficientAD-main"
TEST_DIR = PROJECT_ROOT / "mydataset" / "my_product" / "test"
TRAIN_DIR = PROJECT_ROOT / "mydataset" / "my_product" / "train"
OUTPUT_DIR = PROJECT_ROOT / "weight_sweep_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, str(EFFICIENTAD_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
from efficientad_tools import EfficientADPredictor, ModelArtifacts


def collect_scores(predictor):
    """Return (good_st_scores, good_ae_scores, broken_st_scores, broken_ae_scores)."""
    good_st, good_ae = [], []
    for p in sorted((TEST_DIR / "good").glob("*.png")):
        img = cv2.imread(str(p))
        if img is None: continue
        pred = predictor.predict(img, include_features=False)
        good_st.append(pred.student_teacher_score)
        good_ae.append(pred.autoencoder_score)

    broken_st, broken_ae = [], []
    for p in sorted((TEST_DIR / "broken").glob("*.png")):
        img = cv2.imread(str(p))
        if img is None: continue
        pred = predictor.predict(img, include_features=False)
        broken_st.append(pred.student_teacher_score)
        broken_ae.append(pred.autoencoder_score)

    return (np.array(good_st), np.array(good_ae),
            np.array(broken_st), np.array(broken_ae))


def evaluate_weights(good_st, good_ae, broken_st, broken_ae, w_st, w_ae):
    """
    For weight pair (w_st, w_ae), find best threshold (max accuracy),
    return (fpr, fnr, accuracy).
    """
    good_c = w_st * good_st + w_ae * good_ae
    broken_c = w_st * broken_st + w_ae * broken_ae

    # Find best threshold
    all_scores = np.concatenate([good_c, broken_c])
    all_labels = np.concatenate([np.zeros(len(good_c)), np.ones(len(broken_c))])

    # Sort by score, scan thresholds
    idx = np.argsort(all_scores)
    scores_sorted = all_scores[idx]
    labels_sorted = all_labels[idx]

    best_acc = 0.0
    best_fpr = 1.0
    best_fnr = 1.0

    for i in range(len(scores_sorted) - 1):
        thresh = (scores_sorted[i] + scores_sorted[i + 1]) / 2.0
        preds = (all_scores >= thresh).astype(int)
        tp = np.sum((preds == 1) & (all_labels == 1))
        tn = np.sum((preds == 0) & (all_labels == 0))
        fp = np.sum((preds == 1) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))
        acc = (tp + tn) / len(all_labels)
        if acc > best_acc:
            best_acc = acc
            best_fpr = fp / max(fp + tn, 1)  # FPR
            best_fnr = fn / max(fn + tp, 1)  # FNR

    return best_fpr, best_fnr, best_acc


def sweep_model(good_st, good_ae, broken_st, broken_ae, n_steps=41):
    """Sweep ST×AE in [0,1] with n_steps points. Return grids."""
    ws = np.linspace(0, 1, n_steps)
    fpr_grid = np.full((n_steps, n_steps), np.nan)
    fnr_grid = np.full((n_steps, n_steps), np.nan)
    acc_grid = np.full((n_steps, n_steps), np.nan)

    for i, w_st in enumerate(ws):
        for j, w_ae in enumerate(ws):
            if w_st == 0 and w_ae == 0:
                continue
            fpr, fnr, acc = evaluate_weights(
                good_st, good_ae, broken_st, broken_ae, w_st, w_ae)
            fpr_grid[j, i] = fpr
            fnr_grid[j, i] = fnr
            acc_grid[j, i] = acc

    return ws, fpr_grid, fnr_grid, acc_grid


# ── plot ──────────────────────────────────────────────────────────────

def plot_combined(models_data, ws, output_path):
    """3 models × 3 metrics in a 3×3 grid."""
    metrics = [
        ("fpr", "FPR (False Positive Rate)", "Reds"),
        ("fnr", "FNR (False Negative Rate)", "Blues"),
        ("acc", "Accuracy", "RdYlGn"),
    ]
    n_models = len(models_data)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(n_models, n_metrics,
                              figsize=(5.5 * n_metrics, 5.2 * n_models))

    # Ensure 2D axes array
    if n_models == 1:
        axes = axes.reshape(1, -1)

    for row, data in enumerate(models_data):
        for col, (key, title, cmap_name) in enumerate(metrics):
            ax = axes[row, col]
            grid = data[key]
            masked = np.ma.masked_where(np.isnan(grid), grid)

            vmin, vmax = 0.0, float(np.nanmax(grid))
            if key == "acc":
                vmin = float(np.nanmin(grid))

            if key == "fpr":
                cmap = LinearSegmentedColormap.from_list(
                    "fpr_cmap", ["#F7FBFF", "#EF6C5E"], N=256)
            elif key == "fnr":
                cmap = LinearSegmentedColormap.from_list(
                    "fnr_cmap", ["#F7FBFF", "#3A6FB5"], N=256)
            else:
                cmap = LinearSegmentedColormap.from_list(
                    "acc_cmap", ["#D73027", "#FFFFBF", "#1A9850"], N=256)

            im = ax.pcolormesh(ws, ws, masked, cmap=cmap,
                               vmin=vmin, vmax=vmax,
                               shading="auto", rasterized=True)

            # Mark best point
            if key == "acc":
                best_idx = np.unravel_index(np.nanargmax(grid), grid.shape)
                best_marker = "*"
                best_color = "black"
            elif key == "fpr":
                best_idx = np.unravel_index(np.nanargmin(grid), grid.shape)
                best_marker = "v"
                best_color = "darkred"
            else:
                best_idx = np.unravel_index(np.nanargmin(grid), grid.shape)
                best_marker = "v"
                best_color = "darkblue"

            best_w_ae = ws[best_idx[0]]
            best_w_st = ws[best_idx[1]]
            best_val = grid[best_idx]
            ax.plot(best_w_st, best_w_ae, marker=best_marker, color=best_color,
                    markersize=16, markeredgecolor="white",
                    markeredgewidth=1.5, zorder=5)
            val_str = f"{best_val*100:.1f}%"
            ax.annotate(f" best=({best_w_st:.2f},{best_w_ae:.2f})\n val={val_str}",
                        xy=(best_w_st, best_w_ae),
                        xytext=(0.05, 0.92), textcoords="axes fraction",
                        fontsize=7, fontweight="bold", color=best_color,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  alpha=0.85, ec=best_color))

            # Mark current config (ST=0, AE=1)
            cur_w_st, cur_w_ae = 0.0, 1.0
            cur_val = grid[(np.abs(ws - cur_w_ae)).argmin(),
                           (np.abs(ws - cur_w_st)).argmin()]
            ax.plot(cur_w_st, cur_w_ae, marker="D", color="magenta",
                    markersize=10, markeredgecolor="white",
                    markeredgewidth=1, zorder=5)
            ax.annotate(f"cur={cur_val*100:.1f}%",
                        xy=(0.02, 0.82), textcoords="axes fraction",
                        fontsize=7, color="magenta",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  alpha=0.85, ec="magenta"))

            ax.set_xlabel("ST weight", fontsize=9)
            ax.set_ylabel("AE weight", fontsize=9)
            ax.set_title(f"{data['label']} — {title}", fontsize=10,
                         fontweight="bold")
            ax.set_xticks(np.arange(0, 1.1, 0.25))
            ax.set_yticks(np.arange(0, 1.1, 0.25))

            plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

    fig.suptitle(
        "ST x AE Weight Sweep — FPR, FNR & Accuracy at Optimal Threshold\n"
        "v=best  diamond=current (ST=0,AE=1)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    models = [
        {"name": "14", "bs": 4},
        {"name": "15", "bs": 6},
        {"name": "16", "bs": 1},
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_data = []
    all_results = {}

    for m in models:
        label = f"Model {m['name']} (bs={m['bs']})"
        print(f"\n{'='*55}")
        print(f"  {label}")
        print(f"{'='*55}")

        model_dir = EFFICIENTAD_DIR / "output" / m["name"]
        artifacts = ModelArtifacts.from_output(
            str(model_dir), dataset="mvtec_ad", product="my_product",
            source_dir=str(EFFICIENTAD_DIR))
        predictor = EfficientADPredictor.load(
            artifacts, device=device, train_dir=str(TRAIN_DIR))

        print("  Collecting scores...")
        good_st, good_ae, broken_st, broken_ae = collect_scores(predictor)
        print(f"    Good: {len(good_st)}  Broken: {len(broken_st)}")

        print("  Sweeping weights (41×41 = 1681 combos)...")
        ws, fpr, fnr, acc = sweep_model(
            good_st, good_ae, broken_st, broken_ae, n_steps=41)

        # Find best
        best_acc_idx = np.unravel_index(np.nanargmax(acc), acc.shape)
        best_fpr_idx = np.unravel_index(np.nanargmin(fpr), fpr.shape)
        best_fnr_idx = np.unravel_index(np.nanargmin(fnr), fnr.shape)

        print(f"    Best Acc  : {acc[best_acc_idx]*100:.2f}% "
              f"@ ST={ws[best_acc_idx[1]]:.2f} AE={ws[best_acc_idx[0]]:.2f}")
        print(f"    Best FPR  : {fpr[best_fpr_idx]*100:.2f}% "
              f"@ ST={ws[best_fpr_idx[1]]:.2f} AE={ws[best_fpr_idx[0]]:.2f}")
        print(f"    Best FNR  : {fnr[best_fnr_idx]*100:.2f}% "
              f"@ ST={ws[best_fnr_idx[1]]:.2f} AE={ws[best_fnr_idx[0]]:.2f}")

        models_data.append({
            "label": label, "bs": m["bs"],
            "fpr": fpr, "fnr": fnr, "acc": acc,
        })
        all_results[m["name"]] = {
            "ws": ws.tolist(),
            "fpr": fpr.tolist(),
            "fnr": fnr.tolist(),
            "acc": acc.tolist(),
        }

        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── plot ──
    out_path = OUTPUT_DIR / "sweep_fpr_fnr_heatmap.png"
    plot_combined(models_data, ws, out_path)

    # Save JSON
    json_path = OUTPUT_DIR / "sweep_fpr_fnr.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── summary table ──
    print(f"\n{'='*75}")
    print("  BEST WEIGHT SUMMARY (max accuracy)")
    print(f"{'='*75}")
    print(f"  {'Model':<20} {'Best ST':>8} {'Best AE':>8} "
          f"{'Acc':>8} {'FPR':>8} {'FNR':>8}")
    print(f"  {'-'*55}")
    for data in models_data:
        acc = data["acc"]
        fpr = data["fpr"]
        fnr = data["fnr"]
        idx = np.unravel_index(np.nanargmax(acc), acc.shape)
        print(f"  {data['label']:<20} "
              f"{ws[idx[1]]:>8.2f} {ws[idx[0]]:>8.2f} "
              f"{acc[idx]*100:>7.2f}% {fpr[idx]*100:>7.2f}% "
              f"{fnr[idx]*100:>7.2f}%")

    print(f"\n  Current config (ST=0, AE=1):")
    print(f"  {'Model':<20} {'Acc':>8} {'FPR':>8} {'FNR':>8}")
    print(f"  {'-'*45}")
    for data in models_data:
        cur_j = (np.abs(ws - 1.0)).argmin()  # AE=1
        cur_i = (np.abs(ws - 0.0)).argmin()  # ST=0
        print(f"  {data['label']:<20} "
              f"{data['acc'][cur_j,cur_i]*100:>7.2f}% "
              f"{data['fpr'][cur_j,cur_i]*100:>7.2f}% "
              f"{data['fnr'][cur_j,cur_i]*100:>7.2f}%")

    print(f"\n  JSON : {json_path}")
    print(f"  Plot : {out_path}")


if __name__ == "__main__":
    main()
