#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sweep ST × AE weights for models 14/15/16 on broken-sample detection rate.
Plots a heatmap: ST weight vs AE weight vs recall@95-specificity.
"""
import os, sys, json, time
from pathlib import Path
from collections import defaultdict

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

# ── collect good & broken scores for a given predictor ────────────────

def collect_scores(predictor):
    """Return (good_scores, broken_scores) lists."""
    good_dir = TEST_DIR / "good"
    broken_dir = TEST_DIR / "broken"

    good_scores = []
    for p in sorted(good_dir.glob("*.png")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        pred = predictor.predict(img, include_features=False)
        good_scores.append((pred.student_teacher_score, pred.autoencoder_score))

    broken_scores = []
    for p in sorted(broken_dir.glob("*.png")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        pred = predictor.predict(img, include_features=False)
        broken_scores.append((pred.student_teacher_score, pred.autoencoder_score))

    return good_scores, broken_scores


# ── sweep weights ─────────────────────────────────────────────────────

def compute_recall(good, broken, w_st, w_ae, specificity=0.95):
    """
    For given (w_st, w_ae), compute combined scores:
      score = w_st * st + w_ae * ae
    Find threshold at given specificity on good samples,
    return recall on broken samples.
    """
    good_combined = np.array([w_st * st + w_ae * ae for st, ae in good])
    broken_combined = np.array([w_st * st + w_ae * ae for st, ae in broken])

    # threshold at given specificity
    n_good = len(good_combined)
    n_allow_fp = max(1, int(n_good * (1 - specificity)))
    threshold = np.sort(good_combined)[-n_allow_fp]

    recall = np.mean(broken_combined >= threshold)
    return recall, threshold


def sweep_model(model_name, model_dir, good_scores, broken_scores,
                n_steps=21):
    """Sweep ST and AE weights from 0.0 to 1.0."""
    ws = np.linspace(0, 1, n_steps)
    recall_grid = np.full((n_steps, n_steps), np.nan)

    for i, w_st in enumerate(ws):
        for j, w_ae in enumerate(ws):
            if w_st == 0 and w_ae == 0:
                continue  # meaningless
            recall, thresh = compute_recall(good_scores, broken_scores,
                                            w_st, w_ae)
            recall_grid[j, i] = recall  # row=AE, col=ST

    return ws, recall_grid


# ── plot ──────────────────────────────────────────────────────────────

def plot_heatmaps(models_data, ws, output_path):
    """Draw 3 heatmaps side by side."""
    n_models = len(models_data)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5),
                              squeeze=False)

    vmin = min(np.nanmin(d["grid"]) for d in models_data)
    vmax = max(np.nanmax(d["grid"]) for d in models_data)
    # pad range slightly
    vmin = max(0, vmin - 0.02)
    vmax = min(1, vmax + 0.02)

    # custom colormap
    colors = ["#2166AC", "#F7F7F7", "#B2182B"]  # blue -> white -> red
    cmap = LinearSegmentedColormap.from_list("bwr_custom", colors, N=256)

    best_per_model = []

    for idx, data in enumerate(models_data):
        ax = axes[0, idx]
        grid = data["grid"]
        # mask nan (w_st=w_ae=0)
        masked = np.ma.masked_where(np.isnan(grid), grid)

        im = ax.pcolormesh(ws, ws, masked, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="auto", edgecolors="none")

        # mark best point
        best_idx = np.unravel_index(np.nanargmax(grid), grid.shape)
        best_w_ae = ws[best_idx[0]]
        best_w_st = ws[best_idx[1]]
        best_recall = grid[best_idx]
        best_per_model.append((best_w_st, best_w_ae, best_recall))
        ax.plot(best_w_st, best_w_ae, marker="*", color="lime",
                markersize=18, markeredgecolor="black", markeredgewidth=1,
                zorder=5)
        ax.annotate(f" best\n({best_w_st:.1f},{best_w_ae:.1f})\n"
                    f" R={best_recall*100:.1f}%",
                    xy=(best_w_st, best_w_ae),
                    xytext=(best_w_st + 0.12, best_w_ae + 0.06),
                    fontsize=8, fontweight="bold", color="lime",
                    bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.8))

        # mark current (ST=0, AE=1)
        if 0.0 in ws and 1.0 in ws:
            current_recall = grid[ws == 1.0, :][0, ws == 0.0][0]
            ax.plot(0.0, 1.0, marker="D", color="cyan", markersize=10,
                    markeredgecolor="black", markeredgewidth=1, zorder=5)
            ax.annotate(f"cur\nR={current_recall*100:.1f}%",
                        xy=(0.0, 1.0),
                        xytext=(0.02, 0.92), fontsize=7, color="cyan",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black",
                                  alpha=0.7))

        ax.set_xlabel("ST weight", fontsize=11)
        ax.set_ylabel("AE weight", fontsize=11)
        ax.set_title(f"{data['label']}\n(batch_size={data['bs']})",
                     fontsize=12, fontweight="bold")
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))

    # colorbar
    cbar = fig.colorbar(im, ax=axes[0, :], fraction=0.015, pad=0.04,
                        location="right")
    cbar.set_label("Broken Recall @ 95% Specificity", fontsize=11)

    fig.suptitle(
        "ST × AE Weight Sweep — Broken Sample Detection Rate\n"
        "Threshold set at 95% specificity on good samples  |  "
        f"★ = best  ◆ = current (ST=0,AE=1)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    return best_per_model


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
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        model_dir = EFFICIENTAD_DIR / "output" / m["name"]
        artifacts = ModelArtifacts.from_output(
            str(model_dir), dataset="mvtec_ad", product="my_product",
            source_dir=str(EFFICIENTAD_DIR),
        )
        predictor = EfficientADPredictor.load(
            artifacts, device=device, train_dir=str(TRAIN_DIR),
        )

        print("  Collecting scores...")
        good, broken = collect_scores(predictor)
        print(f"    Good: {len(good)}  Broken: {len(broken)}")

        print("  Sweeping weights...")
        ws, grid = sweep_model(label, model_dir, good, broken, n_steps=21)
        print(f"    Best recall: {np.nanmax(grid)*100:.2f}%")

        models_data.append({
            "label": label,
            "bs": m["bs"],
            "grid": grid,
        })
        all_results[m["name"]] = {
            "ws": ws.tolist(),
            "grid": grid.tolist(),
        }

        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── plot ──
    out_path = OUTPUT_DIR / "weight_sweep_heatmap.png"
    best = plot_heatmaps(models_data, ws, out_path)

    # ── summary ──
    print(f"\n{'='*70}")
    print("  BEST WEIGHT COMBINATIONS PER MODEL")
    print(f"{'='*70}")
    print(f"  {'Model':<20} {'ST weight':>10} {'AE weight':>10} {'Recall':>10}")
    print(f"  {'-'*50}")
    for i, m in enumerate(models):
        w_st, w_ae, rec = best[i]
        print(f"  {m['name']} (bs={m['bs']}):           "
              f"{w_st:>8.2f}  {w_ae:>8.2f}  {rec*100:>8.1f}%")

    # Save JSON
    json_path = OUTPUT_DIR / "sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    print(f"Heatmap saved to: {out_path}")


if __name__ == "__main__":
    main()
