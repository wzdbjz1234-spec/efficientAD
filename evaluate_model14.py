#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate model 14 on mydataset/my_product test set.

Reports:
  1. CPU / GPU single-inference average latency
  2. Inference accuracy (optimal threshold, confusion matrix)
  3. Heatmaps of ST & STAE error for every misclassified sample
"""

import os, sys, time, json, argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── project paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
EFFICIENTAD_DIR = PROJECT_ROOT / "EfficientAD-main"
MODEL_DIR = EFFICIENTAD_DIR / "output" / "16"
TEST_DIR = PROJECT_ROOT / "mydataset" / "my_product" / "test"
TRAIN_DIR = PROJECT_ROOT / "mydataset" / "my_product" / "train"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results_16"
HEATMAP_DIR = OUTPUT_DIR / "misclassified_heatmaps"

sys.path.insert(0, str(EFFICIENTAD_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from efficientad_tools import EfficientADPredictor, ModelArtifacts

# ── helpers ────────────────────────────────────────────────────────────

GPU_AVAILABLE = torch.cuda.is_available()


def load_predictor(device: str):
    """Load model 14 predictor on the requested device."""
    artifacts = ModelArtifacts.from_output(
        str(MODEL_DIR), dataset="mvtec_ad", product="my_product",
        source_dir=str(EFFICIENTAD_DIR),
    )
    # norm_params.json will be computed on first run if missing
    return EfficientADPredictor.load(
        artifacts, device=device, train_dir=str(TRAIN_DIR),
    )


def discover_test_images():
    """Return (good_paths, broken_paths) sorted."""
    good = sorted((TEST_DIR / "good").glob("*.png"))
    broken = sorted((TEST_DIR / "broken").glob("*.png"))
    return good, broken


def benchmark_inference(predictor, image_paths, warmup=5, repeats=3):
    """Measure per-image inference time. Returns list of seconds."""
    # pre-load all images as numpy arrays
    images_np = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is not None:
            images_np.append(img)

    # warmup
    for img in images_np[:min(warmup, len(images_np))]:
        _ = predictor.predict(img, include_features=False)

    times = []
    for _ in range(repeats):
        for img in images_np:
            t0 = time.perf_counter()
            _ = predictor.predict(img, include_features=False)
            times.append(time.perf_counter() - t0)

    return times


def evaluate_all(predictor, good_paths, broken_paths):
    """
    Run inference on every test image, return list of dicts:
      {path, label: 'good'|'broken', score, st_score, ae_score}
    """
    results = []
    total = len(good_paths) + len(broken_paths)

    for i, (paths, label) in enumerate([
        (good_paths, "good"),
        (broken_paths, "broken"),
    ]):
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            pred = predictor.predict(img, include_features=False)
            results.append({
                "path": str(p),
                "filename": p.name,
                "label": label,
                "score": pred.score,
                "st_score": pred.student_teacher_score,
                "ae_score": pred.autoencoder_score,
            })

    return results


def find_best_threshold(results):
    """Find threshold that maximizes accuracy."""
    scores = np.array([r["score"] for r in results])
    labels = np.array([1 if r["label"] == "broken" else 0 for r in results])

    # sort by score
    idx = np.argsort(scores)
    scores_sorted = scores[idx]
    labels_sorted = labels[idx]

    best_acc = 0.0
    best_thresh = 0.0
    best_tp = best_tn = best_fp = best_fn = 0

    # scan thresholds between scores
    for i in range(len(scores_sorted) - 1):
        thresh = (scores_sorted[i] + scores_sorted[i + 1]) / 2.0
        preds = (scores >= thresh).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        acc = (tp + tn) / len(labels)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            best_tp, best_tn, best_fp, best_fn = tp, tn, fp, fn

    return {
        "threshold": float(best_thresh),
        "accuracy": float(best_acc),
        "tp": best_tp, "tn": best_tn,
        "fp": best_fp, "fn": best_fn,
    }


def get_misclassified(results, threshold):
    """Return results where prediction != ground truth."""
    mis = []
    for r in results:
        pred_anomaly = r["score"] >= threshold
        true_anomaly = r["label"] == "broken"
        if pred_anomaly != true_anomaly:
            mis.append(r)
    return mis


# ── heatmap drawing ────────────────────────────────────────────────────

def draw_st_stae_heatmap(image_path, predictor, output_path):
    """
    Draw a multi-panel figure showing ST and AE error heatmaps:
      Row 0: Input image (full width)
      Row 1: ST map overlay | AE map overlay | Combined map overlay
      Row 2: ST diff top-8 channels (hot colormap)
      Row 3: AE diff top-8 channels (hot colormap)
      Row 4: ST mean diff overlay + standalone  |  AE mean diff overlay + standalone
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"  SKIP: cannot read {image_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]

    # get prediction WITH features
    pred = predictor.predict(img_bgr, include_features=True)
    features = pred.features

    st_map = pred.student_teacher_map   # [H, W]
    ae_map = pred.autoencoder_map       # [H, W]
    combined_map = pred.anomaly_map     # [H, W]

    # per-channel diffs
    st_diff = features.student_teacher_diff  # [384, H_f, W_f]
    ae_diff = features.autoencoder_diff       # [384, H_f, W_f]

    st_ch_mean = st_diff.mean(axis=(1, 2))   # [384]
    ae_ch_mean = ae_diff.mean(axis=(1, 2))   # [384]

    top_k = 8
    n_cols = max(top_k, 4)
    st_top = np.argsort(st_ch_mean)[::-1][:top_k]
    ae_top = np.argsort(ae_ch_mean)[::-1][:top_k]

    # ── build figure: 8 rows (input + 3 maps + 2×channel rows + 2×summary rows) ──
    fig = plt.figure(figsize=(3.5 * n_cols, 22))

    # Row 0: Input image  ──────────────────────────────────────────────
    ax_in = plt.subplot2grid((8, n_cols), (0, 0), colspan=n_cols)
    ax_in.imshow(img_rgb)
    ax_in.set_title(f"Input: {Path(image_path).name}  ({w_orig}x{h_orig})",
                    fontsize=13, fontweight="bold")
    ax_in.axis("off")

    # Helper to draw overlay
    def draw_overlay(ax, bg, overlay, title, cmap="hot"):
        ax.imshow(bg)
        vmax = max(float(overlay.max()), 1e-8)
        im = ax.imshow(overlay, cmap=cmap, alpha=0.55, vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
        return im

    # Row 1: ST / AE / Combined map overlays  ──────────────────────────
    n_maps = 3
    for j, (map_data, map_title) in enumerate([
        (st_map, f"ST Map  max={st_map.max():.5f}"),
        (ae_map, f"AE Map  max={ae_map.max():.5f}"),
        (combined_map, f"Combined Map  max={combined_map.max():.5f}"),
    ]):
        c0 = j * (n_cols // 3)
        cs = n_cols // 3
        ax = plt.subplot2grid((8, n_cols), (1, c0), colspan=cs)
        im = draw_overlay(ax, img_rgb, map_data, map_title)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

    # Helper: draw a row of top-K channel diffs
    def draw_channel_row(grid_row, diff, ch_mean, top_indices, label, cmap="hot"):
        for i, ch_idx in enumerate(top_indices):
            ch = diff[ch_idx]  # [H_f, W_f]
            ch_r = cv2.resize(ch, (w_orig, h_orig),
                              interpolation=cv2.INTER_LINEAR)
            ax = plt.subplot2grid((8, n_cols), (grid_row, i))
            ax.imshow(ch_r, cmap=cmap)
            ax.set_title(f"{label} ch{ch_idx}\nμ={ch_mean[ch_idx]:.4f}",
                         fontsize=7)
            ax.axis("off")

    # Row 2: ST diff top-8 channels  ───────────────────────────────────
    draw_channel_row(2, st_diff, st_ch_mean, st_top, "ST", "hot")

    # Row 3: AE diff top-8 channels  ───────────────────────────────────
    draw_channel_row(3, ae_diff, ae_ch_mean, ae_top, "AE", "hot")

    # Row 4: ST mean diff across all 384 channels  ─────────────────────
    st_mean = st_diff.mean(axis=0)
    st_mean = cv2.resize(st_mean, (w_orig, h_orig),
                         interpolation=cv2.INTER_LINEAR)

    ax4a = plt.subplot2grid((8, n_cols), (4, 0), colspan=n_cols // 2)
    ax4a.imshow(img_rgb)
    im4a = ax4a.imshow(st_mean, cmap="hot", alpha=0.55)
    ax4a.set_title(f"ST Mean Diff (384 ch)\nmax={st_mean.max():.6f}",
                   fontsize=10, fontweight="bold")
    ax4a.axis("off")

    ax4b = plt.subplot2grid((8, n_cols), (4, n_cols // 2),
                            colspan=n_cols - n_cols // 2)
    ax4b.imshow(st_mean, cmap="hot")
    ax4b.set_title(f"ST Mean Diff\nmax={st_mean.max():.6f}  "
                   f"μ={float(st_mean.mean()):.6f}",
                   fontsize=10)
    ax4b.axis("off")
    plt.colorbar(im4a, ax=ax4b, fraction=0.04, pad=0.03)

    # Row 5: AE mean diff across all 384 channels  ─────────────────────
    ae_mean = ae_diff.mean(axis=0)
    ae_mean = cv2.resize(ae_mean, (w_orig, h_orig),
                         interpolation=cv2.INTER_LINEAR)

    ax5a = plt.subplot2grid((8, n_cols), (5, 0), colspan=n_cols // 2)
    ax5a.imshow(img_rgb)
    im5a = ax5a.imshow(ae_mean, cmap="hot", alpha=0.55)
    ax5a.set_title(f"AE Mean Diff (384 ch)\nmax={ae_mean.max():.6f}",
                   fontsize=10, fontweight="bold")
    ax5a.axis("off")

    ax5b = plt.subplot2grid((8, n_cols), (5, n_cols // 2),
                            colspan=n_cols - n_cols // 2)
    ax5b.imshow(ae_mean, cmap="hot")
    ax5b.set_title(f"AE Mean Diff\nmax={ae_mean.max():.6f}  "
                   f"μ={float(ae_mean.mean()):.6f}",
                   fontsize=10)
    ax5b.axis("off")
    plt.colorbar(im5a, ax=ax5b, fraction=0.04, pad=0.03)

    # Row 6+7: ST vs AE diff per-channel bar chart ─────────────────────
    # top-30 channels comparison
    n_show = min(30, len(st_ch_mean))
    ax6 = plt.subplot2grid((8, n_cols), (6, 0), colspan=n_cols)
    x = np.arange(n_show)
    w = 0.35
    ax6.bar(x - w/2, st_ch_mean[:n_show], w, label="ST diff", color="#E74C3C",
            alpha=0.85)
    ax6.bar(x + w/2, ae_ch_mean[:n_show], w, label="AE diff", color="#3498DB",
            alpha=0.85)
    ax6.set_xlabel("Channel index", fontsize=9)
    ax6.set_ylabel("Mean squared diff", fontsize=9)
    ax6.set_title(f"Per-Channel Diff: Top-{n_show} ST channels  |  "
                  f"ST μ={st_ch_mean.mean():.6f}  AE μ={ae_ch_mean.mean():.6f}",
                  fontsize=10, fontweight="bold")
    ax6.legend(fontsize=9)
    ax6.tick_params(labelsize=7)

    # Row 7: ST vs AE score distribution bar
    ax7 = plt.subplot2grid((8, n_cols), (7, 0), colspan=n_cols)
    scores = [pred.student_teacher_score, pred.autoencoder_score, pred.score]
    labels_s = ["ST Score", "AE Score", "Combined Score"]
    colors = ["#E74C3C", "#3498DB", "#2ECC71"]
    bars = ax7.barh(labels_s, scores, color=colors, alpha=0.85, height=0.5)
    for bar, val in zip(bars, scores):
        ax7.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.6f}", va="center", fontsize=10, fontweight="bold")
    ax7.set_xlabel("Score", fontsize=9)
    ax7.set_title("ST / AE / Combined Score Breakdown", fontsize=11,
                  fontweight="bold")
    ax7.tick_params(labelsize=9)

    plt.tight_layout(pad=1.5)
    fig.suptitle(
        f"ST–STAE Error Heatmap  |  {Path(image_path).name}  |  "
        f"label={'broken' if 'broken' in str(image_path) else 'good'}  |  "
        f"score={pred.score:.6f}  (ST={pred.student_teacher_score:.5f}  "
        f"AE={pred.autoencoder_score:.5f})",
        fontsize=14, fontweight="bold", y=1.01,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap: {output_path}")


# ── main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate model 14")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-heatmaps", action="store_true",
                        help="Skip drawing misclassified heatmaps")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Skip GPU benchmark")
    parser.add_argument("--gpu-only", action="store_true",
                        help="Skip CPU benchmark")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(HEATMAP_DIR, exist_ok=True)

    good_paths, broken_paths = discover_test_images()
    all_paths = good_paths + broken_paths
    print(f"Test images: {len(good_paths)} good + {len(broken_paths)} broken "
          f"= {len(all_paths)} total")

    # ── 1. CPU benchmark ────────────────────────────────────────────
    cpu_times = None
    cpu_arr = None
    if not args.gpu_only:
        print("\n" + "=" * 60)
        print("  CPU BENCHMARK")
        print("=" * 60)
        try:
            cpu_pred = load_predictor("cpu")
            cpu_times = benchmark_inference(cpu_pred, all_paths[:30],
                                             warmup=3, repeats=3)
            cpu_arr = np.array(cpu_times)
            print(f"  Samples : {len(cpu_times)}")
            print(f"  Mean    : {cpu_arr.mean()*1000:.2f} ms")
            print(f"  Median  : {np.median(cpu_arr)*1000:.2f} ms")
            print(f"  Std     : {cpu_arr.std()*1000:.2f} ms")
            print(f"  Min     : {cpu_arr.min()*1000:.2f} ms")
            print(f"  Max     : {cpu_arr.max()*1000:.2f} ms")
            print(f"  FPS     : {1.0/cpu_arr.mean():.1f}")
            del cpu_pred
            if GPU_AVAILABLE:
                torch.cuda.empty_cache()
        except Exception as e:
                print(f"  CPU benchmark failed: {e}")
                cpu_times = None

    # ── 2. GPU benchmark ────────────────────────────────────────────
    gpu_times = None
    gpu_arr = None
    if GPU_AVAILABLE and not args.cpu_only:
        print("\n" + "=" * 60)
        print("  GPU BENCHMARK (CUDA)")
        print("=" * 60)
        try:
            gpu_pred = load_predictor("cuda")
            # FP32 benchmark (FP16 requires converting all input/norm tensors too)
            gpu_times = benchmark_inference(gpu_pred, all_paths[:30],
                                             warmup=5, repeats=5)
            gpu_arr = np.array(gpu_times)
            print(f"  Samples : {len(gpu_times)}")
            print(f"  Mean    : {gpu_arr.mean()*1000:.2f} ms")
            print(f"  Median  : {np.median(gpu_arr)*1000:.2f} ms")
            print(f"  Std     : {gpu_arr.std()*1000:.2f} ms")
            print(f"  Min     : {gpu_arr.min()*1000:.2f} ms")
            print(f"  Max     : {gpu_arr.max()*1000:.2f} ms")
            print(f"  FPS     : {1.0/gpu_arr.mean():.1f}")
            del gpu_pred
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  GPU benchmark failed: {e}")
            gpu_times = None

    # ── 3. Full evaluation (accuracy) ───────────────────────────────
    print("\n" + "=" * 60)
    print("  FULL EVALUATION")
    print("=" * 60)

    eval_device = "cuda" if GPU_AVAILABLE else "cpu"
    eval_pred = load_predictor(eval_device)
    results = evaluate_all(eval_pred, good_paths, broken_paths)

    # compute accuracy
    best = find_best_threshold(results)
    scores_all = np.array([r["score"] for r in results])
    labels_all = np.array([1 if r["label"] == "broken" else 0 for r in results])

    print(f"\n  Best threshold : {best['threshold']:.6f}")
    print(f"  Accuracy        : {best['accuracy']*100:.2f}%")
    print(f"  TP (broken→broken) : {best['tp']}")
    print(f"  TN (good→good)     : {best['tn']}")
    print(f"  FP (good→broken)   : {best['fp']}  ← false positives")
    print(f"  FN (broken→good)   : {best['fn']}  ← false negatives")

    precision = best['tp'] / max(best['tp'] + best['fp'], 1)
    recall = best['tp'] / max(best['tp'] + best['fn'], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    print(f"  Precision       : {precision*100:.2f}%")
    print(f"  Recall          : {recall*100:.2f}%")
    print(f"  F1 score        : {f1*100:.2f}%")

    # per-class accuracy
    good_correct = best['tn']
    good_total = len(good_paths)
    broken_correct = best['tp']
    broken_total = len(broken_paths)
    print(f"\n  Good accuracy   : {good_correct}/{good_total} "
          f"({100*good_correct/max(good_total,1):.1f}%)")
    print(f"  Broken accuracy : {broken_correct}/{broken_total} "
          f"({100*broken_correct/max(broken_total,1):.1f}%)")

    # ── score distribution ──
    good_scores = np.array([r["score"] for r in results if r["label"] == "good"])
    broken_scores = np.array([r["score"] for r in results if r["label"] == "broken"])
    print(f"\n  Good scores   : min={good_scores.min():.5f}  "
          f"mean={good_scores.mean():.5f}  max={good_scores.max():.5f}")
    print(f"  Broken scores : min={broken_scores.min():.5f}  "
          f"mean={broken_scores.mean():.5f}  max={broken_scores.max():.5f}")

    # ── 4. Misclassified heatmaps ───────────────────────────────────
    mis = get_misclassified(results, best["threshold"])
    print(f"\n  Misclassified  : {len(mis)} samples "
          f"({best['fp']} FP + {best['fn']} FN)")

    if mis and not args.skip_heatmaps:
        print("\n" + "=" * 60)
        print(f"  DRAWING HEATMAPS FOR {len(mis)} MISCLASSIFIED SAMPLES")
        print("=" * 60)

        for i, m in enumerate(mis):
            fname = Path(m["path"]).stem
            pred_label = "broken" if m["score"] >= best["threshold"] else "good"
            out_name = (
                f"{i+1:03d}_true-{m['label']}_pred-{pred_label}_"
                f"score-{m['score']:.6f}_{fname}.png"
            )
            out_path = HEATMAP_DIR / out_name
            print(f"\n[{i+1}/{len(mis)}] {m['filename']}  "
                  f"true={m['label']}  pred={pred_label}  score={m['score']:.6f}")
            draw_st_stae_heatmap(m["path"], eval_pred, str(out_path))

    # ── 5. Save summary ─────────────────────────────────────────────
    summary = {
        "model": "14",
        "dataset": "mydataset/my_product",
        "test_images": {
            "good": len(good_paths),
            "broken": len(broken_paths),
            "total": len(all_paths),
        },
        "cpu_latency_ms": {
            "mean": float(cpu_arr.mean() * 1000) if cpu_times else None,
            "median": float(np.median(cpu_arr) * 1000) if cpu_times else None,
            "std": float(cpu_arr.std() * 1000) if cpu_times else None,
            "min": float(cpu_arr.min() * 1000) if cpu_times else None,
            "max": float(cpu_arr.max() * 1000) if cpu_times else None,
            "fps": float(1.0 / cpu_arr.mean()) if cpu_times else None,
            "n_samples": len(cpu_times) if cpu_times else 0,
        } if cpu_times is not None else None,
        "gpu_latency_ms": {
            "mean": float(gpu_arr.mean() * 1000) if gpu_times else None,
            "median": float(np.median(gpu_arr) * 1000) if gpu_times else None,
            "std": float(gpu_arr.std() * 1000) if gpu_times else None,
            "min": float(gpu_arr.min() * 1000) if gpu_times else None,
            "max": float(gpu_arr.max() * 1000) if gpu_times else None,
            "fps": float(1.0 / gpu_arr.mean()) if gpu_times else None,
            "n_samples": len(gpu_times) if gpu_times else 0,
        } if gpu_times is not None else None,
        "accuracy": best,
        "per_class": {
            "good_accuracy": float(good_correct / max(good_total, 1)),
            "broken_accuracy": float(broken_correct / max(broken_total, 1)),
        },
        "score_distribution": {
            "good": {"min": float(good_scores.min()), "mean": float(good_scores.mean()),
                      "max": float(good_scores.max())},
            "broken": {"min": float(broken_scores.min()), "mean": float(broken_scores.mean()),
                        "max": float(broken_scores.max())},
        },
        "misclassified": [
            {
                "filename": m["filename"],
                "true_label": m["label"],
                "pred_label": "broken" if m["score"] >= best["threshold"] else "good",
                "score": float(m["score"]),
                "st_score": float(m["st_score"]),
                "ae_score": float(m["ae_score"]),
            }
            for m in mis
        ],
        "all_results": [
            {
                "filename": r["filename"],
                "true_label": r["label"],
                "score": float(r["score"]),
                "st_score": float(r["st_score"]),
                "ae_score": float(r["ae_score"]),
                "pred_label": "broken" if r["score"] >= best["threshold"] else "good",
            }
            for r in results
        ],
    }

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_path}")

    # ── 6. Print final summary table ─────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY — Model 14 on mydataset/my_product")
    print("=" * 70)
    print(f"  Test set        : {len(good_paths)} good + {len(broken_paths)} broken")
    print(f"  Accuracy        : {best['accuracy']*100:.2f}%")
    print(f"  Precision       : {precision*100:.2f}%")
    print(f"  Recall          : {recall*100:.2f}%")
    print(f"  F1              : {f1*100:.2f}%")
    print(f"  FP (false alarm): {best['fp']}  |  FN (miss)      : {best['fn']}")
    if cpu_times:
        print(f"  CPU latency     : {cpu_arr.mean()*1000:.1f} ms "
              f"(median {np.median(cpu_arr)*1000:.1f} ms, "
              f"{1.0/cpu_arr.mean():.1f} FPS)")
    if gpu_times:
        print(f"  GPU latency     : {gpu_arr.mean()*1000:.1f} ms "
              f"(median {np.median(gpu_arr)*1000:.1f} ms, "
              f"{1.0/gpu_arr.mean():.1f} FPS)")
    if cpu_times and gpu_times:
        speedup = cpu_arr.mean() / gpu_arr.mean()
        print(f"  GPU speedup     : {speedup:.1f}x")
    print(f"  Heatmaps        : {HEATMAP_DIR} ({len(mis)} files)")
    print(f"  Full results    : {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
