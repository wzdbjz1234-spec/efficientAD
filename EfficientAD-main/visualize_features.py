import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from common import get_autoencoder, get_pdn_small

DEFAULT_IMAGE_SIZE = 256
DEFAULT_OUT_CHANNELS = 384
on_gpu = torch.cuda.is_available()

default_transform = transforms.Compose([
    transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_models(teacher_path, student_path):
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
    student = torch.load(student_path, map_location='cpu', weights_only=False)
    teacher.eval()
    student.eval()
    if on_gpu:
        teacher.cuda()
        student.cuda()
    return teacher, student


def load_and_preprocess(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]
    tensor = default_transform(Image.fromarray(img_rgb)).unsqueeze(0)
    if on_gpu:
        tensor = tensor.cuda()
    return img_bgr, img_rgb, tensor, (h_orig, w_orig)


@torch.no_grad()
def extract_features(teacher, student, tensor, teacher_mean=None, teacher_std=None):
    t_out = teacher(tensor)      # [1, 384, 61, 61]
    if teacher_mean is not None:
        t_out = (t_out - teacher_mean) / teacher_std
    s_out = student(tensor)      # [1, 768, 61, 61]
    s_out_teacher = s_out[:, :DEFAULT_OUT_CHANNELS]   # 前 384 通道
    s_out_ae = s_out[:, DEFAULT_OUT_CHANNELS:]         # 后 384 通道
    return t_out, s_out_teacher, s_out_ae


def load_norm_params(norm_cache_path):
    with open(norm_cache_path) as f:
        data = json.load(f)
    def _t(arr):
        t = torch.tensor(arr)
        return t.cuda() if on_gpu else t
    tm = _t(data['teacher_mean']).view(1, -1, 1, 1)
    ts = _t(data['teacher_std']).view(1, -1, 1, 1)
    return tm, ts


def compute_diff_map(t_out, s_out):
    diff = (t_out - s_out) ** 2          # [1, 384, H, W]
    diff_map = torch.mean(diff, dim=1)   # [1, H, W] — 跨 384 通道平均
    return diff, diff_map


def visualize(image_path, teacher_path, student_path, output_path, top_k=8,
              show_ae=False, norm_cache_path=None):
    teacher, student = load_models(teacher_path, student_path)
    img_bgr, img_rgb, tensor, (h_orig, w_orig) = load_and_preprocess(image_path)

    tm = ts = None
    if norm_cache_path and os.path.isfile(norm_cache_path):
        tm, ts = load_norm_params(norm_cache_path)
        print(f"Loaded teacher normalization from {norm_cache_path}")
    else:
        print("WARNING: No norm cache — comparing RAW teacher output with student. "
              "Results may be misleading!")

    t_out, s_out_t, s_out_ae = extract_features(teacher, student, tensor, tm, ts)

    diff_full, diff_map = compute_diff_map(t_out, s_out_t)

    diff_map_np = diff_map[0].cpu().numpy()
    diff_map_resized = cv2.resize(diff_map_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    diff_per_channel = torch.mean(diff_full, dim=[2, 3])[0].cpu().numpy()  # [384]
    top_indices = np.argsort(diff_per_channel)[::-1][:top_k]

    n_cols = top_k + 1
    fig = plt.figure(figsize=(3 * n_cols, 12))

    ax_input = plt.subplot2grid((5, n_cols), (0, 0), colspan=n_cols)
    ax_input.imshow(img_rgb)
    ax_input.set_title(f"Input ({w_orig}×{h_orig})", fontsize=12, fontweight='bold')
    ax_input.axis('off')

    for i, ch_idx in enumerate(top_indices):
        t_ch = t_out[0, ch_idx].cpu().numpy()
        s_ch = s_out_t[0, ch_idx].cpu().numpy()
        d_ch = diff_full[0, ch_idx].cpu().numpy()
        diff_val = diff_per_channel[ch_idx]

        ax_t = plt.subplot2grid((5, n_cols), (1, i), colspan=1)
        ax_t.imshow(t_ch, cmap='viridis')
        ax_t.set_title(f"Teacher ch{ch_idx}", fontsize=8)
        ax_t.axis('off')

        ax_s = plt.subplot2grid((5, n_cols), (2, i), colspan=1)
        ax_s.imshow(s_ch, cmap='viridis')
        ax_s.set_title(f"Student ch{ch_idx}", fontsize=8)
        ax_s.axis('off')

        ax_d = plt.subplot2grid((5, n_cols), (3, i), colspan=1)
        im_d = ax_d.imshow(d_ch, cmap='hot')
        ax_d.set_title(f"Diff ch{ch_idx} ({diff_val:.4f})", fontsize=8)
        ax_d.axis('off')

    ax_map = plt.subplot2grid((5, n_cols), (4, 0), colspan=n_cols // 2)
    ax_map.imshow(img_rgb)
    im_map = ax_map.imshow(diff_map_resized, cmap='hot', alpha=0.5)
    ax_map.set_title("Anomaly Map (overlay)", fontsize=12, fontweight='bold')
    ax_map.axis('off')

    ax_bar = plt.subplot2grid((5, n_cols), (4, n_cols // 2), colspan=n_cols - n_cols // 2)
    ax_bar.imshow(diff_map_resized, cmap='hot')
    ax_bar.set_title(f"Anomaly Map\nmax={diff_map_resized.max():.6f}", fontsize=12,
                     fontweight='bold')
    ax_bar.axis('off')
    cbar = plt.colorbar(im_map, ax=ax_bar, fraction=0.046, pad=0.04)
    cbar.set_label('Anomaly Score', fontsize=9)

    plt.tight_layout(pad=1.5)
    fig.suptitle(f"Teacher vs Student — Feature Map Comparison\n{os.path.basename(image_path)}",
                 fontsize=14, fontweight='bold', y=1.01)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='visualize_features',
        description='Compare teacher vs student feature maps')

    parser.add_argument('--image', required=True, help='Input image')
    parser.add_argument('--teacher', default=os.path.join(
        SCRIPT_DIR, 'output/1/trainings/mvtec_ad/my_product/teacher_final.pth'))
    parser.add_argument('--student', default=os.path.join(
        SCRIPT_DIR, 'output/1/trainings/mvtec_ad/my_product/student_final.pth'))
    parser.add_argument('--output', '-o', default='feature_comparison.png',
                        help='Output image path')
    parser.add_argument('--top-k', type=int, default=8,
                        help='Show top-K channels with largest difference')
    parser.add_argument('--norm-cache', default=os.path.join(
        SCRIPT_DIR, 'output/1/trainings/mvtec_ad/my_product/norm_params.json'),
                        help='Path to norm_params.json for teacher normalization')

    args = parser.parse_args()

    visualize(args.image, args.teacher, args.student, args.output,
              top_k=args.top_k, norm_cache_path=args.norm_cache)
