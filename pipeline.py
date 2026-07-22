import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

EFFICIENTAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EfficientAD-main')
sys.path.insert(0, EFFICIENTAD_DIR)
from common import get_autoencoder, get_pdn_small, ImageFolderWithoutTarget

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
DEFAULT_IMAGE_SIZE = 256
DEFAULT_OUT_CHANNELS = 384

ORB_FEATURES = 5000
RATIO_THRESH = 0.75
MIN_GOOD_MATCHES = 10
RANSAC_THRESH = 5.0

on_gpu = torch.cuda.is_available()

default_transform = transforms.Compose([
    transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

_orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


# ─── ORB matching ───────────────────────────────────────────────────────

def _orb_match(tpl_gray, img_gray):
    kp1, des1 = _orb.detectAndCompute(tpl_gray, None)
    kp2, des2 = _orb.detectAndCompute(img_gray, None)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return None
    matches = _bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < RATIO_THRESH * n.distance]
    if len(good) < MIN_GOOD_MATCHES:
        return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH)
    if H is None:
        return None
    return H


def find_roi(image, template_name):
    tpl_dir = os.path.join(TEMPLATES_DIR, template_name)
    template_img = cv2.imread(os.path.join(tpl_dir, 'template.png'))
    if template_img is None:
        raise FileNotFoundError(f"Template not found: {template_name}")
    with open(os.path.join(tpl_dir, 'roi.json')) as f:
        rx, ry, rw, rh = json.load(f)['roi']

    tpl_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H = _orb_match(tpl_gray, img_gray)
    if H is None:
        return None

    src_corners = np.float32([
        [rx, ry], [rx + rw, ry], [rx + rw, ry + rh], [rx, ry + rh]
    ]).reshape(-1, 1, 2)
    dst_corners = cv2.perspectiveTransform(src_corners, H).reshape(4, 2).astype(int)

    out_w, out_h = int(rw), int(rh)
    dst_rect = np.float32([
        [0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]
    ]).reshape(-1, 1, 2)
    M = cv2.getPerspectiveTransform(dst_corners.astype(np.float32).reshape(-1, 1, 2), dst_rect)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))

    return {'warped': warped, 'corners': dst_corners}


# ─── EfficientAD inference ──────────────────────────────────────────────

def load_models(teacher_path, student_path, ae_path):
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
    student = torch.load(student_path, map_location='cpu', weights_only=False)
    autoencoder = torch.load(ae_path, map_location='cpu', weights_only=False)
    teacher.eval()
    student.eval()
    autoencoder.eval()
    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()
    return teacher, student, autoencoder


def compute_norm_params(teacher, student, autoencoder, train_dir, cache_path=None):
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        def _to_tensor(arr):
            return torch.tensor(arr)
        teacher_mean = _to_tensor(data['teacher_mean']).view(1, -1, 1, 1)
        teacher_std = _to_tensor(data['teacher_std']).view(1, -1, 1, 1)
        q_st_start = _to_tensor(data['q_st_start'])
        q_st_end = _to_tensor(data['q_st_end'])
        q_ae_start = _to_tensor(data['q_ae_start'])
        q_ae_end = _to_tensor(data['q_ae_end'])
        if on_gpu:
            teacher_mean = teacher_mean.cuda()
            teacher_std = teacher_std.cuda()
            q_st_start = q_st_start.cuda()
            q_st_end = q_st_end.cuda()
            q_ae_start = q_ae_start.cuda()
            q_ae_end = q_ae_end.cuda()
        return teacher_mean, teacher_std, q_st_start, q_st_end, q_ae_start, q_ae_end

    dataset = ImageFolderWithoutTarget(
        train_dir, transform=transforms.Lambda(lambda x: default_transform(x)))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    out_channels_ = DEFAULT_OUT_CHANNELS

    mean_outs = []
    for img in tqdm(loader, desc='Teacher normalization'):
        if on_gpu:
            img = img.cuda()
        with torch.no_grad():
            out = teacher(img)
            mean_outs.append(torch.mean(out, dim=[0, 2, 3]).cpu())
    channel_mean = torch.mean(torch.stack(mean_outs), dim=0)[None, :, None, None]
    if on_gpu:
        channel_mean = channel_mean.cuda()

    mean_dists = []
    for img in tqdm(loader, desc='Teacher std'):
        if on_gpu:
            img = img.cuda()
        with torch.no_grad():
            out = teacher(img)
            dist = (out - channel_mean) ** 2
            mean_dists.append(torch.mean(dist, dim=[0, 2, 3]).cpu())
    channel_var = torch.mean(torch.stack(mean_dists), dim=0)[None, :, None, None]
    channel_std = torch.sqrt(channel_var)
    if on_gpu:
        channel_std = channel_std.cuda()

    torch.cuda.empty_cache()

    maps_st_cpu, maps_ae_cpu = [], []
    for img in tqdm(loader, desc='Map normalization'):
        if on_gpu:
            img = img.cuda()
        with torch.no_grad():
            t_out = (teacher(img) - channel_mean) / channel_std
            s_out = student(img)
            a_out = autoencoder(img)
            m_st = torch.mean((t_out - s_out[:, :out_channels_]) ** 2, dim=1, keepdim=True)
            m_ae = torch.mean((a_out - s_out[:, out_channels_:]) ** 2, dim=1, keepdim=True)
            maps_st_cpu.append(m_st.cpu())
            maps_ae_cpu.append(m_ae.cpu())

    maps_st = torch.cat(maps_st_cpu)
    maps_ae = torch.cat(maps_ae_cpu)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)

    if on_gpu:
        q_st_start = q_st_start.cuda()
        q_st_end = q_st_end.cuda()
        q_ae_start = q_ae_start.cuda()
        q_ae_end = q_ae_end.cuda()

    if cache_path:
        save_data = {
            'teacher_mean': channel_mean.cpu().flatten().tolist(),
            'teacher_std': channel_std.cpu().flatten().tolist(),
            'q_st_start': q_st_start.cpu().item(),
            'q_st_end': q_st_end.cpu().item(),
            'q_ae_start': q_ae_start.cpu().item(),
            'q_ae_end': q_ae_end.cpu().item(),
        }
        with open(cache_path, 'w') as f:
            json.dump(save_data, f)

    return channel_mean, channel_std, q_st_start, q_st_end, q_ae_start, q_ae_end


def run_ad_inference(roi_img, teacher, student, autoencoder,
                     teacher_mean, teacher_std,
                     q_st_start, q_st_end, q_ae_start, q_ae_end):
    from PIL import Image
    roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    tensor = default_transform(Image.fromarray(roi_rgb)).unsqueeze(0)
    if on_gpu:
        tensor = tensor.cuda()

    with torch.no_grad():
        teacher_output = teacher(tensor)
        teacher_output = (teacher_output - teacher_mean) / teacher_std
        student_output = student(tensor)
        ae_output = autoencoder(tensor)

        map_st = torch.mean((teacher_output - student_output[:, :DEFAULT_OUT_CHANNELS]) ** 2,
                            dim=1, keepdim=True)
        map_ae = torch.mean((ae_output - student_output[:, DEFAULT_OUT_CHANNELS:]) ** 2,
                            dim=1, keepdim=True)
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
        map_combined = 0.5 * map_st + 0.5 * map_ae

    score = float(map_combined.max().cpu())
    return score


# ─── Visualization ──────────────────────────────────────────────────────

def draw_result(image, corners, score, threshold=None, box_color=None, text_scale=0.9):
    vis = image.copy()

    if threshold is not None:
        is_anomaly = score > threshold
        label = f"{'ANOMALY' if is_anomaly else 'NORMAL'}"
        if box_color is None:
            box_color = (0, 0, 255) if is_anomaly else (0, 255, 0)
    else:
        state = "HIGH" if score > 0.02 else "LOW"
        label = f"SCORE: {score:.4f} ({state})"
        if box_color is None:
            box_color = (0, 0, 255) if score > 0.02 else (0, 255, 0)

    if threshold is not None:
        score_text = f"{label}  |  score={score:.6f}"
    else:
        score_text = label

    pts = corners.reshape(-1, 2)
    cv2.polylines(vis, [pts], isClosed=True, color=box_color, thickness=2)

    text_x, text_y = int(pts[0][0]), int(pts[0][1]) - 10
    text_y = max(text_y, 20)
    (tw, th), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2)
    cv2.rectangle(vis, (text_x, text_y - th - 4), (text_x + tw, text_y + 4),
                  box_color, -1)
    cv2.putText(vis, score_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 2)

    return vis


# ─── Pipeline ───────────────────────────────────────────────────────────

def run_pipeline(image_path, template_name, teacher, student, autoencoder,
                 norm_params, output_path=None, threshold=None):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Step 1: ORB matching → find & crop ROI
    result = find_roi(image, template_name)
    if result is None:
        print(f"[FAIL] {image_path}: ROI not matched")
        return None

    roi_img = result['warped']
    corners = result['corners']

    # Step 2: EfficientAD inference on ROI
    teacher_mean, teacher_std, q_st_start, q_st_end, q_ae_start, q_ae_end = norm_params
    score = run_ad_inference(roi_img, teacher, student, autoencoder,
                             teacher_mean, teacher_std,
                             q_st_start, q_st_end, q_ae_start, q_ae_end)

    # Step 3: Draw
    annotated = draw_result(image, corners, score, threshold=threshold)

    if output_path:
        cv2.imwrite(output_path, annotated)

    return {'score': score, 'annotated': annotated, 'roi_corners': corners}


# ─── CLI ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='pipeline',
                                     description='ROI matching + EfficientAD inspection')
    parser.add_argument('--image', default=None, help='Input image to inspect (single mode)')
    parser.add_argument('--input-dir', default=None, help='Input directory (batch mode)')
    parser.add_argument('--output-dir', default=None, help='Output directory for annotated images (batch mode)')
    parser.add_argument('--template', required=True, help='Template name for ROI matching')
    parser.add_argument('--teacher', default=os.path.join(
        EFFICIENTAD_DIR, 'output/1/trainings/mvtec_ad/my_product/teacher_final.pth'))
    parser.add_argument('--student', default=os.path.join(
        EFFICIENTAD_DIR, 'output/1/trainings/mvtec_ad/my_product/student_final.pth'))
    parser.add_argument('--autoencoder', default=os.path.join(
        EFFICIENTAD_DIR, 'output/1/trainings/mvtec_ad/my_product/autoencoder_final.pth'))
    parser.add_argument('--train-dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'mydataset/my_product/train'))
    parser.add_argument('--norm-cache', default=os.path.join(
        EFFICIENTAD_DIR, 'output/1/trainings/mvtec_ad/my_product/norm_params.json'))
    parser.add_argument('--output', '-o', default=None, help='Output image path (single mode)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Anomaly score threshold (over this = anomaly)')

    args = parser.parse_args()

    if not args.image and not args.input_dir:
        parser.error('Either --image or --input-dir is required')

    print("[1/3] Loading models...")
    teacher, student, autoencoder = load_models(args.teacher, args.student, args.autoencoder)

    print("[2/3] Computing normalization parameters...")
    norm_params = compute_norm_params(teacher, student, autoencoder,
                                      args.train_dir, args.norm_cache)

    if args.input_dir:
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        files = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(exts))
        if not files:
            print(f"No image files found in {args.input_dir}")
            sys.exit(1)
        output_dir = args.output_dir or os.path.join(args.input_dir, 'annotated')
        os.makedirs(output_dir, exist_ok=True)
        print(f"[3/3] Batch processing {len(files)} images → {output_dir}\n")
        ok = fail = 0
        for f in tqdm(files, desc='Processing'):
            img_path = os.path.join(args.input_dir, f)
            out_path = os.path.join(output_dir, os.path.splitext(f)[0] + '.png')
            result = run_pipeline(img_path, args.template,
                                  teacher, student, autoencoder, norm_params,
                                  output_path=out_path, threshold=args.threshold)
            if result is None:
                fail += 1
            else:
                ok += 1
        print(f"\nDone: {ok} ok, {fail} failed/skipped")
    else:
        print(f"[3/3] Running pipeline on {args.image}...")
        result = run_pipeline(args.image, args.template,
                              teacher, student, autoencoder, norm_params,
                              output_path=args.output, threshold=args.threshold)
        if result is None:
            print("Pipeline failed.")
            sys.exit(1)
        print(f"\n{'='*50}")
        label = ('ANOMALY' if (args.threshold and result['score'] > args.threshold)
                 else f"SCORE: {result['score']}")
        print(f"  Result: {label}")
        print(f"  Score : {result['score']:.6f}")
        if args.output:
            print(f"  Output: {args.output}")
        else:
            cv2.imshow("Inspection Result", result['annotated'])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(f"{'='*50}")
