import argparse
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

EFFICIENTAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EfficientAD-main')
sys.path.insert(0, EFFICIENTAD_DIR)
from common import get_autoencoder, get_pdn_small, ImageFolderWithoutTarget

DEFAULT_IMAGE_SIZE = 256
DEFAULT_OUT_CHANNELS = 384
on_gpu = torch.cuda.is_available()

default_transform = transforms.Compose([
    transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

OVERLAY_ALPHA = 0.4


def load_models(teacher_path, student_path, ae_path):
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
    student = torch.load(student_path, map_location='cpu', weights_only=False)
    autoencoder = torch.load(ae_path, map_location='cpu', weights_only=False)
    teacher.eval(); student.eval(); autoencoder.eval()
    if on_gpu:
        teacher.cuda(); student.cuda(); autoencoder.cuda()
    return teacher, student, autoencoder


def compute_norm_params(teacher, student, autoencoder, train_dir, cache_path=None):
    if cache_path and os.path.isfile(cache_path):
        import json
        with open(cache_path) as f:
            data = json.load(f)
        def _t(arr):
            t = torch.tensor(arr)
            return t.cuda() if on_gpu else t
        tm = _t(data['teacher_mean']).view(1, -1, 1, 1)
        ts = _t(data['teacher_std']).view(1, -1, 1, 1)
        return tm, ts, _t(data['q_st_start']), _t(data['q_st_end']), _t(data['q_ae_start']), _t(data['q_ae_end'])

    import json
    dataset = ImageFolderWithoutTarget(train_dir, transform=transforms.Lambda(lambda x: default_transform(x)))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    oc = DEFAULT_OUT_CHANNELS

    means = []; dists = []
    for img in tqdm(loader, desc='Teacher norm'):
        if on_gpu: img = img.cuda()
        with torch.no_grad():
            out = teacher(img)
            means.append(torch.mean(out, dim=[0,2,3]).cpu())
    cm = torch.mean(torch.stack(means), dim=0)[None,:,None,None]
    if on_gpu: cm = cm.cuda()
    for img in tqdm(loader, desc='Teacher std'):
        if on_gpu: img = img.cuda()
        with torch.no_grad():
            d = (teacher(img) - cm) ** 2
            dists.append(torch.mean(d, dim=[0,2,3]).cpu())
    cv = torch.mean(torch.stack(dists), dim=0)[None,:,None,None]
    cs = torch.sqrt(cv)
    if on_gpu: cs = cs.cuda()

    torch.cuda.empty_cache()
    ms = []; ma = []
    for img in tqdm(loader, desc='Map norm'):
        if on_gpu: img = img.cuda()
        with torch.no_grad():
            to = (teacher(img) - cm) / cs
            so = student(img); ao = autoencoder(img)
            ms.append(torch.mean((to - so[:,:oc])**2, dim=1, keepdim=True).cpu())
            ma.append(torch.mean((ao - so[:,oc:])**2, dim=1, keepdim=True).cpu())
    mst = torch.cat(ms); mae = torch.cat(ma)
    qss = torch.quantile(mst, q=0.9); qse = torch.quantile(mst, q=0.995)
    qas = torch.quantile(mae, q=0.9); qae = torch.quantile(mae, q=0.995)
    if on_gpu:
        qss = qss.cuda(); qse = qse.cuda(); qas = qas.cuda(); qae = qae.cuda()

    if cache_path:
        with open(cache_path, 'w') as f:
            json.dump({'teacher_mean': cm.cpu().flatten().tolist(),
                       'teacher_std': cs.cpu().flatten().tolist(),
                       'q_st_start': qss.cpu().item(), 'q_st_end': qse.cpu().item(),
                       'q_ae_start': qas.cpu().item(), 'q_ae_end': qae.cpu().item()}, f)
    return cm, cs, qss, qse, qas, qae


@torch.no_grad()
def infer_one(roi_img, teacher, student, autoencoder, cm, cs, qss, qse, qas, qae):
    roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = roi_img.shape[:2]
    tensor = default_transform(Image.fromarray(roi_rgb)).unsqueeze(0)
    if on_gpu: tensor = tensor.cuda()

    to = (teacher(tensor) - cm) / cs
    so = student(tensor)
    ao = autoencoder(tensor)
    ms = torch.mean((to - so[:, :DEFAULT_OUT_CHANNELS])**2, dim=1, keepdim=True)
    ma = torch.mean((ao - so[:, DEFAULT_OUT_CHANNELS:])**2, dim=1, keepdim=True)
    ms = 0.1 * (ms - qss) / (qse - qss)
    ma = 0.1 * (ma - qas) / (qae - qas)
    mc = 0.5 * ms + 0.5 * ma

    mc = torch.nn.functional.pad(mc, (4,4,4,4))
    mc = torch.nn.functional.interpolate(mc, (h_orig, w_orig), mode='bilinear')
    score = float(mc.max().cpu())
    amap = mc[0,0].cpu().numpy()
    return score, amap


def generate_overlay(roi_img, amap):
    amap_norm = np.clip(amap / (amap.max() + 1e-8), 0, 1)
    heatmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(roi_img, 1 - OVERLAY_ALPHA, heatmap, OVERLAY_ALPHA, 0)
    return overlay


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='batch_infer')
    parser.add_argument('--input-dir', required=True, help='Directory with cropped ROI images')
    parser.add_argument('--output-dir', default=None, help='Output dir for anomaly maps (optional)')
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
    parser.add_argument('--threshold', type=float, default=None)

    args = parser.parse_args()

    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    files = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(exts))
    if not files:
        print("No images found.")
        sys.exit(0)

    print("[1/2] Loading models...")
    teacher, student, autoencoder = load_models(args.teacher, args.student, args.autoencoder)
    print("[2/2] Loading normalization params...")
    cm, cs, qss, qse, qas, qae = compute_norm_params(teacher, student, autoencoder,
                                                      args.train_dir, args.norm_cache)

    out_dir = args.output_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    scores = []
    print(f"\n{'filename':<45} {'score':>12}  {'judgment'}")
    print("-" * 75)

    for f in tqdm(files, desc='Inferring'):
        img = cv2.imread(os.path.join(args.input_dir, f))
        if img is None:
            print(f"[SKIP] Cannot read {f}")
            continue
        score, amap = infer_one(img, teacher, student, autoencoder, cm, cs, qss, qse, qas, qae)
        scores.append((f, score))

        judgment = ""
        if args.threshold is not None:
            judgment = "ANOMALY" if score > args.threshold else "NORMAL"
        print(f"{f:<45} {score:>12.6f}  {judgment}")

        if out_dir:
            overlay = generate_overlay(img, amap)
            out_path = os.path.join(out_dir, os.path.splitext(f)[0] + '.png')
            cv2.imwrite(out_path, overlay)

    if scores:
        vals = [s[1] for s in scores]
        print(f"\n{'='*50}")
        print(f"  Total   : {len(vals)}")
        print(f"  Min     : {min(vals):.6f}")
        print(f"  Max     : {max(vals):.6f}")
        print(f"  Mean    : {np.mean(vals):.6f}")
        print(f"  Median  : {np.median(vals):.6f}")
        print(f"  Std     : {np.std(vals):.6f}")
        if args.threshold is not None:
            n_anomaly = sum(1 for v in vals if v > args.threshold)
            print(f"  Anomaly : {n_anomaly}/{len(vals)}")
        print(f"{'='*50}")
