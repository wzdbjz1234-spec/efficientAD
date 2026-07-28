import os, sys, time, cv2, json, numpy as np, torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EfficientAD-main'))
from common import get_pdn_small, get_pdn_medium, get_pdn_tiny, get_autoencoder, get_autoencoder_tiny

DEFAULT_IMAGE_SIZE = 256
transform = transforms.Compose([
    transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

LOCAL_TEACHER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'EfficientAD-main', 'models', 'teacher_small.pth')
IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mydataset', 'my_product', 'train', 'good')

OUT_CHANNELS = 384
DEVICE = torch.device('cpu')

configs = {
    'tiny': {
        'student': get_pdn_tiny,    # internal channels halved
        'ae': get_autoencoder_tiny, # internal channels halved
    },
    'small': {
        'student': get_pdn_small,
        'ae': get_autoencoder,
    },
    'medium': {
        'student': get_pdn_medium,  # student bigger, still using same teacher
        'ae': get_autoencoder,
    },
}

images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])[:30]


def bench_model(label, cfg):
    student_fn, ae_fn = cfg['student'], cfg['ae']

    teacher = get_pdn_small(OUT_CHANNELS)
    teacher.load_state_dict(torch.load(LOCAL_TEACHER, map_location='cpu', weights_only=False))
    teacher.eval()

    student = student_fn(2 * OUT_CHANNELS)
    student.eval()

    ae = ae_fn(OUT_CHANNELS)
    ae.eval()

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    ae_params = sum(p.numel() for p in ae.parameters())
    total = teacher_params + student_params + ae_params

    @torch.no_grad()
    def infer(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = transform(Image.fromarray(rgb)).unsqueeze(0)
        to = teacher(t)
        so = student(t)
        ao = ae(t)
        ms = torch.mean((to - so[:, :OUT_CHANNELS]) ** 2, dim=1, keepdim=True)
        ma = torch.mean((ao - so[:, OUT_CHANNELS:]) ** 2, dim=1, keepdim=True)
        return float((0.5 * ms + 0.5 * ma).max())

    # warmup
    for fname in images[:3]:
        img = cv2.imread(os.path.join(IMG_DIR, fname))
        if img is not None:
            _ = infer(img)

    times = []
    for fname in images:
        img = cv2.imread(os.path.join(IMG_DIR, fname))
        if img is None:
            continue
        t0 = time.perf_counter()
        _ = infer(img)
        times.append(time.perf_counter() - t0)

    arr = np.array(times)
    return {
        'label': label,
        'teacher': teacher_params,
        'student': student_params,
        'ae': ae_params,
        'total': total,
        'mean': arr.mean(),
        'median': np.median(arr),
        'std': arr.std(),
        'min': arr.min(),
        'max': arr.max(),
        'fps': 1.0 / arr.mean(),
        'n': len(times),
    }


if __name__ == '__main__':
    print("=" * 75)
    print(f"Model Size Comparison (CPU, {len(images)} images)")
    print("=" * 75)
    print(f"{'Variant':<8} {'Teacher':>12} {'Student':>12} {'AE':>10} {'Total':>12} {'Mean':>10} {'FPS':>8} {'vs small':>8}")
    print("-" * 75)

    results = []
    for label, cfg in configs.items():
        r = bench_model(label, cfg)
        results.append(r)

    base_fps = results[1]['fps']  # small as baseline
    for r in results:
        ratio = r['fps'] / base_fps
        print(f"{r['label']:<8} {r['teacher']:>10,} {r['student']:>10,} {r['ae']:>9,} "
              f"{r['total']:>10,} {r['mean']:>8.4f}s {r['fps']:>7.1f} {ratio:>7.2f}x")

    print("-" * 75)
    print(f"  tiny = student internal channels halved (128→64, 256→128)")
    print(f"  Teacher unchanged (reuses pre-trained models/teacher_small.pth)")
    print(f"{'=' * 75}")
