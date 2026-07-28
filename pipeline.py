import argparse
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

from efficientad_tools import EfficientADPredictor, ModelArtifacts
from roi_mask import apply_masks, load_roi_config

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
EFFICIENTAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EfficientAD-main')

ORB_FEATURES = 5000
RATIO_THRESH = 0.75
MIN_GOOD_MATCHES = 10
RANSAC_THRESH = 5.0

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
    roi, masks = load_roi_config(os.path.join(tpl_dir, 'roi.json'))
    rx, ry, rw, rh = roi

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
    warped = apply_masks(warped, masks)

    return {'warped': warped, 'corners': dst_corners}


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

def run_pipeline(image_path, template_name, predictor,
                 output_path=None, threshold=None):
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
    score = predictor.predict(roi_img).score

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
    parser.add_argument('--model', default='1', help='Model dir under output/ (1 or 2 etc.)')
    parser.add_argument('--dataset', default='mvtec_ad')
    parser.add_argument('--product', default='my_product')
    parser.add_argument('--source-dir', default=EFFICIENTAD_DIR)
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--teacher', default=None)
    parser.add_argument('--student', default=None)
    parser.add_argument('--autoencoder', default=None)
    parser.add_argument('--norm-cache', default=None)
    parser.add_argument('--train-dir', default=None,
                        help='Training images used only if norm_params.json is missing')
    parser.add_argument('--output', '-o', default=None, help='Output image path (single mode)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Anomaly score threshold (over this = anomaly)')

    args = parser.parse_args()

    if not args.image and not args.input_dir:
        parser.error('Either --image or --input-dir is required')

    print("[1/2] Loading model output...")
    artifacts = ModelArtifacts.from_output(
        args.model,
        dataset=args.dataset,
        product=args.product,
        source_dir=args.source_dir,
    ).with_overrides(
        teacher=args.teacher,
        student=args.student,
        autoencoder=args.autoencoder,
        norm_params=args.norm_cache,
    )
    predictor = EfficientADPredictor.load(
        artifacts,
        device=args.device,
        train_dir=args.train_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'mydataset',
            args.product,
            'train',
        ),
    )

    if args.input_dir:
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        files = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(exts))
        if not files:
            print(f"No image files found in {args.input_dir}")
            sys.exit(1)
        output_dir = args.output_dir or os.path.join(args.input_dir, 'annotated')
        os.makedirs(output_dir, exist_ok=True)
        print(f"[2/2] Batch processing {len(files)} images → {output_dir}\n")
        ok = fail = 0
        for f in tqdm(files, desc='Processing'):
            img_path = os.path.join(args.input_dir, f)
            out_path = os.path.join(output_dir, os.path.splitext(f)[0] + '.png')
            result = run_pipeline(img_path, args.template, predictor,
                                  output_path=out_path, threshold=args.threshold)
            if result is None:
                fail += 1
            else:
                ok += 1
        print(f"\nDone: {ok} ok, {fail} failed/skipped")
    else:
        print(f"[2/2] Running pipeline on {args.image}...")
        result = run_pipeline(args.image, args.template, predictor,
                              output_path=args.output, threshold=args.threshold)
        if result is None:
            print("Pipeline failed.")
            sys.exit(1)
        print(f"\n{'='*50}")
        label = ('ANOMALY' if (args.threshold is not None
                               and result['score'] >= args.threshold)
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
