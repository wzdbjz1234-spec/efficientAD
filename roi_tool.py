import argparse
import os
import cv2
import numpy as np

from roi_mask import apply_masks, load_roi_config, save_roi_config, select_mask

ORB_FEATURES = 5000
RATIO_THRESH = 0.75
MIN_GOOD_MATCHES = 10
RANSAC_THRESH = 5.0

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
os.makedirs(TEMPLATES_DIR, exist_ok=True)

_orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def create_template(template_name, image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: cannot read {image_path}")
        return

    window_name = f"Select ROI for '{template_name}' | ENTER=confirm C=cancel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(window_name, image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("Cancelled.")
        return

    cropped = image[int(y):int(y + h), int(x):int(x + w)]
    print("Select the area to exclude from training. Press C/ESC for no mask.")
    mask = select_mask(cropped)
    masks = [] if mask is None else [mask]

    tpl_dir = os.path.join(TEMPLATES_DIR, template_name)
    os.makedirs(tpl_dir, exist_ok=True)
    cv2.imwrite(os.path.join(tpl_dir, 'template.png'), image)
    save_roi_config(
        (x, y, w, h), masks, os.path.join(tpl_dir, 'roi.json'))
    print(
        f"Template '{template_name}' saved. ROI=({x},{y},{w},{h}), "
        f"training masks={len(masks)}")


def list_templates():
    names = []
    for name in sorted(os.listdir(TEMPLATES_DIR)):
        d = os.path.join(TEMPLATES_DIR, name)
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, 'template.png')) and os.path.isfile(os.path.join(d, 'roi.json')):
            names.append(name)
    return names


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

    inliers = mask.ravel().sum() if mask is not None else 0
    return H, inliers


def crop_roi(image_input, template_name):
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            print(f"Error: cannot read {image_input}")
            return None
    else:
        image = image_input

    tpl_dir = os.path.join(TEMPLATES_DIR, template_name)
    template_img = cv2.imread(os.path.join(tpl_dir, 'template.png'))
    if template_img is None:
        print(f"Error: template image not found for '{template_name}'")
        return None

    roi, masks = load_roi_config(os.path.join(tpl_dir, 'roi.json'))
    rx, ry, rw, rh = roi

    tpl_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    result = _orb_match(tpl_gray, img_gray)
    if result is None:
        print("Warning: failed to match features")
        return None

    H, inliers = result

    src_corners = np.float32([
        [rx, ry],
        [rx + rw, ry],
        [rx + rw, ry + rh],
        [rx, ry + rh]
    ]).reshape(-1, 1, 2)

    dst_corners = cv2.perspectiveTransform(src_corners, H)

    h_img, w_img = image.shape[:2]
    for pt in dst_corners.reshape(-1, 2):
        if pt[0] < 0 or pt[0] >= w_img or pt[1] < 0 or pt[1] >= h_img:
            print("Warning: mapped ROI is out of image bounds")
            return None

    out_w, out_h = int(rw), int(rh)
    dst_rect = np.float32([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ]).reshape(-1, 1, 2)

    M = cv2.getPerspectiveTransform(dst_corners, dst_rect)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))

    return apply_masks(warped, masks)


def batch_crop(input_dir, template_name, output_dir):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]
    if not files:
        print("No images found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    ok = skip = 0
    for f in files:
        cropped = crop_roi(os.path.join(input_dir, f), template_name)
        if cropped is None:
            print(f"[SKIP] {f}")
            skip += 1
        else:
            out = os.path.join(output_dir, os.path.splitext(f)[0] + '.png')
            cv2.imwrite(out, cropped)
            ok += 1
    print(f"Done: {ok} ok, {skip} skipped")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='roi_tool')
    sub = parser.add_subparsers(dest='command', required=True)

    p_create = sub.add_parser('create', help='Create a new template with ROI')
    p_create.add_argument('template_name')
    p_create.add_argument('image_path')

    sub.add_parser('list', help='List all saved templates')

    p_crop = sub.add_parser('crop', help='Crop ROI from a single image')
    p_crop.add_argument('template_name')
    p_crop.add_argument('image_path')
    p_crop.add_argument('--output', '-o', default=None, help='Output path')

    p_batch = sub.add_parser('batch', help='Batch crop ROI from a directory')
    p_batch.add_argument('template_name')
    p_batch.add_argument('input_dir')
    p_batch.add_argument('output_dir')

    args = parser.parse_args()

    if args.command == 'create':
        create_template(args.template_name, args.image_path)

    elif args.command == 'list':
        names = list_templates()
        if names:
            print('Templates:')
            for n in names:
                print(f'  {n}')
        else:
            print('No templates.')

    elif args.command == 'crop':
        cropped = crop_roi(args.image_path, args.template_name)
        if cropped is not None:
            out = args.output or os.path.splitext(args.image_path)[0] + '_cropped.png'
            cv2.imwrite(out, cropped)
            print(f'Saved: {out}')
        else:
            print('ROI not matched.')

    elif args.command == 'batch':
        batch_crop(args.input_dir, args.template_name, args.output_dir)
