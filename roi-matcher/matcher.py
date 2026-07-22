import json
import os
import sys

import cv2
import numpy as np


TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

MATCH_METHOD = cv2.TM_CCOEFF_NORMED
MATCH_THRESHOLD = 0.8


def _template_dir(template_name):
    return os.path.join(TEMPLATES_DIR, template_name)


def _template_image_path(template_name):
    return os.path.join(_template_dir(template_name), 'template.png')


def _roi_json_path(template_name):
    return os.path.join(_template_dir(template_name), 'roi.json')


def _load_template(template_name):
    template_dir = _template_dir(template_name)
    if not os.path.isdir(template_dir):
        raise FileNotFoundError(
            f"Template '{template_name}' not found at {template_dir}")

    roi_path = _roi_json_path(template_name)
    if not os.path.isfile(roi_path):
        raise FileNotFoundError(
            f"ROI config not found for template '{template_name}'")

    template_img = cv2.imread(_template_image_path(template_name))
    if template_img is None:
        raise FileNotFoundError(
            f"Template image not found for '{template_name}'")

    with open(roi_path, 'r') as f:
        roi_data = json.load(f)

    x, y, w, h = roi_data['roi']
    return template_img, (x, y, w, h)


def list_templates():
    if not os.path.isdir(TEMPLATES_DIR):
        return []
    templates = []
    for name in os.listdir(TEMPLATES_DIR):
        template_dir = os.path.join(TEMPLATES_DIR, name)
        if os.path.isdir(template_dir):
            roi_path = os.path.join(template_dir, 'roi.json')
            img_path = os.path.join(template_dir, 'template.png')
            if os.path.isfile(roi_path) and os.path.isfile(img_path):
                templates.append(name)
    return sorted(templates)


def create_template(template_name, image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    window_name = f"Select ROI for '{template_name}' - ENTER to confirm, C to cancel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(window_name, image, showCrosshair=True, fromCenter=False)

    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("ROI selection cancelled (empty region).")
        return None

    template_dir = _template_dir(template_name)
    os.makedirs(template_dir, exist_ok=True)

    cv2.imwrite(_template_image_path(template_name), image)

    roi_data = {'roi': [int(x), int(y), int(w), int(h)]}
    with open(_roi_json_path(template_name), 'w') as f:
        json.dump(roi_data, f, indent=2)

    print(f"Template '{template_name}' saved: ROI=({x},{y},{w},{h})")
    return roi_data


def crop_roi(image_path, template_name):
    if isinstance(image_path, np.ndarray):
        input_img = image_path
    else:
        input_img = cv2.imread(image_path)
        if input_img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

    template_img, (rx, ry, rw, rh) = _load_template(template_name)

    result = cv2.matchTemplate(input_img, template_img, MATCH_METHOD)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < MATCH_THRESHOLD:
        return None

    tx, ty = max_loc
    crop_x = int(tx + rx)
    crop_y = int(ty + ry)
    crop_w = int(rw)
    crop_h = int(rh)

    h, w = input_img.shape[:2]
    crop_x = max(0, crop_x)
    crop_y = max(0, crop_y)
    crop_x = min(crop_x, w - 1)
    crop_y = min(crop_y, h - 1)
    crop_w = min(crop_w, w - crop_x)
    crop_h = min(crop_h, h - crop_y)

    if crop_w <= 0 or crop_h <= 0:
        return None

    cropped = input_img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    return cropped


def crop_roi_batch(input_dir, template_name, output_dir):
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(valid_exts)
    ]

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    skipped = 0
    processed = 0
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')

        cropped = crop_roi(input_path, template_name)
        if cropped is None:
            print(f"[SKIP] {filename}: ROI not matched (confidence < {MATCH_THRESHOLD})")
            skipped += 1
            continue

        cv2.imwrite(output_path, cropped)
        processed += 1

    print(f"Done: {processed} processed, {skipped} skipped")
