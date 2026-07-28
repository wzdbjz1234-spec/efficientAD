"""Shared ROI-mask helpers.

Mask coordinates are stored relative to the cropped ROI, not the source image.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


# OpenCV uses BGR. These values become approximately zero after ImageNet
# normalization, so masked pixels cannot leak source-image content.
IMAGENET_MEAN_BGR = (104, 116, 124)


def validate_rect(rect, image_width, image_height, name):
    x, y, width, height = map(int, rect)
    if width <= 0 or height <= 0:
        raise ValueError(f'{name} width and height must be positive')
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError(
            f'{name} ({x},{y},{width},{height}) is outside '
            f'{image_width}x{image_height}')
    return x, y, width, height


def load_roi_config(path):
    with Path(path).open(encoding='utf-8') as handle:
        data = json.load(handle)

    roi = tuple(map(int, data['roi']))
    raw_masks = data.get('masks')
    if raw_masks is None:
        raw_mask = data.get('mask')
        raw_masks = [] if raw_mask is None else [raw_mask]
    masks = [tuple(map(int, mask)) for mask in raw_masks]
    return roi, masks


def save_roi_config(roi, masks, path):
    payload = {
        'roi': list(map(int, roi)),
        'masks': [list(map(int, mask)) for mask in masks],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def select_mask(cropped_image):
    window_name = (
        'Select training mask inside ROI | ENTER/SPACE: confirm | '
        'C/ESC: no mask'
    )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    rect = cv2.selectROI(
        window_name, cropped_image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    rect = tuple(map(int, rect))
    if rect[2] <= 0 or rect[3] <= 0:
        return None
    return rect


def apply_masks(image, masks, fill=IMAGENET_MEAN_BGR):
    """Return a copy with every masked rectangle replaced by a neutral value."""
    result = image.copy()
    image_height, image_width = result.shape[:2]
    for mask in masks:
        x, y, width, height = validate_rect(
            mask, image_width, image_height, 'Mask')
        if result.ndim == 2:
            result[y:y + height, x:x + width] = int(round(sum(fill) / len(fill)))
        else:
            channels = result.shape[2]
            value = fill[:channels]
            if len(value) < channels:
                value = value + (255,) * (channels - len(value))
            result[y:y + height, x:x + width] = value
    return result


def valid_mask_from_config(roi, masks):
    """Build a uint8 mask where 1 participates in training and 0 is ignored."""
    _, _, roi_width, roi_height = roi
    valid = np.ones((roi_height, roi_width), dtype=np.uint8)
    for mask in masks:
        x, y, width, height = validate_rect(
            mask, roi_width, roi_height, 'Mask')
        valid[y:y + height, x:x + width] = 0
    return valid
