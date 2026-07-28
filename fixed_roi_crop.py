import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from roi_mask import (
    apply_masks,
    load_roi_config,
    save_roi_config,
    select_mask,
    validate_rect,
)


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

ROI_FILE = Path(__file__).resolve().parent / 'roi_config.json'


def read_image(path):
    """Read an image while supporting non-ASCII Windows paths."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def write_image(path, image):
    """Write an image while supporting non-ASCII Windows paths."""
    suffix = path.suffix.lower()
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        return False
    encoded.tofile(path)
    return True


def select_fixed_roi_and_masks(reference_path):
    image = read_image(reference_path)
    if image is None:
        raise ValueError(f'Cannot read reference image: {reference_path}')

    window_name = 'Select fixed ROI | ENTER/SPACE: confirm | C/ESC: cancel'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    x, y, width, height = cv2.selectROI(
        window_name, image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    roi = tuple(map(int, (x, y, width, height)))
    if width <= 0 or height <= 0:
        return None, []

    cropped = crop_fixed_roi(image, roi)
    print('Select the area to exclude from training. Press C/ESC for no mask.')
    mask = select_mask(cropped)
    return roi, [] if mask is None else [mask]


def crop_fixed_roi(image, roi, masks=None):
    x, y, width, height = roi
    image_height, image_width = image.shape[:2]
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        return None
    cropped = image[y:y + height, x:x + width].copy()
    return apply_masks(cropped, masks or [])


def find_images(input_dir, output_dir, recursive):
    iterator = input_dir.rglob('*') if recursive else input_dir.iterdir()
    output_dir = output_dir.resolve()
    images = []

    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            path.resolve().relative_to(output_dir)
        except ValueError:
            images.append(path)

    return sorted(images)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Select fixed ROI and apply the same crop to every image.')
    parser.add_argument('--reference', '-r', required=True,
                        help='Image displayed for selecting the fixed ROI')
    parser.add_argument('--input-dir', '-i', required=True,
                        help='Directory containing source images')
    parser.add_argument('--output-dir', '-o', required=True,
                        help='Directory where cropped images are saved')
    parser.add_argument('--recursive', action='store_true',
                        help='Also process images in subdirectories')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite files that already exist in output-dir')
    parser.add_argument('--roi', default=None,
                        help='Fixed ROI as x,y,w,h (skips GUI selection)')
    parser.add_argument('--mask', action='append', default=None,
                        help='Training mask inside cropped ROI as x,y,w,h; may be repeated')
    parser.add_argument('--save-roi', default=None,
                        help='Save selected ROI to a JSON file after GUI selection')
    parser.add_argument('--load-roi', default=None,
                        help='Load ROI from a JSON file (skips GUI)')
    return parser.parse_args()


def parse_roi_string(s):
    parts = s.split(',')
    if len(parts) != 4:
        raise ValueError('ROI must be x,y,w,h (4 comma-separated integers)')
    return tuple(map(int, parts))


def resolve_roi_and_masks(reference_path, args):
    if args.load_roi:
        roi, saved_masks = load_roi_config(args.load_roi)
        masks = ([parse_roi_string(value) for value in args.mask]
                 if args.mask is not None else saved_masks)
        print(f'Loaded ROI from file: {args.load_roi}')
        print(f'ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}')
        return roi, masks

    if args.roi:
        roi = parse_roi_string(args.roi)
        masks = [parse_roi_string(value) for value in (args.mask or [])]
        print(f'Using ROI from command line: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}')
        return roi, masks

    roi, masks = select_fixed_roi_and_masks(reference_path)
    if roi is None:
        return None, []

    if args.save_roi:
        save_roi_config(roi, masks, args.save_roi)
        print(f'ROI and mask saved to: {args.save_roi}')
    elif args.load_roi is None:
        save_roi_config(roi, masks, ROI_FILE)
        print(f'ROI and mask saved to: {ROI_FILE}')
    return roi, masks


def main():
    args = parse_args()
    reference_path = Path(args.reference).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not reference_path.is_file():
        print(f'[ERROR] Reference image not found: {reference_path}')
        return 1
    if not input_dir.is_dir():
        print(f'[ERROR] Input directory not found: {input_dir}')
        return 1
    if input_dir == output_dir:
        print('[ERROR] Input and output directories must be different.')
        return 1

    print(f'Reference: {reference_path}')
    if not args.roi and not args.load_roi:
        print('Drag a rectangle, then press ENTER or SPACE to confirm.')
    roi, masks = resolve_roi_and_masks(reference_path, args)
    if roi is None:
        print('ROI selection cancelled.')
        return 1

    x, y, width, height = roi
    validate_rect(roi, *read_image(reference_path).shape[1::-1], 'ROI')
    for mask in masks:
        validate_rect(mask, width, height, 'Mask')
    print(f'Selected ROI: x={x}, y={y}, width={width}, height={height}')
    if masks:
        for index, mask in enumerate(masks, start=1):
            print(f'Training mask {index}: x={mask[0]}, y={mask[1]}, '
                  f'width={mask[2]}, height={mask[3]}')
    else:
        print('Training mask: none')

    image_paths = find_images(input_dir, output_dir, args.recursive)
    if not image_paths:
        print(f'[ERROR] No supported images found in: {input_dir}')
        return 1

    saved = skipped = failed = 0
    for source_path in image_paths:
        relative_path = source_path.relative_to(input_dir)
        destination_path = output_dir / relative_path

        if destination_path.exists() and not args.overwrite:
            print(f'[SKIP] Output already exists: {destination_path}')
            skipped += 1
            continue

        image = read_image(source_path)
        if image is None:
            print(f'[FAIL] Cannot read: {source_path}')
            failed += 1
            continue

        cropped = crop_fixed_roi(image, roi, masks)
        if cropped is None:
            image_height, image_width = image.shape[:2]
            print(f'[FAIL] Image too small ({image_width}x{image_height}): {source_path}')
            failed += 1
            continue

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if not write_image(destination_path, cropped):
            print(f'[FAIL] Cannot write: {destination_path}')
            failed += 1
            continue

        print(f'[OK] {source_path.name} -> {destination_path}')
        saved += 1

    print(f'Finished: {saved} saved, {skipped} skipped, {failed} failed')
    print(f'Output: {output_dir}')
    return 0 if saved > 0 or skipped > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
