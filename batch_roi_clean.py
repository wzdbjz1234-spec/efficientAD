import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roi_tool import crop_roi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CROPPED_DIR = os.path.join(BASE_DIR, 'cropped')

SOURCE_DIRS = {
    '7-7': os.path.join(DATA_DIR, '7-7', '7-7'),
    '7-14': os.path.join(DATA_DIR, '7-14', '7-14'),
}

TEMPLATE_NAME = 'my_product'

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CROPPED_DIR, exist_ok=True)


def extract_prefix_number(filename):
    m = re.search(r'([Kk]\d)_(\d+)', filename)
    if m:
        return m.group(1).upper(), m.group(2)
    return None, None


def main():
    import cv2

    total_ok = 0
    total_fail = 0

    for date_label, src_dir in SOURCE_DIRS.items():
        if not os.path.isdir(src_dir):
            print(f"[SKIP] Directory not found: {src_dir}")
            continue

        files = sorted([f for f in os.listdir(src_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        print(f"\n{'='*60}")
        print(f"Processing {date_label}: {len(files)} images")
        print(f"{'='*60}")

        ok = fail = 0
        for f in files:
            src_path = os.path.join(src_dir, f)
            prefix, num = extract_prefix_number(f)
            if num is None:
                print(f"[SKIP] Cannot extract number from: {f}")
                fail += 1
                continue

            cropped = crop_roi(src_path, TEMPLATE_NAME)
            if cropped is None:
                print(f"[FAIL] {f}")
                fail += 1
                continue

            out_name = f"{date_label}_{prefix}_{num}.png"
            cv2.imwrite(os.path.join(DATA_DIR, out_name), cropped)
            cv2.imwrite(os.path.join(CROPPED_DIR, out_name), cropped)
            ok += 1

        print(f"\n{date_label}: {ok} ok, {fail} failed")
        total_ok += ok
        total_fail += fail

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_ok} ok, {total_fail} failed")
    print(f"Output to: {DATA_DIR}")
    print(f"Output to: {CROPPED_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
