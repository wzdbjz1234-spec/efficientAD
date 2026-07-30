import tempfile
import unittest
from pathlib import Path

import numpy as np

from fixed_roi_crop import crop_fixed_roi, select_multiple_masks
from roi_mask import (
    IMAGENET_MEAN_BGR,
    load_roi_config,
    save_roi_config,
    valid_mask_from_config,
)


class RoiMaskTests(unittest.TestCase):
    def test_config_round_trip(self):
        roi = (10, 20, 30, 40)
        masks = [(2, 3, 4, 5), (12, 13, 6, 7)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'roi.json'
            save_roi_config(roi, masks, path)
            self.assertEqual(load_roi_config(path), (roi, masks))

    def test_mask_is_relative_to_cropped_roi(self):
        image = np.zeros((8, 9, 3), dtype=np.uint8)
        image[:, :] = (1, 2, 3)
        cropped = crop_fixed_roi(image, (2, 1, 5, 6), [(1, 2, 2, 3)])

        self.assertEqual(cropped.shape, (6, 5, 3))
        self.assertTrue(np.all(cropped[2:5, 1:3] == IMAGENET_MEAN_BGR))
        self.assertTrue(np.all(cropped[0, 0] == (1, 2, 3)))

    def test_valid_mask_excludes_only_selected_area(self):
        valid = valid_mask_from_config(
            (100, 200, 6, 5),
            [(1, 1, 3, 2)],
        )
        self.assertEqual(int(valid.sum()), 24)
        self.assertTrue(np.all(valid[1:3, 1:4] == 0))
        self.assertEqual(valid[0, 0], 1)

    def test_multiple_masks_can_be_selected_until_cancelled(self):
        image = np.zeros((10, 12, 3), dtype=np.uint8)
        selected = iter([(1, 2, 3, 4), (7, 1, 2, 5), None])
        previews = []

        def selector(preview):
            previews.append(preview.copy())
            return next(selected)

        masks = select_multiple_masks(image, selector=selector)

        self.assertEqual(masks, [(1, 2, 3, 4), (7, 1, 2, 5)])
        self.assertEqual(len(previews), 3)
        self.assertTrue(
            np.all(previews[1][2:6, 1:4] == IMAGENET_MEAN_BGR))
        self.assertTrue(
            np.all(previews[2][1:6, 7:9] == IMAGENET_MEAN_BGR))


if __name__ == '__main__':
    unittest.main()
