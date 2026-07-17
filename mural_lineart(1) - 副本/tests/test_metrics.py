import unittest

import numpy as np

from losses.metrics import (
    cldice,
    edge_f1,
    evaluate_lineart,
    hausdorff95,
    to_line_mask,
    valid_hallucination_rate,
)


def line_image(size=32, column=16):
    image = np.full((size, size), 255, dtype=np.uint8)
    image[5:27, column] = 0
    return image


class LineMetricTests(unittest.TestCase):
    def test_dark_pixels_are_foreground_in_supported_ranges(self):
        uint = np.array([[0, 255]], dtype=np.uint8)
        expected = np.array([[True, False]])
        np.testing.assert_array_equal(to_line_mask(uint), expected)
        np.testing.assert_array_equal(to_line_mask(uint / 255.0), expected)
        np.testing.assert_array_equal(to_line_mask(uint / 127.5 - 1.0), expected)

    def test_identical_prediction_is_perfect(self):
        image = line_image()
        self.assertEqual(edge_f1(image, image), (1.0, 1.0, 1.0))
        self.assertEqual(cldice(image, image), 1.0)
        self.assertEqual(hausdorff95(image, image), 0.0)

    def test_one_pixel_shift_is_within_f1_tolerance_and_has_distance(self):
        gt = line_image(column=16)
        pred = line_image(column=17)
        self.assertEqual(edge_f1(pred, gt, tolerance=0)[2], 0.0)
        self.assertEqual(edge_f1(pred, gt, tolerance=1)[2], 1.0)
        self.assertEqual(hausdorff95(pred, gt), 1.0)

    def test_missing_line_has_finite_diagonal_hd95_penalty(self):
        gt = line_image()
        empty = np.full_like(gt, 255)
        self.assertEqual(hausdorff95(empty, gt), np.hypot(*gt.shape))

    def test_report_contains_six_ablation_metrics(self):
        gt = line_image()
        pred = gt.copy()
        pred[10, 3] = 0
        damage = np.zeros_like(gt, dtype=bool)
        damage[:, 12:20] = True
        report = evaluate_lineart(pred, gt, damage)
        self.assertEqual(
            set(report),
            {
                "hole_precision",
                "hole_recall",
                "hole_f1",
                "hole_cldice",
                "hole_hd95",
                "valid_hallucination_rate",
            },
        )
        self.assertEqual(report["hole_precision"], 1.0)
        self.assertEqual(report["hole_recall"], 1.0)
        self.assertEqual(report["hole_f1"], 1.0)
        self.assertGreater(report["valid_hallucination_rate"], 0.0)

    def test_hallucination_uses_length_and_tolerance(self):
        gt = line_image(column=16)
        damage = np.zeros_like(gt, dtype=bool)

        shifted = line_image(column=17)
        self.assertEqual(
            valid_hallucination_rate(shifted, gt, damage, tolerance=1), 0.0
        )

        hallucinated = gt.copy()
        hallucinated[8:24, 3] = 0
        rate = valid_hallucination_rate(hallucinated, gt, damage, tolerance=1)
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 1.0)

    def test_cldice_decreases_when_line_is_broken(self):
        complete = line_image()
        broken = complete.copy()
        broken[14:18, 16] = 255
        self.assertLess(cldice(broken, complete), cldice(complete, complete))


if __name__ == "__main__":
    unittest.main()
