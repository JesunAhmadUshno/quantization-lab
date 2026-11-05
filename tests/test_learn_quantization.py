"""Unit tests for Level 1 Learn quantization. Hand-computed expectations."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-1-learn"))
from quantize import (  # noqa: E402
    QuantizationError, argmax_first, build_task_samples, build_weight_matrix,
    error_metrics, evaluate_task, flatten, linear_scores, pack_int4,
    quantize_symmetric, reshape, round_half_away, size_bytes, unpack_int4,
)

FLAT = [1.0, -0.5, 0.25, -0.5, 1.0, -0.25,
        0.25, -0.75, 1.0, -0.75, -0.25, -1.0]
INT8_CODES = [127, -64, 32, -64, 127, -32, 32, -95, 127, -95, -32, -127]
INT4_CODES = [7, -4, 2, -4, 7, -2, 2, -5, 7, -5, -2, -7]
INT4_PACKED = [199, 194, 231, 178, 183, 158]


class TestRoundingAndScales(unittest.TestCase):
    def test_round_half_away(self):
        self.assertEqual(round_half_away(3.5), 4)
        self.assertEqual(round_half_away(-3.5), -4)
        self.assertEqual(round_half_away(2.4), 2)
        self.assertEqual(round_half_away(-2.6), -3)
        self.assertEqual(round_half_away(0.0), 0)

    def test_int8_scale(self):
        # max abs is 1.0, qmax 127, scale is 1/127.
        q = quantize_symmetric(FLAT, 8)
        self.assertAlmostEqual(q["scale"], 1.0 / 127.0, places=15)
        self.assertEqual(q["qmax"], 127)

    def test_int4_scale(self):
        q = quantize_symmetric(FLAT, 4)
        self.assertAlmostEqual(q["scale"], 1.0 / 7.0, places=15)
        self.assertEqual(q["qmax"], 7)

    def test_zero_tensor_scale_is_one(self):
        q = quantize_symmetric([0.0, 0.0], 8)
        self.assertEqual(q["scale"], 1.0)
        self.assertEqual(q["codes"], [0, 0])

    def test_bad_bits_rejected(self):
        with self.assertRaises(QuantizationError):
            quantize_symmetric(FLAT, 5)

    def test_empty_rejected(self):
        with self.assertRaises(QuantizationError):
            quantize_symmetric([], 8)


class TestCodes(unittest.TestCase):
    def test_int8_codes_hand_computed(self):
        # code is round half away of w times 127: -0.5 gives -64
        # (63.5 rounds away), 0.25 gives 32 (31.75), -0.75 gives
        # -95 (95.25), 0.25 and -0.25 as above.
        self.assertEqual(quantize_symmetric(FLAT, 8)["codes"], INT8_CODES)

    def test_int4_codes_hand_computed(self):
        # code is round half away of w times 7: -0.5 gives -4
        # (3.5 rounds away), 0.25 gives 2 (1.75), -0.75 gives -5.
        self.assertEqual(quantize_symmetric(FLAT, 4)["codes"], INT4_CODES)

    def test_int8_dequant_spot_values(self):
        dq = quantize_symmetric(FLAT, 8)["dequantized"]
        self.assertEqual(dq[0], 1.0)
        self.assertAlmostEqual(dq[1], -64.0 / 127.0, places=15)
        self.assertAlmostEqual(dq[2], 32.0 / 127.0, places=15)
        self.assertEqual(dq[11], -1.0)

    def test_int4_dequant_spot_values(self):
        dq = quantize_symmetric(FLAT, 4)["dequantized"]
        self.assertAlmostEqual(dq[1], -4.0 / 7.0, places=15)
        self.assertAlmostEqual(dq[7], -5.0 / 7.0, places=15)
        self.assertEqual(dq[8], 1.0)


class TestPackingAndSizes(unittest.TestCase):
    def test_pack_int4_hand_computed(self):
        # (7, -4): 7 plus 12 shifted left 4 is 199. (2, -4): 194.
        # (7, -2): 7 plus 14 shifted is 231. (2, -5): 178.
        # (7, -5): 183. (-2, -7): 14 plus 9 shifted is 158.
        self.assertEqual(pack_int4(INT4_CODES), INT4_PACKED)

    def test_unpack_round_trip(self):
        self.assertEqual(unpack_int4(INT4_PACKED, 12), INT4_CODES)

    def test_pack_rejects_out_of_range(self):
        with self.assertRaises(QuantizationError):
            pack_int4([8])

    def test_sizes(self):
        # 12 weights: FP32 12 times 4 is 48, no scale. INT8 12
        # payload plus 4 scale is 16. INT4 6 payload plus 4 is 10.
        self.assertEqual(size_bytes(12, 32),
                         {"payload_bytes": 48, "scale_bytes": 0, "total_bytes": 48})
        self.assertEqual(size_bytes(12, 8),
                         {"payload_bytes": 12, "scale_bytes": 4, "total_bytes": 16})
        self.assertEqual(size_bytes(12, 4),
                         {"payload_bytes": 6, "scale_bytes": 4, "total_bytes": 10})

    def test_odd_int4_count_wastes_half_byte(self):
        self.assertEqual(size_bytes(9, 4)["payload_bytes"], 5)


class TestErrors(unittest.TestCase):
    def test_int8_errors_hand_computed(self):
        # Absolute errors: 0 for the four weights at 1.0 in abs,
        # 0.5/127 for the two -0.5 weights (error 0.003937...), and
        # 0.25/127 for the six 0.25 and 0.75 weights (0.001969...).
        # Mean is (2 times 0.0039370079 plus 6 times 0.0019685039) / 12.
        m = error_metrics(FLAT, quantize_symmetric(FLAT, 8)["dequantized"])
        self.assertAlmostEqual(m["max_abs_error"], 0.5 / 127.0, places=12)
        self.assertAlmostEqual(m["mean_abs_error"], 0.0016404199475066, places=12)
        self.assertAlmostEqual(m["total_abs_error"], 0.0196850393700787, places=12)
        self.assertAlmostEqual(m["mse"], 4.520842375018067e-06, places=12)

    def test_int4_errors_hand_computed(self):
        # Absolute errors: 0 for 1.0 weights, 1/14 (0.071429) for the
        # two -0.5 weights, 1/28 (0.035714) for the six others that
        # are not exact. Mean is (2/14 plus 6/28) / 12 = 0.029762.
        m = error_metrics(FLAT, quantize_symmetric(FLAT, 4)["dequantized"])
        self.assertAlmostEqual(m["max_abs_error"], 1.0 / 14.0, places=12)
        self.assertAlmostEqual(m["mean_abs_error"], 0.0297619047619048, places=12)
        self.assertAlmostEqual(m["mse"], 0.0014880952380952, places=12)


class TestTask(unittest.TestCase):
    def test_weights_flat_exact(self):
        self.assertEqual(flatten(build_weight_matrix()), FLAT)

    def test_fp32_scores_sample7_hand_computed(self):
        # Input [0, 1, -2]: row scores are 0 -0.5 -0.5 = -1.0,
        # 0 + 1 + 0.5 = 1.5, 0 -0.75 -2 = -2.75, 0 -0.25 + 2 = 1.75.
        scores = linear_scores(build_weight_matrix(), [0, 1, -2])
        self.assertEqual(scores, [-1.0, 1.5, -2.75, 1.75])
        self.assertEqual(argmax_first(scores), 3)

    def test_fp32_task_accuracy(self):
        ev = evaluate_task(build_weight_matrix(), build_task_samples())
        self.assertEqual(ev["predictions"], [0, 1, 2, 3, 0, 1, 3, 0])
        self.assertEqual(ev["correct"], 8)
        self.assertEqual(ev["accuracy"], 1.0)

    def test_int8_task_accuracy(self):
        dq = quantize_symmetric(FLAT, 8)["dequantized"]
        ev = evaluate_task(reshape(dq, 4, 3), build_task_samples())
        self.assertEqual(ev["predictions"], [0, 1, 2, 3, 0, 1, 3, 0])
        self.assertEqual(ev["accuracy"], 1.0)

    def test_int4_task_accuracy(self):
        dq = quantize_symmetric(FLAT, 4)["dequantized"]
        ev = evaluate_task(reshape(dq, 4, 3), build_task_samples())
        self.assertEqual(ev["predictions"], [0, 1, 2, 3, 0, 1, 3, 0])
        self.assertEqual(ev["accuracy"], 1.0)

    def test_argmax_tie_goes_first(self):
        self.assertEqual(argmax_first([1.0, 1.0, 0.0]), 0)


if __name__ == "__main__":
    unittest.main()
