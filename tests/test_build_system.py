"""Unit tests for Level 2 Build. Hand-computed expectations where noted."""
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-2-build"))
sys.path.insert(0, os.path.join(ROOT, "level-1-learn"))
import build_system as bs  # noqa: E402
from quantize import pack_int4, quantize_symmetric  # noqa: E402


class TestPerChannel(unittest.TestCase):
    def test_int8_scales_and_codes_hand_computed(self):
        # Column 0 max abs 1.0, scale 1/127, codes 127 and 32
        # (0.25 times 127 is 31.75, rounds to 32). Column 1 max
        # abs 2.0, scale 2/127, codes -32 (-0.5 times 127/2 is
        # -31.75, rounds half away to -32) and 127.
        q = bs.quantize_per_channel_matrix(
            np.array([[1.0, -0.5], [0.25, 2.0]]), 8)
        self.assertAlmostEqual(q["scales"][0], 1.0 / 127.0, places=15)
        self.assertAlmostEqual(q["scales"][1], 2.0 / 127.0, places=15)
        self.assertEqual(q["codes"].tolist(), [[127, -32], [32, 127]])
        self.assertAlmostEqual(q["dequantized"][1, 0], 32.0 / 127.0, places=15)

    def test_int4_scales_and_codes_hand_computed(self):
        # Column 0 scale 1/7: codes 7 and 2 (0.25 times 7 is 1.75).
        # Column 1 scale 2/7: codes -2 (-0.5 times 7/2 is -1.75)
        # and 7.
        q = bs.quantize_per_channel_matrix(
            np.array([[1.0, -0.5], [0.25, 2.0]]), 4)
        self.assertAlmostEqual(q["scales"][0], 1.0 / 7.0, places=15)
        self.assertAlmostEqual(q["scales"][1], 2.0 / 7.0, places=15)
        self.assertEqual(q["codes"].tolist(), [[7, -2], [2, 7]])

    def test_clip_percentile_hand_computed(self):
        # Absolute values 2.0 and 1.0; the 50th percentile is 1.5,
        # so clipping gives 1.5 and -1.0, scale 1.5/127, codes 127
        # and -85 (-1.0 times 127/1.5 is -84.67, rounds to -85).
        q = bs.quantize_per_channel_matrix(
            np.array([[2.0], [-1.0]]), 8, clip_percentile=50.0)
        self.assertAlmostEqual(q["scales"][0], 1.5 / 127.0, places=15)
        self.assertEqual(q["codes"].tolist(), [[127], [-85]])

    def test_zero_column_scale_is_one(self):
        q = bs.quantize_per_channel_matrix(np.zeros((2, 2)), 8)
        self.assertEqual(q["scales"].tolist(), [1.0, 1.0])
        self.assertEqual(q["codes"].tolist(), [[0, 0], [0, 0]])

    def test_bad_bits_rejected(self):
        with self.assertRaises(bs.BuildError):
            bs.quantize_per_channel_matrix(np.ones((2, 2)), 5)


class TestPerTensorReuse(unittest.TestCase):
    def test_per_tensor_matches_learn_exactly(self):
        # The Build per-tensor path is the Learn function applied
        # per matrix, so codes and scale must equal Learn output.
        matrix = np.array([[1.0, -0.5, 0.25], [-0.5, 1.0, -0.25]])
        flat = [float(v) for v in matrix.ravel()]
        learn = quantize_symmetric(flat, 8)
        q = bs.quantize_per_tensor_matrix(matrix, 8)
        self.assertEqual(q["codes"].ravel().tolist(), learn["codes"])
        self.assertAlmostEqual(float(q["scales"][0]), learn["scale"], places=15)


class TestPacking(unittest.TestCase):
    def test_pack_int4_hand_computed(self):
        # Reused Learn packing: (7, -4) packs to 199, (2, -4) to
        # 194, as in the Learn README hand checks.
        self.assertEqual(pack_int4([7, -4, 2, -4]), [199, 194])

    def test_int4_variant_payload_is_packed(self):
        model = bs.init_model()
        q, deq = {}, {}
        for key in bs.WEIGHT_KEYS:
            q[key] = bs.quantize_per_channel_matrix(model[key], 4)
            deq[key] = q[key]["dequantized"]
        entries = bs.variant_entries(model, q, "int4_per_channel_packed")
        code_entries = [e for e in entries if e[1] == "int4_packed"]
        self.assertEqual(len(code_entries), 2)
        # W1 has 512 codes, so 256 packed bytes. W2 has 128
        # codes, so 64 packed bytes.
        sizes = sorted(e[3].__len__() for e in code_entries)
        self.assertEqual(sizes, [64, 256])


class TestQLABRoundTrip(unittest.TestCase):
    def _round_trip(self, variant, model, quantized):
        entries = bs.variant_entries(model, quantized, variant)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.qlab")
            size = bs.save_qlab(path, entries)
            self.assertEqual(size, os.path.getsize(path))
            header, data = bs.load_qlab(path)
        loaded = bs.model_from_qlab(header, data, variant)
        return loaded

    def test_fp32_round_trip(self):
        model = bs.init_model()
        loaded = self._round_trip("fp32", model, None)
        for key in bs.ALL_KEYS:
            self.assertTrue(np.allclose(loaded[key], model[key], atol=1e-6))

    def test_int8_per_channel_round_trip(self):
        model = bs.init_model()
        quantized, deq, _ = bs.calibrated_model(
            model, 8, "per_channel", bs.make_split(bs.CALIB_SEED, 20)[0])
        loaded = self._round_trip("int8_per_channel", model, quantized)
        for key in bs.WEIGHT_KEYS:
            self.assertTrue(np.allclose(loaded[key], deq[key], atol=1e-6))

    def test_int4_packed_round_trip(self):
        model = bs.init_model()
        quantized, deq, _ = bs.calibrated_model(
            model, 4, "per_channel", bs.make_split(bs.CALIB_SEED, 20)[0])
        loaded = self._round_trip("int4_per_channel_packed", model, quantized)
        for key in bs.WEIGHT_KEYS:
            self.assertTrue(np.allclose(loaded[key], deq[key], atol=1e-6))

    def test_bad_magic_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.qlab")
            with open(path, "wb") as fh:
                fh.write(b"GGUF" + b"\x00" * 32)
            with self.assertRaises(bs.BuildError):
                bs.load_qlab(path)


class TestPruning(unittest.TestCase):
    def test_mask_hand_computed(self):
        # Absolute values 0.1, 0.5, 0.05, 0.9. Sparsity 0.5
        # prunes floor(0.5 times 4) = 2 smallest: 0.05 and 0.1.
        matrix = np.array([[0.1, -0.5], [0.05, 0.9]])
        self.assertEqual(bs.pruning_mask(matrix, 0.5).tolist(),
                         [[False, True], [False, True]])
        self.assertEqual(bs.prune_matrix(matrix, 0.5).tolist(),
                         [[0.0, -0.5], [0.0, 0.9]])

    def test_sparsity_zero_keeps_everything(self):
        matrix = np.array([[0.1, -0.5], [0.05, 0.9]])
        self.assertEqual(bs.prune_matrix(matrix, 0.0).tolist(),
                         matrix.tolist())

    def test_pruned_counts_on_trained_model(self):
        model = bs.pruned_model(bs.init_model(), 0.5)
        init = bs.init_model()
        total = sum(init[k].size for k in bs.WEIGHT_KEYS)
        pruned_zeros = sum(int((model[k] == 0).sum()) for k in bs.WEIGHT_KEYS)
        expected = sum(int(np.floor(0.5 * init[k].size)) for k in bs.WEIGHT_KEYS)
        self.assertEqual(total, 640)
        self.assertEqual(pruned_zeros, expected)

    def test_bad_sparsity_rejected(self):
        with self.assertRaises(bs.BuildError):
            bs.pruning_mask(np.ones((2, 2)), 1.0)


class TestModelAndData(unittest.TestCase):
    def test_split_shapes_and_determinism(self):
        X1, y1 = bs.make_split(bs.TEST_SEED, bs.TEST_N)
        X2, y2 = bs.make_split(bs.TEST_SEED, bs.TEST_N)
        self.assertEqual(X1.shape, (400, 16))
        self.assertTrue((X1 == X2).all() and (y1 == y2).all())
        self.assertEqual(sorted(np.bincount(y1).tolist()), [100, 100, 100, 100])

    def test_training_is_deterministic(self):
        X, y = bs.make_split(bs.TRAIN_SEED, 100)
        m1 = bs.train_model(bs.init_model(), X, y, epochs=5)
        m2 = bs.train_model(bs.init_model(), X, y, epochs=5)
        for key in bs.ALL_KEYS:
            self.assertTrue((m1[key] == m2[key]).all())

    def test_trained_model_beats_chance(self):
        datasets = bs.build_datasets()
        model = bs.train_model(bs.init_model(), *datasets["train"])
        acc = bs.accuracy(model, *datasets["test"])
        self.assertGreater(acc, 0.80)
        self.assertLess(acc, 1.0)

    def test_calibration_returns_a_named_candidate(self):
        datasets = bs.build_datasets()
        model = bs.train_model(bs.init_model(), *datasets["train"])
        best = bs.calibrate_matrix(model, "W2", 8, "per_channel",
                                   datasets["calibration"][0])
        self.assertIn(best["clip_percentile"], bs.CALIB_CANDIDATES)
        self.assertGreaterEqual(best["calibration_logit_mse"], 0.0)


if __name__ == "__main__":
    unittest.main()
