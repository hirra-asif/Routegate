"""Scientific invariants for cancellation controls, using unittest only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from routegate import cancellationcontrols as cc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT.parents[1] / "research_audit_20260910/routegate-data"
DATA_ROOT = Path(os.environ.get("ROUTEGATE_CANCELLATION_DATA_ROOT", str(DEFAULT_DATA)))
HAVE_REAL_DATA = (DATA_ROOT / cc.BANK_RELATIVE_PATH / "validated_direction_bank.npz").is_file()


def analytic_fixture() -> dict[str, np.ndarray]:
    # Deliberately non-unit repair vectors: q must divide by r.r, not use cosine.
    repair = np.zeros((3, 2048), dtype=np.float32)
    safety = np.zeros_like(repair)
    repair[:, 0] = (1.1, 0.9, 1.05)
    safety[:, 0] = (-0.35, -0.4, -0.3)
    safety[:, 1] = (0.8, 0.85, 0.9)
    r64, s64 = repair.astype(np.float64), safety.astype(np.float64)
    q = np.einsum("ij,ij->i", r64, s64) / np.einsum("ij,ij->i", r64, r64)
    perp = (s64-q[:, None]*r64).astype(np.float32)
    random = np.zeros((3, 3, 2048), dtype=np.float32)
    for draw in range(3):
        random[draw, :, draw+2] = np.linalg.norm(perp.astype(np.float64), axis=1)
    return {"repair": repair, "safety": safety, "q": q,
            "safety_perp": perp, "random_orthogonal": random,
            "zero": np.zeros_like(repair)}


def intended(arrays: dict, arm: str, state: int) -> np.ndarray:
    repair, second, ar, ass = cc.arm_components(arrays, arm, state)
    return ar*repair.astype(np.float64) + ass*second.astype(np.float64)


class IntendedEditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arrays = analytic_fixture()

    def test_frozen_inventory_and_baseline(self) -> None:
        self.assertEqual(len(set(cc.ARM_IDS)), 9)
        self.assertEqual(cc.EXPECTED_GENERATIONS, 162)
        self.assertEqual(tuple(s-1 for s in cc.STATES), cc.BUDGET_REFERENCE_STATES)
        self.assertEqual(cc.HISTORICAL_LAYERS, (31, 32, 33))
        for state in cc.STATES:
            r, s, ar, ass = cc.arm_components(self.arrays, "baseline", state)
            self.assertEqual((ar, ass), (0.0, 0.0))
            self.assertFalse(np.any(r))
            self.assertFalse(np.any(s))

    def test_attenuation_matches_projection_not_whole_delta(self) -> None:
        for index, state in enumerate(cc.STATES):
            r = self.arrays["repair"][index].astype(np.float64)
            original = intended(self.arrays, "original_combination", state)
            reduced = intended(self.arrays, "attenuated_repair", state)
            alone = intended(self.arrays, "orthogonal_only", state)
            np.testing.assert_allclose(original, reduced+alone, atol=1e-8, rtol=0)
            self.assertAlmostEqual(float(original@r), float(reduced@r), places=8)
            self.assertGreater(float(np.linalg.norm(original-reduced)), 0.05)
            self.assertLess(float(np.linalg.norm(reduced)),
                            float(np.linalg.norm(intended(self.arrays, "repair_only", state))))

    def test_orthogonal_full_retains_repair_projection(self) -> None:
        for index, state in enumerate(cc.STATES):
            r = self.arrays["repair"][index].astype(np.float64)
            full = intended(self.arrays, "orthogonal_full_repair", state)
            repair = intended(self.arrays, "repair_only", state)
            self.assertAlmostEqual(float(full@r), float(repair@r), places=8)

    def test_random_controls_match_norm_and_projection(self) -> None:
        for index, state in enumerate(cc.STATES):
            r = self.arrays["repair"][index].astype(np.float64)
            original = intended(self.arrays, "original_combination", state)
            for draw in range(1, 4):
                random = intended(self.arrays, f"random_orthogonal_{draw}", state)
                self.assertAlmostEqual(float(original@r), float(random@r), places=8)
                self.assertAlmostEqual(float(np.linalg.norm(original)),
                                       float(np.linalg.norm(random)), places=8)
                self.assertGreater(float(np.linalg.norm(original-random)), 0.05)

    def test_q_is_projection_coefficient_and_components_not_normalised(self) -> None:
        for index, state in enumerate(cc.STATES):
            r = self.arrays["repair"][index].astype(np.float64)
            s = self.arrays["safety"][index].astype(np.float64)
            cosine = float(r@s) / float(np.linalg.norm(r)*np.linalg.norm(s))
            q = float(self.arrays["q"][index])
            self.assertGreater(abs(q-cosine), 0.01)
            returned_r, returned_s, ar, ass = cc.arm_components(self.arrays, "original_combination", state)
            np.testing.assert_array_equal(returned_r, self.arrays["repair"][index])
            np.testing.assert_array_equal(returned_s, self.arrays["safety"][index])
            self.assertEqual((ar, ass), (0.15, 0.1))
            self.assertLess(float(np.linalg.norm(intended(self.arrays, "original_combination", state))), 0.3)

    def test_validation_rejects_scientifically_wrong_vectors(self) -> None:
        self.assertTrue(cc.validate_control_bank(self.arrays)["all_passed"])
        bad = {key: value.copy() for key, value in self.arrays.items()}
        bad["random_orthogonal"][0, 0, 0] += 0.1
        with self.assertRaises(cc.CancellationControlsError):
            cc.validate_control_bank(bad)
        bad = {key: value.copy() for key, value in self.arrays.items()}
        bad["q"][0] = 0.3
        with self.assertRaises(cc.CancellationControlsError):
            cc.validate_control_bank(bad)

    def test_unknown_arm_or_state_fails_closed(self) -> None:
        with self.assertRaises(cc.CancellationControlsError):
            cc.arm_components(self.arrays, "random_orthogonal_4", 32)
        for state in (31, 35, 32.0, True):
            with self.assertRaises(cc.CancellationControlsError):
                cc.arm_components(self.arrays, "repair_only", state)

    def test_changed_selection_manifest_rejected_before_data_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            path = repo / cc.SELECTION_RELATIVE_PATH
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"row_ids": list(cc.SELECTED_ROW_IDS[:-1])}))
            with self.assertRaisesRegex(cc.CancellationControlsError, "SHA256 mismatch"):
                cc.load_selected_rows(repo, repo / "absent_data")


@unittest.skipUnless(HAVE_REAL_DATA, "Set ROUTEGATE_CANCELLATION_DATA_ROOT for original-bank checks")
class OriginalInputTests(unittest.TestCase):
    def test_original_bank_is_preserved_and_controls_repeat(self) -> None:
        arrays, metadata = cc.build_control_bank(DATA_ROOT / cc.BANK_RELATIVE_PATH)
        again, second_metadata = cc.build_control_bank(DATA_ROOT / cc.BANK_RELATIVE_PATH)
        self.assertEqual(metadata, second_metadata)
        self.assertTrue(metadata["checks"]["all_passed"])
        self.assertEqual(metadata["expected_fresh_generations"], 162)
        with np.load(DATA_ROOT / cc.BANK_RELATIVE_PATH / "validated_direction_bank.npz",
                     allow_pickle=False) as bank:
            np.testing.assert_array_equal(arrays["repair"], bank["repair_direction"])
            np.testing.assert_array_equal(arrays["safety"], bank["safety_direction"])
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], again[key])
            self.assertFalse(arrays[key].flags.writeable)

    def test_pinned_selection_loads_only_prompt_fields(self) -> None:
        rows = cc.load_selected_rows(REPO_ROOT, DATA_ROOT)
        self.assertEqual([row["row_id"] for row in rows], list(cc.SELECTED_ROW_IDS))
        self.assertEqual(len(rows), 18)
        self.assertEqual(cc.selected_rows_digest([row["row_id"] for row in rows]), cc.SELECTED_ROWS_SHA256)
        for row in rows:
            self.assertEqual(set(row), {"row_id", "prompt_text", "prompt_sha256", "row_class", "source_family"})
            self.assertEqual(row["row_class"], "H")
            self.assertTrue(row["prompt_text"])

    def test_tampered_bank_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = DATA_ROOT / cc.BANK_RELATIVE_PATH
            shutil.copyfile(source / "direction_bank_manifest.json", directory / "direction_bank_manifest.json")
            (directory / "validated_direction_bank.npz").write_bytes(
                (source / "validated_direction_bank.npz").read_bytes()+b"tampered")
            with self.assertRaisesRegex(cc.CancellationControlsError, "SHA256 mismatch"):
                cc.build_control_bank(directory)


if __name__ == "__main__":
    unittest.main()
