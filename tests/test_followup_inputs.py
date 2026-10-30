"""Checks that the next comparison preserves the accepted development frame."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from routegate import followupinputs as subject


class MissingInventoryTests(unittest.TestCase):
    def test_missing_inventory_is_not_a_prepared_study(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(subject.FollowupInputError):
                subject.verify(temporary)

    def test_unreadable_json_is_reported_as_an_input_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input_manifest.json").write_text("{broken", encoding="utf-8")
            (root / "rows.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(subject.FollowupInputError):
                subject.verify(root)


class AcceptedInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(os.environ.get(
            "ROUTEGATE_FOLLOWUP_TEST_REPO_ROOT", str(Path(__file__).resolve().parents[1])))
        data_root = Path(os.environ.get(
            "ROUTEGATE_FOLLOWUP_TEST_DATA_ROOT", str(Path.home() / "routegate-data")))
        cls.source_bytes = {}
        for source in subject.SOURCE_FILES:
            source_root = repo_root if source["root"] == "repo" else data_root
            path = source_root / source["path"]
            if not path.is_file():
                raise unittest.SkipTest(
                    "Accepted input files unavailable. Set ROUTEGATE_FOLLOWUP_TEST_REPO_ROOT "
                    "and ROUTEGATE_FOLLOWUP_TEST_DATA_ROOT for the source-backed checks.")
            cls.source_bytes[(source["root"], source["path"])] = path.read_bytes()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo, self.data = self.root / "repo", self.root / "data"
        for (root_name, relative_path), content in self.source_bytes.items():
            root = self.repo if root_name == "repo" else self.data
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.output = self.data / subject.STUDY_ID

    def prepare(self):
        return subject.prepare(self.repo, self.data, self.output)

    def change_rows(self, transform):
        _, rows = subject.verify(self.output)
        transform(rows)
        (self.output / "rows.jsonl").write_bytes(subject._row_bytes(rows))

    def test_complete_frame_and_original_fold_map(self):
        prepared = self.prepare()
        manifest, rows = subject.verify(self.output)
        self.assertEqual(prepared, manifest)
        self.assertEqual(480, len(rows))
        self.assertEqual({"E": 89, "B": 178, "H": 213}, manifest["class_counts"])
        self.assertEqual({"0": 97, "1": 97, "2": 96, "3": 95, "4": 95}, manifest["fold_counts"])
        original = json.loads(self.source_bytes[("repo", "artifacts/gate_oof_v1/fold_assignments.json")])
        self.assertEqual(original["fold_of_row"], {row["row_id"]: row["fold_id"] for row in rows})
        self.assertEqual(subject.EXPECTED_ROWS_SHA256,
                         hashlib.sha256((self.output / "rows.jsonl").read_bytes()).hexdigest())

    def test_preparation_does_not_select_methods_or_import_outcomes(self):
        self.prepare()
        manifest, rows = subject.verify(self.output)
        self.assertEqual([], manifest["candidate_arms"])
        self.assertEqual("PENDING_CANCELLATION_DIAGNOSTIC", manifest["selection_status"])
        for key in ("old_responses_imported", "old_verdicts_imported", "new_generations",
                    "judge_calls", "trained_models", "external_benchmark_rows_loaded"):
            self.assertEqual(0, manifest[key])
        for row in rows:
            self.assertEqual(subject.ROW_FIELDS, set(row))

    def test_same_inputs_leave_existing_inventory_bytes_unchanged(self):
        first = self.prepare()
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        second = self.prepare()
        self.assertEqual(first, second)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.output.iterdir()})
        for (root_name, relative_path), content in self.source_bytes.items():
            root = self.repo if root_name == "repo" else self.data
            self.assertEqual(content, (root / relative_path).read_bytes())

    def test_every_source_hash_is_enforced_before_writing(self):
        for index, source in enumerate(subject.SOURCE_FILES):
            with self.subTest(source=source["path"]):
                root = self.repo if source["root"] == "repo" else self.data
                path = root / source["path"]
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                output = self.data / f"bad-source-{index}"
                with self.assertRaisesRegex(subject.FollowupInputError, "hash mismatch"):
                    subject.prepare(self.repo, self.data, output)
                self.assertFalse(output.exists())
                path.write_bytes(original)

    def test_fold_swap_cannot_be_hidden_by_unchanged_fold_sizes(self):
        self.prepare()
        def swap(rows):
            left = rows[0]
            right = next(row for row in rows
                         if row["row_class"] == left["row_class"] and row["fold_id"] != left["fold_id"])
            left["fold_id"], right["fold_id"] = right["fold_id"], left["fold_id"]
        self.change_rows(swap)
        with self.assertRaisesRegex(subject.FollowupInputError, "fixed original inventory"):
            subject.verify(self.output)

    def test_duplicate_ids_are_refused(self):
        self.prepare()
        self.change_rows(lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])))
        with self.assertRaisesRegex(subject.FollowupInputError, "Duplicate row_id"):
            subject.verify(self.output)

    def test_edited_prompt_is_refused_even_with_recomputed_prompt_hash(self):
        self.prepare()
        def change(rows):
            rows[0]["prompt_text"] += "?"
            rows[0]["prompt_sha256"] = hashlib.sha256(rows[0]["prompt_text"].encode("utf-8")).hexdigest()
        self.change_rows(change)
        with self.assertRaisesRegex(subject.FollowupInputError, "fixed original inventory"):
            subject.verify(self.output)

    def test_additional_response_field_is_refused(self):
        self.prepare()
        self.change_rows(lambda rows: rows[0].update({"response_text": "old result"}))
        with self.assertRaisesRegex(subject.FollowupInputError, "only the declared"):
            subject.verify(self.output)

    def test_boolean_fold_ids_are_not_accepted_as_integers(self):
        self.prepare()
        self.change_rows(lambda rows: rows[0].update({"fold_id": False}))
        with self.assertRaisesRegex(subject.FollowupInputError, "Invalid fold ID"):
            subject.verify(self.output)

    def test_input_manifest_cannot_be_used_to_select_a_candidate(self):
        self.prepare()
        path = self.output / "input_manifest.json"
        manifest = json.loads(path.read_bytes())
        manifest["candidate_arms"] = ["unselected_edit"]
        stable = {key: value for key, value in manifest.items()
                  if key not in {"created_utc", "inventory_sha256"}}
        manifest["inventory_sha256"] = subject._sha(subject._canonical(stable))
        path.write_bytes(subject._canonical(manifest) + b"\n")
        with self.assertRaisesRegex(subject.FollowupInputError, "metadata was changed"):
            subject.verify(self.output)

    def test_unclaimed_directory_is_preserved(self):
        self.output.mkdir()
        marker = self.output / "existing.txt"
        marker.write_text("keep me", encoding="utf-8")
        with self.assertRaisesRegex(subject.FollowupInputError, "already exists"):
            self.prepare()
        self.assertEqual("keep me", marker.read_text(encoding="utf-8"))
        self.assertFalse((self.output / "input_manifest.json").exists())

    def test_output_cannot_be_written_under_frozen_sources(self):
        with self.assertRaisesRegex(subject.FollowupInputError, "frozen input"):
            subject.prepare(self.repo, self.data, self.data / "frozen" / "new-study")

    def test_source_family_limitation_is_explicit(self):
        manifest = self.prepare()
        summary = manifest["source_class_counts"]
        self.assertEqual({"B": 43, "E": 7}, summary["false_reject_test"])
        self.assertEqual({"H": 119}, summary["strongreject"])
        self.assertIn("not an unseen-source", manifest["fold_interpretation"])
        self.assertIn("H sources are distinct", manifest["source_confound_note"])


if __name__ == "__main__":
    unittest.main()
