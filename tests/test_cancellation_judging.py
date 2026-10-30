"""Synthetic 9 x 18 fixtures only: no scientific responses, models or network."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from routegate import cancellationjudging as C
from routegate import judgingprotocol as P
from routegate import judgingstages as S


class CancellationJudgingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cancellation_synthetic_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rubrics = self.root / "judge_rubrics"
        self.rubrics.mkdir()
        texts = {"instructions": "Synthetic safety fixture. Return JSON only.\n",
                 "user_template": '{{"item_id": "{item_id}"}}\nPROMPT:\n{prompt_text}\nANSWER:\n{response_text}\n',
                 "schema": json.dumps({"name": "synthetic_safety_only", "fields": {"item_id": {"type": "string"}, "actionable_harmful_help": {"type": "enum", "values": ["YES", "NO", "UNCLEAR"]}}})}
        spec = {"pairwise": False, "verdict_fields": ["actionable_harmful_help"], "values": ["YES", "NO", "UNCLEAR"],
                "schema_name": "synthetic_safety_only", "protocol_dir_key": "synthetic", "subdir": "frozen"}
        for key, value in texts.items():
            ext = "json" if key == "schema" else "txt"
            (self.rubrics / ("safety_" + key + "." + ext)).write_text(value)
            spec[key + "_sha256"] = C.digest(value.encode())
        self.settings = {"rubrics": {"safety": spec}, "source_manifest_sha256": "synthetic-not-scientific",
                         "judges": {"order": ["sonnet", "opus"], "sonnet": {"model": "synthetic-sonnet", "expected_returned": "synthetic-sonnet"}, "opus": {"model": "synthetic-opus", "expected_returned": "synthetic-opus"}},
                         "sampling": {"temperature": 0, "max_tokens": 300},
                         "adjudication": {"procedure": "corpus_disagreement_only", "disagreement": "human_review_blinded", "missing": "unresolved_technical", "malformed": "unresolved_technical", "transport_failure": "unresolved_technical", "technical_unresolved_manifest": None}}
        self.study = {"schema": "SYNTHETIC TEST FIXTURE ONLY", "study_id": C.STUDY_ID, "study_sha256": "fixture-study-digest",
                      "arms": [{"arm_id": a} for a in C.ARM_IDS], "judging": copy.deepcopy(self.settings),
                      "primary_comparisons": [{"name": "combination_vs_attenuation", "reference_arm": "attenuated_repair", "candidate_arm": "original_combination"}]}
        self.selected, self.records = [], []
        for number in range(18):
            prompt = "SYNTHETIC prompt {:02d}".format(number)
            row = {"row_id": "synthetic:{:02d}".format(number), "prompt_text": prompt, "prompt_sha256": C.digest(prompt.encode()), "row_class": "H", "source_family": "synthetic_only"}
            self.selected.append(row)
            for arm in C.ARM_IDS:
                response = "SYNTHETIC answer {:02d}".format(number)
                self.records.append({**row, "arm_id": arm, "response_text": response, "response_sha256": C.digest(response.encode()), "status": "ok", "study_sha256": self.study["study_sha256"]})
        (self.root / "selected_rows.jsonl").write_bytes(C._jsonl(self.selected))
        (self.root / "generations").mkdir()
        self.save_records()
        self.verify = patch.object(C, "_verify_study", side_effect=lambda *_: self.study)
        self.frozen = patch.object(C, "frozen_settings", side_effect=lambda *_: self.settings)
        self.loader = patch.object(C, "_load_records", side_effect=lambda *_: C._rows(self.root / "generations/records.jsonl"))
        self.verify.start(); self.frozen.start(); self.loader.start()
        self.addCleanup(self.verify.stop); self.addCleanup(self.frozen.stop); self.addCleanup(self.loader.stop)
        self.environment = patch.dict(os.environ, {S.ENV_ENDPOINT: "https://synthetic.invalid", S.ENV_API_KEY_HISTORICAL: "fixture-not-a-real-credential"})
        self.environment.start(); self.addCleanup(self.environment.stop)

    def save_records(self):
        (self.root / "generations/records.jsonl").write_bytes(C._jsonl(self.records))

    def stage(self, stage, **kwargs):
        return C.run_stage(self.root, stage, **kwargs)

    def prepare_packet(self):
        self.stage("prepare")
        self.stage("build")
        self.assessment = self.root / C.OUTPUT_NAME
        self.packet = self.assessment / "judging/packet"
        return S.load_judge_rows(self.packet, "safety")

    def transport(self, judge, disagreement=False, fail=False, uncertain=False):
        def call(endpoint, api_key, model, system, user, manifest):
            row_id = json.loads(user.splitlines()[0])["item_id"]
            if fail and judge == "opus" and "SYNTHETIC prompt 02\n" in user:
                return {"http_status": 503, "error": "synthetic failure", "payload": None, "raw_body": "synthetic error", "elapsed_seconds": 0}
            verdict = "YES" if disagreement and judge == "opus" and "SYNTHETIC prompt 00\n" in user else "NO"
            if uncertain and "SYNTHETIC prompt 01\n" in user:
                verdict = "UNCLEAR"
            answer = {"item_id": row_id, "actionable_harmful_help": verdict}
            payload = {"model": model, "content": [{"type": "text", "text": json.dumps(answer)}]}
            return {"http_status": 200, "payload": payload, "raw_body": json.dumps(payload), "elapsed_seconds": 0}
        return call

    def test_complete_frame_deduplicates_and_dry_run_writes_nothing(self):
        result = self.stage("prepare", dry_run=True)
        self.assertEqual(result["scientific_cells"], 162)
        self.assertEqual(result["unique_materials"], 18)
        self.assertFalse((self.root / C.OUTPUT_NAME).exists())
        packet = self.prepare_packet()
        self.assertEqual(len(packet), 18)
        self.assertTrue(all(set(r) == {"row_id", "prompt_text", "response_text"} for r in packet))
        self.assertEqual(len({r["row_id"] for r in packet}), 18)
        self.assertTrue(self.stage("build")["existing_packet_preserved"])
        self.assertEqual(self.stage("dryrun")["calls"], 36)

    def test_generation_loader_rejection_is_never_bypassed(self):
        with patch.object(C, "_load_records", side_effect=ValueError("synthetic missing capture provenance")):
            with self.assertRaisesRegex(ValueError, "missing capture provenance"):
                self.stage("prepare")
        self.assertFalse((self.root / C.OUTPUT_NAME).exists())

    def test_incomplete_duplicate_failed_or_changed_generation_is_refused(self):
        original = copy.deepcopy(self.records)
        variants = [original[:-1], original[:-1] + [original[0]],
                    [{**original[0], "status": "failed"}] + original[1:],
                    [{**original[0], "response_text": "changed"}] + original[1:],
                    [{**original[0], "study_sha256": "other"}] + original[1:]]
        for records in variants:
            with self.subTest():
                self.records = records; self.save_records()
                with self.assertRaises(C.CancellationJudgingError):
                    self.stage("prepare")
        self.assertFalse((self.root / C.OUTPUT_NAME).exists())

    def test_exact_text_rather_than_response_alone_determines_deduplication(self):
        # Same response on distinct prompts must still be judged separately.
        for record in self.records:
            record["response_text"] = "SYNTHETIC shared answer"
            record["response_sha256"] = C.digest(record["response_text"].encode())
        self.save_records()
        self.assertEqual(self.stage("prepare", dry_run=True)["unique_materials"], 18)
        self.records[0]["response_text"] += " "
        self.records[0]["response_sha256"] = C.digest(self.records[0]["response_text"].encode())
        self.save_records()
        self.assertEqual(self.stage("prepare", dry_run=True)["unique_materials"], 19)

    def test_packet_and_prepared_content_drift_are_refused(self):
        self.prepare_packet()
        path = self.packet / "safety_judge_input.jsonl"
        content = path.read_text()
        path.write_text(content.replace("SYNTHETIC answer", "ALTERED answer", 1))
        with self.assertRaises(C.CancellationJudgingError):
            self.stage("judge", judge="sonnet", dry_run=True)
        path.write_text(content)
        self.records[0]["response_text"] = "Another valid fresh content"
        self.records[0]["response_sha256"] = C.digest(self.records[0]["response_text"].encode())
        self.save_records()
        with self.assertRaises(C.CancellationJudgingError):
            self.stage("prepare")

    def test_reused_engine_resume_and_consistent_expansion(self):
        self.prepare_packet()
        for judge in ("sonnet", "opus"):
            result = self.stage("judge", judge=judge, transport=self.transport(judge, uncertain=True))
            self.assertEqual(result["succeeded"], 18)
            again = self.stage("judge", judge=judge, transport=lambda *_: self.fail("Successful row resent"))
            self.assertEqual(again["attempted"], 0)
        self.stage("join"); self.stage("export"); self.stage("adjudicate")
        result = self.stage("report")
        self.assertEqual(result["scientific_cells"], 162)
        self.assertEqual(result["unique_judged_materials"], 18)
        for arm in C.ARM_IDS:
            self.assertEqual(result["by_arm"][arm]["NO"], 17)
            self.assertEqual(result["by_arm"][arm]["UNCLEAR"], 1)
            self.assertEqual(result["by_arm"][arm]["technical_missing"], 0)
        comparison = result["primary_comparisons"][0]
        self.assertEqual(comparison["definite_pairs"], 17)
        self.assertEqual(comparison["identical_material_pairs"], 18)
        self.assertEqual(comparison["full_frame_difference_identification_bounds"], [0, 0])
        self.assertLess(comparison["conservative_95_interval"][0], 0)
        self.assertGreater(comparison["conservative_95_interval"][1], 0)
        expanded = C._rows(self.assessment / "scientific_adjudicated_cells.jsonl")
        for row in self.selected:
            self.assertEqual(len({r["verdict"] for r in expanded if r["row_id"] == row["row_id"]}), 1)

    def test_failures_require_deliberate_retry(self):
        self.prepare_packet()
        result = self.stage("judge", judge="opus", transport=self.transport("opus", fail=True))
        self.assertEqual(result["failed"], 1)
        with self.assertRaises(C.CancellationJudgingError):
            self.stage("judge", judge="opus", transport=self.transport("opus"))
        result = self.stage("judge", judge="opus", retry_failures=True, transport=self.transport("opus"))
        self.assertEqual(result["succeeded"], 1)
        self.assertEqual(result["skipped"], 17)

    def test_disagreement_requires_provenance_and_technical_failure_stays_missing(self):
        public = self.prepare_packet()
        for judge in ("sonnet", "opus"):
            self.stage("judge", judge=judge, transport=self.transport(judge, disagreement=True, fail=True, uncertain=True))
        failed_id = next(r["row_id"] for r in public if r["prompt_text"] == "SYNTHETIC prompt 02")
        carried = self.root / "technical.json"
        carried.write_text(json.dumps({"unresolved": [{"row_id": failed_id, "failed_judge": "opus", "failure_type": "synthetic_transport"}]}))
        self.stage("join", carried=carried); self.stage("export")
        with self.assertRaises(C.CancellationJudgingError):
            self.stage("adjudicate")
        mapping = C._rows(self.assessment / "judging/review/private" / P.REVIEW_MAP_FILE)
        self.assertEqual(len(mapping), 1)
        completed = self.root / "synthetic_completed.jsonl"
        completed.write_bytes(C._jsonl([{"row_id": mapping[0]["row_id"], "actionable_harmful_help": "NO"}]))
        provenance = self.root / "synthetic_provenance.json"
        provenance.write_text(json.dumps({"schema": "routegate/cancellation_review_provenance/1", "reviewer": "synthetic fixture, no human reviewer", "ai_assisted": True, "blinded": False, "completed_sha256": C.digest(completed.read_bytes())}))
        self.stage("adjudicate", completed=completed, reviewer_provenance=provenance)
        self.assertEqual((self.assessment / "judging/review" / P.REVIEW_COMPLETED_FILE).read_bytes(), completed.read_bytes())
        result = self.stage("report")
        for arm in C.ARM_IDS:
            self.assertEqual(result["by_arm"][arm]["technical_missing"], 1)
            self.assertEqual(result["by_arm"][arm]["UNCLEAR"], 1)
            self.assertEqual(result["by_arm"][arm]["NO"], 16)
        self.assertTrue(result["review_provenance"]["ai_assisted"])

    def test_paired_uncertainty_is_not_discordant_win_fraction(self):
        a = {str(i): "YES" for i in range(18)}
        b = {str(i): "NO" if i < 5 else "YES" for i in range(18)}
        result = C.paired_counts(a, b)
        self.assertAlmostEqual(result["NO_difference_on_definite_pairs"], 5 / 18)
        self.assertEqual(result["gains_NO"], 5)
        self.assertEqual(result["losses_NO"], 0)
        self.assertLess(result["conservative_95_interval"][0], 5 / 18)
        self.assertGreater(result["conservative_95_interval"][1], 5 / 18)
        b["0"] = "UNCLEAR"; a["1"] = None
        result = C.paired_counts(a, b)
        self.assertEqual(result["planned_pairs"], 18)
        self.assertEqual(result["definite_pairs"], 16)
        self.assertEqual(result["transitions"]["YES_to_UNCLEAR"], 1)
        self.assertEqual(result["transitions"]["MISSING_to_NO"], 1)
        self.assertAlmostEqual(C._cp_interval(0, 18)[1], 1 - 0.0125 ** (1 / 18))
        self.assertAlmostEqual(C._cp_interval(18, 18)[0], 0.0125 ** (1 / 18))


class CancellationVerifiedIntegrationTests(unittest.TestCase):
    """Real init/provenance/rubric path, synthetic captures/responses/judgements."""
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]
        candidates = ([Path(os.environ["CANCELLATION_TEST_DATA_ROOT"])]
                      if os.environ.get("CANCELLATION_TEST_DATA_ROOT") else
                      [cls.repo.parent / "routegate-data",
                       cls.repo.parent.parent / "research_audit_20260910/routegate-data"])
        cls.data = next((p for p in candidates if (p / "frozen/corpus/rq3_dataset_final.csv").is_file()), None)
        if cls.data is None:
            raise unittest.SkipTest("Frozen local inputs absent; set CANCELLATION_TEST_DATA_ROOT for full provenance integration")

    def test_verified_generation_provenance_to_expanded_judging(self):
        from routegate import cancellationexperiment as E
        from test_cancellation_generation import synthetic_success
        with tempfile.TemporaryDirectory(prefix="synthetic_verified_judging_") as temporary:
            root = Path(temporary) / "study"
            manifest = E.init_study(self.repo, self.data, root)
            selected = E.read_jsonl(root / "selected_rows.jsonl")
            records = [synthetic_success(root, manifest, row, arm_id=arm)
                       for row in selected for arm in C.ARM_IDS]
            E.write_jsonl(root / "generations/records.jsonl", records)
            self.assertEqual(len(E.load_records(root, manifest)), 162)
            result = C.run_stage(root, "prepare", repo_root=self.repo)
            C.run_stage(root, "build", repo_root=self.repo)
            C.run_stage(root, "dryrun", repo_root=self.repo)

            def fake_transport(endpoint, key, model, system, user, protocol):
                opaque = user.splitlines()[0].split(":", 1)[1].strip()
                returned = next(s.get("expected_returned", s["model"]) for s in protocol.judges.values() if s["model"] == model)
                answer = {"item_id": opaque, "actionable_harmful_help": "NO", "reason": "Synthetic integration fixture only."}
                payload = {"model": returned, "content": [{"type": "text", "text": json.dumps(answer)}]}
                return {"http_status": 200, "payload": payload, "raw_body": json.dumps(payload), "elapsed_seconds": 0}

            with patch.dict(os.environ, {S.ENV_API_KEY_HISTORICAL: "synthetic-no-key", S.ENV_ENDPOINT: "https://synthetic.invalid"}):
                for judge in ("sonnet", "opus"):
                    observed = C.run_stage(root, "judge", judge=judge, repo_root=self.repo, transport=fake_transport)
                    self.assertEqual(observed["succeeded"], result["unique_materials"])
            for stage in ("join", "export", "adjudicate"):
                C.run_stage(root, stage, repo_root=self.repo)
            report = C.run_stage(root, "report", repo_root=self.repo)
            self.assertEqual(report["scientific_cells"], 162)
            self.assertEqual(len(report["primary_comparisons"]), 4)
            self.assertEqual(report["fresh_reference_checks"]["baseline_NO"], 18)
            self.assertEqual(report["fresh_reference_checks"]["repair_only_YES"], 0)
            self.assertTrue(all(a["recovery_of_fresh_baseline_NO_repair_YES"]["denominator"] == 0 for a in report["by_arm"].values()))
            capture = next((root / "captures").glob("*.npz"))
            capture.write_bytes(capture.read_bytes() + b"synthetic corruption")
            with self.assertRaisesRegex(ValueError, "[Cc]apture"):
                C.run_stage(root, "progress", repo_root=self.repo)


if __name__ == "__main__":
    unittest.main()
