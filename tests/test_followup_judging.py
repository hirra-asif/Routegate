"""Offline synthetic tests: no language-model generation or paid judging."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from routegate import followupjudging as F
from routegate import judgingprotocol as P
from routegate import judgingstages as S
from routegate.validation import ValidationError

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ROUTEGATE_TEST_DATA", "/workspace/scratch/ec6ae4ab5e10/research_audit_20260910/routegate-data"))


class FollowupJudgingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="followup_synthetic_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = F.frozen_settings(REPO)
        # Synthetic text and schemas retain the engine contract without depending
        # on any uploaded prompt or on the location of private rubric files.
        self.rubric_dir = self.root / "judge_rubrics"
        self.rubric_dir.mkdir()
        for rubric, spec in self.settings["rubrics"].items():
            fields = {"item_id": {"type": "string"}}
            if rubric == "safety":
                fields["actionable_harmful_help"] = {"type": "enum", "values": ["YES", "NO", "UNCLEAR"]}
            elif rubric == "axis2":
                fields["response_behaviour"] = {"type": "enum", "values": spec["values"]}
            else:
                fields.update(preference={"type": "enum", "values": ["A_BETTER", "B_BETTER", "EQUIVALENT"]}, difference={"type": "enum", "values": ["MATERIAL", "MINOR", "NONE"]})
            template = '{{"item_id":"{item_id}"}}\n{prompt_text}\n' + ("{response_a}\n{response_b}\n" if spec["pairwise"] else "{response_text}\n")
            texts = {"instructions": "SYNTHETIC " + rubric, "user_template": template, "schema": json.dumps({"name": spec["schema_name"], "fields": fields})}
            for part, content in texts.items():
                name = rubric + "_" + part + (".json" if part == "schema" else ".txt")
                (self.rubric_dir / name).write_text(content)
                spec[part + "_sha256"] = F.digest(content.encode())
        self.rows = []
        for klass, count in (("E", 89), ("B", 178), ("H", 213)):
            for number in range(count):
                # Repeat one prompt deliberately: dedup is by material, whereas
                # scientific row IDs and E/B/H grouping still remain distinct.
                prompt = "SYNTHETIC shared prompt"
                self.rows.append({"row_id": "fixture_{}_{}".format(klass, number), "row_class": klass, "prompt_text": prompt, "prompt_sha256": F.digest(prompt.encode()), "source_family": "fixture", "fold_id": number % 5})
        self.plan = {"study_sha256": "synthetic_inventory", "inventory_sha256": "synthetic_inventory", "plan_sha256": "synthetic_plan", "arm_ids": ["baseline", "candidate_a", "candidate_b"], "candidate_arms": ["candidate_a", "candidate_b"], "generation_settings": {"max_new_tokens": 768}, "generation_settings_sha256": "synthetic_generation", "arm_settings_sha256": {a: "synthetic_" + a for a in ("baseline", "candidate_a", "candidate_b")}, "judging": self.settings, "rubric_files": {p.name: F.digest(p.read_bytes()) for p in self.rubric_dir.iterdir()}, "primary_comparisons": [{"name": "a_vs_baseline", "reference_arm": "baseline", "candidate_arm": "candidate_a"}]}
        self.records = []
        for row in self.rows:
            for arm in self.plan["arm_ids"]:
                record = {**row, "arm_id": arm, "status": "ok", "study_sha256": self.plan["study_sha256"], "plan_sha256": self.plan["plan_sha256"], "generation_settings_sha256": self.plan["generation_settings_sha256"], "arm_settings_sha256": self.plan["arm_settings_sha256"][arm], "response_text": "SYNTHETIC answer", "response_sha256": F.digest(b"SYNTHETIC answer"), "hit_max_new_tokens": False, "eos_reached": True, "completion_token_count": 5}
                record["record_sha256"] = F.digest(F._json(record))
                self.records.append(record)
        (self.root / "generations").mkdir()
        self.save_records()
        self.patch = patch.object(F, "verify_plan", side_effect=lambda *_: (self.plan, self.rows))
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.env = patch.dict(os.environ, {S.ENV_ENDPOINT: "https://synthetic.invalid", S.ENV_API_KEY_HISTORICAL: "fixture-not-a-real-key"})
        self.env.start(); self.addCleanup(self.env.stop)
        self.requests = {"sonnet": [], "opus": []}

    def save_records(self):
        (self.root / "generations/records.jsonl").write_bytes(F._jsonl(self.records))

    def stage(self, stage, **kwargs):
        return F.run_stage(self.root, stage, **kwargs)

    def packet(self):
        self.stage("prepare"); self.stage("build")
        return self.root / F.OUTPUT_NAME

    def transport(self, judge, fail=False, conflict=False):
        def call(endpoint, key, model, system, user, manifest):
            self.requests[judge].append((model, system, user))
            opaque = json.loads(user.splitlines()[0])["item_id"]
            if fail and system.endswith("safety"):
                return {"http_status": 503, "error": "synthetic failure", "payload": None, "raw_body": "fixture", "elapsed_seconds": 0}
            answer = {"item_id": opaque}
            if system.endswith("safety"):
                answer["actionable_harmful_help"] = "YES" if conflict else "NO"
            elif system.endswith("axis2"):
                answer["response_behaviour"] = "SA"
            else:
                answer.update(preference="EQUIVALENT", difference="NONE")
            payload = {"model": manifest.judges[judge]["expected_returned"], "content": [{"type": "text", "text": json.dumps(answer)}]}
            return {"http_status": 200, "payload": payload, "raw_body": json.dumps(payload), "elapsed_seconds": 0}
        return call

    def test_full_frame_dedup_blindness_and_dry_run(self):
        result = self.stage("prepare", dry_run=True)
        self.assertEqual(result["scientific_cells"], 1440)
        self.assertEqual(result["unique_materials"], 3)
        self.assertEqual(result["planned_calls"], 6)
        self.assertFalse((self.root / F.OUTPUT_NAME).exists())
        root = self.packet()
        self.assertTrue(self.stage("build")["existing_packet_preserved"])
        for rubric in F.RUBRICS:
            rows = S.load_judge_rows(root / "judging/packet", rubric)
            self.assertEqual(len(rows), 1)
            self.assertNotIn("arm_id", rows[0])
            self.assertNotIn("row_class", rows[0])
        self.assertEqual(self.stage("dryrun")["calls"], 6)

    def test_both_judges_receive_same_material_and_expansion_is_consistent(self):
        root = self.packet()
        for judge in ("sonnet", "opus"):
            result = self.stage("judge", judge=judge, transport=self.transport(judge))
            self.assertEqual(result["succeeded"], 3)
            again = self.stage("judge", judge=judge, transport=lambda *_: self.fail("Successful call repeated"))
            self.assertEqual(again["attempted"], 0)
        self.assertEqual([r[1:] for r in self.requests["sonnet"]], [r[1:] for r in self.requests["opus"]])
        self.stage("join"); self.stage("export"); self.stage("adjudicate"); self.stage("report")
        original = (root / "judging/adjudicated/adjudication_summary.json").read_bytes()
        self.assertTrue(self.stage("adjudicate")["existing_adjudication_preserved"])
        self.assertEqual((root / "judging/adjudicated/adjudication_summary.json").read_bytes(), original)
        manifest, cells = F.verify_scoring_inputs(root)
        self.assertEqual(len(cells), 1440)
        self.assertEqual(manifest["inventory_sha256"], self.plan["study_sha256"])
        self.assertEqual(sum(r["rubric"] == "axis2" for r in cells), 445)
        self.assertEqual(sum(r["rubric"] == "pairwise_quality" for r in cells), 356)
        self.assertEqual(sum(r["rubric"] == "safety" for r in cells), 639)
        self.assertEqual(sum(r["verdict"] is None for r in cells), 0)
        self.assertTrue(all(r["verdict"] == "EQUIVALENT|NONE" for r in cells if r["rubric"] == "pairwise_quality"))
        self.assertEqual(len({r["verdict"] for r in cells if r["rubric"] == "axis2"}), 1)
        from routegate import followupscoring as scoring, followupinputs
        with patch.object(followupinputs, "verify", return_value=({"inventory_sha256": self.plan["inventory_sha256"]}, self.rows)):
            report = scoring.build_report(self.root, root, self.root / "scoring")
        self.assertTrue((self.root / "scoring/report.json").is_file())


    def test_partial_duplicate_changed_hash_and_cap_metadata_refused(self):
        original = copy.deepcopy(self.records)
        variants = [original[:-1], original[:-1] + [original[0]], [{**original[0], "response_text": "changed"}] + original[1:], [{**original[0], "plan_sha256": "other"}] + original[1:], [{**original[0], "hit_max_new_tokens": True}] + original[1:]]
        for records in variants:
            with self.subTest():
                self.records = records; self.save_records()
                with self.assertRaises(ValidationError):
                    self.stage("prepare")
        self.assertFalse((self.root / F.OUTPUT_NAME).exists())

    def test_deliberate_retry_reuses_successes(self):
        self.packet()
        first = self.stage("judge", judge="opus", transport=self.transport("opus", fail=True))
        self.assertEqual(first["failed"], 1)
        with self.assertRaisesRegex(ValidationError, "retry-failures"):
            self.stage("judge", judge="opus", transport=self.transport("opus"))
        retry = self.stage("judge", judge="opus", retry_failures=True, transport=self.transport("opus"))
        self.assertEqual(retry["succeeded"], 1)
        self.assertEqual(retry["skipped"], 2)

    def test_eos_at_token_cap_retains_both_flags(self):
        record = self.records[0]
        record.update(completion_token_count=768, eos_reached=True, hit_max_new_tokens=True)
        record["record_sha256"] = F.digest(F._json({k: v for k, v in record.items() if k != "record_sha256"}))
        self.save_records()
        self.assertEqual(self.stage("prepare", dry_run=True)["scientific_cells"], 1440)

    def test_generation_record_helper_binds_plan_and_preserves_diagnostics(self):
        record = F.generation_record(self.plan, self.rows[0], "candidate_a", "Actual fixture output", 768, True, True, {"forward_passes": 2})
        self.assertEqual(record["metadata"]["forward_passes"], 2)
        self.assertEqual(record["record_sha256"], F.digest(F._json({k: v for k, v in record.items() if k != "record_sha256"})))
        with self.assertRaises(ValidationError):
            F.generation_record(self.plan, self.rows[0], "undeclared", "text", 1, True, False)

    def test_technical_failure_remains_missing_when_expanded(self):
        root = self.packet()
        self.stage("judge", judge="sonnet", transport=self.transport("sonnet"))
        self.stage("judge", judge="opus", transport=self.transport("opus", fail=True))
        opaque = S.load_judge_rows(root / "judging/packet", "safety")[0]["row_id"]
        carried = self.root / "technical.json"
        carried.write_text(json.dumps({"unresolved": [{"row_id": opaque, "failed_judge": "opus", "failure_type": "synthetic_transport"}]}))
        self.stage("join", carried=carried); self.stage("export"); self.stage("adjudicate")
        summary = self.stage("report")
        self.assertEqual(summary["technically_unresolved_cells"], 639)
        _, cells = F.verify_scoring_inputs(root)
        self.assertTrue(all(r["verdict"] is None for r in cells if r["rubric"] == "safety"))

    def test_review_requires_and_preserves_ai_assisted_provenance(self):
        root = self.packet()
        self.stage("judge", judge="sonnet", transport=self.transport("sonnet"))
        self.stage("judge", judge="opus", transport=self.transport("opus", conflict=True))
        self.stage("join"); self.stage("export")
        with self.assertRaisesRegex(ValidationError, "provenance"):
            self.stage("adjudicate")
        mapping = F._rows(root / "judging/review/private" / P.REVIEW_MAP_FILE)
        self.assertEqual(len(mapping), 1)
        completed = self.root / "fixture_completed.jsonl"
        completed.write_bytes(F._jsonl([{"row_id": mapping[0]["row_id"], "actionable_harmful_help": "UNCLEAR"}]))
        provenance = self.root / "fixture_provenance.json"
        provenance.write_text(json.dumps({"schema": "routegate/development_followup_review_provenance/1", "reviewer": "Synthetic fixture, no human reviewer", "ai_assisted": True, "blinded": False, "completed_sha256": F.digest(completed.read_bytes())}))
        self.stage("adjudicate", completed=completed, reviewer_provenance=provenance); self.stage("report")
        self.assertTrue(F._read(root / "review_provenance.json")["ai_assisted"])
        _, cells = F.verify_scoring_inputs(root)
        self.assertTrue(all(r["verdict"] == "UNCLEAR" and not r["technically_unresolved"] for r in cells if r["rubric"] == "safety"))

    def test_binding_is_prospective_explicit_and_keeps_carried_unknowns(self):
        root = self.root / "future_inventory"
        root.mkdir()
        report = self.root / "diagnostic/assessment_v1/scientific_report.json"
        report.parent.mkdir(parents=True)
        diagnostic = {"study_sha256": "diagnostic_fixture", "by_arm": {"baseline": {"technical_missing": 1, "unresolved_review": 0, "UNCLEAR": 2}}}
        report.write_bytes(F._json(diagnostic))
        source = "src/routegate/runtime.py"
        request = {"candidate_arms": [{"arm_id": "explicit_candidate", "description": "Synthetic plan only", "settings": {"coefficient": 0.1}}],
                   "generation_settings": {"model": "fixture", "revision": "fixture", "dtype": "float32", "max_new_tokens": 768, "do_sample": False, "enable_thinking": False},
                   "generation_code_hashes": {source: F.digest((REPO / source).read_bytes())},
                   "primary_comparisons": [{"name": "candidate_vs_baseline", "reference_arm": "baseline", "candidate_arm": "explicit_candidate"}],
                   "diagnostic_report": str(report), "diagnostic_interpretation": "One carried unknown is retained; this is a synthetic prospective plan."}
        path = self.root / "request.json"
        path.write_bytes(F._json(request))
        self.patch.stop()
        with patch.object(F, "_inventory", return_value=({"inventory_sha256": "fixture_inventory"}, self.rows)), patch.object(F.C, "run_stage", return_value=diagnostic), patch.object(F, "frozen_settings", return_value=self.settings), patch.object(F, "_rubric_files", return_value={p.name: p.read_bytes() for p in self.rubric_dir.iterdir()}):
            plan = F.bind_plan(root, path, repo_root=REPO, dry_run=True)
            self.assertFalse((root / F.PLAN_FILE).exists())
            self.assertEqual(plan["candidate_arms"], ["explicit_candidate"])
            self.assertEqual(plan["diagnostic_report"]["technical_missing_by_arm"], {"baseline": 1})
            bound = F.bind_plan(root, path, repo_root=REPO)
            actual, _ = F.verify_plan(root, REPO)
            self.assertEqual(actual, bound)
            # Existing outputs cannot be retroactively attached to a new plan.
            (root / "generations").mkdir()
            (root / "generations/records.jsonl").write_text("")
            with self.assertRaisesRegex(ValidationError, "before any generation"):
                F.bind_plan(root, path, repo_root=REPO)

    def test_pairwise_orientation_tampering_is_refused(self):
        root = self.packet()
        path = root / "judging/packet/private" / P.ROW_MAP_FILE
        rows = F._rows(path)
        pair = next(r for r in rows if r["rubric"] == "pairwise_quality")
        pair["response_a_origin"] = pair["response_b_origin"]
        path.write_bytes(F._jsonl(rows))
        with self.assertRaisesRegex(ValidationError, "orientation"):
            self.stage("judge", judge="sonnet", dry_run=True)

    def test_scoring_rejects_modified_expanded_labels(self):
        root = self.packet()
        for judge in ("sonnet", "opus"):
            self.stage("judge", judge=judge, transport=self.transport(judge))
        self.stage("join"); self.stage("export"); self.stage("adjudicate"); self.stage("report")
        path = root / "scientific_adjudicated_cells.jsonl"
        rows = F._rows(path); rows[0]["verdict"] = "DR"
        path.write_bytes(F._jsonl(rows))
        with self.assertRaisesRegex(ValidationError, "Expanded verdict"):
            F.verify_scoring_inputs(root)


if __name__ == "__main__":
    unittest.main()
