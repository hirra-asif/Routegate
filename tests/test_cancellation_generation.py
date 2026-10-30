"""Cancellation runner integrity and optional real-Torch CPU hook tests.

Uses unittest rather than pytest. The Torch tests must run on Myriad before
model generation when Torch is unavailable in the local preparation runtime.
No language-model checkpoint, API, historical judgement or GPU is required.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


def toy_arrays():
    """Three distinct 4D direction pairs, retaining an orthogonal control axis."""
    import numpy as np
    repair = np.repeat(np.array([[1., 0., 0., 0.]], dtype=np.float32), 3, axis=0)
    safety = np.array([[-.8, .6, 0., 0.], [-.6, .8, 0., 0.],
                       [-.4, 0., np.sqrt(.84), 0.]], dtype=np.float32)
    q = safety[:, 0].astype(np.float64)
    perp = safety.copy()
    perp[:, 0] = 0.
    random = np.zeros((3, 3, 4), dtype=np.float32)
    for draw in range(3):
        random[draw, :, 3] = np.linalg.norm(perp, axis=1)
    return {'repair': repair, 'safety': safety, 'q': q,
            'safety_perp': perp, 'random_orthogonal': random,
            'zero': np.zeros_like(repair)}


if torch is not None:
    class ToyBlock(torch.nn.Module):
        """Simple varying affine blocks expose changed downstream budgets."""
        def __init__(self, index):
            super().__init__()
            self.register_buffer('bias', torch.tensor([
                .003*(index+1), .005*(index+1), .007*(index+1), .011*(index+1)]))
            self.factor = 1.0 + .002*(index % 3)

        def forward(self, hidden_states, **unused):
            return (hidden_states*self.factor + self.bias,)

    class ToyBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(16, 4)
            self.layers = torch.nn.ModuleList(ToyBlock(i) for i in range(35))
            with torch.no_grad():
                self.embed_tokens.weight.copy_(
                    torch.arange(64, dtype=torch.float32).reshape(16, 4)/32)

        def forward(self, input_ids):
            hidden = self.embed_tokens(input_ids)
            for layer in self.layers:
                hidden = layer(hidden)[0]
            return hidden

    class ToyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = ToyBase()

        def forward(self, input_ids, **unused):
            return self.model(input_ids)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def synthetic_success(study_root, manifest, row, arm_id='baseline', **changes):
    """Write test-only provenance and return an explicitly synthetic success.

    Call only inside a TemporaryDirectory: this is an integrity/judging fixture,
    never evidence of an actual model run. It creates valid NPZ capture bytes but
    does not invoke a model, tokenizer or judge.
    """
    import io
    import numpy as np
    from routegate import cancellationexperiment as experiment
    root = Path(study_root)
    environment = {'python': 'synthetic-test', 'torch': 'synthetic-test',
                   'transformers': 'synthetic-test', 'numpy': np.__version__,
                   'cuda': None, 'gpu_names': [], 'matmul_allow_tf32': False,
                   'cudnn_allow_tf32': False, 'deterministic_algorithms': False,
                   'seed': 0, 'note': 'SYNTHETIC TEST FIXTURE; no model execution'}
    environment_path = root/'execution_environment.json'
    if not environment_path.exists():
        experiment.write_json(environment_path, environment)
    environment_hash = experiment.digest(environment)
    capture_path = root/'captures'/(digest(row['row_id'])+'.npz')
    if not capture_path.exists():
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **{str(state): np.ones(2048, dtype=np.float32)
                                      for state in (31, 32, 33, 34)})
        experiment.atomic_bytes(capture_path, buffer.getvalue())
    capture_hash = hashlib.sha256(capture_path.read_bytes()).hexdigest()
    metadata_path = capture_path.with_suffix('.json')
    if not metadata_path.exists():
        experiment.write_json(metadata_path, {
            'row_id': row['row_id'], 'study_sha256': manifest['study_sha256'],
            'prompt_sha256': row['prompt_sha256'], 'execution_sha256': environment_hash,
            'sha256': capture_hash, 'states': [31, 32, 33, 34], 'generated_tokens': 0,
            'prompt_token_count': 3, 'created_utc': '2000-01-01T00:00:00+00:00'})
    planned = arm_id != 'baseline'
    delta = .1 if planned else 0.
    diagnostics = {}
    for state in (32, 33, 34):
        metrics = {'l2': delta, 'repair_coefficient': delta, 'perpendicular_l2': 0.}
        diagnostics[str(state)] = {
            'layer': state-1, 'output_state': state, 'budget_reference_state': state-1,
            'frozen_unsteered_input_norm': float(np.sqrt(2048)),
            'live_dtype': 'torch.bfloat16', 'live_before_l2': 1.,
            'live_after_l2': 1.+delta, 'intended_delta': metrics.copy(),
            'cast_delta': metrics.copy(), 'realised_delta': metrics.copy(),
            'changed_coordinates': int(planned), 'edit_planned': planned,
            'plan_status': 'ok' if planned else 'no_intervention',
            'planned_edit_rounded_to_identity': False}
    response = 'SYNTHETIC TEST FIXTURE: no model response was generated.'
    record = {**row, 'arm_id': arm_id, 'study_sha256': manifest['study_sha256'],
              'execution_sha256': environment_hash, 'capture_sha256': capture_hash,
              'generation_fingerprint': manifest['generation_fingerprint'],
              'response_text': response, 'response_sha256': digest(response),
              'prompt_token_count': 3, 'completion_token_count': 5,
              'status': 'ok', 'error_message': '',
              'hooks_expected': 3 if planned else 0, 'hooks_fired': 3 if planned else 0,
              'layer_diagnostics': diagnostics, 'per_state': {}}
    record.update(changes)
    record['record_sha256'] = changes.get('record_sha256', experiment.digest({
        key: value for key, value in record.items() if key != 'record_sha256'}))
    return record


@unittest.skipUnless(torch is not None, 'Torch absent; run CPU toy hook tests on Myriad before generation')
class CancellationHookTests(unittest.TestCase):
    def setUp(self):
        from routegate import cancellationexperiment as experiment
        self.experiment = experiment
        self.arrays = toy_arrays()
        self.model = ToyCausalLM().eval().requires_grad_(False)
        self.ids = torch.tensor([[1, 5, 9]], dtype=torch.long)

    def capture(self, controller):
        controller.reset_run(mode='capture', example_id='toy', capture_name='toy')
        self.model(self.ids)
        return dict(controller.capture_for('toy'))

    def test_three_sites_use_unsteered_predecessor_budgets(self):
        from routegate.cancellationcontrols import STATES, BUDGET_REFERENCE_STATES
        controller = self.experiment.make_controller(self.model, self.arrays)
        try:
            capture = self.capture(controller)
            self.assertEqual(sorted(capture), [31, 32, 33, 34])
            frozen = {state: value.clone() for state, value in capture.items()}
            plans = self.experiment.build_plans(self.arrays, 'original_combination', capture)
            self.assertEqual(sorted(plans), list(STATES))
            expected_norms = {state: float(torch.linalg.vector_norm(
                frozen[state-1].float()).item()) for state in STATES}
            controller.reset_run(mode='patch', example_id='toy', plans=plans)
            self.model(self.ids)
            controller.verify_applied()
            for state, reference_state in zip(STATES, BUDGET_REFERENCE_STATES):
                self.assertEqual(plans[state].budget_reference_state, reference_state)
                self.assertEqual(plans[state].reference_norm, expected_norms[state])
                self.assertTrue(torch.equal(capture[reference_state], frozen[reference_state]))
            self.assertEqual(dict(controller.run.hits), {state: 1 for state in STATES})
        finally:
            controller.close()

    def test_only_final_prompt_token_and_prefill_are_edited(self):
        controller = self.experiment.make_controller(self.model, self.arrays)
        try:
            controller.reset_run(mode='off')
            original = self.model(self.ids).detach().clone()
            decode_ids = torch.tensor([[2]], dtype=torch.long)
            untouched_decode = self.model(decode_ids).detach().clone()
            capture = self.capture(controller)
            plans = self.experiment.build_plans(self.arrays, 'repair_only', capture)
            controller.reset_run(mode='patch', example_id='toy', plans=plans)
            edited = self.model(self.ids)
            self.assertTrue(torch.equal(original[:, :-1], edited[:, :-1]))
            self.assertFalse(torch.equal(original[:, -1], edited[:, -1]))
            # A second top-level forward is decode, even with sequence length 1.
            decoded = self.model(decode_ids)
            self.assertTrue(torch.equal(untouched_decode, decoded))
            self.assertEqual(dict(controller.run.hits), {32: 1, 33: 1, 34: 1})
            controller.verify_applied()
            controller.reset_run(mode='off')
            self.assertTrue(torch.equal(original, self.model(self.ids)))
            self.assertEqual(dict(controller.run.hits), {})
        finally:
            controller.close()
        self.assertFalse(controller.installed)

    def test_baseline_has_no_write_but_quantised_positive_edit_has_a_hit(self):
        self.model = self.model.to(dtype=torch.bfloat16)
        controller = self.experiment.make_controller(self.model, self.arrays)
        try:
            capture = self.capture(controller)
            controller.reset_run(mode='off')
            original = self.model(self.ids).detach().clone()
            baseline = self.experiment.build_plans(self.arrays, 'baseline', capture)
            self.assertTrue(all(not plan.applies_anything for plan in baseline.values()))
            controller.reset_run(mode='patch', example_id='toy', plans=baseline)
            self.assertTrue(torch.equal(original, self.model(self.ids)))
            controller.verify_applied()
            self.assertEqual(sum(controller.run.hits.values()), 0)
            # A tiny but positive intended edit should reach all hooks even if
            # bfloat16 addition absorbs every component into the hidden state.
            tiny_capture = {state: vector.clone() for state, vector in capture.items()}
            for state in (31, 32, 33):
                tiny_capture[state] = torch.full_like(tiny_capture[state], 1e-7)
            tiny = self.experiment.build_plans(self.arrays, 'repair_only', tiny_capture)
            self.assertTrue(all(plan.applies_anything for plan in tiny.values()))
            self.assertTrue(all(plan.applied_delta_norm > 0 for plan in tiny.values()))
            controller.reset_run(mode='patch', example_id='toy', plans=tiny)
            realised = self.model(self.ids)
            self.assertTrue(torch.equal(original, realised))
            controller.verify_applied()
            self.assertEqual(dict(controller.run.hits), {32: 1, 33: 1, 34: 1})
        finally:
            controller.close()


class CancellationPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        from routegate import cancellationexperiment as experiment
        from routegate.cancellationcontrols import load_selected_rows
        cls.experiment = experiment
        cls.repo = Path(__file__).resolve().parents[1]
        candidates = [Path(os.environ['CANCELLATION_TEST_DATA_ROOT'])] \
            if os.environ.get('CANCELLATION_TEST_DATA_ROOT') else [
                cls.repo.parent/'routegate-data',
                cls.repo.parent.parent/'research_audit_20260910'/'routegate-data']
        cls.data = next((p for p in candidates
                         if (p/'frozen/corpus/rq3_dataset_final.csv').is_file()), None)
        if cls.data is None:
            raise unittest.SkipTest('Frozen test inputs absent; set CANCELLATION_TEST_DATA_ROOT')
        cls.row = load_selected_rows(cls.repo, cls.data)[0]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='cancellation-runner-test-')
        self.addCleanup(self.temporary.cleanup)
        self.study = Path(self.temporary.name)/'study'
        self.manifest = self.experiment.init_study(self.repo, self.data, self.study)

    def success(self, **changes):
        return synthetic_success(self.study, self.manifest, self.row, **changes)

    def test_verified_empty_study_reopens_and_detects_success(self):
        self.assertEqual(self.experiment.verify_study(self.study, repo_root=self.repo), self.manifest)
        self.assertEqual(self.experiment.load_records(self.study), {})
        record = self.success()
        self.experiment.save_success(self.study, record)
        loaded = self.experiment.load_records(self.study)
        self.assertEqual(loaded[(record['row_id'], record['arm_id'])], record)

    def test_duplicate_success_is_rejected_even_when_identical(self):
        record = self.success()
        self.experiment.save_success(self.study, record)
        with self.assertRaises(ValueError):
            self.experiment.save_success(self.study, copy.deepcopy(record))
        self.assertEqual(len(self.experiment.load_records(self.study)), 1)

    def test_failed_attempt_does_not_satisfy_resume_and_can_be_retried(self):
        failed = self.success(status='error', error_message='Synthetic transient failure')
        self.experiment.record_attempt(self.study, failed)
        self.assertEqual(self.experiment.load_records(self.study), {})
        retried = self.success()
        self.experiment.save_success(self.study, retried)
        self.assertEqual(len(self.experiment.load_records(self.study)), 1)
        with self.assertRaises(ValueError):
            self.experiment.save_success(self.study, failed)

    def test_success_rejects_wrong_study_or_response_hash_or_key(self):
        for changes in [
            {'study_sha256': '0'*64},
            {'response_sha256': '0'*64},
            {'prompt_sha256': '0'*64},
            {'row_id': 'not-a-selected-row'},
            {'arm_id': 'undeclared_arm'},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.experiment.save_success(self.study, self.success(**changes))
        self.assertEqual(self.experiment.load_records(self.study), {})

    def test_provenance_fields_and_record_digest_are_required(self):
        for field in ('record_sha256', 'execution_sha256', 'capture_sha256',
                      'generation_fingerprint', 'layer_diagnostics'):
            record = self.success()
            record.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.experiment.save_success(self.study, record)
        for changes in ({'record_sha256': '0'*64},
                        {'generation_fingerprint': '0'*64},
                        {'execution_sha256': 'not-a-digest'},
                        {'capture_sha256': 'f'*63},
                        {'hooks_fired': 3}, {'hooks_expected': 3}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.experiment.save_success(self.study, self.success(**changes))
        self.assertEqual(self.experiment.load_records(self.study), {})

    def test_layer_mapping_and_planned_edit_checks_reject_rehashed_records(self):
        for field, value in [('layer', 32), ('budget_reference_state', 32),
                             ('edit_planned', True), ('plan_status', 'ok'),
                             ('live_dtype', 'torch.float32'),
                             ('changed_coordinates', 1), ('live_before_l2', -1.),
                             ('planned_edit_rounded_to_identity', True)]:
            diagnostic = self.success()['layer_diagnostics']
            diagnostic['32'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.experiment.save_success(self.study, self.success(layer_diagnostics=diagnostic))
        diagnostic = self.success()['layer_diagnostics']
        diagnostic['32']['realised_delta']['l2'] = .1
        with self.assertRaisesRegex(ValueError, 'Baseline'):
            self.experiment.save_success(self.study, self.success(layer_diagnostics=diagnostic))

    def test_nonbaseline_allows_realised_zero_when_positive_edit_was_planned(self):
        record = self.success(arm_id='repair_only')
        for diagnostic in record['layer_diagnostics'].values():
            diagnostic['changed_coordinates'] = 0
            diagnostic['live_after_l2'] = diagnostic['live_before_l2']
            diagnostic['realised_delta'] = {'l2': 0., 'repair_coefficient': 0., 'perpendicular_l2': 0.}
            diagnostic['planned_edit_rounded_to_identity'] = True
        record['record_sha256'] = self.experiment.digest({
            key: value for key, value in record.items() if key != 'record_sha256'})
        self.experiment.save_success(self.study, record)
        self.assertEqual(len(self.experiment.load_records(self.study)), 1)

    def test_capture_and_environment_bytes_are_checked_on_save_and_resume(self):
        record = self.success()
        capture = self.study/'captures'/(digest(self.row['row_id'])+'.npz')
        capture_bytes = capture.read_bytes()
        capture.unlink()
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.experiment.save_success(self.study, record)
        self.assertFalse((self.study/'generations/records.jsonl').exists())
        capture.write_bytes(capture_bytes)
        self.experiment.save_success(self.study, record)
        paths = (capture, capture.with_suffix('.json'), self.study/'execution_environment.json')
        for path in paths:
            original = path.read_bytes()
            try:
                if path == capture:
                    path.write_bytes(original+b'tampered')
                else:
                    value = json.loads(original)
                    value['row_id' if path == capture.with_suffix('.json') else 'torch'] = 'changed'
                    self.experiment.write_json(path, value)
                with self.subTest(path=path.name), self.assertRaises(ValueError):
                    self.experiment.load_records(self.study)
            finally:
                path.write_bytes(original)
        self.assertEqual(len(self.experiment.load_records(self.study)), 1)

    def test_capture_metadata_binds_prompt_study_execution_and_state_inventory(self):
        record = self.success()
        path = self.study/'captures'/(digest(self.row['row_id'])+'.json')
        metadata = json.loads(path.read_text())
        for field, value in [('study_sha256', '0'*64), ('prompt_sha256', '0'*64),
                             ('execution_sha256', '0'*64), ('sha256', '0'*64),
                             ('states', [32, 33, 34]), ('generated_tokens', 1),
                             ('prompt_token_count', 4)]:
            self.experiment.write_json(path, {**metadata, field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.experiment.save_success(self.study, record)
        self.experiment.write_json(path, metadata)
        self.assertEqual(self.experiment.load_records(self.study), {})

    def test_changed_model_revision_is_rejected_after_manifest_rehash(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['generation']['revision'] = 'a-different-model-revision'
        manifest['study_sha256'] = self.experiment.digest({
            key: value for key, value in manifest.items() if key != 'study_sha256'})
        self.experiment.write_json(self.study/'study_manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'Generation settings'):
            self.experiment.verify_study(self.study, repo_root=self.repo)

    def test_budget_declaration_tamper_changes_manifest_identity(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['design']['budget'] = 'use a fresh live budget after every previous edit'
        self.experiment.write_json(self.study/'study_manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'manifest hash'):
            self.experiment.verify_study(self.study, repo_root=self.repo)

    def test_disk_duplicates_and_torn_json_are_refused(self):
        record = self.success()
        self.experiment.save_success(self.study, record)
        path = self.study/'generations/records.jsonl'
        self.experiment.write_jsonl(path, [record, record])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.experiment.load_records(self.study)
        self.experiment.write_jsonl(path, [record])
        with path.open('a', encoding='utf-8') as handle:
            handle.write('{"row_id":')
        with self.assertRaisesRegex(ValueError, 'Invalid JSONL'):
            self.experiment.load_records(self.study)

    def test_a_second_writer_cannot_claim_the_same_study(self):
        with self.experiment.study_lock(self.study):
            with self.assertRaisesRegex(ValueError, 'Another generation process'):
                with self.experiment.study_lock(self.study):
                    self.fail('second writer acquired an already-held lock')
        with self.experiment.study_lock(self.study):
            pass

    def test_init_refuses_nonempty_output_without_altering_its_contents(self):
        other = Path(self.temporary.name)/'occupied'
        other.mkdir()
        sentinel = other/'user-file.txt'
        sentinel.write_text('keep exactly\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.experiment.init_study(self.repo, self.data, other)
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'keep exactly\n')
        self.assertEqual(list(other.iterdir()), [sentinel])


if __name__ == '__main__':
    unittest.main()
