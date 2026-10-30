#!/usr/bin/env python3
"""Measure the two steering arithmetic paths on cached development activations.

No language model is loaded. No response is generated. No API is called.
Imports and calls the existing actuator, plan builder and addition operation.
Uses the frozen 480 x 49 x 2048 development array, read-only, on the CPU.
This is a numerical diagnostic, not a sealed-set behavioural experiment.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def compare(repo, data, proposal, output, activation_path=None, actuator_dir=None, paths_file=None):
    output = output.resolve()
    require(not output.is_relative_to(data), 'Do not write this audit into the frozen data directory.')
    tracked = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        require(expected is None or actual == expected, 'File differs from the audited snapshot: ' + str(path))
        tracked[str(path)] = actual
        return path

    spec = read_json(bind(proposal))
    for rel in ('src/routegate/actuator.py', 'src/routegate/steering.py',
                'src/routegate/intervention.py', 'src/routegate/finalrunner.py', 'src/routegate/hooks.py'):
        bind(repo / rel, spec['source_hashes'][rel])
    deployment = read_json(bind(repo / 'manifests/routegate_gate_deployment_v1.json'))
    source = deployment['sources']['residual_activations']
    paths_file = paths_file or repo / 'configs/paths.local.json'
    if activation_path is None:
        require(paths_file.is_file(), 'Missing configs/paths.local.json. Supply --activation with the existing residual_final_prompt_token.npy path.')
        paths = read_json(bind(paths_file))
        require(source['paths_key'] in paths, 'The path configuration does not contain extraction_dir.')
        extraction = Path(paths[source['paths_key']]).expanduser()
        if not extraction.is_absolute():
            extraction = repo / extraction
        activation_path = extraction / source['path']
    if not activation_path.is_file():
        raise ValueError('Cached activations are absent: ' + str(activation_path) + '. Do not generate replacements; supply the existing cache path.')
    bind(activation_path, source['sha256'])
    cube = np.load(str(activation_path), mmap_mode='r', allow_pickle=False)
    require(cube.shape == (480, 49, 2048) and cube.dtype == np.float32, 'Wrong activation array shape or dtype.')
    actuator_dir = actuator_dir or data / 'frozen/rq2_actuator'
    require(actuator_dir.is_dir(), 'Frozen actuator directory absent: ' + str(actuator_dir))
    bind(repo / 'manifests/rq2_actuator.json')
    for path in sorted(actuator_dir.rglob('*')):
        if path.is_file():
            bind(path)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Run inside the existing Apptainer container, which supplies PyTorch. No GPU allocation is needed.') from exc
    torch.set_num_threads(1)
    sys.path.insert(0, str(repo / 'src'))
    from routegate.actuator import PRIMARY_VARIANT, load_frozen_actuator
    from routegate.steering import build_steering_plan
    from routegate.intervention import apply_delta_at_position

    actuator = load_frozen_actuator(PRIMARY_VARIANT, directory=actuator_dir,
                                   manifest_path=repo / 'manifests/rq2_actuator.json')
    require(tuple(actuator.layers) == (31, 32, 33), 'Unexpected intervention layers.')
    require(PRIMARY_VARIANT == 'consensus23_global_frozen', 'Wrong actuator variant.')
    rows = []
    for index in range(cube.shape[0]):
        for layer in (31, 32, 33):
            state = layer + 1
            target = torch.from_numpy(np.array(cube[index, state], dtype=np.float32, copy=True))
            reference = torch.from_numpy(np.array(cube[index, layer], dtype=np.float32, copy=True))
            require(bool(torch.isfinite(target).all() and torch.isfinite(reference).all()), 'Non-finite cached activation.')
            historical = actuator.applied_delta(layer, target, reference_vector=reference, coefficient=0.15)
            plan = build_steering_plan(state=state, direction=actuator.direction_for_state(state),
                                       strength=0.15, reference_vector=reference, budget_reference='block_input')
            require(plan.status == 'ok' and plan.delta is not None, 'The final plan produced no usable edit.')
            current = plan.delta
            hn, cn = historical.double().norm(), current.double().norm()
            require(float(hn) > 0 and float(cn) > 0, 'Zero-norm edit.')
            difference = current.double() - historical.double()
            historical_bf16 = historical.to(torch.bfloat16)
            current_bf16 = current.to(torch.bfloat16)
            # Use the exact repository addition function. Cached target is an
            # unsteered development state, not a propagated edited trajectory.
            target_bf16 = target.reshape(1, 1, -1).to(torch.bfloat16)
            old_written = apply_delta_at_position(target_bf16, 0, historical)
            new_written = apply_delta_at_position(target_bf16, 0, current)
            rows.append({
                'array_row_index': index, 'layer_zero_based': layer, 'stored_state': state,
                'historical_delta_norm': float(hn), 'final_delta_norm': float(cn),
                'delta_relative_l2_difference': float(difference.norm() / hn),
                'delta_max_absolute_difference': float(difference.abs().max()),
                'delta_cosine': float(torch.dot(historical.double(), current.double()) / (hn * cn)),
                'delta_float32_different_coordinates': int(torch.count_nonzero(historical.view(torch.int32) != current.view(torch.int32))),
                'delta_bfloat16_different_coordinates': int(torch.count_nonzero(historical_bf16.view(torch.int16) != current_bf16.view(torch.int16))),
                'written_bfloat16_different_coordinates': int(torch.count_nonzero(old_written.view(torch.int16) != new_written.view(torch.int16))),
                'written_max_absolute_difference': float((old_written.float() - new_written.float()).abs().max()),
                'target_exactly_representable_in_bfloat16': bool(torch.equal(target_bf16.reshape(-1).float(), target)),
            })
    by_layer = {}
    for layer in (31, 32, 33):
        selected = [r for r in rows if r['layer_zero_based'] == layer]
        by_layer[str(layer)] = {
            'vectors': len(selected),
            'max_delta_relative_l2_difference': max(r['delta_relative_l2_difference'] for r in selected),
            'median_delta_relative_l2_difference': float(np.median([r['delta_relative_l2_difference'] for r in selected])),
            'minimum_delta_cosine': min(r['delta_cosine'] for r in selected),
            'vectors_with_bfloat16_delta_difference': sum(r['delta_bfloat16_different_coordinates'] > 0 for r in selected),
            'vectors_with_bfloat16_written_difference': sum(r['written_bfloat16_different_coordinates'] > 0 for r in selected),
            'maximum_changed_written_coordinates': max(r['written_bfloat16_different_coordinates'] for r in selected),
            'maximum_written_absolute_difference': max(r['written_max_absolute_difference'] for r in selected),
        }
    require(len(rows) == 1440, 'Expected 480 prompts times three layers.')
    cast_same = all(r['delta_bfloat16_different_coordinates'] == 0 for r in rows)
    summary = {
        'status': 'MEASUREMENT_COMPLETE', 'measurement_population': '480 cached development prompts, all three intervention layers',
        'sealed_prompts_measured': 0, 'new_generations': 0, 'new_judge_calls': 0, 'gpu_used': False,
        'variant': PRIMARY_VARIANT, 'strength': 0.15, 'torch_version': torch.__version__,
        'arithmetic_backend': 'CPU float32; identical reference vectors and reduction backend in both paths',
        'historical_function': 'FrozenActuator.applied_delta', 'final_function': 'build_steering_plan',
        'write_function': 'apply_delta_at_position with a bfloat16 target tensor',
        'all_cached_delta_vectors_equal_after_bfloat16_cast': cast_same,
        'per_layer': by_layer,
        'interpretation': ('The two paths give identical bfloat16 deltas on all measured development states.' if cast_same
                           else 'Some measured development states receive different bfloat16 deltas. This establishes a numerical difference, not a behavioural effect.'),
        'limitations': [
            'No sealed activation vectors are present in this input. Do not claim sealed responses are unchanged by the arithmetic choice.',
            'Written-state comparisons use cached unsteered states. At later intervention layers they are not a simulation of upstream edit propagation.',
            'CPU reductions are used in both paths to isolate arithmetic order. This is not a historical GPU reproduction.',
            'No tolerance is selected after seeing results. This script makes no pass/fail decision about scientific acceptability.',
            'Any numerical difference must be interpreted before proposing new generations. Existing outputs and frozen code remain unchanged.',
        ],
        'frozen_files_changed': 0,
    }
    for path, expected in tracked.items():
        require(sha(path) == expected, 'A source file changed during the comparison: ' + path)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'per_vector_arithmetic.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump(output / 'arithmetic_summary.json', summary)
    dump(output / 'arithmetic_input_hashes.json', tracked)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--data-root', type=Path, default=Path.home() / 'routegate-data')
    parser.add_argument('--proposal', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--activation', type=Path, help='Existing cache only; its frozen checksum must match.')
    parser.add_argument('--actuator-dir', type=Path)
    parser.add_argument('--paths-file', type=Path)
    args = parser.parse_args()
    repo, data = args.repo.resolve(), args.data_root.resolve()
    proposal = args.proposal or repo / 'docs/final_evaluation/consistency_review_v1.json'
    output = args.output or repo / 'outputs/thesis_results/consistency_review_v1/arithmetic'
    result = compare(repo, data, proposal, output, args.activation, args.actuator_dir, args.paths_file)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise SystemExit('Arithmetic check stopped: ' + str(exc))
