#!/usr/bin/env python3
"""Export three already-cached E/B activation states for further geometry plots.

Reads the existing array and frozen row assignments. No model is loaded, no
response is generated and no classifier is fitted. The output contains no
prompt or response text. Python 3.6+ syntax; NumPy is the only extra dependency.
"""
import argparse
import csv
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ARRAY_SHA = 'af89872d5decd4dc432ec579a941e6247bc7de63ff5690432e9f5812a994e2d8'
FOLDS_SHA = 'be3961a264f3435e92bfb369da7460b3903d1c6abaff6fd45daec610cd8d1d85'
STATES = [24, 32, 45]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def export_bundle(activation_path, fold_path, destination):
    if sha256(activation_path) != ARRAY_SHA:
        raise ValueError('The activation array differs from the frozen development array.')
    if sha256(fold_path) != FOLDS_SHA:
        raise ValueError('The fold assignments differ from the corrected geometry run.')
    rows = sorted([r for r in read_rows(fold_path) if r['repeat'] == '0'],
                  key=lambda r: int(r['position']))
    if len(rows) != 267 or len({r['row_id'] for r in rows}) != 267:
        raise ValueError('Expected 267 unique E/B prompts.')
    if sum(r['final_ebh'] == 'E' for r in rows) != 89 or sum(r['final_ebh'] == 'B' for r in rows) != 178:
        raise ValueError('Unexpected E/B labels.')
    indices = np.array([int(r['array_row_index']) for r in rows], dtype=np.int64)
    if len(np.unique(indices)) != 267 or indices.min() < 0 or indices.max() >= 480:
        raise ValueError('Invalid original array indices.')
    cube = np.load(str(activation_path), mmap_mode='r', allow_pickle=False)
    if cube.shape != (480, 49, 2048) or cube.dtype != np.float32:
        raise ValueError('Unexpected activation shape or dtype.')
    # The exact row-state-unit values are copied, with no centring or scaling.
    selected = np.stack([cube[indices, state, :] for state in STATES], axis=1)
    if not np.isfinite(selected).all():
        raise ValueError('Non-finite activation value.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='routegate_geometry_export_') as tmp:
        root = Path(tmp)
        array_file = root / 'eb_activations_states24_32_45.npz'
        np.savez_compressed(str(array_file), activations=selected,
                            states=np.array(STATES, dtype=np.int64),
                            array_row_indices=indices,
                            y=np.array([int(r['y_true']) for r in rows], dtype=np.int64))
        row_fields = ['row_id', 'position', 'array_row_index', 'final_ebh',
                      'y_true', 'fold', 'source_family', 'topic', 'task_frame']
        with (root / 'rows.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=row_fields)
            writer.writeheader()
            writer.writerows({k: r[k] for k in row_fields} for r in rows)
        (root / 'cv_assignments.csv').write_bytes(fold_path.read_bytes())
        metadata = {
            'schema': 'routegate/probe_geometry_small_export/1',
            'source_activation_sha256': ARRAY_SHA, 'source_folds_sha256': FOLDS_SHA,
            'source_array_shape': list(cube.shape), 'export_shape': list(selected.shape),
            'axes': ['E/B prompt', 'selected state', 'hidden coordinate'],
            'states': STATES, 'layers_zero_based': [s - 1 for s in STATES],
            'counts': {'E': 89, 'B': 178},
            'row_order': 'rows.csv; position in the original corrected E/B analysis',
            'preprocessing': 'none; original float32 activation values copied exactly',
            'selection': 'Descriptive comparison at states 24, 32 and the previously selected state 45.',
            'model_loaded': False, 'classifier_fitted': False,
            'files': {p.name: sha256(p) for p in sorted(root.iterdir()) if p.is_file()},
        }
        (root / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        # Exclusive creation protects a previous export from accidental replacement.
        with zipfile.ZipFile(str(destination), 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.iterdir()):
                archive.write(str(path), path.name)
    with zipfile.ZipFile(str(destination)) as archive:
        if archive.testzip() is not None:
            raise ValueError('Export archive failed its integrity check.')
    return selected.shape


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activation', type=Path,
                        help='Override the array path; its frozen checksum must still match.')
    parser.add_argument('--folds', type=Path,
                        default=repo / 'outputs/residual_geometry/corrected/cv_assignments.csv')
    parser.add_argument('--output', type=Path,
                        default=Path.home() / 'routegate_probe_geometry_inputs.zip')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('An export already exists: ' + str(args.output) +
                         '\nUse that file, or select a different --output path.')
    activation = args.activation
    if activation is None:
        config = repo / 'configs/paths.local.json'
        if not config.is_file():
            raise SystemExit('No configs/paths.local.json found. Supply --activation /absolute/path/to/the/cache.npy')
        paths = json.loads(config.read_text())
        extraction = Path(paths['extraction_dir']).expanduser()
        if not extraction.is_absolute():
            extraction = repo / extraction
        activation = extraction / 'residual_final_prompt_token.npy'
    if not activation.is_file() or not args.folds.is_file():
        raise SystemExit('Missing cached array or corrected fold file.\nArray: ' +
                         str(activation) + '\nFolds: ' + str(args.folds))
    shape = export_bundle(activation, args.folds, args.output)
    print('Saved ' + str(args.output))
    print('Shape: ' + str(shape) + '; ' + str(args.output.stat().st_size) + ' bytes.')
    print('Cached E/B values only. No model run or classifier fit.')


if __name__ == '__main__':
    main()
