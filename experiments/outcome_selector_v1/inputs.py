"""Load only the hash-bound development audit and original activation frame."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from selector_core import favourable


PINS = {
    'cases': '319181ac55caecc3550722cfe83384d46d921399df0465ba513a5dab66d3f6c1',
    'folds': 'f654c0bb6c1ccfcdaf924960ee92f67921c645169083fb586681f157972a2cd7',
    'development_manifest': '9d97bb2f89785e3edd6b5f982a8cb7eb6fbfdfef8f1436cb305342c224f599ca',
    'extraction_manifest': 'a3b7444b1a2782b5bc04142f3545ac5d71ad57a5f546edc315d7d51d14947abd',
    'activation': 'af89872d5decd4dc432ec579a941e6247bc7de63ff5690432e9f5812a994e2d8'
}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def development_path(path):
    path = Path(path).resolve()
    if any(s in str(path).lower() for s in ('sealed', 'final_evaluation', 'final_benchmark')):
        raise ValueError('External evaluation path is not an input to this study: ' + str(path))
    return path


def verified_file(path, expected, provenance):
    path = development_path(path)
    if not path.is_file():
        raise ValueError('Missing existing input: ' + str(path))
    actual = sha256(path)
    if actual != expected:
        raise ValueError('Input digest differs from the reviewed development version: ' + str(path))
    provenance[str(path)] = {'sha256': actual, 'bytes': path.stat().st_size}
    return path


def align_rows(rows, extraction, fold_map, expected_n=480):
    by_id = {}
    for row in rows:
        rid = row['row_id']
        if rid in by_id:
            raise ValueError('Duplicate audit row ID')
        by_id[rid] = row
    if len(by_id) != expected_n or set(by_id) != set(fold_map) or len(extraction) != expected_n:
        raise ValueError('Audit, folds and activation inventory differ')
    positions = []
    seen = set()
    for i, item in enumerate(extraction):
        rid = item['row_id']
        if rid in seen or rid not in by_id or int(item['row_index']) != i:
            raise ValueError('Extraction row order or identity is invalid')
        seen.add(rid)
        row = by_id[rid]
        if type(row['fold']) is not int or type(fold_map[rid]) is not int or row['fold'] != fold_map[rid] or row['fold'] not in range(5):
            raise ValueError('Frozen fold mismatch: ' + rid)
        if item['final_ebh'] != row['group'] or item['source_family'] != row['source_family'] or item['prompt_text'] != row['prompt_text']:
            raise ValueError('Audit does not match the frozen extraction: ' + rid)
        prompt_hash = hashlib.sha256(row['prompt_text'].encode('utf-8')).hexdigest()
        if item['prompt_sha256'] != prompt_hash:
            raise ValueError('Prompt digest mismatch')
        for arm in ('fixed', 'dynamic'):
            computed = favourable(row['group'], row[arm + '_verdict'])
            if computed is not row[arm + '_favourable']:
                raise ValueError('Endpoint differs from the frozen audit mapping: ' + rid)
        for arm in ('baseline', 'fixed', 'dynamic'):
            if hashlib.sha256(row['responses'][arm].encode('utf-8')).hexdigest() != row[arm + '_response_sha256']:
                raise ValueError('Recorded response identity mismatch: ' + rid)
            cell = row.get('adjudicated_cells', {}).get(arm)
            if cell and cell.get('response_sha256') and cell['response_sha256'] != row[arm + '_response_sha256']:
                raise ValueError('Judgement refers to a different response: ' + rid)
        positions.append(row)
    return positions


def load_inputs(repo, audit, paths_file=None):
    repo, audit = Path(repo).resolve(), development_path(audit)
    provenance = {}
    local_path = Path(paths_file) if paths_file else repo / 'configs/paths.local.json'
    if not local_path.is_file():
        raise ValueError('Missing local input paths: ' + str(local_path))
    paths = json.loads(local_path.read_text())
    # Record path configuration; its unrelated entries are not read as data.
    provenance[str(local_path.resolve())] = {'sha256': sha256(local_path), 'bytes': local_path.stat().st_size}
    spec_path = verified_file(repo / 'manifests/development_inputs.json', PINS['development_manifest'], provenance)
    spec = json.loads(spec_path.read_text())
    cases_path = verified_file(audit / 'cases.jsonl', PINS['cases'], provenance)
    fold_path = verified_file(repo / 'artifacts/gate_oof_v1/fold_assignments.json', PINS['folds'], provenance)
    extraction = Path(paths['extraction_dir'])
    manifest_path = verified_file(extraction / spec['extraction_manifest']['filename'], PINS['extraction_manifest'], provenance)
    activation_path = verified_file(extraction / spec['activation']['filename'], PINS['activation'], provenance)
    rows = [json.loads(line) for line in cases_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    manifest = list(csv.DictReader(io.StringIO(manifest_path.read_text(encoding='utf-8-sig'))))
    fold_map = json.loads(fold_path.read_text())['fold_of_row']
    rows = align_rows(rows, manifest, fold_map)
    counts = {g:sum(r['group'] == g for r in rows) for g in 'EBH'}
    if counts != {'E':89, 'B':178, 'H':213}:
        raise ValueError('Development class counts changed')
    states = np.load(str(activation_path), mmap_mode='r', allow_pickle=False)
    if states.shape != (480,49,2048) or states.dtype != np.dtype('float32'):
        raise ValueError('Wrong frozen activation shape or dtype')
    x = np.ascontiguousarray(states[:,32,:], dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite cached activation')
    return rows, x, provenance


def verify_unchanged(provenance):
    for name, record in provenance.items():
        if sha256(name) != record['sha256']:
            raise ValueError('Input changed during the run: ' + name)
