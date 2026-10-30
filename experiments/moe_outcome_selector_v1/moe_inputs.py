"""Load the hash-bound THREE-LAYER static outcome table, residual state 45, router logits and the original folds.

Nothing here loads Qwen, touches the network, reads sealed material or writes a frozen artifact.
"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

# Declared digests. A mismatch is a hard stop, never a substitution.
PINS = {
    'static_master': '525e62e0a6fdac3200111deae146d699ea2db9b3459dab2e57b09c705b11b9e4',
    'residual_array': 'af89872d5decd4dc432ec579a941e6247bc7de63ff5690432e9f5812a994e2d8',
    'extraction_manifest': 'a3b7444b1a2782b5bc04142f3545ac5d71ad57a5f546edc315d7d51d14947abd',
    'router_logits': '4f2d73102e07a59d923c9470235771cce54642570637fb21d0d30a91ab3d1ee7',
    'fold_assignments': '8c5426ca8bb5fe50a359683de431a05f2092252517f477954f8b6085e851e441',
}
# Read for provenance and for the descriptive historical reproductions only.
CONTEXT_PINS = {
    'routing_completed_rows': '27e29b7db4559cabb4b9e61d3fb4c313c1e26eed64342a627719177cf1cce522',
    'routing_row_ids': '8d0ab2458cd735773f0fbeac50dd85271072f4eb77caf53f053c3749cac92839',
    'routing_artifact_hashes': None,
    'routing_analysis_oof': None,
    'routing_analysis_thresholds': None,
    'residual_thresholds': 'e7d56cfe1b2bb763fc52a8dd86a30254375f81b594ed8c942c3fe24715a35507',
    'deployment_manifest': None,
}
EXPECTED_CLASS_COUNTS = {'E': 89, 'B': 178, 'H': 213}
AXIS2 = ('SA', 'PA-S', 'PA-N', 'DR', 'IR', 'NR', 'UM')
QUALITY_OUTCOMES = ('EQUIVALENT', 'BEFORE_BETTER', 'AFTER_BETTER')
QUALITY_STRENGTHS = ('NONE', 'MINOR', 'MATERIAL')
HARM_VERDICTS = ('YES', 'NO')


class InputError(RuntimeError):
    """A named, actionable input failure. Never resolved by regenerating or substituting data."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def development_path(path):
    text = str(Path(path)).lower()
    for forbidden in ('sealed', 'final_evaluation', 'final_benchmark', 'external_380'):
        if forbidden in text:
            raise InputError('Held-out evaluation material is not an input to this study: ' + str(path))
    return Path(path)


def verified_file(path, expected, provenance, role):
    path = development_path(path)
    if not path.is_file():
        raise InputError('Missing required input: ' + str(path) + ' (' + role + ')')
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise InputError(
            'Input digest differs from the declared version: ' + str(path)
            + '\n  expected ' + expected + '\n  observed ' + actual
            + '\n  Locate the hash-matching artifact; do not regenerate or substitute data.')
    provenance[str(path)] = {'sha256': actual, 'bytes': path.stat().st_size, 'role': role}
    return path


def locate_by_hash(expected, roots):
    """If a declared path has moved, find the same artifact by digest within the project/data roots."""
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for candidate in root.rglob('*'):
            if not candidate.is_file():
                continue
            text = str(candidate).lower()
            if any(f in text for f in ('sealed', 'final_evaluation', 'final_benchmark')):
                continue
            try:
                if candidate.stat().st_size > 0 and sha256(candidate) == expected:
                    return candidate
            except OSError:
                continue
    return None


def read_csv_rows(path):
    return list(csv.DictReader(io.StringIO(Path(path).read_text(encoding='utf-8-sig'))))


def validate_labels(row):
    """Unrecognised or inconsistent labels raise; no mapping is guessed."""
    rid, group = row['row_id'], row['frozen_class']
    axis2 = row['final_after_axis2']
    quality = (row['final_quality_outcome'], row['final_difference_strength'])
    harm = row['final_actionable_harmful_help']
    if group == 'E':
        if axis2 not in AXIS2:
            raise InputError('Unrecognised E axis2 verdict ' + repr(axis2) + ' at ' + rid)
        if quality != ('', '') or harm != '':
            raise InputError('E row carries out-of-group quality/safety cells at ' + rid)
    elif group == 'B':
        if quality[0] not in QUALITY_OUTCOMES or quality[1] not in QUALITY_STRENGTHS:
            raise InputError('Unrecognised B quality label ' + repr(quality) + ' at ' + rid)
        if harm != '':
            raise InputError('B row carries an out-of-group safety cell at ' + rid)
    elif group == 'H':
        if harm not in HARM_VERDICTS:
            raise InputError('Unrecognised H safety verdict ' + repr(harm) + ' at ' + rid)
        if axis2 != '' or quality != ('', ''):
            raise InputError('H row carries out-of-group axis2/quality cells at ' + rid)
    else:
        raise InputError('Unrecognised frozen_class ' + repr(group) + ' at ' + rid)


def load_inputs(repo, paths_file=None):
    repo = Path(repo).resolve()
    provenance, notes = {}, []
    paths_path = Path(paths_file) if paths_file else repo / 'configs/paths.local.json'
    if not paths_path.is_file():
        raise InputError('Missing local input paths: ' + str(paths_path))
    provenance[str(paths_path.resolve())] = {'sha256': sha256(paths_path), 'bytes': paths_path.stat().st_size,
                                             'role': 'configured data locations'}
    paths = json.loads(paths_path.read_text())
    roots = [repo, Path(os.environ.get('ROUTEGATE_DATA_DIR', str(Path.home() / 'routegate-data')))]

    def resolve(expected, declared, role):
        declared = Path(declared)
        if declared.is_file() and sha256(declared) == expected:
            return verified_file(declared, expected, provenance, role)
        found = locate_by_hash(expected, roots)
        if found is None:
            raise InputError('Missing required input for ' + role + '. Declared path: ' + str(declared)
                             + '\n  No file with digest ' + expected + ' was found under '
                             + ', '.join(str(r) for r in roots)
                             + '\n  Do not regenerate or substitute data.')
        notes.append('Declared path moved; used the hash-matching artifact for ' + role + ': ' + str(found))
        return verified_file(found, expected, provenance, role)

    master_path = resolve(PINS['static_master'], paths['static_steering_master'], 'three-layer static outcome table')
    extraction = Path(paths['extraction_dir'])
    manifest_path = resolve(PINS['extraction_manifest'], extraction / 'manifest.csv', 'residual extraction row order')
    residual_path = resolve(PINS['residual_array'], extraction / 'residual_final_prompt_token.npy', 'residual activations')
    logits_path = resolve(PINS['router_logits'], repo / 'outputs/routing_extraction_v1/router_logits.npy', 'router logits')
    fold_path = resolve(PINS['fold_assignments'], repo / 'outputs/routegate_residual_v1/fold_assignments.csv',
                        'original five outer folds')
    completed_path = resolve(CONTEXT_PINS['routing_completed_rows'],
                             repo / 'outputs/routing_extraction_v1/completed_rows.jsonl', 'router logits row index')

    master = read_csv_rows(master_path)
    manifest = read_csv_rows(manifest_path)
    folds = read_csv_rows(fold_path)
    completed = [json.loads(line) for line in completed_path.read_text().splitlines() if line.strip()]

    for name, table in (('static master', master), ('extraction manifest', manifest),
                        ('fold assignments', folds), ('router row index', completed)):
        if len(table) != 480:
            raise InputError(name + ' has ' + str(len(table)) + ' rows, expected 480')

    master_by = {}
    for row in master:
        if row['row_id'] in master_by:
            raise InputError('Duplicate row_id in the static master: ' + row['row_id'])
        validate_labels(row)
        master_by[row['row_id']] = row
    fold_by = {}
    for row in folds:
        if row['row_id'] in fold_by:
            raise InputError('Duplicate row_id in the fold assignments: ' + row['row_id'])
        fold = int(row['outer_fold'])
        if fold not in range(5):
            raise InputError('Fold outside 0..4 at ' + row['row_id'])
        fold_by[row['row_id']] = (fold, row['final_ebh'])
    completed_by = {row['row_id']: row for row in completed}
    if not (set(master_by) == set(fold_by) == set(completed_by) == {r['row_id'] for r in manifest}):
        raise InputError('The four inputs do not describe the same 480 row identities')

    # Join strictly by verified row id and row index; file order is never assumed to agree.
    rows = []
    for position, item in enumerate(manifest):
        rid = item['row_id']
        if int(item['row_index']) != position:
            raise InputError('Extraction manifest row_index is not its position at ' + rid)
        if int(completed_by[rid]['row_index']) != position:
            raise InputError('Router logits row_index disagrees with the residual row order at ' + rid)
        record, fold = master_by[rid], fold_by[rid]
        if not (record['frozen_class'] == item['final_ebh'] == fold[1] == completed_by[rid]['final_ebh']):
            raise InputError('Class label disagrees across inputs at ' + rid)
        prompt_hash = hashlib.sha256(item['prompt_text'].encode('utf-8')).hexdigest()
        if prompt_hash != item['prompt_sha256']:
            raise InputError('Extraction manifest prompt digest mismatch at ' + rid)
        if hashlib.sha256(record['prompt_text'].encode('utf-8')).hexdigest() != item['prompt_sha256']:
            raise InputError('Static master prompt text does not match the extracted prompt at ' + rid)
        if completed_by[rid]['prompt_sha256'] != item['prompt_sha256']:
            raise InputError('Router logits prompt digest mismatch at ' + rid)
        if int(record['intervention_hit_count']) != 3:
            raise InputError('Row was not produced by the three-layer intervention at ' + rid)
        rows.append({
            'row_id': rid, 'row_index': position, 'group': record['frozen_class'], 'fold': fold[0],
            'source': record['source'], 'prompt_sha256': item['prompt_sha256'],
            'final_after_axis2': record['final_after_axis2'],
            'final_quality_outcome': record['final_quality_outcome'],
            'final_difference_strength': record['final_difference_strength'],
            'final_actionable_harmful_help': record['final_actionable_harmful_help'],
            'repair_status': record['repair_status'],
            'after_hit_max_tokens': record['after_hit_max_tokens'] == 'True',
            'after_hit_eos': record['after_hit_eos'] == 'True',
            'after_tokens': int(record['after_tokens']) if record['after_tokens'] else None,
        })

    counts = {g: sum(r['group'] == g for r in rows) for g in 'EBH'}
    if counts != EXPECTED_CLASS_COUNTS:
        raise InputError('Development class counts changed: ' + json.dumps(counts))

    states = np.load(str(residual_path), mmap_mode='r', allow_pickle=False)
    if states.shape != (480, 49, 2048) or states.dtype != np.dtype('float32'):
        raise InputError('Residual array is ' + str(states.shape) + '/' + str(states.dtype)
                         + ', expected (480, 49, 2048) float32')
    residual = np.ascontiguousarray(states[:, 45, :], dtype=np.float64)  # state 45 = block 44 output
    logits = np.load(str(logits_path), mmap_mode='r', allow_pickle=False)
    if logits.shape != (480, 48, 128) or logits.dtype != np.dtype('float32'):
        raise InputError('Router logits array is ' + str(logits.shape) + '/' + str(logits.dtype)
                         + ', expected (480, 48, 128) float32')
    routing = np.ascontiguousarray(np.asarray(logits, dtype=np.float64).reshape(480, 6144, order='C'))
    if residual.shape != (480, 2048) or routing.shape != (480, 6144):
        raise InputError('Feature matrices have the wrong shape after extraction')
    if not np.isfinite(residual).all():
        raise InputError('Non-finite residual feature')
    if not np.isfinite(routing).all():
        raise InputError('Non-finite router logit feature')
    features = {'residual_only': residual, 'routing_only': routing,
                'combined': np.ascontiguousarray(np.hstack([residual, routing]))}
    return rows, features, provenance, notes


def load_historical(repo, provenance):
    """Recorded scores and thresholds for the two descriptive historical references."""
    repo = Path(repo).resolve()
    oof = verified_file(repo / 'outputs/routegate_routing_analysis_v1/oof_predictions.csv',
                        CONTEXT_PINS['routing_analysis_oof'], provenance, 'historical pooled OOF scores')
    thresholds = verified_file(repo / 'outputs/routegate_routing_analysis_v1/thresholds.json',
                               CONTEXT_PINS['routing_analysis_thresholds'], provenance, 'historical thresholds')
    residual_thresholds = verified_file(repo / 'outputs/routegate_residual_v1/thresholds.json',
                                        CONTEXT_PINS['residual_thresholds'], provenance, 'frozen residual gate thresholds')
    return read_csv_rows(oof), json.loads(Path(thresholds).read_text()), json.loads(Path(residual_thresholds).read_text())


def verify_unchanged(provenance):
    changed = [name for name, record in provenance.items() if sha256(name) != record['sha256']]
    if changed:
        raise InputError('Inputs changed during the run: ' + ', '.join(changed))
    return {'inputs_checked': len(provenance), 'changed': 0}
