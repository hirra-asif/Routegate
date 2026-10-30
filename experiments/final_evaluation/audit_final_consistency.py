#!/usr/bin/env python3
"""Read-only audit of final judging, gate decisions and proposed label corrections.

Only writes a separate reporting directory. Makes no API or model calls, imports
no corrected verdicts, and leaves the frozen adjudications and groups unchanged.
Python 3.9+; standard library only.
"""
import argparse
import copy
import csv
import hashlib
import html
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ('baseline', 'dynamic_raw', 'safety_first_routegate')
NAMES = {'baseline': 'Baseline', 'dynamic_raw': 'Dynamic steering',
         'safety_first_routegate': 'Safety-first RouteGate'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def csv_dump(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return (row['screen_row_id'], row['arm_id'], row['rubric'])


def group_for(permission, behaviour):
    if permission is None:
        return 'permission_unresolved'
    if permission == 'AMBIGUOUS':
        return 'permission_ambiguous'
    if permission == 'PROHIBITED':
        return 'H'
    require(permission == 'PERMITTED', 'Unrecognised permission')
    if behaviour in ('DR', 'IR'):
        return 'E'
    if behaviour == 'SA':
        return 'B'
    return 'permitted_baseline_unresolved' if behaviour is None else 'permitted_other'


def audit(repo, data, proposal_file, output):
    require(not output.is_relative_to(data), 'The reporting output cannot be inside the evidence data directory.')
    proposal = read_json(proposal_file)
    require(proposal['status'] == 'PROPOSED_NOT_IMPORTED', 'This script handles the proposed review only.')
    tracked = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        if expected is not None:
            require(actual == expected, 'Input differs from the audited snapshot: ' + str(path))
        tracked[str(path)] = actual
        return path

    bind(proposal_file)
    for rel, expected in proposal['source_hashes'].items():
        bind(repo / rel, expected)
    judging = data / 'final_evaluation_v5_judging_v1'
    rows = {}
    for phase in ('baseline', 'responses'):
        path = bind(judging / phase / 'judging/adjudicated/adjudicated_cells.jsonl',
                    proposal['adjudication_sha256'][phase])
        rows[phase] = read_rows(path)
    baseline = {key(row): row for row in rows['baseline']}
    responses = {key(row): row for row in rows['responses']}
    require(len(baseline) == 760 and len(responses) == 861, 'Unexpected adjudication count or duplicate cells.')
    eligibility = read_json(bind(judging / 'responses/inputs/eligibility.json'))
    groups = {r['row_id']: r['group'] for r in eligibility}
    require(len(groups) == 380, 'Expected 380 eligibility rows.')
    for rid, group in groups.items():
        permission = baseline[(rid, 'prompt', 'axis1')]['verdict']
        behaviour = baseline[(rid, 'baseline', 'axis2')]['verdict']
        require(group_for(permission, behaviour) == group, 'Stored eligibility does not reproduce.')

    generations = {}
    for arm in ARMS:
        rel = 'final_evaluation_v5/sealed__' + arm + '/records.jsonl'
        path = bind(data / rel, proposal['generation_sha256'][rel])
        generations[arm] = {r['item_id']: r for r in read_rows(path)}
        require(set(generations[arm]) == set(groups), 'Generation identity coverage mismatch.')
        require(all(r['condition'] == arm for r in generations[arm].values()), 'Wrong system condition.')
    for rid in groups:
        require(len({generations[arm][rid]['prompt_sha256'] for arm in ARMS}) == 1,
                'Arms did not receive the same prompt.')

    prompt_audit = {}
    for phase in ('baseline', 'responses'):
        root = judging / phase
        protocol = read_json(bind(root / 'protocol.json'))
        packet, rubric_text = {}, {}
        for name, spec in protocol['rubrics'].items():
            packet[name] = {r['row_id']: r for r in read_rows(bind(root / 'judging/packet' / (name + '_judge_input.jsonl')))}
            texts = []
            for part, suffix in (('instructions', '.txt'), ('user_template', '.txt'), ('schema', '.json')):
                path = bind(root / 'frozen/judge_rubrics' / (name + '_' + part + suffix), spec[part + '_sha256'])
                texts.append(path.read_text(encoding='utf-8'))
            rubric_text[name] = texts
        counts = Counter()
        pair_digests = defaultdict(dict)
        attempts = 0
        for path in sorted((root / 'judging/runs').glob('packet__*/*s.jsonl')):
            if path.name not in ('results.jsonl', 'failures.jsonl'):
                continue
            for rec in read_rows(bind(path)):
                rubric, judge, rid = rec['rubric'], rec['judge'], rec['row_id']
                row = packet[rubric][rid]
                instructions, template, _ = rubric_text[rubric]
                values = {k: v for k, v in row.items() if k != 'row_id'}
                user = template.format(item_id=rid, **values)
                model = protocol['judges'][judge]['model']
                require(rec['model_requested'] == model, 'Wrong judge model.')
                require(rec['temperature'] == 0.0 and rec['max_tokens'] == 300, 'Changed judge settings.')
                body = {'model': model, 'max_tokens': 300, 'temperature': 0.0,
                        'system': instructions, 'messages': [{'role': 'user', 'content': user}]}
                require(text_sha(json.dumps(body, ensure_ascii=False)) == rec['request_sha256'],
                        'Saved request hash does not match the frozen prompt: ' + rid)
                digest = text_sha(json.dumps({k: v for k, v in body.items() if k != 'model'}, ensure_ascii=False))
                previous = pair_digests[(rubric, rid)].get(judge)
                require(previous in (None, digest), 'A retry changed the judging prompt.')
                pair_digests[(rubric, rid)][judge] = digest
                attempts += 1
                if path.name == 'results.jsonl':
                    counts[judge] += 1
        both = [entry for entry in pair_digests.values() if len(entry) == 2]
        require(all(entry['sonnet'] == entry['opus'] for entry in both), 'Judges received different prompts or settings.')
        prompt_audit[phase] = {'attempts_verified': attempts, 'successful_records': dict(counts),
                               'paired_request_rows': len(both), 'prompt_or_setting_mismatches': 0,
                               'retry_prompt_changes': 0}
    require(prompt_audit['responses']['successful_records'] == {'opus': 861, 'sonnet': 861},
            'Response judging is incomplete.')
    cross_phase = {}
    for name in ('axis1', 'axis2', 'safety', 'pairwise_quality'):
        for part in ('instructions.txt', 'user_template.txt', 'schema.json'):
            paths = [bind(judging / phase / 'frozen/judge_rubrics' / (name + '_' + part))
                     for phase in ('baseline', 'responses')]
            cross_phase[name + '_' + part] = paths[0].read_bytes() == paths[1].read_bytes()
    require(all(cross_phase.values()), 'Rubric changed between phases.')

    # Re-discover every exact-text conflict, including agreements and non-E cells.
    text_groups = defaultdict(list)
    for phase in rows:
        for row in rows[phase]:
            if row['rubric'] not in ('axis2', 'safety') or row['verdict'] is None:
                continue
            rid, arm = row['screen_row_id'], row['arm_id']
            digest = text_sha(generations[arm][rid]['response_text'])
            text_groups[(rid, row['rubric'], digest)].append((phase, row))
    conflicts = {k: v for k, v in text_groups.items() if len({r['verdict'] for _, r in v}) > 1}
    proposed_keys = {(i['row_id'], i['rubric'], i['response_sha256']) for i in proposal['items']}
    require(set(conflicts) == proposed_keys, 'The review does not cover exactly all current exact-text conflicts.')
    proposed_rows = copy.deepcopy(rows)
    proposed_index = {phase: {key(r): r for r in proposed_rows[phase]} for phase in proposed_rows}
    changes = []
    for item in proposal['items']:
        rid = item['row_id']
        require(text_sha(item['prompt']) == item['prompt_sha256'], 'Proposal prompt hash mismatch.')
        require(text_sha(item['saved_reply']) == item['response_sha256'], 'Proposal reply hash mismatch.')
        actual = conflicts[(rid, item['rubric'], item['response_sha256'])]
        expected_refs = {(a['phase'], a['arm'], a['judged_row_id'], a['verdict']) for a in item['assessments']}
        actual_refs = {(phase, row['arm_id'], row['judged_row_id'], row['verdict']) for phase, row in actual}
        require(expected_refs == actual_refs, 'Proposal assessment bindings changed.')
        for phase, old in actual:
            new = proposed_index[phase][key(old)]
            new['verdict'] = item['proposed_verdict']
            if new['verdict'] != old['verdict']:
                changes.append({'row_id': rid, 'phase': phase, 'arm': old['arm_id'],
                                'judged_row_id': old['judged_row_id'], 'original_label': old['verdict'],
                                'proposed_label': new['verdict'], 'original_label_source': old['label_source'],
                                'proposal_source': 'AI_PROPOSED_ERRATUM_NOT_IMPORTED'})
    # These proposals must not alter eligibility or the number of paired prompts.
    for rid, group in groups.items():
        permission = proposed_index['baseline'][(rid, 'prompt', 'axis1')]['verdict']
        behaviour = proposed_index['baseline'][(rid, 'baseline', 'axis2')]['verdict']
        require(group_for(permission, behaviour) == group, 'Proposed baseline correction would change eligibility; a full re-analysis is required.')
    for matches in text_groups.values():
        revised = {proposed_index[phase][key(row)]['verdict'] for phase, row in matches}
        require(len(revised) == 1, 'An exact-text disagreement would remain.')

    impacts = []
    for arm in ARMS[1:]:
        for group, rubric in (('E', 'axis2'), ('B', 'pairwise_quality'), ('H', 'safety'), ('permitted_other', 'axis2')):
            ids = [rid for rid in groups if groups[rid] == group]
            before = Counter(responses[(rid, arm, rubric)]['verdict'] for rid in ids)
            after = Counter(proposed_index['responses'][(rid, arm, rubric)]['verdict'] for rid in ids)
            for label in sorted(set(before) | set(after)):
                impacts.append({'arm': arm, 'group': group, 'label': label, 'denominator': len(ids),
                                'original_count': before[label], 'proposed_count': after[label]})
            if group in ('B', 'H'):
                require(before == after, 'The Axis-2 review changed another rubric.')
    e_ids = [rid for rid in groups if groups[rid] == 'E']
    e_summary = {}
    for arm in ARMS[1:]:
        def substantive(index, rid):
            return index[(rid, arm, 'axis2')]['verdict'] in ('SA', 'PA-S')
        e_summary[arm] = {'n': len(e_ids), 'original_substantive': sum(substantive(responses, rid) for rid in e_ids),
                          'proposed_substantive': sum(substantive(proposed_index['responses'], rid) for rid in e_ids),
                          'proposed_substantive_with_changed_reply': sum(
                              substantive(proposed_index['responses'], rid) and
                              generations[arm][rid]['response_text'] != generations['baseline'][rid]['response_text']
                              for rid in e_ids)}

    gate_counts = defaultdict(Counter)
    gate_rows = []
    for rid, group in groups.items():
        record = generations['safety_first_routegate'][rid]
        e_pass = record['gate_p_E'] >= record['threshold_E']
        h_pass = record['gate_p_H'] < record['threshold_H']
        expected_fire = e_pass and h_pass
        require(record['gate_fires'] == expected_fire, 'Gate telemetry contradicts the thresholds.')
        require(record['hooks_fired'] == (3 if expected_fire else 0), 'Gate/hook mismatch.')
        if not expected_fire:
            require(record['response_text'] == generations['baseline'][rid]['response_text'], 'Gate-off reply differs from baseline.')
        category = ('allowed' if expected_fire else 'blocked_H_only' if e_pass
                    else 'blocked_E_only' if h_pass else 'blocked_both')
        gate_counts[group][category] += 1
        gate_rows.append({'row_id': rid, 'group': group, 'E_score': record['gate_p_E'],
                          'H_score': record['gate_p_H'], 'threshold_E': record['threshold_E'],
                          'threshold_H': record['threshold_H'], 'decision_category': category})
    alphas = [generations['dynamic_raw'][rid]['applied_alpha_R'] for rid in e_ids]
    require(all(generations['dynamic_raw'][rid]['hooks_fired'] == 2 for rid in groups), 'A dynamic edit did not fire.')
    summary = {'status': 'PROPOSED_NOT_IMPORTED', 'judge_prompt_audit': prompt_audit,
               'cross_phase_rubrics_identical': cross_phase,
               'conflicting_text_groups': len(conflicts), 'assessments_reviewed': sum(len(v) for v in conflicts.values()),
               'proposed_changed_labels': len(changes), 'group_counts_unchanged': dict(Counter(groups.values())),
               'E_results_if_proposals_accepted': e_summary,
               'gate_decisions_by_group': {g: dict(v) for g, v in gate_counts.items()},
               'dynamic_E_coefficients': {'n': len(alphas), 'minimum': min(alphas),
                                           'median': statistics.median(alphas), 'maximum': max(alphas), 'ceiling': 0.15},
               'original_files_changed': 0, 'new_model_calls': 0, 'new_judge_calls': 0,
               'review_is_blinded': False, 'independent_human_review': False,
               'numerical_arithmetic_measurement': 'Separate compare_steering_arithmetic.py; no activation measurement is inferred here.'}
    for path, expected in tracked.items():
        require(sha(path) == expected, 'Input changed during audit: ' + path)
    output.mkdir(parents=True, exist_ok=True)
    dump(output / 'audit_summary.json', summary)
    dump(output / 'input_hashes.json', tracked)
    dump(output / 'proposed_changes.json', changes)
    csv_dump(output / 'proposed_label_changes.csv', changes)
    csv_dump(output / 'outcomes_if_accepted.csv', impacts)
    csv_dump(output / 'gate_exclusion_reasons.csv', gate_rows)
    write_review_html(output / 'LABEL_REVIEW.html', proposal, summary)
    dump(output / 'output_hashes.json', {p.name: sha(p) for p in sorted(output.iterdir())
                                      if p.is_file() and p.name != 'output_hashes.json'})
    return summary


def write_review_html(path, proposal, summary):
    esc = html.escape
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>RouteGate label consistency review</title>',
             '<style>body{font:17px/1.6 system-ui,sans-serif;color:#233640;max-width:980px;margin:40px auto;padding:0 20px}',
             'h1,h2{color:#2F6595}table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #ccd9df;text-align:left}',
             'pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.6 system-ui;background:#f5f8f9;padding:18px}',
             '.notice{border-left:5px solid #B47B48;padding:12px;background:#fff8ef}section{margin:42px 0}code{overflow-wrap:anywhere}</style>',
             '<h1>Six proposed label corrections</h1><div class="notice"><strong>Proposed; not imported.</strong> ',
             'These decisions were prepared by an AI assistant with access to the existing labels and results. They are not an independent, blinded human review. ',
             'The frozen evidence remains unchanged. Review the complete text and the qualification for each case.</div>',
             '<p>Six identical-text groups contain 13 assessments. The proposals change seven labels across both stages, ',
             'preserve every eligibility group, and leave B quality and H safety labels unchanged.</p>',
             '<table><tr><th>System</th><th>Recorded E substantive</th><th>If proposals accepted</th></tr>']
    for arm, stats in summary['E_results_if_proposals_accepted'].items():
        parts.append('<tr><td>%s</td><td>%s/%s</td><td>%s/%s</td></tr>' % (
            esc(NAMES[arm]), stats['original_substantive'], stats['n'], stats['proposed_substantive'], stats['n']))
    parts.append('</table><p>UM means the behaviour cannot be reliably classified under the rubric. It is a decided uncertainty label, not an API failure.</p>')
    for number, item in enumerate(proposal['items'], 1):
        parts.append('<section><h2>%d. %s</h2><p><strong>Proposed label: %s</strong> · confidence: %s</p><p>%s</p><p><strong>Qualification:</strong> %s</p>' % (
            number, esc(item['row_id']), esc(item['proposed_verdict']), esc(item['confidence']), esc(item['reason']), esc(item['review_note'])))
        parts.append('<table><tr><th>Stage</th><th>System</th><th>Original</th><th>Proposed</th><th>Original source</th></tr>')
        for a in item['assessments']:
            parts.append('<tr>' + ''.join('<td>%s</td>' % esc(str(v)) for v in (
                a['phase'], NAMES[a['arm']], a['verdict'], item['proposed_verdict'], a['label_source'])) + '</tr>')
        parts.append('</table><h3>Request</h3><pre>%s</pre><h3>Identical saved reply</h3><pre>%s</pre><p>Reply SHA-256: <code>%s</code></p></section>' % (
            esc(item['prompt']), esc(item['saved_reply']), item['response_sha256']))
    parts.append('</html>')
    Path(path).write_text(''.join(parts), encoding='utf-8')


def main():
    repo_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=repo_default)
    parser.add_argument('--data-root', type=Path, default=Path.home() / 'routegate-data')
    parser.add_argument('--proposal', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    repo, data = args.repo.resolve(), args.data_root.resolve()
    proposal = args.proposal or repo / 'docs/final_evaluation/consistency_review_v1.json'
    output = args.output or repo / 'outputs/thesis_results/consistency_review_v1'
    result = audit(repo, data, proposal.resolve(), output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    print('Review: ' + str(output.resolve() / 'LABEL_REVIEW.html'))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise SystemExit('Audit stopped: ' + str(exc))
