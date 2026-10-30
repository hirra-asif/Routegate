"""Tables and report for the three-layer static selection comparison."""
import csv
import json
from pathlib import Path

import numpy as np

from moe_core import (HEADS, OUTCOME_OF, group_mask, outcome_arrays, paired_differences,
                      score_policy, within_class_quality)

NOTES = [
    'Exploratory analysis of development data that has already been inspected. Nested fitting prevents '
    'direct fold leakage; it does not make this a fresh confirmatory test and establishes no generalisation.',
    'Every policy chooses between the same saved baseline response and the same saved static response. '
    'No new generations, no model loading, no GPU, no judge API calls, no label changes.',
    'The intervention is consensus23_global_frozen at strength 0.15 on zero-based block outputs 31-33, applied at '
    'the final prompt token during prefill. It is NOT historical_d1 and NOT the later two-layer controller edit.',
    'H is "harmful static responses selected /213": a property of the selected static response alone. This master '
    'contains no matched baseline H safety judgement, so this is NOT newly harmful assistance and NOT a '
    'complete-system harm rate. The old /212 baseline-NO population is not imported and no baseline NO label is invented.',
    'no-edit contributes zero selected-static events by construction. That does not establish that the baseline '
    'responses are safe; it only means no static response was selected.',
    'E counts are retained substantive static answers (SA or PA-S) out of the whole 89-prompt E inventory. The single '
    'UM row stays in the inventory and is reported as unknown in selected/withheld counts; it is never counted as repair.',
    'Profiles constrain OBSERVED inner-validation outcomes. They are exploratory settings, not deployment '
    'tolerances, not guarantees and not safety certificates. All nine arm/profile combinations are reported.',
    'Routing features are raw pre-selection router logits and can encode source and topic. Pooled prediction quality '
    'alone is therefore insufficient; within-class out-of-fold quality is reported for each head.',
    'The two historical references used thresholds calibrated on POOLED development out-of-fold predictions, unlike '
    'this nested comparison. Differences against them do not isolate a method improvement.',
    'No causal identification of safety experts, no expert intervention, no novelty claim.',
    'Token-cap limitation, reported as recorded and with no row excluded or relabelled: 199 of 480 static responses '
    'hit the generation token cap and 282 ended on EOS.',
    'No external 380 evaluation, no all-development deployment refit, no neural-network search.',
]


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flat(name, reading):
    row = {'policy': name, 'edits': reading['edits']}
    for group in 'EBH':
        row['edits_' + group] = reading['edits_by_group'][group]
    e, b, h = reading['E_substantive_selected'], reading['B_material_loss'], reading['H_harmful_static_selected']
    row.update(E_substantive_selected=e['positive'], E_population=e['population'],
               E_unknown_selected=reading['E_unknown_selected'], E_unknown_withheld=reading['E_unknown_withheld'],
               B_material_loss=b['positive'], B_population=b['population'], B_unknown_selected=b['unknown'],
               H_harmful_static_selected=h['positive'], H_population=h['population'], H_unknown_selected=h['unknown'])
    return row


def table(rows_out):
    header = ('| Policy | Edits / 480 | Edits E/B/H | E substantive selected / 89 | B material loss / 178 | '
              'Harmful static selected / 213 |')
    lines = [header, '|---|---:|---|---:|---:|---:|']
    for row in rows_out:
        e_extra = '' if not row['E_unknown_selected'] else ' (+' + str(row['E_unknown_selected']) + ' unknown)'
        b_extra = '' if not row['B_unknown_selected'] else ' (+' + str(row['B_unknown_selected']) + ' unknown)'
        h_extra = '' if not row['H_unknown_selected'] else ' (+' + str(row['H_unknown_selected']) + ' unknown)'
        lines.append('| ' + row['policy'] + ' | ' + str(row['edits']) + ' | '
                     + '/'.join(str(row['edits_' + g]) for g in 'EBH') + ' | '
                     + str(row['E_substantive_selected']) + e_extra + ' | '
                     + str(row['B_material_loss']) + b_extra + ' | '
                     + str(row['H_harmful_static_selected']) + h_extra + ' |')
    return '\n'.join(lines)


def make_reports(output, rows, actions, fold_records, design, provenance, checks, historical, scores_by_arm):
    output = Path(output)
    results = {name: score_policy(rows, action) for name, action in actions.items()}
    order = (['no_edit', 'always_fixed']
             + [a['name'] + '__' + p['name'] for a in design['arms'] for p in design['profiles']]
             + [k for k in actions if k.startswith('historical_')])
    order = [k for k in order if k in results] + [k for k in results if k not in order]
    flat_rows = [flat(name, results[name]) for name in order]
    write_csv(output / 'summary.csv', flat_rows)

    paired = {}
    for arm in ('routing_only', 'combined'):
        for profile in design['profiles']:
            key = arm + '__' + profile['name']
            reference = 'residual_only__' + profile['name']
            paired[key + '_vs_' + reference] = paired_differences(rows, actions[key], actions[reference])
    for name in order:
        if name not in ('always_fixed',):
            paired[name + '_vs_always_fixed'] = paired_differences(rows, actions[name], actions['always_fixed'])
    (output / 'paired_differences.json').write_text(json.dumps(paired, indent=2) + '\n')

    paired_rows = []
    for key, record in paired.items():
        policy, reference = key.split('_vs_')
        entry = {'policy': policy, 'reference': reference, 'edits_policy': record['edits_policy'],
                 'edits_reference': record['edits_reference'], 'edits_in_common': record['edits_in_common']}
        for field in ('E_substantive_gained', 'E_substantive_lost', 'E_unknown_selected_gained',
                      'E_unknown_selected_lost', 'B_material_loss_introduced', 'B_material_loss_avoided',
                      'B_unknown_selected_gained', 'B_unknown_selected_avoided', 'H_harmful_static_introduced',
                      'H_harmful_static_avoided', 'H_unknown_selected_gained', 'H_unknown_selected_avoided'):
            entry['n_' + field] = len(record[field])
            entry[field] = ';'.join(record[field])
        entry.update(net_E_substantive=record['net_E_substantive'],
                     net_B_material_loss=record['net_B_material_loss'],
                     net_H_harmful_static=record['net_H_harmful_static'])
        paired_rows.append(entry)
    write_csv(output / 'paired_differences.csv', paired_rows)

    quality_rows = []
    for arm, score in scores_by_arm.items():
        for entry in within_class_quality(rows, score):
            quality_rows.append({'arm': arm, **entry})
    write_csv(output / 'within_class_quality.csv', quality_rows)

    per_fold = []
    for record in fold_records:
        arm, fold = record['arm'], record['outer_fold']
        index = np.asarray(record['test_indices'])
        subset = [rows[i] for i in index]
        for profile, selected in record['rules'].items():
            action = np.asarray(record['actions'][profile], dtype=bool)
            reading = score_policy(subset, action)
            per_fold.append({
                'arm': arm, 'outer_fold': fold, 'profile': profile,
                'test_rows': record['test_rows'], 'training_rows': record['training_rows'],
                'chosen_C_E': record['head_selection'][0]['chosen_C'],
                'chosen_C_B': record['head_selection'][1]['chosen_C'],
                'chosen_C_H': record['head_selection'][2]['chosen_C'],
                'rule_no_edit': bool(selected['rule'].get('no_edit', False)),
                'threshold_E': None if selected['rule'].get('no_edit') else selected['rule']['thresholds'][0],
                'threshold_B': None if selected['rule'].get('no_edit') else selected['rule']['thresholds'][1],
                'threshold_H': None if selected['rule'].get('no_edit') else selected['rule']['thresholds'][2],
                'validation_E_captured': selected['validation']['E_captured'],
                'validation_B_upper': selected['validation']['B_upper'],
                'validation_H_upper': selected['validation']['H_upper'],
                'validation_edits': selected['validation']['edits'],
                'validation_B_population': selected['validation']['B_population'],
                'validation_H_population': selected['validation']['H_population'],
                **{k: v for k, v in flat(arm + '__' + profile, reading).items() if k != 'policy'},
            })
    write_csv(output / 'results_by_fold.csv', per_fold)

    group_counts = {g: int(group_mask(rows, g).sum()) for g in 'EBH'}
    target_counts = []
    for record in fold_records:
        for head_index, head in enumerate(HEADS):
            entry = record['head_selection'][head_index]
            for candidate in entry['candidates']:
                target_counts.append({'arm': record['arm'], 'outer_fold': record['outer_fold'], 'head': head,
                                      'C': candidate['C'], 'pooled_inner_log_loss': candidate['pooled_inner_log_loss'],
                                      'chosen': candidate['C'] == entry['chosen_C'],
                                      'inner_known_labels': ';'.join(str(f['known_labels']) for f in candidate['fits']),
                                      'inner_positives': ';'.join(str(f['positives']) for f in candidate['fits']),
                                      'inner_kinds': ';'.join(f['kind'] for f in candidate['fits'])})
    write_csv(output / 'head_selection.csv', target_counts)

    convergence = [c for record in fold_records for c in record['convergence']]
    write_csv(output / 'convergence.csv', convergence)
    iterations = [c['iterations'] for c in convergence if c['iterations'] is not None]
    convergence_summary = {'fits': len(convergence), 'logistic_fits': len(iterations),
                           'constant_fits': len(convergence) - len(iterations),
                           'max_iter': design['estimator']['logistic_regression']['max_iter'],
                           'max_iterations_used': max(iterations) if iterations else None,
                           'median_iterations': float(np.median(iterations)) if iterations else None,
                           'fits_at_cap': sum(1 for i in iterations if i >= design['estimator']['logistic_regression']['max_iter'])}

    row_rows = []
    for i, row in enumerate(rows):
        entry = {'row_id': row['row_id'], 'row_index': row['row_index'], 'group': row['group'],
                 'fold': row['fold'], 'source': row['source'],
                 'final_after_axis2': row['final_after_axis2'],
                 'final_quality_outcome': row['final_quality_outcome'],
                 'final_difference_strength': row['final_difference_strength'],
                 'final_actionable_harmful_help': row['final_actionable_harmful_help'],
                 'static_outcome': OUTCOME_OF[row['group']](row),
                 'after_hit_max_tokens': row['after_hit_max_tokens'], 'after_hit_eos': row['after_hit_eos']}
        for arm, score in scores_by_arm.items():
            for head_index, head in enumerate(HEADS):
                entry[arm + '_score_' + head] = float(score[i, head_index])
        for name in order:
            entry['action_' + name] = int(bool(np.asarray(actions[name])[i]))
        row_rows.append(entry)
    write_csv(output / 'per_row_scores_actions.csv', row_rows)

    summary = {
        'status': 'DEVELOPMENT_COMPARISON_COMPLETE',
        'study_id': design['study_id'], 'rows': len(rows), 'class_counts': group_counts,
        'intervention': design['intervention'],
        'H_endpoint_name': 'harmful static responses selected /213',
        'exploratory': True, 'exploratory_statement': design['exploratory_statement'],
        'new_generations': 0, 'api_calls': 0, 'gpu_used': False, 'label_changes': 0,
        'external_380_access': False, 'deployment_refit': False, 'expert_interventions': 0,
        'neural_network_search': False,
        'notes': NOTES, 'policies': {name: results[name] for name in order},
        'within_class_quality': quality_rows, 'convergence': convergence_summary,
        'historical_references': historical, 'checks': checks, 'provenance': provenance,
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')

    report = ['# Do native MoE routing features improve selection of the frozen three-layer static repair?', '',
              'One frozen static repair: `consensus23_global_frozen`, strength 0.15, zero-based block outputs 31-33, '
              'final prompt token during prefill. 480 development rows (E=89, B=178, H=213), the original five outer '
              'folds, nested inner validation. Every policy chooses between the same saved baseline response and the '
              'same saved static response.', '',
              '**H is "harmful static responses selected /213"** - a property of the selected static response alone. '
              'This outcome table contains no matched baseline H safety judgement, so this is not newly harmful '
              'assistance and not a complete-system harm rate. no-edit scores zero selected-static events by '
              'construction, which does not establish that those baselines are safe.', '',
              '## All nine arm/profile combinations, with references', '', table(flat_rows), '',
              '## Per-fold results', '',
              'Full detail in `results_by_fold.csv` (chosen C per head, chosen thresholds, inner-validation outcomes '
              'and per-fold endpoint counts).', '']

    fold_lines = ['| Arm | Profile | Fold | Edits | E subst. | B mat. loss | Harmful static | C (E/B/H) |',
                  '|---|---|---:|---:|---:|---:|---:|---|']
    for entry in per_fold:
        fold_lines.append('| ' + entry['arm'] + ' | ' + entry['profile'] + ' | ' + str(entry['outer_fold']) + ' | '
                          + str(entry['edits']) + ' | ' + str(entry['E_substantive_selected'])
                          + ('' if not entry['E_unknown_selected'] else '+' + str(entry['E_unknown_selected']) + 'u')
                          + ' | ' + str(entry['B_material_loss']) + ' | ' + str(entry['H_harmful_static_selected'])
                          + ' | ' + '/'.join(str(entry['chosen_C_' + g]) for g in 'EBH') + ' |')
    report += fold_lines + ['', '## Historical references (descriptive only)', '',
                            'Both used thresholds calibrated on pooled development out-of-fold predictions, unlike this '
                            'nested comparison, so differences against them do not isolate a method improvement.', '']
    hist_lines = ['| Reference | Edits | E substantive / 89 | B material loss / 178 | Harmful static / 213 | Reproduced |',
                  '|---|---:|---:|---:|---:|---|']
    for name, record in historical.items():
        hist_lines.append('| ' + name + ' | ' + str(record['observed']['edits']) + ' | '
                          + str(record['observed']['E_substantive_selected'])
                          + ('' if not record['observed']['E_unknown_selected'] else '+'
                             + str(record['observed']['E_unknown_selected']) + ' unknown')
                          + ' | ' + str(record['observed']['B_material_loss']) + ' | '
                          + str(record['observed']['H_harmful_static_selected']) + ' | '
                          + ('YES' if record['matches_expected'] else 'NO') + ' |')
    report += hist_lines + ['', '## Within-class out-of-fold quality', '',
                            'Each head scored only inside its own class. Routing features can encode source and topic, '
                            'so pooled separation is not evidence of within-class ordering.', '']
    q_lines = ['| Arm | Head | Class | Class rows | Known | Positives | AUC | AP |', '|---|---|---|---:|---:|---:|---:|---:|']
    for entry in quality_rows:
        auc = 'undefined' if entry['auc'] is None else format(entry['auc'], '.3f')
        ap = 'undefined' if entry['average_precision'] is None else format(entry['average_precision'], '.3f')
        q_lines.append('| ' + entry['arm'] + ' | ' + entry['head'] + ' | ' + entry['class'] + ' | '
                       + str(entry['class_rows']) + ' | ' + str(entry['known_rows']) + ' | '
                       + str(entry['positives']) + ' | ' + auc + ' | ' + ap + ' |')
    report += q_lines + ['', '## Paired prompt-level differences versus the matched residual-only arm', '',
                         'Exact row identifiers are in `paired_differences.json` and `paired_differences.csv`.', '']
    p_lines = ['| Arm | Profile | Edits (arm vs residual) | E gained | E lost | E net | B introduced | B avoided | '
               'Harmful introduced | Harmful avoided | Unknown transitions |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for arm in ('routing_only', 'combined'):
        for profile in design['profiles']:
            key = arm + '__' + profile['name'] + '_vs_residual_only__' + profile['name']
            record = paired[key]
            unknown = ('E +' + str(len(record['E_unknown_selected_gained'])) + '/-'
                       + str(len(record['E_unknown_selected_lost'])) + ', B +'
                       + str(len(record['B_unknown_selected_gained'])) + '/-'
                       + str(len(record['B_unknown_selected_avoided'])) + ', H +'
                       + str(len(record['H_unknown_selected_gained'])) + '/-'
                       + str(len(record['H_unknown_selected_avoided'])))
            p_lines.append('| ' + arm + ' | ' + profile['name'] + ' | ' + str(record['edits_policy']) + ' vs '
                           + str(record['edits_reference']) + ' | ' + str(len(record['E_substantive_gained'])) + ' | '
                           + str(len(record['E_substantive_lost'])) + ' | ' + str(record['net_E_substantive']) + ' | '
                           + str(len(record['B_material_loss_introduced'])) + ' | '
                           + str(len(record['B_material_loss_avoided'])) + ' | '
                           + str(len(record['H_harmful_static_introduced'])) + ' | '
                           + str(len(record['H_harmful_static_avoided'])) + ' | ' + unknown + ' |')
    report += p_lines + ['', '## Checks', '']
    for name, value in checks.items():
        report.append('- **' + name + '**: ' + (json.dumps(value, default=str) if not isinstance(value, str) else value))
    report += ['', '## Convergence', '',
               '- ' + str(convergence_summary['logistic_fits']) + ' logistic fits and '
               + str(convergence_summary['constant_fits']) + ' recorded constant-head fallbacks',
               '- max_iter=' + str(convergence_summary['max_iter']) + '; largest used '
               + str(convergence_summary['max_iterations_used']) + '; median '
               + str(convergence_summary['median_iterations']) + '; fits at the cap '
               + str(convergence_summary['fits_at_cap']),
               '', '## Reading notes', '']
    report += ['- ' + note for note in NOTES]
    (output / 'REPORT.md').write_text('\n'.join(report) + '\n')
    return summary, flat_rows
