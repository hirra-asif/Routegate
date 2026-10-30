"""Comparison tables, per-fold results, exact gained/lost prompts and ranking diagnostics."""
import html
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from reporting import write_csv
from selector_core import binary_loss, outcome_arrays, score_policy


HEADS = ['opportunity', 'benign_loss', 'new_harm']
SHORT = {'opportunity': 'E_fixed_substantive', 'benign_loss': 'B_material_loss', 'new_harm': 'H_new_harm'}

NOTES = [
    'Development selection among already recorded baseline and fixed-edit replies. No new generations, no judge calls, no label changes, CPU only.',
    'All three selectors choose between the same baseline response and the same fixed repair edit: strength 0.15 at zero-based layers 31-32, read from the frozen unsteered state 32 (output of zero-based layer 31).',
    'class_selector and outcome_selector are the existing linear arms, refitted here by the same code and verified to reproduce experiments/outcome_selector_v1 exactly.',
    'nn_outcome_selector uses the identical joint-outcome targets, folds, threshold grid, budget profiles and scoring; only the model family changes to one hidden layer of 16 ReLU units with L2 regularisation, averaged over three fixed seeds.',
    'E counts are retained fixed-edit substantive answers out of the whole 89-prompt E inventory. Baseline-selected E behaviour is not assigned a label, so this is NOT a complete-system repair rate.',
    'H new harm is counted over the 212 known non-harmful H baselines. The single H row with an unknown baseline is never folded into that denominator and is reported separately.',
    'Unknown outcomes stay unknown. A selected unknown B quality or unknown H baseline is reported as unknown, never as a favourable result.',
    'Profiles constrain inner-validation observed-plus-unknown damage only. They are exploratory settings, not deployment tolerances or safety guarantees. All three are reported; none is chosen from outer results.',
    'The reversed H-class ranking figure is a descriptive control showing whether the new-harm head orders harm at all. Every method and profile keeps the gate new_harm_score <= t_H; no gate was reversed after seeing outcomes.',
    'Every scaler, network, alpha, C and threshold excludes its outer evaluation fold. There is no final all-development refit and no external evaluation.',
    'The design follows inspection of these development outcomes. Nested fitting prevents direct fold leakage but this remains exploratory, not a fresh confirmatory test.'
]


def flat_counts(reading):
    e, b, h = (reading[k] for k in ('E_fixed_substantive_selected', 'B_material_loss', 'H_new_harm_baseline_NO'))
    return {'edits': reading['edits'], 'edits_E': reading['edits_by_class']['E'], 'edits_B': reading['edits_by_class']['B'],
            'edits_H': reading['edits_by_class']['H'],
            'E_retained': e['positive'], 'E_population': e['population'], 'E_unknown_selected': e['unknown'],
            'B_material_loss': b['positive'], 'B_population': b['population'], 'B_unknown_selected': b['unknown'],
            'H_new_harm': h['positive'], 'H_population': h['population'], 'H_unknown_selected': h['unknown'],
            'H_any_harm_known': reading['H_any_harm']['positive'], 'H_any_harm_unknown': reading['H_any_harm']['unknown'],
            'H_selected_unknown_baseline': reading['H_selected_unknown_baseline']}


def per_fold_table(rows, actions):
    out = []
    for name, action in actions.items():
        action = np.asarray(action, dtype=bool)
        for fold in sorted({r['fold'] for r in rows}):
            ids = [i for i, r in enumerate(rows) if r['fold'] == fold]
            reading = score_policy([rows[i] for i in ids], action[ids])
            out.append({'policy': name, 'fold': fold, 'rows': len(ids), **flat_counts(reading)})
    return out


def outcome_flags(rows):
    y = outcome_arrays(rows)
    return {'E': y[:, 0], 'B': y[:, 1], 'H': y[:, 2]}


def difference_sets(rows, action, reference, label, reference_label):
    """Exact prompts whose selection differs, and the outcome each difference carries."""
    action = np.asarray(action, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    y = outcome_arrays(rows)
    nn_only = [i for i in range(len(rows)) if action[i] and not reference[i]]
    ref_only = [i for i in range(len(rows)) if reference[i] and not action[i]]

    def ids(indices, group, column, want):
        picked = []
        for i in indices:
            if rows[i]['group'] != group:
                continue
            value = y[i, column]
            if want == 'positive' and value == 1.0:
                picked.append(rows[i]['row_id'])
            elif want == 'unknown' and not np.isfinite(value):
                picked.append(rows[i]['row_id'])
        return picked

    result = {'policy': label, 'reference': reference_label,
              'edits_policy': int(action.sum()), 'edits_reference': int(reference.sum()),
              'edits_in_common': int((action & reference).sum()),
              'selected_by_policy_only': [rows[i]['row_id'] for i in nn_only],
              'selected_by_reference_only': [rows[i]['row_id'] for i in ref_only]}
    result['E_substantive_gained'] = ids(nn_only, 'E', 0, 'positive')
    result['E_substantive_lost'] = ids(ref_only, 'E', 0, 'positive')
    result['E_unknown_selected_gained'] = ids(nn_only, 'E', 0, 'unknown')
    result['E_unknown_selected_lost'] = ids(ref_only, 'E', 0, 'unknown')
    result['B_material_loss_incurred'] = ids(nn_only, 'B', 1, 'positive')
    result['B_material_loss_avoided'] = ids(ref_only, 'B', 1, 'positive')
    result['B_unknown_selected_gained'] = ids(nn_only, 'B', 1, 'unknown')
    result['B_unknown_selected_avoided'] = ids(ref_only, 'B', 1, 'unknown')
    result['H_new_harm_incurred'] = ids(nn_only, 'H', 2, 'positive')
    result['H_new_harm_avoided'] = ids(ref_only, 'H', 2, 'positive')
    result['H_unknown_selected_gained'] = ids(nn_only, 'H', 2, 'unknown')
    result['H_unknown_selected_avoided'] = ids(ref_only, 'H', 2, 'unknown')
    result['net_E_substantive'] = len(result['E_substantive_gained']) - len(result['E_substantive_lost'])
    result['net_B_material_loss'] = len(result['B_material_loss_incurred']) - len(result['B_material_loss_avoided'])
    result['net_H_new_harm'] = len(result['H_new_harm_incurred']) - len(result['H_new_harm_avoided'])
    return result


def harmful_prompt_sets(rows, actions):
    """Which H prompts actually regress under each policy, and whether they are the same ones."""
    y = outcome_arrays(rows)
    per_policy = {}
    for name, action in actions.items():
        action = np.asarray(action, dtype=bool)
        per_policy[name] = {
            'newly_harmful_row_ids': sorted(rows[i]['row_id'] for i in range(len(rows))
                                            if action[i] and rows[i]['group'] == 'H' and y[i, 2] == 1.0),
            'selected_unknown_baseline_row_ids': sorted(rows[i]['row_id'] for i in range(len(rows))
                                                        if action[i] and rows[i]['group'] == 'H'
                                                        and rows[i]['baseline_verdict'] not in ('YES', 'NO'))}
    always = set(per_policy['always_fixed']['newly_harmful_row_ids'])
    overlaps = []
    for name, record in per_policy.items():
        if name in ('no_edit', 'always_fixed'):
            continue
        method, profile = name.split('__')
        mine = set(record['newly_harmful_row_ids'])
        for other_name, other in per_policy.items():
            if other_name == name or '__' not in other_name or other_name.split('__')[1] != profile:
                continue
            theirs = set(other['newly_harmful_row_ids'])
            overlaps.append({'profile': profile, 'policy': name, 'compared_with': other_name,
                             'policy_count': len(mine), 'compared_count': len(theirs),
                             'shared_row_ids': sorted(mine & theirs),
                             'policy_only_row_ids': sorted(mine - theirs),
                             'compared_only_row_ids': sorted(theirs - mine),
                             'same_prompts_regress': bool(mine == theirs)})
        per_policy[name]['fraction_of_always_fixed_harms'] = len(mine & always) / len(always) if always else None
        per_policy[name]['outside_always_fixed_harms'] = sorted(mine - always)
    return {'per_policy': per_policy, 'always_fixed_newly_harmful_row_ids': sorted(always), 'profile_overlaps': overlaps}


def ranking_quality(rows, scores_by_method):
    """Out-of-fold ranking of each gate score against its own outcome, inside the relevant class.

    The reversed figures are a descriptive control. No gate direction is changed anywhere.
    """
    y = outcome_arrays(rows)
    groups = np.asarray([r['group'] for r in rows])
    baseline_no = np.asarray([r['baseline_verdict'] == 'NO' for r in rows])
    domains = {'opportunity': (groups == 'E'), 'benign_loss': (groups == 'B'), 'new_harm': (groups == 'H') & baseline_no}
    table = []
    for method, scores in scores_by_method.items():
        for column, head in enumerate(HEADS):
            domain = domains[head] & np.isfinite(y[:, column])
            target = y[domain, column]
            score = scores[domain, column]
            positives = int(target.sum())
            record = {'method': method, 'head': head, 'target': SHORT[head],
                      'within_class_rows': int(domain.sum()), 'positives': positives,
                      'negatives': int(domain.sum()) - positives,
                      'score_min': float(score.min()) if len(score) else None,
                      'score_max': float(score.max()) if len(score) else None,
                      'auc': None, 'average_precision': None, 'binary_log_loss': None,
                      'reversed_auc': None, 'reversed_average_precision': None,
                      'reversed_is_descriptive_control_only': True, 'gate_direction_unchanged': True}
            if 0 < positives < int(domain.sum()):
                record['auc'] = float(roc_auc_score(target, score))
                record['average_precision'] = float(average_precision_score(target, score))
                record['reversed_auc'] = float(roc_auc_score(target, -score))
                record['reversed_average_precision'] = float(average_precision_score(target, -score))
            full = np.full(len(rows), np.nan)
            full[domain] = target
            record['binary_log_loss'] = binary_loss(full, scores[:, column])
            table.append(record)
    return table


def class_membership_quality(rows, scores_by_method):
    groups = np.asarray([r['group'] for r in rows])
    table = []
    for method, scores in scores_by_method.items():
        for column, group in enumerate('EBH'):
            target = (groups == group).astype(float)
            table.append({'method': method, 'head': HEADS[column], 'membership_class': group,
                          'rows': len(rows), 'positives': int(target.sum()),
                          'auc': float(roc_auc_score(target, scores[:, column]))})
    return table


def markdown_table(header, alignment, body):
    return ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(alignment) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in body]


def build_report(output, rows, actions, results, per_fold, differences, harm_sets, ranking, convergence, checks, design):
    lines = ['# Does a small nonlinear selector beat the linear selectors?', '',
             'One fixed repair edit (strength 0.15, zero-based layers 31-32), 480 development prompts, frozen unsteered state 32, five original outer folds.',
             'Every selector chooses between the same saved baseline reply and the same saved fixed-edit reply.', '']
    body = []
    for name, reading in results.items():
        counts = flat_counts(reading)
        method, profile = (name.split('__') + ['-'])[:2]
        body.append([method, profile, counts['edits'],
                     '{} / 89'.format(counts['E_retained']),
                     '{} + {} unknown / 178'.format(counts['B_material_loss'], counts['B_unknown_selected']),
                     '{} + {} unknown / 212'.format(counts['H_new_harm'], counts['H_unknown_selected']),
                     counts['H_selected_unknown_baseline']])
    lines += ['## Headline comparison', ''] + markdown_table(
        ['Method', 'Validation budget', 'Edits / 480', 'E substantive retained', 'B material loss', 'H newly harmful', 'H unknown-baseline rows selected'],
        ['---', '---', '---:', '---:', '---:', '---:', '---:'], body) + ['']

    lines += ['## Results by outer fold', '']
    fold_body = [[r['policy'], r['fold'], r['edits'], r['E_retained'], '{}+{}u'.format(r['B_material_loss'], r['B_unknown_selected']),
                  '{}+{}u'.format(r['H_new_harm'], r['H_unknown_selected'])] for r in per_fold]
    lines += markdown_table(['Policy', 'Fold', 'Edits', 'E retained', 'B material loss', 'H new harm'],
                            ['---', '---:', '---:', '---:', '---:', '---:'], fold_body) + ['']

    lines += ['## Exact prompts gained and lost by the network, against each linear selector', '']
    diff_body = []
    for d in differences:
        diff_body.append([d['policy'], d['reference'], d['edits_policy'], d['edits_reference'],
                          len(d['E_substantive_gained']), len(d['E_substantive_lost']), d['net_E_substantive'],
                          len(d['B_material_loss_incurred']), len(d['B_material_loss_avoided']),
                          len(d['H_new_harm_incurred']), len(d['H_new_harm_avoided'])])
    lines += markdown_table(['Network policy', 'Linear reference', 'Edits (net)', 'Edits (ref)', 'E gained', 'E lost', 'E net',
                             'B losses incurred', 'B losses avoided', 'H harms incurred', 'H harms avoided'],
                            ['---', '---', '---:', '---:', '---:', '---:', '---:', '---:', '---:', '---:', '---:'], diff_body)
    lines += ['', 'The row identifiers of every gained and lost prompt are in `gained_lost_prompts.json` and `gained_lost_prompts.csv`; they key into `cases.jsonl` of the frozen audit.', '']

    lines += ['## Do the same H prompts regress?', '']
    harm_body = [[o['profile'], o['policy'], o['compared_with'], o['policy_count'], o['compared_count'],
                  len(o['shared_row_ids']), 'yes' if o['same_prompts_regress'] else 'no']
                 for o in harm_sets['profile_overlaps'] if o['policy'].startswith('nn_')]
    lines += markdown_table(['Profile', 'Network policy', 'Compared with', 'Network harms', 'Reference harms', 'Shared prompts', 'Same prompts regress'],
                            ['---', '---', '---', '---:', '---:', '---:', '---'], harm_body) + ['']

    lines += ['## Within-class prediction quality (out of fold)', '',
              'AUC of each gate score against its own outcome, computed only inside that head\'s class. The reversed column is a descriptive ranking control: it shows what the score would achieve read backwards. No gate was reversed.', '']
    rank_body = []
    for r in ranking:
        fmt = lambda v: '-' if v is None else '{:.3f}'.format(v)
        rank_body.append([r['method'], r['head'], '{} ({} pos)'.format(r['within_class_rows'], r['positives']),
                          fmt(r['auc']), fmt(r['average_precision']), fmt(r['reversed_auc']), '{:.4f}'.format(r['binary_log_loss'])])
    lines += markdown_table(['Method', 'Head', 'Within-class rows', 'AUC', 'Average precision', 'Reversed AUC (control)', 'Log loss'],
                            ['---', '---', '---:', '---:', '---:', '---:', '---:'], rank_body) + ['']

    lines += ['## Model convergence', '',
              '- Network fits performed: {}'.format(convergence['network_fits']),
              '- Converged before max_iter={}: {}'.format(design['nn']['max_iter'], convergence['converged_before_max_iter']),
              '- Hit the iteration cap: {}'.format(convergence['hit_max_iter']),
              '- Median iterations used: {}'.format(convergence['median_iterations']),
              '- Largest final training loss: {}'.format(None if convergence['max_final_training_loss'] is None else round(convergence['max_final_training_loss'], 6)), '']

    lines += ['## Checks', ''] + ['- {}: {}'.format(k, v) for k, v in checks.items()] + ['']
    lines += ['## Reading notes', ''] + ['- ' + n for n in NOTES] + ['']
    (Path(output) / 'REPORT.md').write_text('\n'.join(lines) + '\n')

    page = ['<!doctype html><meta charset="utf-8"><title>Selector capacity comparison</title>',
            '<style>body{font:16px/1.6 system-ui;max-width:1200px;padding:28px;margin:auto;color:#263743}h1,h2{color:#244b78}',
            'table{border-collapse:collapse;font-size:13px;margin-bottom:18px}th,td{padding:7px 10px;border-bottom:1px solid #d7e2e8;text-align:left}',
            'th{background:#e8f4f5}img{max-width:100%}code{background:#f2f6f8;padding:1px 4px}</style>']
    in_table = False
    for line in lines:
        if line.startswith('|'):
            cells = [c.strip() for c in line.strip('|').split('|')]
            if set(''.join(cells)) <= set('-:'):
                continue
            if not in_table:
                page.append('<div style="overflow:auto"><table>')
                in_table = True
                tag = 'th'
            else:
                tag = 'td'
            page.append('<tr>' + ''.join('<' + tag + '>' + html.escape(c) + '</' + tag + '>' for c in cells) + '</tr>')
        else:
            if in_table:
                page.append('</table></div>')
                in_table = False
            if line.startswith('# '):
                page.append('<h1>' + html.escape(line[2:]) + '</h1>')
            elif line.startswith('## '):
                page.append('<h2>' + html.escape(line[3:]) + '</h2>')
            elif line.startswith('- '):
                page.append('<p>&bull; ' + html.escape(line[2:]) + '</p>')
            elif line.strip():
                page.append('<p>' + html.escape(line) + '</p>')
    if in_table:
        page.append('</table></div>')
    (Path(output) / 'REPORT.html').write_text(''.join(page))


def write_outputs(output, rows, actions, design):
    output = Path(output)
    results = {name: score_policy(rows, action) for name, action in actions.items()}
    write_csv(output / 'summary.csv', [{'policy': n, **flat_counts(r)} for n, r in results.items()])
    per_fold = per_fold_table(rows, actions)
    write_csv(output / 'results_by_fold.csv', per_fold)
    return results, per_fold


def make_plot(output, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    directory = Path(output) / 'plots'
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    colour = {'class_selector': '#30669A', 'outcome_selector': '#2597A5', 'nn_outcome_selector': '#9B4F96',
              'no_edit': '#667786', 'always_fixed': '#BD8248'}
    markers = {'zero_validation_damage': 'o', 'H1pct_B2pct_validation': 's', 'H5pct_B5pct_validation': '^'}
    max_damage = [0, 0]
    for name, r in results.items():
        parts = name.split('__')
        method, profile = parts[0], parts[1] if len(parts) > 1 else None
        y = r['E_fixed_substantive_selected']['positive']
        for j, key in enumerate(('H_new_harm_baseline_NO', 'B_material_loss')):
            d = r[key]
            max_damage[j] = max(max_damage[j], d['positive'] + d['unknown'])
            axes[j].errorbar(d['positive'], y, xerr=[[0], [d['unknown']]], fmt=markers.get(profile, 'D'), ms=8,
                             color=colour[method], capsize=3, alpha=.85)
            if method in ('no_edit', 'always_fixed'):
                axes[j].annotate('No edit' if method == 'no_edit' else 'Always fixed', (d['positive'], y),
                                 xytext=(5, 7 if y == 0 else -16), textcoords='offset points', fontsize=9, color=colour[method])
    for j, (ax, label) in enumerate(zip(axes, ('Newly harmful H responses (of 212)', 'Material B answer losses (of 178)'))):
        ax.set_xlabel(label + ' - lower is better')
        ax.set_ylabel('Fixed-edit E substantive answers retained (of 89)')
        ax.set_xlim(-.6, max_damage[j] + 2)
        ax.set_ylim(-3, 70)
        ax.grid(alpha=.15)
    fig.suptitle('Linear versus one-hidden-layer selection of the same fixed repair edit', fontweight='bold', fontsize=13, y=.98)
    handles = [Line2D([0], [0], color=colour[k], marker='o', linestyle='', label=v) for k, v in
               [('class_selector', 'Linear class-based'), ('outcome_selector', 'Linear outcome-based'), ('nn_outcome_selector', 'Network outcome-based (16 ReLU)')]]
    handles += [Line2D([0], [0], color='#667786', marker=m, linestyle='', label=l) for m, l in
                [('o', 'Zero validation damage'), ('s', 'H 1% / B 2% validation budget'), ('^', 'H 5% / B 5% validation budget')]]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .06), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .015, 'Nested out-of-fold development predictions; no new generations. Rightward bars add selected unknown outcomes.\n'
                       'E measures retained fixed-edit answers, not complete-system repair. Profiles are validation settings, not safety guarantees.',
             ha='center', fontsize=9, color='#526b79')
    fig.subplots_adjust(left=.08, right=.98, top=.88, bottom=.30, wspace=.32)
    for suffix in ('svg', 'pdf', 'png'):
        fig.savefig(directory / ('capacity_tradeoff.' + suffix), dpi=220)
    plt.close(fig)
