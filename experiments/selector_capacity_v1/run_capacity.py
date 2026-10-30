#!/usr/bin/env python3
"""Linear versus one-hidden-layer selection of the same fixed repair edit.

CPU only. Saved responses only. No new generations, no judge calls, no label changes.
"""
import argparse
import csv
import io
import json
import platform
import sys
import tarfile
from pathlib import Path

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

import capacity_reporting
from capacity_reporting import (build_report, class_membership_quality, difference_sets,
                                harmful_prompt_sets, make_plot, ranking_quality, write_outputs)
from inputs import load_inputs, sha256, verify_unchanged, development_path
from nn_core import METHOD as NN_METHOD, convergence_summary, fit_outer_nn
from reporting import write_csv
from run_comparison import output_allowed, write_json
from selector_core import fit_outer, outcome_arrays, score_policy

HERE = Path(__file__).resolve().parent
LINEAR_METHODS = ('class_selector', 'outcome_selector')


def load_designs(repo):
    design = json.loads((HERE / 'design.json').read_text())
    base_path = development_path(repo / 'experiments/outcome_selector_v1/design.json')
    actual = sha256(base_path)
    if actual != design['linear_design_sha256']:
        raise ValueError('The reused linear design has changed: ' + str(base_path))
    base = json.loads(base_path.read_text())
    for name, expected in design['reused_code_sha256'].items():
        path = repo / 'experiments/outcome_selector_v1' / name
        if sha256(path) != expected:
            raise ValueError('Reused module has changed since this design was registered: ' + str(path))
    for key in ('head_order', 'threshold_quantiles', 'profiles'):
        if base[key] != design[key]:
            raise ValueError('Shared evaluation procedure differs from outcome_selector_v1: ' + key)
    if base['read_state'] != design['read_state'] or base['edit'] != design['edit']:
        raise ValueError('The read state or the steering edit differs from outcome_selector_v1')
    return design, base


def target_consistency(rows):
    """The frozen inventory the two completed linear arms were scored against."""
    y = outcome_arrays(rows)
    everything = np.ones(len(rows), dtype=bool)
    fixed = score_policy(rows, everything)
    observed = {
        'rows': len(rows),
        'class_counts': {g: int(sum(r['group'] == g for r in rows)) for g in 'EBH'},
        'E_fixed_substantive': fixed['E_fixed_substantive_selected']['positive'],
        'E_population': fixed['E_fixed_substantive_selected']['population'],
        'B_material_loss': fixed['B_material_loss']['positive'],
        'B_unknown': fixed['B_material_loss']['unknown'],
        'B_population': fixed['B_material_loss']['population'],
        'H_new_harm': fixed['H_new_harm_baseline_NO']['positive'],
        'H_known_non_harmful_baselines': fixed['H_new_harm_baseline_NO']['population'],
        'H_unknown_baseline_rows': int(sum(r['group'] == 'H' and r['baseline_verdict'] not in ('YES', 'NO') for r in rows)),
        'unknown_targets_by_head': [int((~np.isfinite(y[:, j])).sum()) for j in range(3)]}
    expected = {'rows': 480, 'class_counts': {'E': 89, 'B': 178, 'H': 213},
                'E_fixed_substantive': 64, 'E_population': 89, 'B_material_loss': 16, 'B_unknown': 2,
                'B_population': 178, 'H_new_harm': 10, 'H_known_non_harmful_baselines': 212,
                'H_unknown_baseline_rows': 1}
    for key, value in expected.items():
        if observed[key] != value:
            raise ValueError('Frozen inventory differs from the completed comparison: {} is {}, expected {}'.format(key, observed[key], value))
    return observed


def split_separation(rows, fitted_by_method):
    """Assert out-of-fold coverage and training/evaluation disjointness for every fitted model."""
    all_ids = [r['row_id'] for r in rows]
    folds = [r['fold'] for r in rows]
    records = []
    for method, fitted_folds in fitted_by_method.items():
        covered = np.zeros(len(rows), dtype=int)
        for fitted in fitted_folds:
            test_idx = list(fitted['test_indices'])
            covered[test_idx] += 1
            test_ids = {all_ids[i] for i in test_idx}
            train_ids = {all_ids[i] for i in range(len(rows)) if i not in set(test_idx)}
            if test_ids & train_ids:
                raise ValueError('Training and evaluation rows overlap')
            if {folds[i] for i in test_idx} != {fitted['outer_fold']}:
                raise ValueError('Evaluation fold membership is wrong')
            if fitted['outer_fold'] in fitted['training_folds']:
                raise ValueError('The outer fold was used for training')
            for history in fitted['head_selection']:
                for candidate in history['candidates']:
                    for fit in candidate['fits']:
                        if fitted['outer_fold'] in fit['training_folds'] or fit['validation_fold'] in fit['training_folds']:
                            raise ValueError('Inner fold leakage')
            records.append({'method': method, 'outer_fold': fitted['outer_fold'],
                            'training_rows': len(train_ids), 'evaluation_rows': len(test_ids),
                            'training_ids_sha256': fitted['training_ids_sha256'],
                            'evaluation_ids_sha256': fitted['test_ids_sha256']})
        if not np.all(covered == 1):
            raise ValueError('Each row must be evaluated exactly once out of fold')
    return records


def reproduces_completed_run(reference, rows, actions):
    """The two linear arms must reproduce experiments/outcome_selector_v1 decision for decision."""
    path = development_path(Path(reference)) / 'selected_actions.csv'
    if not path.is_file():
        return {'status': 'REFERENCE_NOT_PRESENT', 'path': str(path)}
    previous = {}
    for row in csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig'))):
        previous[(row['policy'], row['row_id'])] = row['selected_action']
    order = [r['row_id'] for r in rows]
    mismatches, compared = [], 0
    for name, action in actions.items():
        if name.split('__')[0] not in LINEAR_METHODS:
            continue
        for i, row_id in enumerate(order):
            key = (name, row_id)
            if key not in previous:
                return {'status': 'REFERENCE_POLICY_MISSING', 'policy': name, 'path': str(path)}
            compared += 1
            if previous[key] != ('fixed' if action[i] else 'baseline'):
                mismatches.append({'policy': name, 'row_id': row_id})
    return {'status': 'IDENTICAL' if not mismatches else 'DIFFERS', 'decisions_compared': compared,
            'mismatches': mismatches, 'reference': str(path)}


def run(args):
    if args.stage == 'status':
        summary = args.output / 'summary.json'
        if summary.is_file() and (args.output / 'COMPLETE.json').is_file():
            value = json.loads(summary.read_text())
            value['output'] = str(args.output)
            print(json.dumps(value, indent=2))
        else:
            print(json.dumps({'status': 'NO_COMPLETE_COMPARISON', 'output': str(args.output),
                              'next_command': 'bash experiments/selector_capacity_v1/run.sh run'}, indent=2))
        return
    print('[inputs] Verifying the frozen audit, fold map, cached activations and the reused linear design.', flush=True)
    design, base_design = load_designs(args.repo)
    rows, x, provenance = load_inputs(args.repo, args.audit, args.paths)
    inventory = target_consistency(rows)
    code_hashes = {p.name: sha256(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py', '.json', '.sh', '.md')}
    print(json.dumps({'stage': 'preflight', 'rows': len(rows), 'feature_shape': list(x.shape),
                      'read_state': design['read_state'], 'fixed_strength': design['edit']['strength'],
                      'layers': design['edit']['zero_based_layers'], 'folds': design['outer_folds'],
                      'methods': design['methods'], 'nn_hidden_units': design['nn']['hidden_layer_sizes'],
                      'nn_alpha_values': design['nn']['alpha_values'], 'nn_seeds': design['nn']['seeds'],
                      'frozen_inventory': inventory, 'new_generations': 0, 'api_calls': 0, 'uses_GPU': False,
                      'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__}, indent=2), flush=True)
    if args.stage == 'preflight':
        return

    out = output_allowed(args.output, provenance)
    archive = out.parent / (out.name + '.tar.gz')
    if archive.exists():
        raise ValueError('Output archive already exists and was preserved: ' + str(archive))
    out.mkdir(parents=True)
    write_json(out / 'run_manifest.json', {'status': 'STARTED', 'design': design, 'linear_design': base_design,
                                           'source_files': provenance, 'code_sha256': code_hashes,
                                           'frozen_inventory': inventory, 'python': platform.python_version(),
                                           'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
                                           'new_generations': 0, 'api_calls': 0, 'uses_GPU': False, 'label_changes': 0})
    models = out / 'fold_models'
    models.mkdir()

    actions = {'no_edit': np.zeros(len(rows), dtype=bool), 'always_fixed': np.ones(len(rows), dtype=bool)}
    scores_by_method, fitted_by_method, prediction_rows = {}, {}, []
    for method in design['methods']:
        for profile in design['profiles']:
            actions[method + '__' + profile['name']] = np.zeros(len(rows), dtype=bool)
        scores_by_method[method] = np.full((len(rows), 3), np.nan)
        fitted_by_method[method] = []
        for outer in range(design['outer_folds']):
            print('[fit] {} | outer fold {} of 0-{}'.format(method, outer, design['outer_folds'] - 1), flush=True)
            if method == NN_METHOD:
                fitted, weights = fit_outer_nn(x, rows, outer, design)
                np.savez_compressed(str(models / (method + '_fold_' + str(outer) + '_weights.npz')), **weights)
                record = {k: v for k, v in fitted.items() if k not in ('inner_validation_scores', 'inner_validation_indices')}
                write_json(models / (method + '_fold_' + str(outer) + '_inner_validation.json'),
                           {'outer_fold': outer, 'row_ids': [rows[i]['row_id'] for i in fitted['inner_validation_indices']],
                            'seed_averaged_inner_scores': fitted['inner_validation_scores']})
            else:
                fitted = fit_outer(x, rows, outer, method, base_design)
                record = fitted
            write_json(models / (method + '_fold_' + str(outer) + '.json'), record)
            fitted_by_method[method].append(record)
            indices = np.asarray(fitted['test_indices'])
            scores_by_method[method][indices] = np.asarray(fitted['test_scores'])
            for profile, values in fitted['actions'].items():
                actions[method + '__' + profile][indices] = values
            for local, i in enumerate(indices):
                prediction_rows.append({'row_id': rows[i]['row_id'], 'group': rows[i]['group'], 'fold': rows[i]['fold'],
                                        'outer_fold': outer, 'method': method,
                                        'opportunity_score': fitted['test_scores'][local][0],
                                        'benign_loss_score': fitted['test_scores'][local][1],
                                        'new_harm_score': fitted['test_scores'][local][2],
                                        'training_folds': ' '.join(str(v) for v in fitted['training_folds'])})
        if not np.isfinite(scores_by_method[method]).all():
            raise ValueError('Each row must be predicted once out of fold for ' + method)

    separation = split_separation(rows, fitted_by_method)
    write_csv(out / 'split_separation.csv', separation)
    write_csv(out / 'prediction_scores.csv', prediction_rows)

    decisions = []
    for name, action in actions.items():
        for i, row in enumerate(rows):
            chosen = 'fixed' if action[i] else 'baseline'
            decisions.append({'row_id': row['row_id'], 'fold': row['fold'], 'group': row['group'], 'policy': name,
                              'selected_action': chosen, 'selected_response_sha256': row[chosen + '_response_sha256'],
                              'fixed_verdict': row['fixed_verdict'], 'baseline_verdict': row['baseline_verdict'],
                              'E_baseline_behaviour_scored': False})
    write_csv(out / 'selected_actions.csv', decisions)

    results, per_fold = write_outputs(out, rows, actions, design)

    differences = []
    for profile in design['profiles']:
        nn_name = NN_METHOD + '__' + profile['name']
        for reference in LINEAR_METHODS:
            reference_name = reference + '__' + profile['name']
            differences.append(difference_sets(rows, actions[nn_name], actions[reference_name], nn_name, reference_name))
        differences.append(difference_sets(rows, actions[nn_name], actions['always_fixed'], nn_name, 'always_fixed'))
    write_json(out / 'gained_lost_prompts.json', differences)
    long_rows = []
    for d in differences:
        for key, values in d.items():
            if isinstance(values, list):
                for row_id in values:
                    row = next(r for r in rows if r['row_id'] == row_id)
                    long_rows.append({'policy': d['policy'], 'reference': d['reference'], 'category': key,
                                      'row_id': row_id, 'group': row['group'], 'fold': row['fold'],
                                      'baseline_verdict': row['baseline_verdict'], 'fixed_verdict': row['fixed_verdict']})
    write_csv(out / 'gained_lost_prompts.csv', long_rows or [{'policy': '', 'reference': '', 'category': 'none',
                                                              'row_id': '', 'group': '', 'fold': '',
                                                              'baseline_verdict': '', 'fixed_verdict': ''}])

    harm_sets = harmful_prompt_sets(rows, actions)
    write_json(out / 'h_regression_prompts.json', harm_sets)
    ranking = ranking_quality(rows, scores_by_method)
    write_csv(out / 'within_class_quality.csv', ranking)
    membership = class_membership_quality(rows, scores_by_method)
    write_csv(out / 'class_membership_quality.csv', membership)
    convergence = convergence_summary(fitted_by_method[NN_METHOD])
    write_json(out / 'convergence.json', convergence)

    reproduction = reproduces_completed_run(args.reference, rows, actions)
    checks = {
        'split_separation': 'PASS: every row evaluated exactly once out of fold; no training row shares its fold with its evaluation fold; inner folds disjoint from outer',
        'target_and_scoring_consistency': 'PASS: {} E substantive of {}, {} B material losses (+{} unknown) of {}, {} H new harms of {} known non-harmful baselines, {} unknown-baseline H row kept separate'.format(
            inventory['E_fixed_substantive'], inventory['E_population'], inventory['B_material_loss'], inventory['B_unknown'],
            inventory['B_population'], inventory['H_new_harm'], inventory['H_known_non_harmful_baselines'], inventory['H_unknown_baseline_rows']),
        'linear_arms_reproduce_outcome_selector_v1': reproduction['status'] + ' ({} decisions compared)'.format(reproduction.get('decisions_compared', 0)),
        'model_convergence': '{} of {} network fits converged before max_iter={}; median {} L-BFGS iterations'.format(
            convergence['converged_before_max_iter'], convergence['network_fits'], design['nn']['max_iter'], convergence['median_iterations']),
        'seed_handling': 'all scores are the mean of {} fixed seeds {} at inner validation and at outer evaluation; no seed selected'.format(
            len(design['nn']['seeds']), design['nn']['seeds']),
        'label_changes': 0, 'new_generations': 0, 'judge_api_calls': 0, 'uses_GPU': False,
        'external_evaluation': False, 'final_all_data_refit': False}
    write_json(out / 'checks.json', {'checks': checks, 'reproduction_detail': reproduction,
                                     'split_separation_records': separation, 'frozen_inventory': inventory})
    if reproduction['status'] not in ('IDENTICAL', 'REFERENCE_NOT_PRESENT'):
        raise ValueError('The linear arms did not reproduce the completed comparison: ' + json.dumps(reproduction['mismatches'][:5]))

    plot_status = 'created'
    try:
        make_plot(out, results)
    except ImportError:
        plot_status = 'matplotlib unavailable; all numerical results complete'
    build_report(out, rows, actions, results, per_fold, differences, harm_sets, ranking, convergence, checks, design)
    if plot_status == 'created':
        with (out / 'REPORT.html').open('a') as handle:
            handle.write('<h2>Observed development trade-offs</h2><img src="plots/capacity_tradeoff.png" alt="Repair opportunity against preservation damage for the three selectors">')

    summary = {'status': 'CAPACITY_COMPARISON_COMPLETE', 'rows': len(rows), 'methods': design['methods'],
               'new_generations': 0, 'api_calls': 0, 'final_refit': False, 'plot_status': plot_status,
               'checks': checks, 'convergence': {k: v for k, v in convergence.items() if k != 'non_converged_fits'},
               'notes': capacity_reporting.NOTES, 'policies': results}
    write_json(out / 'summary.json', summary)

    verify_unchanged(provenance)
    if any(sha256(HERE / name) != value for name, value in code_hashes.items()):
        raise ValueError('Code changed during the run')
    write_json(out / 'COMPLETE.json', {'status': 'COMPLETE', 'rows': len(rows), 'methods': design['methods'],
                                       'outer_prediction_rows': len(prediction_rows), 'new_generations': 0,
                                       'api_calls': 0, 'frozen_files_changed': 0})
    sums = [sha256(p) + '  ' + p.relative_to(out).as_posix() for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS']
    (out / 'SHA256SUMS').write_text('\n'.join(sums) + '\n')
    if archive.exists():
        raise ValueError('Comparison completed, but the archive path already exists and was preserved: ' + str(archive))
    with tarfile.open(str(archive), 'w:gz') as handle:
        handle.add(str(out), arcname=out.name)
    headline = {name: {'edits': r['edits'], 'E_of_89': r['E_fixed_substantive_selected']['positive'],
                       'B_material_loss_of_178': r['B_material_loss']['positive'],
                       'B_unknown': r['B_material_loss']['unknown'],
                       'H_new_harm_of_212': r['H_new_harm_baseline_NO']['positive']} for name, r in results.items()}
    print(json.dumps({'status': 'CAPACITY_COMPARISON_COMPLETE', 'output': str(out), 'download_archive': str(archive),
                      'checks': checks, 'headline': headline}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('preflight', 'run', 'status'))
    parser.add_argument('--repo', type=Path, default=HERE.parents[1])
    parser.add_argument('--audit', type=Path, default=Path.home() / 'routegate-data/controller_error_audit_v1')
    parser.add_argument('--output', type=Path, default=Path.home() / 'routegate-data/selector_capacity_v1')
    parser.add_argument('--reference', type=Path, default=Path.home() / 'routegate-data/outcome_selector_v1')
    parser.add_argument('--paths', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        try:
            run(args)
        except (ValueError, OSError, KeyError) as error:
            print('STOP: ' + str(error), file=sys.stderr)
            raise SystemExit(1)


if __name__ == '__main__':
    main()
