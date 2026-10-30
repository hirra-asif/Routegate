#!/usr/bin/env python3
"""CPU checks for the nonlinear selector arm. Synthetic vectors and synthetic labels only."""
import copy
import json
import unittest
from pathlib import Path

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from capacity_reporting import (difference_sets, harmful_prompt_sets, per_fold_table,
                                ranking_quality, flat_counts)
from nn_core import convergence_summary, fit_nn_head, fit_outer_nn, predict_nn_head
from run_capacity import split_separation, target_consistency
from selector_core import apply_rule, outcome_arrays, score_policy, targets
from test_selector import fixture

HERE = Path(__file__).resolve().parent
DESIGN = json.loads((HERE / 'design.json').read_text())
FAST = dict(DESIGN, nn=dict(DESIGN['nn'], max_iter=120))


class CapacityTests(unittest.TestCase):
    def test_declared_architecture_search_and_seeds(self):
        nn = DESIGN['nn']
        self.assertEqual(nn['hidden_layer_sizes'], [16])
        self.assertEqual(nn['activation'], 'relu')
        self.assertEqual(len(nn['seeds']), 3)
        self.assertEqual(len(set(nn['seeds'])), 3)
        self.assertEqual(len(nn['alpha_values']), 3)
        self.assertTrue(all(a > 0 for a in nn['alpha_values']))
        self.assertTrue(DESIGN['preregistered_before_running'])
        self.assertEqual(DESIGN['head_order'], ['opportunity', 'benign_loss', 'new_harm'])

    def test_targets_are_the_existing_outcome_definitions(self):
        rows, _ = fixture(60)
        np.testing.assert_array_equal(np.nan_to_num(targets(rows, 'outcome_selector'), nan=-9),
                                      np.nan_to_num(outcome_arrays(rows), nan=-9))

    def test_prediction_is_the_plain_mean_of_the_three_seeds(self):
        rows, x = fixture(60, 12)
        y = outcome_arrays(rows)[:, 0]
        with threadpool_limits(limits=1):
            model = fit_nn_head(x, y, 100.0, FAST['nn'])
        self.assertEqual(model['kind'], 'mlp')
        self.assertEqual(len(model['_arrays']['nets']), 3)
        z = (x - model['_arrays']['mean']) / model['_arrays']['scale']
        manual = np.mean([expit(np.maximum(z @ w1 + b1, 0.0) @ w2 + b2)
                          for w1, b1, w2, b2 in model['_arrays']['nets']], axis=0)
        np.testing.assert_allclose(predict_nn_head(model, x), manual, rtol=0, atol=1e-12)
        self.assertEqual([f['seed'] for f in model['seed_fits']], DESIGN['nn']['seeds'])

    def test_seeds_are_not_identical_so_averaging_does_something(self):
        rows, x = fixture(60, 12)
        y = outcome_arrays(rows)[:, 0]
        with threadpool_limits(limits=1):
            model = fit_nn_head(x, y, 10.0, FAST['nn'])
        z = (x - model['_arrays']['mean']) / model['_arrays']['scale']
        per_seed = [expit(np.maximum(z @ w1 + b1, 0.0) @ w2 + b2) for w1, b1, w2, b2 in model['_arrays']['nets']]
        self.assertGreater(float(np.max(np.abs(per_seed[0] - per_seed[1]))), 0.0)

    def test_constant_head_matches_the_linear_fallback(self):
        rows, x = fixture(30, 6)
        y = np.zeros(len(rows))
        model = fit_nn_head(x, y, 10.0, FAST['nn'])
        self.assertEqual(model['kind'], 'constant')
        self.assertAlmostEqual(model['value'], 1.0 / (len(rows) + 2))
        np.testing.assert_allclose(predict_nn_head(model, x), np.full(len(rows), model['value']))

    def test_unknown_targets_are_dropped_not_imputed(self):
        rows, x = fixture(60, 6)
        y = outcome_arrays(rows)[:, 0].copy()
        y[:6] = np.nan
        model = fit_nn_head(x, y, 100.0, FAST['nn'])
        self.assertEqual(model['unknown_training_rows'], 6)
        self.assertEqual(model['known_training_rows'], len(rows) - 6)

    def test_outer_fold_labels_and_features_never_enter_fitting(self):
        rows, x = fixture(75, 10)
        with threadpool_limits(limits=1):
            first, first_weights = fit_outer_nn(x, rows, 0, FAST)
            changed, changed_x = copy.deepcopy(rows), x.copy()
            for i, row in enumerate(changed):
                if row['fold'] == 0:
                    row['fixed_verdict'] = {'E': 'DR' if row['fixed_verdict'] == 'SA' else 'SA',
                                            'B': 'BEFORE_BETTER|MATERIAL', 'H': 'YES'}[row['group']]
                    changed_x[i] = 1e5
            second, second_weights = fit_outer_nn(changed_x, changed, 0, FAST)
        self.assertEqual(first['models'], second['models'])
        self.assertEqual(first['head_selection'], second['head_selection'])
        self.assertEqual(first['rules'], second['rules'])
        self.assertEqual(sorted(first_weights), sorted(second_weights))
        for key in first_weights:
            np.testing.assert_array_equal(first_weights[key], second_weights[key])
        self.assertNotIn(0, first['training_folds'])

    def test_inner_folds_are_disjoint_from_their_validation_fold(self):
        rows, x = fixture(75, 10)
        with threadpool_limits(limits=1):
            fitted, _ = fit_outer_nn(x, rows, 2, FAST)
        for history in fitted['head_selection']:
            for candidate in history['candidates']:
                self.assertEqual(len(candidate['fits']), 4)
                for fit in candidate['fits']:
                    self.assertNotIn(fit['validation_fold'], fit['training_folds'])
                    self.assertNotIn(2, fit['training_folds'])
                    self.assertEqual(len(fit['seed_fits']), 3)

    def test_alpha_ties_favour_stronger_regularisation(self):
        candidates = [{'alpha': a, 'loss': 0.5} for a in (10.0, 100.0, 1000.0)]
        self.assertEqual(min(candidates, key=lambda c: (c['loss'], -c['alpha']))['alpha'], 1000.0)
        candidates[0]['loss'] = 0.4
        self.assertEqual(min(candidates, key=lambda c: (c['loss'], -c['alpha']))['alpha'], 10.0)

    def test_split_separation_rejects_a_leaking_record(self):
        rows, x = fixture(75, 10)
        with threadpool_limits(limits=1):
            fitted, _ = fit_outer_nn(x, rows, 0, FAST)
        leaked = copy.deepcopy(fitted)
        leaked['training_folds'] = [0, 1, 2, 3, 4]
        with self.assertRaises(ValueError):
            split_separation(rows, {'nn_outcome_selector': [leaked]})
        partial = copy.deepcopy(fitted)
        with self.assertRaises(ValueError):
            split_separation(rows, {'nn_outcome_selector': [partial]})  # only one of five folds covered

    def test_gate_direction_and_reversed_control_are_separate(self):
        scores = np.array([[0.9, 0.1, 0.2], [0.9, 0.1, 0.8]])
        rule = {'no_edit': False, 'thresholds': [0.5, 0.5, 0.5]}
        np.testing.assert_array_equal(apply_rule(scores, rule), np.array([True, False]))
        rows, _ = fixture(90, 6)
        rng = np.random.default_rng(3)
        table = ranking_quality(rows, {'m': rng.random((len(rows), 3))})
        for record in table:
            self.assertTrue(record['reversed_is_descriptive_control_only'])
            self.assertTrue(record['gate_direction_unchanged'])
            if record['auc'] is not None:
                self.assertAlmostEqual(record['auc'] + record['reversed_auc'], 1.0, places=9)

    def test_within_class_domains_exclude_other_classes_and_unknowns(self):
        rows, _ = fixture(90, 6)
        unknown_e = next(i for i, r in enumerate(rows) if r['group'] == 'E')
        rows[unknown_e]['fixed_verdict'] = 'UM'  # an unknown E judgement
        scores = np.full((len(rows), 3), 0.5)
        scores[:, 0] = np.linspace(0, 1, len(rows))
        table = {r['head']: r for r in ranking_quality(rows, {'m': scores})}
        all_e = sum(1 for r in rows if r['group'] == 'E')
        self.assertEqual(table['opportunity']['within_class_rows'], all_e - 1)
        self.assertEqual(table['benign_loss']['within_class_rows'],
                         sum(1 for r in rows if r['group'] == 'B'))
        self.assertEqual(table['new_harm']['within_class_rows'],
                         sum(1 for r in rows if r['group'] == 'H' and r['baseline_verdict'] == 'NO'))

    def test_gained_and_lost_prompts_are_exact(self):
        rows, _ = fixture(90, 6)
        e_rows = [i for i, r in enumerate(rows) if r['group'] == 'E' and r['fixed_verdict'] == 'SA']
        policy = np.zeros(len(rows), dtype=bool)
        reference = np.zeros(len(rows), dtype=bool)
        policy[e_rows[0]] = True
        reference[e_rows[1]] = True
        diff = difference_sets(rows, policy, reference, 'policy', 'reference')
        self.assertEqual(diff['E_substantive_gained'], [rows[e_rows[0]]['row_id']])
        self.assertEqual(diff['E_substantive_lost'], [rows[e_rows[1]]['row_id']])
        self.assertEqual(diff['net_E_substantive'], 0)
        self.assertEqual(diff['edits_in_common'], 0)

    def test_same_h_prompts_regress_detection(self):
        rows, _ = fixture(90, 6)
        harmful = [i for i, r in enumerate(rows) if r['group'] == 'H' and r['fixed_verdict'] == 'YES' and r['baseline_verdict'] == 'NO']
        self.assertGreaterEqual(len(harmful), 2)
        a = np.zeros(len(rows), dtype=bool); a[harmful[0]] = True
        b = np.zeros(len(rows), dtype=bool); b[harmful[0]] = True
        c = np.zeros(len(rows), dtype=bool); c[harmful[1]] = True
        actions = {'no_edit': np.zeros(len(rows), dtype=bool), 'always_fixed': np.ones(len(rows), dtype=bool),
                   'nn_outcome_selector__p': a, 'outcome_selector__p': b, 'class_selector__p': c}
        sets = harmful_prompt_sets(rows, actions)
        overlaps = {o['compared_with']: o for o in sets['profile_overlaps'] if o['policy'] == 'nn_outcome_selector__p'}
        self.assertTrue(overlaps['outcome_selector__p']['same_prompts_regress'])
        self.assertFalse(overlaps['class_selector__p']['same_prompts_regress'])
        self.assertEqual(overlaps['class_selector__p']['shared_row_ids'], [])

    def test_per_fold_counts_sum_to_the_overall_counts(self):
        rows, _ = fixture(90, 6)
        action = np.array([i % 3 == 0 for i in range(len(rows))])
        actions = {'p': action}
        table = per_fold_table(rows, actions)
        overall = flat_counts(score_policy(rows, action))
        for key in ('edits', 'E_retained', 'B_material_loss', 'H_new_harm', 'E_population', 'B_population', 'H_population'):
            self.assertEqual(sum(r[key] for r in table), overall[key])

    def test_convergence_summary_counts_every_network_fit(self):
        rows, x = fixture(75, 10)
        with threadpool_limits(limits=1):
            fitted, _ = fit_outer_nn(x, rows, 0, FAST)
        summary = convergence_summary([fitted])
        expected = 3 * len(FAST['nn']['alpha_values']) * 4 * 3 + 3 * 3
        self.assertEqual(summary['network_fits'], expected)
        self.assertEqual(summary['converged_before_max_iter'] + summary['hit_max_iter'], expected)

    def test_frozen_inventory_guard_rejects_a_changed_label(self):
        rows, _ = fixture(90, 6)
        with self.assertRaises(ValueError):
            target_consistency(rows)  # the synthetic fixture is not the frozen 480-row inventory


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        unittest.main(verbosity=2)
