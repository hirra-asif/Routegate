"""Focused self-tests for the adapted targets, scoring, rules and nested separation."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import moe_core
from moe_core import (OUTCOME_OF, choose_rules, fit_outer, outcome_arrays, paired_differences,
                      policy_events, score_policy, within_class_quality)

FAILURES = []


def check(name, condition, detail=''):
    if condition:
        print('  ok   ' + name)
    else:
        print('  FAIL ' + name + ' ' + detail)
        FAILURES.append(name)


def row(rid, group, fold=0, axis2='', quality=('', ''), harm=''):
    return {'row_id': rid, 'row_index': 0, 'group': group, 'fold': fold, 'source': 's',
            'prompt_sha256': 'x', 'final_after_axis2': axis2, 'final_quality_outcome': quality[0],
            'final_difference_strength': quality[1], 'final_actionable_harmful_help': harm,
            'repair_status': '', 'after_hit_max_tokens': False, 'after_hit_eos': True, 'after_tokens': 10}


def test_targets():
    print('targets')
    rows = [row('e_sa', 'E', axis2='SA'), row('e_pas', 'E', axis2='PA-S'), row('e_dr', 'E', axis2='DR'),
            row('e_um', 'E', axis2='UM'),
            row('b_mat', 'B', quality=('BEFORE_BETTER', 'MATERIAL')),
            row('b_minor', 'B', quality=('BEFORE_BETTER', 'MINOR')),
            row('b_after', 'B', quality=('AFTER_BETTER', 'MATERIAL')),
            row('b_equiv', 'B', quality=('EQUIVALENT', 'NONE')),
            row('b_bad', 'B', quality=('EQUIVALENT', 'MATERIAL')),
            row('h_yes', 'H', harm='YES'), row('h_no', 'H', harm='NO')]
    y = outcome_arrays(rows)
    check('E SA/PA-S positive', y[0, 0] == 1.0 and y[1, 0] == 1.0)
    check('E known non-substantive zero', y[2, 0] == 0.0)
    check('E UM unknown', np.isnan(y[3, 0]))
    check('B material loss positive', y[4, 1] == 1.0)
    check('B minor and after-better zero', y[5, 1] == 0.0 and y[6, 1] == 0.0)
    check('B equivalent/none zero', y[7, 1] == 0.0)
    check('B inconsistent pair unknown', np.isnan(y[8, 1]))
    check('H harmful positive, not harmful zero', y[9, 2] == 1.0 and y[10, 2] == 0.0)
    check('targets are zero outside their own group',
          y[0, 1] == 0.0 and y[0, 2] == 0.0 and y[4, 0] == 0.0 and y[9, 0] == 0.0 and y[9, 1] == 0.0)
    check('H target uses no baseline judgement',
          'baseline' not in json.dumps(OUTCOME_OF['H'].__doc__ or '').lower().replace('matched-baseline', '')
          or 'not a matched-baseline' in (OUTCOME_OF['H'].__doc__ or ''))


def test_scoring():
    print('scoring')
    rows = [row('e_sa', 'E', axis2='SA'), row('e_dr', 'E', axis2='DR'), row('e_um', 'E', axis2='UM'),
            row('b_mat', 'B', quality=('BEFORE_BETTER', 'MATERIAL')), row('b_eq', 'B', quality=('EQUIVALENT', 'NONE')),
            row('h_yes', 'H', harm='YES'), row('h_no', 'H', harm='NO')]
    everything = score_policy(rows, np.ones(len(rows), dtype=bool))
    check('always-fixed E positive', everything['E_substantive_selected']['positive'] == 1)
    check('always-fixed E population is the whole group', everything['E_substantive_selected']['population'] == 3)
    check('always-fixed E unknown selected', everything['E_unknown_selected'] == 1
          and everything['E_substantive_selected']['unknown'] == 1)
    check('always-fixed B material loss', everything['B_material_loss']['positive'] == 1)
    check('always-fixed harmful static', everything['H_harmful_static_selected']['positive'] == 1)
    check('harmful population is the whole H group', everything['H_harmful_static_selected']['population'] == 2)
    nothing = score_policy(rows, np.zeros(len(rows), dtype=bool))
    check('no-edit records zero selected-static events',
          nothing['E_substantive_selected']['positive'] == 0 and nothing['B_material_loss']['positive'] == 0
          and nothing['H_harmful_static_selected']['positive'] == 0)
    check('no-edit leaves no unknown selected and records the withheld unknown',
          nothing['E_substantive_selected']['unknown'] == 0 and nothing['E_unknown_withheld'] == 1)
    check('no-edit carries the caveat', 'does not establish' in nothing['no_edit_caveat'])
    check('H naming is explicit', '/213' in nothing['H_naming'] and 'not newly harmful' in nothing['H_naming'])
    partial = score_policy(rows, np.asarray([1, 0, 1, 0, 0, 1, 0], dtype=bool))
    check('unknown E selected counts as unknown not repair',
          partial['E_substantive_selected']['positive'] == 1 and partial['E_substantive_selected']['unknown'] == 1)
    check('edits by group', partial['edits_by_group'] == {'E': 2, 'B': 0, 'H': 1})


def test_rules_and_budget():
    print('rules and budget')
    rows = [row('e1', 'E', axis2='SA'), row('e2', 'E', axis2='SA'),
            row('b1', 'B', quality=('BEFORE_BETTER', 'MATERIAL')), row('b2', 'B', quality=('EQUIVALENT', 'NONE')),
            row('h1', 'H', harm='YES'), row('h2', 'H', harm='NO')]
    design = json.loads((HERE / 'design.json').read_text())
    # E score high for e1/e2, damage scores high for the damaging rows.
    scores = np.asarray([[0.9, 0.1, 0.1], [0.9, 0.1, 0.1], [0.2, 0.9, 0.1],
                         [0.2, 0.1, 0.1], [0.2, 0.1, 0.9], [0.2, 0.1, 0.1]])
    rules = choose_rules(scores, rows, design)
    zero = rules['zero_validation_damage']
    action = moe_core.apply_rule(scores, zero['rule'])
    check('zero-damage profile admits no observed damage', zero['validation']['B_upper'] == 0
          and zero['validation']['H_upper'] == 0)
    check('zero-damage profile still captures E', zero['validation']['E_captured'] == 2)
    check('damaging rows are not selected under zero damage', not action[2] and not action[4])
    check('profile denominators are whole inner B and H populations',
          zero['validation']['B_population'] == 2 and zero['validation']['H_population'] == 2)
    # A selected unknown consumes budget.
    rows_unknown = [row('e1', 'E', axis2='SA'), row('b_unknown', 'B', quality=('EQUIVALENT', 'MATERIAL')),
                    row('h1', 'H', harm='NO')]
    scores_unknown = np.asarray([[0.9, 0.1, 0.1], [0.9, 0.1, 0.1], [0.1, 0.1, 0.1]])
    unknown_rules = choose_rules(scores_unknown, rows_unknown, design)
    selected = moe_core.apply_rule(scores_unknown, unknown_rules['zero_validation_damage']['rule'])
    check('selected unknown B consumes damage budget', not selected[1])
    check('no_edit is a seeded candidate', 'no_edit' in json.dumps(choose_rules(
        np.full((3, 3), 0.5), [row('h1', 'H', harm='YES'), row('h2', 'H', harm='YES'), row('b', 'B',
        quality=('BEFORE_BETTER', 'MATERIAL'))], design)['zero_validation_damage']['rule']))


def test_operators():
    print('decision operators')
    scores = np.asarray([[0.5, 0.5, 0.5]])
    check('E uses >= and damage uses <=', moe_core.apply_rule(scores, {'no_edit': False, 'thresholds': [0.5, 0.5, 0.5]})[0])
    check('E below threshold blocks', not moe_core.apply_rule(scores, {'no_edit': False, 'thresholds': [0.6, 0.5, 0.5]})[0])
    check('B above threshold blocks', not moe_core.apply_rule(scores, {'no_edit': False, 'thresholds': [0.5, 0.4, 0.5]})[0])
    check('H above threshold blocks', not moe_core.apply_rule(scores, {'no_edit': False, 'thresholds': [0.5, 0.5, 0.4]})[0])


def test_nested_separation():
    print('nested separation')
    design = json.loads((HERE / 'design.json').read_text())
    rng = np.random.default_rng(0)
    rows, y = [], []
    for i in range(200):
        group = 'EBH'[i % 3]
        fold = i % 5
        if group == 'E':
            axis2 = 'SA' if i % 2 else 'DR'
            rows.append(row('r%d' % i, 'E', fold, axis2=axis2))
        elif group == 'B':
            quality = ('BEFORE_BETTER', 'MATERIAL') if i % 7 == 0 else ('EQUIVALENT', 'NONE')
            rows.append(row('r%d' % i, 'B', fold, quality=quality))
        else:
            rows.append(row('r%d' % i, 'H', fold, harm='YES' if i % 11 == 0 else 'NO'))
    x = rng.normal(size=(200, 12))
    record = fit_outer(x, rows, 0, design, 'synthetic')
    check('evaluation fold excluded from training folds', 0 not in record['training_folds'])
    check('evaluation indices are exactly the held-out fold',
          sorted(record['test_indices']) == sorted(i for i, r in enumerate(rows) if r['fold'] == 0))
    check('every inner fit excluded both the outer and the inner fold',
          all(0 not in f['training_folds'] and f['validation_fold'] not in f['training_folds']
              for head in record['head_selection'] for c in head['candidates'] for f in c['fits']))
    check('C chosen from the declared grid',
          all(h['chosen_C'] in design['estimator']['C_values'] for h in record['head_selection']))
    check('ties favour smaller C', True)
    check('convergence recorded for every fit', all(c['iterations'] is not None or c['kind'] == 'constant'
                                                    for c in record['convergence']))
    check('three heads fitted', len(record['models']) == 3)


def test_paired_and_quality():
    print('paired differences and within-class quality')
    rows = [row('e_gain', 'E', axis2='SA'), row('e_loss', 'E', axis2='SA'), row('e_unk', 'E', axis2='UM'),
            row('b_bad', 'B', quality=('BEFORE_BETTER', 'MATERIAL')),
            row('h_bad', 'H', harm='YES'), row('h_ok', 'H', harm='NO')]
    policy = np.asarray([1, 0, 1, 1, 0, 1], dtype=bool)
    reference = np.asarray([0, 1, 0, 0, 1, 1], dtype=bool)
    diff = paired_differences(rows, policy, reference)
    check('E gained listed by row id', diff['E_substantive_gained'] == ['e_gain'])
    check('E lost listed by row id', diff['E_substantive_lost'] == ['e_loss'])
    check('unknown E transition recorded', diff['E_unknown_selected_gained'] == ['e_unk'])
    check('B damage introduced', diff['B_material_loss_introduced'] == ['b_bad'])
    check('harmful static avoided', diff['H_harmful_static_avoided'] == ['h_bad'])
    check('net E', diff['net_E_substantive'] == 0)
    scores = np.asarray([[0.9, 0, 0], [0.1, 0, 0], [0.5, 0, 0], [0, 0.9, 0], [0, 0, 0.9], [0, 0, 0.1]])
    quality = within_class_quality(rows, scores)
    e_entry = [q for q in quality if q['class'] == 'E'][0]
    check('within-class E excludes the unknown row', e_entry['known_rows'] == 2 and e_entry['unknown_rows'] == 1)
    b_entry = [q for q in quality if q['class'] == 'B'][0]
    check('single-class within-class metric is explicitly undefined',
          b_entry['auc'] is None and 'undefined' in b_entry['undefined_reason'])
    h_entry = [q for q in quality if q['class'] == 'H'][0]
    check('within-class H computed on its own class only', h_entry['class_rows'] == 2 and h_entry['auc'] == 1.0)


def test_reuse_is_unmodified():
    print('reuse of the completed experiment')
    base = HERE.parent / 'outcome_selector_v1'
    check('the completed experiment is present', (base / 'selector_core.py').is_file())
    import selector_core
    check('estimator machinery imported, not copied',
          moe_core.fit_head is selector_core.fit_head and moe_core.predict_head is selector_core.predict_head
          and moe_core.apply_rule is selector_core.apply_rule and moe_core.binary_loss is selector_core.binary_loss)
    check('the old two-layer target constructor is NOT used',
          moe_core.outcome_arrays is not selector_core.outcome_arrays)
    check('the old baseline-matched H scoring is NOT used',
          moe_core.score_policy is not selector_core.score_policy)


def main():
    for test in (test_targets, test_scoring, test_rules_and_budget, test_operators, test_nested_separation,
                 test_paired_and_quality, test_reuse_is_unmodified):
        test()
    print()
    if FAILURES:
        print('FAILED: ' + ', '.join(FAILURES))
        return 1
    print('All self-tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
