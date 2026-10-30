#!/usr/bin/env python3
"""Meaningful CPU checks using synthetic vectors and synthetic response labels."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from threadpoolctl import threadpool_limits

from inputs import align_rows, verified_file
from selector_core import (apply_rule, choose_rules, favourable, fit_head, fit_outer,
                           outcome_arrays, paired_policy_counts, predict_head, score_policy)

HERE=Path(__file__).resolve().parent
DESIGN=json.loads((HERE/'design.json').read_text())


def fixture(n=90, width=8):
    rng=np.random.default_rng(17)
    x=rng.normal(size=(n,width))
    rows=[]
    for i in range(n):
        group='EBH'[i%3]
        fixed={'E':'SA' if x[i,0]>0 else 'DR','B':'BEFORE_BETTER|MATERIAL' if x[i,1]>1 else 'EQUIVALENT|NONE',
               'H':'YES' if x[i,2]>.9 else 'NO'}[group]
        row={'row_id':'synthetic_'+str(i),'group':group,'fold':(i//3)%5,
             'source_family':'synthetic','topic':'synthetic_topic','task_frame':'synthetic_task',
             'prompt_text':'Synthetic neutral test prompt '+str(i),'baseline_verdict':None if group=='B' else 'DR' if group=='E' else 'NO',
             'fixed_verdict':fixed,'dynamic_verdict':fixed}
        row['responses']={'baseline':'unchanged baseline '+str(i),'fixed':'fixed reply '+str(i),'dynamic':'fixed reply '+str(i)}
        for arm in row['responses']:
            row[arm+'_response_sha256']=hashlib.sha256(row['responses'][arm].encode()).hexdigest()
        row['fixed_favourable']=row['dynamic_favourable']=favourable(group,fixed)
        rows.append(row)
    return rows,x


class SelectorTests(unittest.TestCase):
    def test_original_endpoint_rules_and_quality_orientation(self):
        self.assertTrue(favourable('E','PA-S'));self.assertFalse(favourable('E','PA-N'))
        self.assertIsNone(favourable('E','UM'));self.assertIsNone(favourable('H','UNCLEAR'))
        self.assertFalse(favourable('B','REFERENCE_BETTER|MATERIAL'))
        self.assertTrue(favourable('B','CANDIDATE_BETTER|MATERIAL'))
        self.assertIsNone(favourable('B','EQUIVALENT|MATERIAL'))

    def test_unknown_harm_not_imputed_and_no_edit_does_not_assert_safety(self):
        rows,_=fixture(15)
        h=[i for i,r in enumerate(rows) if r['group']=='H']
        rows[h[0]]['fixed_verdict']=None
        rows[h[1]]['baseline_verdict']=None
        fixed=score_policy(rows,np.ones(len(rows),bool))
        none=score_policy(rows,np.zeros(len(rows),bool))
        self.assertEqual(fixed['H_new_harm_baseline_NO']['population'],4)
        self.assertEqual(fixed['H_new_harm_baseline_NO']['unknown'],1)
        self.assertEqual(none['H_new_harm_baseline_NO']['positive'],0)
        self.assertEqual(none['H_any_harm']['unknown'],1)
        self.assertEqual(none['E_fixed_substantive_selected']['positive'],0)
        self.assertEqual(none['E_baseline_selected_behaviour_not_scored'],5)

    def test_nested_outer_outcomes_and_features_do_not_enter_model_fitting(self):
        rows,x=fixture()
        with threadpool_limits(limits=1):
            a=fit_outer(x,rows,0,'outcome_selector',DESIGN)
            changed=copy.deepcopy(rows);changed_x=x.copy()
            for i,row in enumerate(changed):
                if row['fold']==0:
                    row['fixed_verdict']={'E':'DR' if row['fixed_verdict']=='SA' else 'SA',
                                          'B':'BEFORE_BETTER|MATERIAL','H':'YES'}[row['group']]
                    changed_x[i]=1e5
            b=fit_outer(changed_x,changed,0,'outcome_selector',DESIGN)
        self.assertEqual(a['models'],b['models'])
        self.assertEqual(a['head_selection'],b['head_selection'])
        self.assertEqual(a['rules'],b['rules'])
        for history in a['head_selection']:
            for candidate in history['candidates']:
                for fit in candidate['fits']:
                    self.assertNotIn(0,fit['training_folds'])
                    self.assertNotIn(fit['validation_fold'],fit['training_folds'])

    def test_historical_gate_and_response_text_never_become_features(self):
        rows,x=fixture(60)
        changed=copy.deepcopy(rows)
        for row in changed:
            row['g_R']=float('nan');row['prompt_text']='changed metadata';row['responses']={'fake':'do not learn from text'}
        with threadpool_limits(limits=1):
            a=fit_outer(x,rows,1,'class_selector',DESIGN)
            b=fit_outer(x,changed,1,'class_selector',DESIGN)
        self.assertEqual(a,b)

    def test_scaler_uses_labelled_training_rows_only_and_saved_model_predicts(self):
        x=np.asarray([[0,1],[1,3],[2,5],[100000,100000]],float);y=np.asarray([0,1,1,np.nan])
        with threadpool_limits(limits=1):model=fit_head(x,y,.01,DESIGN['logistic_regression'])
        np.testing.assert_allclose(model['mean'],[1,3])
        np.testing.assert_allclose(predict_head(model,x),predict_head(json.loads(json.dumps(model)),x))

    def test_no_positives_does_not_produce_a_false_certain_zero(self):
        model=fit_head(np.ones((4,3)),np.zeros(4),.01,DESIGN['logistic_regression'])
        self.assertEqual(model['kind'],'constant');self.assertAlmostEqual(model['value'],1/6)

    def test_thresholds_count_selected_unknown_damage_and_no_edit_is_feasible(self):
        rows,_=fixture(3)
        rows[0]['fixed_verdict']='SA';rows[1]['fixed_verdict']=None;rows[2]['fixed_verdict']=None
        # Identical scores force selecting the E opportunity and both unknown damage cases together.
        rules=choose_rules(np.full((3,3),.5),rows,DESIGN)
        self.assertTrue(rules['zero_validation_damage']['rule']['no_edit'])

    def test_manifest_alignment_and_mutation_protection(self):
        rows,x=fixture(15)
        folds={r['row_id']:r['fold'] for r in rows}
        manifest=[{'row_id':r['row_id'],'row_index':str(i),'final_ebh':r['group'],
                   'source_family':r['source_family'],'prompt_text':r['prompt_text'],
                   'prompt_sha256':hashlib.sha256(r['prompt_text'].encode()).hexdigest()} for i,r in enumerate(rows)]
        self.assertEqual(align_rows(list(reversed(rows)),manifest,folds,15),rows)
        broken=copy.deepcopy(manifest);broken[0]['row_index']='1'
        with self.assertRaises(ValueError):align_rows(rows,broken,folds,15)
        broken=copy.deepcopy(rows);broken[0]['fold']=True
        with self.assertRaises(ValueError):align_rows(broken,manifest,folds,15)
        with self.assertRaises(ValueError):align_rows(rows[:-1]+[rows[0]],manifest,folds,15)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'input.json';p.write_text('original')
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            verified_file(p,digest,{})
            p.write_text('changed')
            with self.assertRaises(ValueError):verified_file(p,digest,{})

    def test_paired_comparison_counts_same_prompts(self):
        rows,_=fixture(15)
        r=paired_policy_counts(rows,np.zeros(15,bool),np.ones(15,bool))
        self.assertEqual(r['E_fixed_substantive_selected']['selector_only_positive'],0)
        self.assertEqual(r['H_new_harm_baseline_NO']['domain'],5)
        self.assertEqual(r['B_material_loss']['selector_only_positive'],0)


if __name__=='__main__':unittest.main(verbosity=2)
