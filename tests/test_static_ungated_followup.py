"""Meaningful CPU checks for the new comparison; no model/API calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from routegate import staticfollowup as U, staticassessment as A, finalrunner as R

REPO=Path(__file__).resolve().parents[1]
RELEASE=json.loads((REPO/"manifests/routegate_sealed_judging_v1_release.json").read_text())


def fake_produced(text="A test response."):
    return {"response_text":text,"prompt_token_count":20,"completion_token_count":5,
            "eos_reached":True,"hit_max_new_tokens":False,"stop_fired":False,
            "stop_reason":"eos","stop_strings":[],"prompt_mode":"chat","hooks_fired":3,
            "applied_by_state":{32:1.0,33:2.0,34:3.0}}


class ExactStaticArithmetic(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(20260911)
        self.vectors={s:(lambda v:v/np.linalg.norm(v))(rng.normal(size=2048).astype(np.float32)) for s in R.STATIC_STATES}
        self.capture={s:torch.tensor(rng.normal(size=2048).astype(np.float32)) for s in (31,32,33,34,44,45)}
        self.actuator=SimpleNamespace(direction_for_state=lambda s:self.vectors[s])
        self.item={"item_id":"unit:1","item_index":0,"prompt":"A test prompt.","prompt_sha256":U.sha(b"A test prompt.")}

    def test_ungated_plan_matches_original_gate_on_path_exactly(self):
        calls=[]
        def generated(*args,**kw):
            calls.append(kw);return fake_produced()
        controller=SimpleNamespace(capture_for=lambda _:self.capture)
        decision=SimpleNamespace(p_E=.9,p_H=.01,fires=True)
        runtime=R.ConditionRuntime(condition=R.COND_ROUTEGATE,steers=True,
            hook_states=U.HOOK_STATES,capture_state=45,apply_blocks=R.STATIC_BLOCKS,hooks_expected=3,
            bundle=SimpleNamespace(decide=lambda x:decision),thresholds=SimpleNamespace(threshold_E=.3,threshold_H=.07),actuator=self.actuator)
        with patch('routegate.runtime.capture_prompt_states'),patch('routegate.runtime.generate_response',side_effect=generated):
            R.execute_item(runtime,model=None,tokenizer=None,controller=controller,prompt=self.item['prompt'],item_id='unit:1',generation=None,prompt_mode='chat',stop_strings=[])
            U.execute_static(model=None,tokenizer=None,controller=controller,actuator=self.actuator,item=self.item,generation=None)
        self.assertEqual(len(calls),2)
        for s in R.STATIC_STATES:
            a,b=calls[0]['plans'][s],calls[1]['plans'][s]
            self.assertTrue(torch.equal(a.delta,b.delta))
            self.assertEqual(a.budget_reference_state,s-1)
            self.assertEqual(a.budget,b.budget)
        self.assertEqual(calls[0]['mode'],calls[1]['mode'])

    def test_missing_unsteered_budget_is_rejected(self):
        capture=dict(self.capture);capture.pop(31)
        with self.assertRaises(U.FollowupError):U.make_plans(self.actuator,capture)

    def test_missing_hook_does_not_become_a_valid_response(self):
        result=fake_produced();result['hooks_fired']=2
        with patch('routegate.runtime.capture_prompt_states'),patch('routegate.runtime.generate_response',return_value=result):
            with self.assertRaises(U.FollowupError):
                U.execute_static(model=None,tokenizer=None,controller=SimpleNamespace(capture_for=lambda _:self.capture),actuator=self.actuator,item=self.item,generation=None)

    def test_no_gate_or_classifier_is_needed(self):
        calls=[]
        with patch('routegate.runtime.capture_prompt_states'),patch('routegate.runtime.generate_response',side_effect=lambda *a,**kw:(calls.append(kw) or fake_produced())):
            U.execute_static(model=None,tokenizer=None,controller=SimpleNamespace(capture_for=lambda _:self.capture),actuator=self.actuator,item=self.item,generation=None)
        self.assertEqual(set(calls[0]['plans']),{32,33,34})


class RecordsAndResume(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.item={'item_id':'unit:1','item_index':0,'prompt':'Test','prompt_sha256':U.sha(b'Test')}
        self.study={'record_sha256':'study','gate_on_reference_ids':['unit:1']}
        self.execution={'version':'test'};self.execution['record_sha256']=U.digest_record(self.execution)
        self.original={'gate_fires':True,'response_text':'A test response.'}
        (self.root/'inputs').mkdir();(self.root/'generations').mkdir()
        (self.root/'execution.json').write_bytes(U.jbytes(self.execution))
        (self.root/'inputs/safety_first_routegate.json').write_bytes(U.jbytes({'unit:1':self.original}))
        self.record=U.make_record(self.item,fake_produced(),self.study,self.execution['record_sha256'],self.original)

    def tearDown(self):self.tmp.cleanup()

    def write(self,records):
        (self.root/'generations/records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))

    def test_valid_resume_and_duplicate_rejection(self):
        self.write([self.record]);self.assertEqual(len(U.load_records(self.root,self.study,[self.item])),1)
        self.write([self.record,self.record])
        with self.assertRaises(U.FollowupError):U.load_records(self.root,self.study,[self.item])

    def test_changed_response_and_execution_are_rejected(self):
        r=copy.deepcopy(self.record);r['response_text']='Changed';self.write([r])
        with self.assertRaises(U.FollowupError):U.load_records(self.root,self.study,[self.item])
        r=copy.deepcopy(self.record);r['execution_sha256']='wrong';r['record_sha256']=U.digest_record(r);self.write([r])
        with self.assertRaises(U.FollowupError):U.load_records(self.root,self.study,[self.item])

    def test_rehashed_wrong_layer_or_zero_edit_is_rejected(self):
        for field,value in [('layers_zero_based',[30,31,32]),('applied_by_state',{'32':0,'33':1,'34':1}),('gate_used',True)]:
            r=copy.deepcopy(self.record);r[field]=value;r['record_sha256']=U.digest_record(r);self.write([r])
            with self.assertRaises(U.FollowupError):U.load_records(self.root,self.study,[self.item])

    def test_torn_line_is_not_silently_ignored(self):
        (self.root/'generations/records.jsonl').write_text(json.dumps(self.record))
        with self.assertRaises(U.FollowupError):U.load_records(self.root,self.study,[self.item])

    def test_gate_on_mismatch_is_preserved_and_reported(self):
        record=U.make_record(self.item,fake_produced('Different text'),self.study,self.execution['record_sha256'],self.original)
        self.write([record]);loaded=U.load_records(self.root,self.study,[self.item])
        check=U.consistency(loaded,self.study)
        self.assertFalse(check['passed']);self.assertEqual(check['mismatched_item_ids'],['unit:1'])

    def test_eos_exactly_at_limit_keeps_both_flags(self):
        p=fake_produced();p['completion_token_count']=768;p['hit_max_new_tokens']=True
        r=U.make_record(self.item,p,self.study,self.execution['record_sha256'],self.original)
        self.assertTrue(r['eos_reached']);self.assertTrue(r['hit_max_new_tokens'])

    def test_different_judges_can_run_together_but_duplicate_or_join_cannot(self):
        with U.lock(self.root,'assessment',shared=True),U.lock(self.root,'judge_sonnet'):
            with U.lock(self.root,'assessment',shared=True),U.lock(self.root,'judge_opus'):
                with self.assertRaises(U.FollowupError):
                    with U.lock(self.root,'assessment'):pass
                with self.assertRaises(U.FollowupError):
                    with U.lock(self.root,'judge_sonnet'):pass


class JointAssessment(unittest.TestCase):
    def plan(self):
        items=[];groups=[];base={};gate={};new={}
        for i,g in enumerate(['E','B','H','permitted_other','permission_unresolved']):
            rid='item:'+str(i);p='Prompt '+str(i)
            items.append({'item_id':rid,'prompt':p,'prompt_sha256':U.sha(p.encode())})
            groups.append({'row_id':rid,'group':g})
            base[rid]={'response_text':'Original '+str(i)}
            gate[rid]={'response_text':base[rid]['response_text'],'gate_fires':False}
            new[rid]={'response_text':('Edited '+str(i)) if g!='B' else base[rid]['response_text']}
        return A.material_plan(items,groups,{'baseline':base,'safety_first_routegate':gate},new,RELEASE)

    def test_identical_material_is_deduplicated_across_all_conditions(self):
        files,classes,refs,mapping=self.plan()
        self.assertEqual(len(mapping),12)
        self.assertEqual(sum(c['rows'] for c in classes.values()),6)
        self.assertEqual(sum(m['identity_verdict'] is not None for m in mapping),3)
        self.assertEqual(len(json.loads(files['inputs/excluded_groups.json'])),1)
        erows=[m for m in mapping if m['group']=='E']
        self.assertEqual(erows[0]['content_id'],erows[1]['content_id'])
        self.assertNotEqual(erows[0]['content_id'],erows[2]['content_id'])

    def test_same_response_for_different_prompts_is_not_merged(self):
        items=[{'item_id':str(i),'prompt':str(i),'prompt_sha256':U.sha(str(i).encode())} for i in (1,2)]
        groups=[{'row_id':str(i),'group':'H'} for i in (1,2)]
        base={str(i):{'response_text':'Same','gate_fires':False} for i in (1,2)}
        _,classes,_,_=A.material_plan(items,groups,{'baseline':base,'safety_first_routegate':base},base,RELEASE)
        self.assertEqual(classes['safety_case']['rows'],2)

    def expanded(self):
        _,_,_,mapping=self.plan();adjudicated=[];seen=set()
        for m in mapping:
            if m['identity_verdict'] is not None or m['content_id'] in seen:continue
            seen.add(m['content_id'])
            val= ('SA' if m['arm_id']==U.CONDITION else 'DR') if m['rubric']=='axis2' else ('YES' if m['arm_id']==U.CONDITION else 'NO')
            adjudicated.append({'screen_row_id':m['content_id'],'verdict':val,'label_source':'SYNTHETIC_TEST','technically_unresolved':False})
        return A.expand(mapping,adjudicated)

    def test_scoring_counts_repair_and_harm_separately(self):
        report=A.summarise(self.expanded());s=report['by_arm'][U.CONDITION]
        self.assertEqual(s['E']['endpoint_count'],1)
        self.assertEqual(s['E']['originally_blocked_E']['strict_repairs'],1)
        self.assertEqual(s['B']['endpoint_count'],0)
        self.assertEqual(s['H']['new_harmful_help']['YES'],1)

    def test_unchanged_answer_cannot_become_a_causal_repair(self):
        expanded=self.expanded()
        for r in expanded:
            if r['group']=='E':r['verdict']='SA'
        report=A.summarise(expanded)
        self.assertEqual(report['by_arm']['safety_first_routegate']['E']['endpoint_count'],1)
        self.assertEqual(report['by_arm']['safety_first_routegate']['E']['substantive_after_fresh_baseline_refusal'],0)

    def test_unknown_is_not_counted_as_safe(self):
        expanded=self.expanded()
        for r in expanded:
            if r['group']=='H' and r['arm_id']==U.CONDITION:r['verdict']='UNCLEAR'
        s=A.summarise(expanded)['by_arm'][U.CONDITION]['H']
        self.assertEqual(s['unknown_endpoint_count'],1);self.assertEqual(s['rate_bounds'],[0,1])

    def test_identical_unknown_material_has_zero_paired_difference(self):
        r={'group':'H','content_id':'same','verdict':None}
        d=A.paired_endpoint({'1':r},{'1':r})
        self.assertEqual(d['full_frame_difference_bounds'],[0,0])

    def test_inconsistent_labels_for_identical_material_are_refused(self):
        a={'group':'H','content_id':'same','verdict':'NO'};b=dict(a,verdict='YES')
        with self.assertRaises(U.FollowupError):A.paired_endpoint({'1':a},{'1':b})

    def test_common_manifest_keeps_the_same_judge_settings(self):
        from routegate import finaljudging as F,judgingprotocol as P
        files,classes,refs,_=self.plan()
        doc=F.protocol_document(RELEASE,'responses',files,classes,['unique_material'],refs)
        m=P.parse_protocol_manifest(doc,Path('test/protocol.json'))
        self.assertEqual(m.raw['judges'],RELEASE['judges'])
        self.assertEqual(m.raw['sampling'],RELEASE['sampling'])


if __name__=='__main__':unittest.main()
