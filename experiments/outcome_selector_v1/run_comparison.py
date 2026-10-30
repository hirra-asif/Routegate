#!/usr/bin/env python3
"""Nested development comparison; saved responses only. Python >=3.8."""
import argparse
import hashlib
import json
import platform
import sys
import tarfile
from pathlib import Path

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from inputs import load_inputs, sha256, verify_unchanged, development_path
from reporting import make_reports, write_csv
from selector_core import fit_outer, score_policy


HERE=Path(__file__).resolve().parent


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def output_allowed(path,provenance):
    path=development_path(path)
    for source in provenance:
        parent=Path(source).resolve().parent
        if path==parent or parent in path.parents:
            raise ValueError('Output must be outside all input directories: '+str(path))
    if path.exists():
        raise ValueError('Output already exists and was preserved: '+str(path)+'. Use status; for a separate run supply a new --output path.')
    return path


def run(args):
    if args.stage=='status':
        summary=args.output/'summary.json'
        if summary.is_file() and (args.output/'COMPLETE.json').is_file():
            value=json.loads(summary.read_text())
            value['output']=str(args.output)
            print(json.dumps(value,indent=2))
        else:
            print(json.dumps({'status':'NO_COMPLETE_COMPARISON','output':str(args.output),'next_command':'bash experiments/outcome_selector_v1/run.sh run'},indent=2))
        return
    print('[inputs] Verifying the existing audit, fold map and cached activation array.',flush=True)
    rows,x,provenance=load_inputs(args.repo,args.audit,args.paths)
    design=json.loads((HERE/'design.json').read_text())
    code_hashes={p.name:sha256(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.json','.sh')}
    print(json.dumps({'stage':'preflight','rows':len(rows),'feature_shape':list(x.shape),
                      'read_state':32,'read_layer':31,'candidate_layers':[31,32],
                      'fixed_strength':.15,'folds':5,'methods':design['methods'],
                      'new_generations':0,'api_calls':0,'uses_GPU':False,
                      'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},indent=2),flush=True)
    if args.stage=='preflight':return
    out=output_allowed(args.output,provenance)
    archive=out.parent/(out.name+'.tar.gz')
    if archive.exists():
        raise ValueError('Output archive already exists and was preserved: '+str(archive))
    out.mkdir(parents=True)
    write_json(out/'run_manifest.json',{'status':'STARTED','design':design,'source_files':provenance,
                                       'code_sha256':code_hashes,'python':platform.python_version(),
                                       'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
                                       'new_generations':0,'api_calls':0,'historical_fold0_checkpoint_root_cause':'not diagnosed by this experiment'})
    models=out/'fold_models';models.mkdir()
    prediction_rows=[]
    actions={'no_edit':np.zeros(len(rows),dtype=bool),'always_fixed':np.ones(len(rows),dtype=bool)}
    for method in design['methods']:
        for profile in design['profiles']:
            actions[method+'__'+profile['name']]=np.zeros(len(rows),dtype=bool)
        assigned=np.zeros(len(rows),dtype=int)
        for outer in range(5):
            print('[fit] {} | outer fold {} of 0–4'.format(method,outer),flush=True)
            fitted=fit_outer(x,rows,outer,method,design)
            write_json(models/(method+'_fold_'+str(outer)+'.json'),fitted)
            indices=np.asarray(fitted['test_indices'])
            assigned[indices]+=1
            for profile,values in fitted['actions'].items():actions[method+'__'+profile][indices]=values
            for local,i in enumerate(indices):
                row=rows[i]
                prediction_rows.append({'row_id':row['row_id'],'outer_fold':outer,'method':method,
                                        'opportunity_score':fitted['test_scores'][local][0],
                                        'benign_loss_score':fitted['test_scores'][local][1],
                                        'new_harm_score':fitted['test_scores'][local][2],
                                        'training_folds':' '.join(str(v) for v in fitted['training_folds'])})
        if not np.all(assigned==1):raise ValueError('Each row must be predicted once out of fold')
    write_csv(out/'prediction_scores.csv',prediction_rows)
    decisions=[]
    for name,action in actions.items():
        for i,row in enumerate(rows):
            chosen='fixed' if action[i] else 'baseline'
            decisions.append({'row_id':row['row_id'],'fold':row['fold'],'group':row['group'],
                              'policy':name,'selected_action':chosen,
                              'selected_response_sha256':row[chosen+'_response_sha256'],
                              'fixed_verdict':row['fixed_verdict'],'baseline_verdict':row['baseline_verdict'],
                              'E_baseline_behaviour_scored':False})
    write_csv(out/'selected_actions.csv',decisions)
    summary=make_reports(out,rows,actions,design)
    # Historical dynamic on its original common frame is context, not a trained feature.
    context={}
    for group in 'EBH':
        r=[row for row in rows if row['group']==group and row['transition']!='unknown']
        context[group]={'common_resolved':len(r),'fixed_favourable':sum(row['fixed_favourable'] for row in r),
                        'dynamic_favourable':sum(row['dynamic_favourable'] for row in r)}
    write_json(out/'historical_dynamic_context.json',{'role':'historical continuously scaled action; not the same binary policy','readings':context})
    verify_unchanged(provenance)
    if any(sha256(HERE/name)!=value for name,value in code_hashes.items()):
        raise ValueError('Code changed during the run')
    (out/'COMPLETE.json').write_text(json.dumps({'status':'COMPLETE','rows':480,'outer_prediction_rows':len(prediction_rows),
                                               'new_generations':0,'api_calls':0,'frozen_files_changed':0},indent=2)+'\n')
    sums=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS':sums.append(sha256(p)+'  '+p.relative_to(out).as_posix())
    (out/'SHA256SUMS').write_text('\n'.join(sums)+'\n')
    if archive.exists():raise ValueError('Comparison completed, but archive path already exists and was preserved: '+str(archive))
    with tarfile.open(str(archive),'w:gz') as handle:handle.add(str(out),arcname=out.name)
    print(json.dumps({'status':'DEVELOPMENT_COMPARISON_COMPLETE','output':str(out),'download_archive':str(archive),
                      'new_generations':0,'api_calls':0,'policies':summary['policies']},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('preflight','run','status'))
    parser.add_argument('--repo',type=Path,default=HERE.parents[1])
    parser.add_argument('--audit',type=Path,default=Path.home()/'routegate-data/controller_error_audit_v1')
    parser.add_argument('--output',type=Path,default=Path.home()/'routegate-data/outcome_selector_v1')
    parser.add_argument('--paths',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        try:run(args)
        except (ValueError,OSError,KeyError) as error:
            print('STOP: '+str(error),file=sys.stderr)
            raise SystemExit(1)


if __name__=='__main__':main()
