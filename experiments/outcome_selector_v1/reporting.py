"""Compact tables and scientific figures for recorded-response selection."""
import csv
import html
import json
from pathlib import Path

import numpy as np

from selector_core import paired_policy_counts, score_policy


NOTES = [
    'Development selection among recorded baseline/fixed replies; no new Qwen generations or API calls.',
    'All selectors choose between the same baseline and the same fixed 0.15 edit at layers 31–32.',
    'E captured means selected fixed replies labelled SA/PA-S. Baseline-selected E behaviour is not scored; this is NOT an overall refusal-repair rate.',
    'The full E inventory has 64 fixed substantive outcomes among 89 prompts. The earlier fixed-versus-dynamic common subset had 62 among 87. No labels have changed.',
    'The full fixed B inventory has 16 material losses and 2 unknowns among 178, unlike the 14 losses among the 171 common resolved rows in the previous comparison.',
    'Profiles constrain inner-validation observed-plus-unknown damage only. They are exploratory settings, not deployment tolerances or safety guarantees.',
    'Missing outcomes are explicit. H new harm requires a baseline NO; the separately reported H-any-harm measure retains unknown baselines.',
    'Every fitted scaler, classifier, C choice and threshold excludes its outer evaluation fold.',
    'The design follows inspection of these development outcomes. This is exploratory validation, not a fresh confirmatory test.',
    'All H sources differ from E/B sources. Topic breakdowns do not establish matched-intent or MoE-specific mechanisms.',
    'Historical dynamic results are context only and use continuously scaled edits. Historical g_R never enters training.',
    'No final profile is automatically selected, no all-development deployment refit is made, and no external evaluation is launched.'
]


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flat(name, reading):
    row = {'policy':name, 'edits':reading['edits']}
    for short,key in [('E','E_fixed_substantive_selected'),('B','B_material_loss'),('H_new','H_new_harm_baseline_NO'),('H_any','H_any_harm')]:
        row.update({short+'_'+k:v for k,v in reading[key].items()})
    return row


def make_reports(output, rows, actions, design):
    results = {name:score_policy(rows, action) for name,action in actions.items()}
    flat_rows = [flat(name,r) for name,r in results.items()]
    write_csv(output/'summary.csv', flat_rows)
    summary = {'status':'DEVELOPMENT_COMPARISON_COMPLETE','rows':len(rows),
               'new_generations':0,'api_calls':0,'final_refit':False,'notes':NOTES,'policies':results}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    paired = {name:paired_policy_counts(rows, action, actions['always_fixed']) for name,action in actions.items()}
    (output/'paired_vs_fixed.json').write_text(json.dumps(paired,indent=2)+'\n')
    stratified = []
    for fields in [('fold',),('source_family',),('topic',),('topic','task_frame')]:
        keys = sorted({tuple(r[f] for f in fields) for r in rows})
        for key in keys:
            ids = [i for i,r in enumerate(rows) if tuple(r[f] for f in fields) == key]
            subset = [rows[i] for i in ids]
            for name, action in actions.items():
                reading = score_policy(subset,np.asarray(action)[ids])
                stratified.append({'stratum_fields':','.join(fields),'stratum':json.dumps(key),**flat(name,reading)})
    write_csv(output/'stratified_results.csv',stratified)
    table = ['| Policy | Edits / 480 | E fixed substantive selected / 89 | B material loss: known + unknown / 178 | H new harm: known + unknown / 212 |',
             '|---|---:|---:|---:|---:|']
    for name,r in results.items():
        e,b,h = (r[k] for k in ('E_fixed_substantive_selected','B_material_loss','H_new_harm_baseline_NO'))
        table.append('| {} | {} | {} | {} + {} unknown | {} + {} unknown |'.format(name,r['edits'],e['positive'],b['positive'],b['unknown'],h['positive'],h['unknown']))
    markdown='# Fixed-edit selector development comparison\n\n'+'\n'.join(table)+'\n\n'+'\n\n'.join(NOTES)+'\n'
    (output/'REPORT.md').write_text(markdown)
    cells=[]
    for i,line in enumerate(table):
        if i==1:continue
        tag='th' if i==0 else 'td'
        cells.append('<tr>'+''.join('<'+tag+'>'+html.escape(c.strip())+'</'+tag+'>' for c in line.strip('|').split('|'))+'</tr>')
    page='<!doctype html><meta charset="utf-8"><title>Fixed-edit selector comparison</title><style>body{font:16px/1.6 system-ui;max-width:1200px;padding:28px;margin:auto;color:#263743}h1{color:#244b78}table{border-collapse:collapse;font-size:14px}th,td{padding:10px;border-bottom:1px solid #d7e2e8;text-align:left}th{background:#e8f4f5}img{max-width:100%}</style><h1>Fixed-edit selector development comparison</h1><div style="overflow:auto"><table>'+''.join(cells)+'</table></div><p><b>E counts are retained fixed-edit substantive answers, not overall repaired refusals.</b></p>'
    page+=''.join('<p>'+html.escape(n)+'</p>' for n in NOTES)
    (output/'REPORT.html').write_text(page)
    try:
        make_plot(output,results)
        with (output/'REPORT.html').open('a') as f:f.write('<h2>Observed development trade-offs</h2><img src="plots/tradeoff.png" alt="Observed repair opportunity and preservation trade-offs">')
        summary['plot_status']='created'
    except ImportError:
        summary['plot_status']='matplotlib unavailable; all numerical results complete'
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def make_plot(output,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    directory=output/'plots';directory.mkdir()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(11.5,5.7))
    colour={'class_selector':'#30669A','outcome_selector':'#2597A5','no_edit':'#667786','always_fixed':'#BD8248'}
    markers={'zero_validation_damage':'o','H1pct_B2pct_validation':'s','H5pct_B5pct_validation':'^'}
    max_damage=[0,0]
    for name,r in results.items():
        parts=name.split('__');method=parts[0];profile=parts[1] if len(parts)>1 else None
        y=r['E_fixed_substantive_selected']['positive']
        for j,key in enumerate(('H_new_harm_baseline_NO','B_material_loss')):
            d=r[key];x=d['positive'];unknown=d['unknown']
            max_damage[j]=max(max_damage[j],x+unknown)
            axes[j].errorbar(x,y,xerr=[[0],[unknown]],fmt=markers.get(profile,'D'),ms=7,color=colour[method],capsize=3,alpha=.85)
            if method in ('no_edit','always_fixed'):
                axes[j].annotate('No edit' if method=='no_edit' else 'Always fixed',(x,y),xytext=(5,7 if y==0 else -16),textcoords='offset points',fontsize=9,color=colour[method])
    for j,(ax,label) in enumerate(zip(axes,('Newly harmful H responses','Material B answer losses'))):
        ax.set_xlabel(label+' (count; lower is better)');ax.set_ylabel('Selected fixed E substantive answers (count)')
        ax.set_xlim(-.6,max_damage[j]+2);ax.set_ylim(-3,70);ax.grid(alpha=.15)
    fig.suptitle('Does selecting the same fixed edit retain repair opportunities with less damage?',fontweight='bold',fontsize=13,y=.98)
    handles=[Line2D([0],[0],color=colour[k],marker='o',linestyle='',label=v) for k,v in [('class_selector','Class-based selector'),('outcome_selector','Outcome-based selector')]]
    handles.extend(Line2D([0],[0],color='#667786',marker=m,linestyle='',label=l) for m,l in [('o','Zero validation damage'),('s','H 1% / B 2% validation budget'),('^','H 5% / B 5% validation budget')])
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.07),ncol=3,frameon=False,fontsize=9)
    fig.text(.5,.015,'Nested development predictions; no new generations. Rightward bars include unknown selected outcomes.\nE measures retained fixed-edit answers; profiles are validation settings, not safety guarantees.',ha='center',fontsize=9,color='#526b79')
    fig.subplots_adjust(left=.08,right=.98,top=.88,bottom=.30,wspace=.32)
    for suffix in ('svg','pdf','png'):fig.savefig(directory/('tradeoff.'+suffix),dpi=220)
    plt.close(fig)
