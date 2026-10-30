#!/usr/bin/env python3
"""Rebuild audited paired-outcome figures; no fitting, judging or model calls."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parent
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=ROOT/'Figures')
OUT=parser.parse_args().output_dir;OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True})
C={'E':'#B47B48','B':'#2F6595','H':'#9E594C'}
def finish(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',facecolor='white')
 plt.close(fig)
def clean(ax):
 ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(axis='y',color='#e7e8ea',linewidth=.6);ax.spines[['top','right']].set_visible(False)
a=json.loads((ROOT/'data/static_audit.json').read_text());r=json.loads((ROOT/'data/static_report.json').read_text())
arms=['baseline','safety_first_routegate','static_ungated'];names=['Baseline','Gated static','Ungated static']
fig,axs=plt.subplots(1,3,figsize=(10.8,4.0));fig.subplots_adjust(wspace=.38,bottom=.25,top=.8)
for ax,g,title,lim in zip(axs,['E','B','H'],['(a) Substantive E answers /38','(b) Material B losses /78','(c) Newly harmful H replies /161'],[28,6,34]):
 vals=[];unks=[]
 for arm in arms:
  d=r['by_arm'][arm][g];vals.append(d['new_harmful_help']['YES'] if g=='H' else d['endpoint_count']);unks.append(d['new_harmful_help']['unknown'] if g=='H' else d['unknown_endpoint_count'])
 bars=ax.bar(range(3),vals,color=C[g],width=.56,alpha=.85)
 for i,(v,u) in enumerate(zip(vals,unks)):
  if u:ax.bar(i,u,bottom=v,width=.56,color='white',edgecolor=C[g],hatch='////',linewidth=1)
  ax.text(i,v+u+.02*lim,str(v)+(f' +{u}u' if u else ''),ha='center',va='bottom',fontsize=10)
 ax.set(xticks=range(3),xticklabels=names,ylim=(0,lim),title=title);ax.tick_params(axis='x',labelrotation=20);clean(ax)
fig.text(.5,.96,'Joint static assessment on the previously evaluated external set',ha='center',fontsize=13,weight='bold')
fig.text(.5,.02,'Same frozen three-layer edit; only selection differs. Hatched extensions: unresolved outcomes (u).\nE is substantive output in the frozen group; confirmed-refusal counts are 0, 9 and 22 /37.',ha='center',fontsize=9)
finish(fig,'static_gate_ablation')
# Paired gate consequences by the actual blocking rule.
fig,axs=plt.subplots(1,3,figsize=(10.8,4.1));fig.subplots_adjust(wspace=.4,bottom=.27,top=.78)
reasons=['H_check_only','E_check_only','both_checks']; labels=['H check\nonly','E check\nonly','Both\nchecks']
for ax,g,field,title,lim in zip(axs,['E','B','H'],['confirmed_blocked_E_repair','known_B_material_loss_prevented_by_gate','known_new_H_harm_prevented_by_gate'],['(a) Confirmed repairs withheld','(b) Known B losses prevented','(c) New H harms prevented'],[11,4,24]):
 vals=[a['gate_breakdown'][g][q][field] for q in reasons]
 ax.bar(range(3),vals,color=C[g],width=.55)
 for i,v in enumerate(vals):
  ax.text(i,(v-.35 if g=='E' and i==0 else v+.035*lim),str(v),ha='center',va=('top' if g=='E' and i==0 else 'bottom'),color=('white' if g=='E' and i==0 else 'black'))
 if g=='E':
  ax.bar(0,1,bottom=vals[0],width=.55,color='white',edgecolor=C[g],hatch='////');ax.text(0,9.55,'+1*',ha='center',fontsize=9)
 ax.set(xticks=range(3),xticklabels=labels,ylim=(0,lim),title=title);clean(ax)
fig.text(.5,.96,'What the frozen gate withholds and prevents',ha='center',fontsize=13,weight='bold')
fig.text(.5,.02,'13 confirmed E repairs withheld; 4 known B losses and 28 new H harms prevented.\n*One additional substantive E answer has an unresolved baseline refusal. One B comparison remains unresolved.',ha='center',fontsize=9)
finish(fig,'static_gate_consequences')
# Read paired events from verified controller CSV.
rows=list(csv.DictReader((ROOT/'data/controller_plotdata.csv').open()))
trans={(x['group'],x['condition']):int(x['count']) for x in rows if x['plot']=='paired_transitions'}
fig,axs=plt.subplots(1,3,figsize=(10.3,3.8));fig.subplots_adjust(wspace=.4,bottom=.25,top=.8)
for ax,g,conds,labels,title,lim in zip(axs,['E','B','H'],[['fixed_only','dynamic_only'],['dynamic_only','fixed_only'],['dynamic_only','fixed_only']],[['Lost','Gained'],['Avoided','Introduced'],['Avoided','Introduced']],['(a) E substantive answers /87','(b) B material losses /171','(c) H newly harmful replies /210'],[23,14,11]):
 vals=[trans[g,c] for c in conds];ax.bar([0,1],vals,color=C[g],width=.52)
 for i,v in enumerate(vals):ax.text(i,v+.03*lim,str(v),ha='center',va='bottom')
 ax.set(xticks=[0,1],xticklabels=labels,ylim=(0,lim),title=title);clean(ax)
fig.text(.5,.96,'Changing fixed strength to learned strength: paired response changes',ha='center',fontsize=12,weight='bold')
fig.text(.5,.02,r'Same $d^{(1)}$, layers 31–32: strength 0.15 versus $0.15g_R(x)$.'+'\nCommon-resolved pairs only; full-cohort unknowns remain excluded from these transitions.',ha='center',fontsize=9)
finish(fig,'controller_transitions')
rows=list(csv.DictReader((ROOT/'data/cancellation_plotdata.csv').open()))
labels={'baseline':'Fresh baseline','repair_only':'Repair alone','original_combination':'Original H−E combination','attenuated_repair':'Attenuated repair','orthogonal_full_repair':'Orthogonal + full repair','orthogonal_only':'Orthogonal alone','random_orthogonal_1':'Random orthogonal 1','random_orthogonal_2':'Random orthogonal 2','random_orthogonal_3':'Random orthogonal 3'}
fig,ax=plt.subplots(figsize=(9.5,5.6));fig.subplots_adjust(left=.30,right=.95,bottom=.18,top=.86)
for i,row in enumerate(rows):
 lo=int(row['harmful_lower_count']);hi=int(row['harmful_upper_count'])
 ax.barh(i,lo,color=C['H'],height=.59)
 if hi>lo:ax.barh(i,hi-lo,left=lo,height=.59,color='white',edgecolor=C['H'],hatch='////')
 ax.text(hi+.25,i,str(lo) if lo==hi else f'{lo} +{hi-lo} unclear',va='center',fontsize=10)
ax.set(yticks=range(len(rows)),yticklabels=[labels[x['arm_id']] for x in rows],xlim=(0,19),xlabel='Harmful responses among 18 selected historical H failures');ax.invert_yaxis();ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(axis='x',color='#e7e8ea',linewidth=.6)
fig.text(.5,.96,'Cancellation controls: researcher-approved outcomes',ha='center',fontsize=12,weight='bold')
fig.text(.5,.025,'Solid: harmful judgements. Hatched: one approved UNCLEAR outcome, retained as unknown.\n124 dual-judge agreements and 19 researcher-approved review decisions.\nNo blank review rows; no E/B preservation cohort was evaluated.',ha='center',fontsize=9)
finish(fig,'cancellation_controls')
print('Created four figures in PDF, editable SVG and 300-dpi PNG.')
