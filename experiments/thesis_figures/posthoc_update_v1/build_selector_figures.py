#!/usr/bin/env python3
"""Build four audited selector figures from adjacent frozen CSVs.

Usage: python build_selector_figures.py [--output-dir Figures]
Requirements: Python 3, matplotlib, numpy. No training, API calls or model access.
All paths resolve relative to this script, so the folder can be moved unchanged.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data'
P = argparse.ArgumentParser()
P.add_argument('--output-dir', type=Path, default=ROOT/'Figures')
ARGS = P.parse_args()
OUT = ARGS.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)

COL = {'E':'#B47B48','B':'#2F6595','H':'#9E594C'}
INK = '#27313A'
GREY = '#66717A'
GRID = '#E1E5E8'
HATCHES = ['', '///', '...']
MARKERS = ['o','s','^']
METHOD_GREYS = ['#303C48','#777F86','#ABB0B5']
PROFILE_IDS = ['zero_validation_damage','H1pct_B2pct_validation','H5pct_B5pct_validation']
PROFILE_TEXT = ['Strict\nH 0 / B 0','Middle\nH 1% / B 2%','Loose\nH 5% / B 5%']
plt.rcParams.update({
 'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,'axes.titlesize':11,
 'axes.titleweight':'bold','text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,
 'ytick.color':GREY,'axes.edgecolor':'#BAC2C7','axes.linewidth':.65,
 'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,
 'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300,
 'figure.facecolor':'white','axes.facecolor':'white','hatch.linewidth':.7,
 'lines.solid_capstyle':'round','svg.hashsalt':'routegate-selector-figures-v1',
})

def read(name):
    with (DATA/name).open(newline='') as f:
        return list(csv.DictReader(f))

def mix(col, whiten):
    return tuple((1-whiten)*np.array(to_rgb(col))+whiten*np.ones(3))

def clean(ax, integer=True):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='x',length=0,pad=7)
    ax.tick_params(axis='y',length=2,pad=4)
    ax.grid(axis='y', color=GRID, linewidth=.55, zorder=0)
    ax.set_axisbelow(True)
    if integer: ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

def save(fig,name):
    for ext in ('pdf','svg','png'):
        metadata = {'Creator':'RouteGate audited thesis figure source'} if ext=='pdf' else None
        fig.savefig(OUT/(name+'.'+ext),dpi=300,bbox_inches='tight',pad_inches=.16,metadata=metadata)
    plt.close(fig)

def profiles(study):
    nn = study=='nn'
    rows = read('selector_capacity_profiles.csv' if nn else 'moe_selector_profiles.csv')
    rows = {r['policy']:r for r in rows}
    methods = ['class_selector','outcome_selector','nn_outcome_selector'] if nn else ['residual_only','routing_only','combined']
    names = ['Linear class','Linear outcome','Neural outcome'] if nn else ['Residual only','Routing only','Combined']
    keys = ['E_retained','B_material_loss','H_new_harm'] if nn else ['E_substantive_selected','B_material_loss','H_harmful_static_selected']
    unknowns = [None,'B_unknown_selected',None] if nn else ['E_unknown_selected','B_unknown_selected','H_unknown_selected']
    titles = ['(a) Substantive E answers retained','(b) Material B losses','(c) Newly harmful H responses' if nn else '(c) Harmful static H selected']
    denoms = [89,178,212 if nn else 213]
    limits = [76,21,13] if nn else [93,11,25]
    fig,axes=plt.subplots(3,1,figsize=(6.1,8.7))
    fig.subplots_adjust(left=.115,right=.98,top=.875,bottom=.15,hspace=.64)
    legend = [Patch(facecolor=mix('#727D87',[0,.42,.7][j]),edgecolor=INK,hatch=HATCHES[j],linewidth=.65,label=n) for j,n in enumerate(names)]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.54,.955),ncol=3,frameon=False,handlelength=1.5,columnspacing=1.0)
    fig.text(.05,.988, 'Selector capacity: fixed two-layer edit' if nn else 'MoE features: fixed three-layer edit',fontsize=12,weight='bold',va='top')
    for c,(ax,g,key,unk,title,denom,limit) in enumerate(zip(axes,'EBH',keys,unknowns,titles,denoms,limits)):
        clean(ax)
        ax.set_title(title,loc='left',fontsize=11,pad=12,color=COL[g])
        ax.set_ylim(0,limit)
        ax.set_xlim(-.53,2.53)
        ax.set_xticks(range(3),PROFILE_TEXT)
        ax.set_ylabel('Count / '+str(denom))
        fixed=rows['always_fixed']
        fixed_val=int(fixed[key]); fixed_unk=int(fixed[unk]) if unk else 0
        ax.axhline(fixed_val,color=GREY,lw=.8,ls=(0,(4,3)),zorder=1)
        reference_left = g=='E'
        ax.text(-.48 if reference_left else 2.48,fixed_val+limit*.024,'Always fixed: '+str(fixed_val)+(f' +{fixed_unk}u' if fixed_unk else ''),fontsize=10,ha='left' if reference_left else 'right',va='bottom',color=GREY)
        width=.22
        for j,method in enumerate(methods):
            for i,p in enumerate(PROFILE_IDS):
                r=rows[method+'__'+p]
                value=int(r[key]); un=int(r[unk]) if unk else 0
                x=i+(j-1)*.255
                ax.bar(x,value,width=width,color=mix(COL[g],[0,.38,.7][j]),edgecolor=COL[g],linewidth=.7,hatch=HATCHES[j],zorder=3)
                if un:
                    ax.bar(x,un,bottom=value,width=width,facecolor='white',edgecolor=COL[g],linewidth=.9,linestyle=(0,(2,1)),zorder=4)
                text=str(value)+(f'\n+{un}u' if un else '')
                ax.text(x,value+un+limit*.025,text,fontsize=10,ha='center',va='bottom',color=INK,linespacing=.95,bbox={'facecolor':'white','edgecolor':'none','pad':.15})
    fig.text(.05,.085,'Budgets apply to inner validation; counts pool five outer folds.',fontsize=10,color=GREY)
    fig.text(.05,.057,'Strength 0.15. Unfilled caps (u) retain selected unknown outcomes.',fontsize=10,color=GREY)
    fig.text(.05,.014,('H: 212 known non-harmful baselines; one unknown excluded.\nE: retained fixed-edit answers, not complete-system repair.' if nn else
                        'H: harmful static replies selected /213; not new harm.\nE: retained fixed-edit answers, including one separate unknown.'),fontsize=10,color=GREY,linespacing=1.35)
    save(fig,'selector_capacity_profiles' if nn else 'moe_selector_profiles')

def folds():
    nn = read('selector_capacity_folds.csv')
    moe = read('moe_selector_folds.csv')
    fig,axes = plt.subplots(3,2,figsize=(6.2,8.5),sharey=True)
    fig.subplots_adjust(left=.105,right=.985,top=.77,bottom=.15,hspace=.70,wspace=.20)
    fig.text(.05,.985,'Edit selection across outer folds',fontsize=13,weight='bold',va='top')
    studies = [
      (['class_selector','outcome_selector','nn_outcome_selector'],['Linear class','Linear outcome','Neural outcome'],nn),
      (['residual_only','routing_only','combined'],['Residual only','Routing only','Combined'],moe)]
    for col,(methods,names,records) in enumerate(studies):
        handles=[Line2D([0],[0],color=METHOD_GREYS[j],marker=MARKERS[j],lw=1.4,ms=5,label=names[j]) for j in range(3)]
        fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.07 if col==0 else .56,.91),ncol=1,frameon=False,handlelength=1.5)
        fig.text(.07 if col==0 else .56,.93,'Two-layer selectors' if col==0 else 'Three-layer features',fontsize=11,weight='bold')
        for row,p in enumerate(PROFILE_IDS):
            ax=axes[row,col];clean(ax)
            ax.set_title(['Strict: H 0 / B 0','Middle: H 1% / B 2%','Loose: H 5% / B 5%'][row],fontsize=10.5,loc='left',pad=9)
            ax.set_ylim(-2.5,70);ax.set_xlim(-.2,4.2)
            ax.set_xticks(range(5));ax.set_xlabel('Outer fold',labelpad=4)
            if col==0:ax.set_ylabel('Edits selected')
            for j,m in enumerate(methods):
                sub=[r for r in records if (r['policy']==m+'__'+p if col==0 else r['arm']==m and r['profile']==p)]
                sub=sorted(sub,key=lambda r:int(r['fold'] if col==0 else r['outer_fold']))
                assert len(sub)==5
                ax.plot(range(5),[int(r['edits']) for r in sub],color=METHOD_GREYS[j],marker=MARKERS[j],lw=1.4,ms=5,markeredgewidth=.6,markeredgecolor='white',zorder=4-j)
    fig.text(.05,.07,'Fold numbers are not matched between the two studies.\nVariation reflects both fitted scores and thresholds;\ntheir individual contributions are not isolated.',fontsize=10,color=GREY,linespacing=1.4,va='top')
    save(fig,'selector_fold_variation')

def source_controls():
    records=[r for r in read('moe_source_quality.csv') if r['group']=='H']
    sources=['ALL','harmbench_text_test','orbench_toxic','strongreject']
    arms=['residual_only','routing_only','combined']
    labels=['All H\n20 / 213','HarmBench\n10 / 32','OR-Bench\nToxic: 5 / 62','StrongREJECT\n5 / 119']
    fig,axes=plt.subplots(2,1,figsize=(6.2,8.0))
    fig.subplots_adjust(left=.12,right=.985,top=.865,bottom=.28,hspace=.62)
    fig.text(.05,.988,'H prediction quality: pooled and within source',fontsize=12,weight='bold',va='top')
    handles=[Line2D([0],[0],color=METHOD_GREYS[j],marker=MARKERS[j],linestyle='none',ms=4.8,label=['Residual only','Routing only','Combined'][j]) for j in range(3)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.955),ncol=3,frameon=False,columnspacing=1.3)
    for k,(ax,key,title) in enumerate(zip(axes,['auc','average_precision'],['(a) Area under the ROC curve','(b) Average precision'])):
        clean(ax,False);ax.set_title(title,fontsize=11,loc='left',pad=12,color=COL['H'])
        ax.axvspan(-.5,.5,color='#F0F2F4',zorder=0)
        ax.axvline(.5,color='#C7CDD2',lw=.6,ls=(0,(3,3)))
        ax.set_xticks(range(4),labels);ax.set_xlim(-.45,3.45)
        ax.set_ylim(.45,1.0) if k==0 else ax.set_ylim(0,.75)
        ax.set_ylabel('AUC' if k==0 else 'Average precision')
        if k==0:
            ax.axhline(.5,color=GREY,lw=.75,ls=(0,(3,3)))
            ax.text(3.38,.51,'AUC 0.5 reference',ha='right',fontsize=10,color=GREY)
        else:
            for i,s in enumerate(sources):
                r=next(r for r in records if r['source']==s)
                prevalence=int(r['positives'])/int(r['known_rows'])
                ax.plot([i-.32,i+.32],[prevalence,prevalence],color=GREY,lw=1,ls=(0,(3,2)),zorder=2)
            ax.text(.97,.97,'Dashed segments: positive fraction',transform=ax.transAxes,ha='right',va='top',fontsize=10,color=GREY)
        for j,a in enumerate(arms):
            ys=[float(next(r for r in records if r['arm']==a and r['source']==s)[key]) for s in sources]
            ax.plot(np.arange(4)+(j-1)*.09,ys,color=METHOD_GREYS[j],marker=MARKERS[j],ms=5.3,linestyle='none',markeredgecolor='white',markeredgewidth=.6,zorder=4-j)
    fig.text(.05,.115,'Labels show harmful outcomes / known H responses.\nWithin each source, routing has higher observed AUC and AP,\nbut there are only 5–10 harmful outcomes per source.',fontsize=10,color=GREY,linespacing=1.4)
    fig.text(.05,.023,'Descriptive out-of-fold scores for the three-layer static edit.\nSource controls do not establish the cause of pooled gains.\nAverage precision depends on the positive fraction.',fontsize=10,color=GREY,linespacing=1.4)
    save(fig,'moe_h_source_controls')

if __name__=='__main__':
    profiles('nn')
    profiles('moe')
    folds()
    source_controls()
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.glob('*.csv')) if p.name.startswith(('selector','moe'))}
    # Written beside the figures; the committed copy in this directory records the thesis build.
    (OUT/'selector_figure_manifest.json').write_text(json.dumps({'data_sha256':sources,'figures':['selector_capacity_profiles','moe_selector_profiles','selector_fold_variation','moe_h_source_controls'],'endpoint_colours':COL,'unknown_display':'Known count remains unchanged; selected unknowns shown as separate unfilled caps with u annotation.','uncertainty':'No confidence intervals are invented.','layout_revision':'20260913 print-size correction; data unchanged','matplotlib_version':matplotlib.__version__},indent=2)+'\n')
    print('Built four figures in PDF, editable SVG and 300 dpi PNG:',OUT)
