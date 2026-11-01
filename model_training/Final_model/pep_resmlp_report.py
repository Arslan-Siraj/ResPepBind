#!/usr/bin/env python3
"""Summarize completed fixed-architecture runs without selecting by test performance."""
import argparse,csv,json,statistics
from pathlib import Path

def rows(path):
    if not path.exists():return []
    with open(path,newline='') as f:return list(csv.DictReader(f))

def stats(values):
    v=[float(x) for x in values if x not in ('',None)]
    return {'mean':statistics.mean(v),'std':statistics.stdev(v) if len(v)>1 else 0.,'n':len(v)} if v else None

def summarize(run):
    v=rows(run/'validation_summary.csv')
    e=rows(run/'ensemble_independent_test_metrics.csv')
    model=rows(run/'model_summary.csv')
    info={'run':run.name,'configuration':model[0] if model else {}, 'validation':{},'independent_ensemble':{}}
    for m in ['acc','f1','auc','aupr','val_loss']:
        key='val_'+m if not m.startswith('val_') else m
        info['validation'][m]=stats([r.get(key) for r in v])
    for r in e:
        if r.get('ensemble_threshold_strategy')!='oof_youden':continue
        info['independent_ensemble'][r['split']]={k:float(r[k]) for k in ('acc','f1','auc','aupr','threshold','loss') if k in r}
    history=[]
    for fd in sorted(run.glob('fold[0-9]*')):
        h=rows(fd/'history.csv')
        best=[x for x in h if str(x.get('is_best','0')) in ('1','1.0')]
        if not best and h:
            # validation_summary has the saved best epoch
            row=next((x for x in v if x.get('fold')==fd.name),None)
            if row:best=[x for x in h if int(float(x['epoch']))==int(float(row['best_epoch']))]
        if best:
            b=best[-1]
            history.append({'fold':fd.name,'best_epoch':b.get('epoch'),
            'train_aupr':b.get('train_aupr'),'val_aupr':b.get('val_aupr'),
            'train_loss':b.get('train_loss'),'val_loss':b.get('val_loss'),
            'loss_gap':b.get('generalization_gap')})
    # Out-of-fold scores, each sample predicted by only its held-out fold.
    yy=[]; pp=[]
    for fd in sorted(run.glob('fold[0-9]*')):
        for record in rows(fd/'val_predictions_best.csv'):
            yy.append(int(float(record['label'])))
            pp.append(float(record['probability']))
    if yy and len(set(yy))==2:
        from sklearn.metrics import roc_auc_score, average_precision_score
        info['oof_metrics']={'n':len(yy),'roc_auc':float(roc_auc_score(yy,pp)),
             'average_precision':float(average_precision_score(yy,pp))}
    else: info['oof_metrics']=None
    info['checkpoint_history']=history
    info['checkpoint_loss_gap']=stats([x['loss_gap'] for x in history])
    return info

def main():
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--run_dir');group.add_argument('--parent_dir');p.add_argument('--out_dir',required=True)
    a=p.parse_args();output=Path(a.out_dir);output.mkdir(parents=True,exist_ok=True)
    runs=[Path(a.run_dir)] if a.run_dir else sorted(p for p in Path(a.parent_dir).iterdir() if p.is_dir() and (p/'validation_summary.csv').exists())
    data=[summarize(x) for x in runs]
    (output/'publication_summary.json').write_text(json.dumps(data,indent=2))
    with open(output/'model_comparison.csv','w',newline='') as f:
        cols=['run','parameters','cv_aupr_mean','cv_aupr_sd','pooled_oof_aupr','pooled_oof_auc','cv_auc_mean','cv_f1_mean','checkpoint_loss_gap_mean','Test251_aupr','Test1440_aupr','Test167_aupr','Leadspep_aupr','NewInd_aupr']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for d in data:
            g=lambda met,key: d['validation'].get(met,{}).get(key,'') if d['validation'].get(met) else ''
            r={'run':d['run'],'parameters':d['configuration'].get('trainable_parameters',''),
            'cv_aupr_mean':g('aupr','mean'),'cv_aupr_sd':g('aupr','std'),'cv_auc_mean':g('auc','mean'),
            'cv_f1_mean':g('f1','mean'),'pooled_oof_aupr':(d['oof_metrics'] or {}).get('average_precision',''),'pooled_oof_auc':(d['oof_metrics'] or {}).get('roc_auc',''),'checkpoint_loss_gap_mean':(d['checkpoint_loss_gap'] or {}).get('mean','')}
            r.update({f'{name}_aupr':d['independent_ensemble'].get(name,{}).get('aupr','') for name in ['Test251','Test1440','Test167','Leadspep','NewInd']})
            w.writerow(r)
    md=['# Fixed-model evaluation summary','', 'Interpretation: models were fixed before evaluation; independent results are descriptive, not tuning criteria.','',
        '| Seed/run | CV AUPR mean ± SD | Checkpoint loss gap | Independent ensemble AUPR (Test251 / Test1440 / Test167 / Leadspep / NewInd) |', '|---|---|---|---|']
    for d in data:
        cv=d['validation'].get('aupr');gap=d['checkpoint_loss_gap']
        auc='NA' if not cv else f"{cv['mean']:.4f} ± {cv['std']:.4f}"
        gp='NA' if not gap else f"{gap['mean']:.4f}"
        vals=' / '.join(f"{d['independent_ensemble'][x]['aupr']:.4f}" if x in d['independent_ensemble'] else 'NA' for x in ['Test251','Test1440','Test167','Leadspep','NewInd'])
        md.append(f"| {d['run']} | {auc} | {gp} | {vals} |")
    md.extend(['','## Important limitations','','- Fold-specific best-epoch validation metrics are used for CV summaries; they are selected on the same fold and may be optimistic. Pooled OOF AUC/AP, if saved, are in JSON and CSV.','- Training/validation loss gap at each selected checkpoint is a descriptive diagnostic, not proof of no overfitting.','- Verify duplicate pairs, peptide/protein homology and leakage before claiming independent generalization.','- Parameter count excludes the pretrained embedding model; 10-fold ensemble weights are ten times the individual classifier weights.','- AUPR is sklearn average precision; comparisons require consistent dataset construction and prevalence.'])
    (output/'publication_summary.md').write_text('\n'.join(md)+'\n')
    print('Wrote',output/'model_comparison.csv',output/'publication_summary.md',output/'publication_summary.json')
if __name__=='__main__':main()
