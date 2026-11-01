#!/usr/bin/env python3
"""Inference with a trained single checkpoint or a complete 10-fold PepResMLP ensemble.
Input: peptide/protein pair FASTA (two sequence lines per >record), embeddings precomputed
by the same encoder and preprocessing as training. Labels are optional.
"""
import argparse, csv, json, sys
from pathlib import Path
from argparse import Namespace
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pep_resmlp_train import (EmbeddingStore, read_fasta_records, _split_sequence_pair,
    _extract_label, _extract_field, sanitize_id, clean_sequence_line,
    collate_batch, build_model, compute_metrics)

from pep_resmlp_train import (
    EmbeddingStore,
    parse_pair_fasta,
    collate_batch,
    build_model,
    compute_metrics,
)

class InferencePairs(Dataset):
    def __init__(self, fasta, embeddings, max_peptide_len, max_protein_len):
        self.store = EmbeddingStore(embeddings)
        self.max_peptide_len = max_peptide_len
        self.max_protein_len = max_protein_len
        self.rows = []

        examples = parse_pair_fasta(fasta)

        for ex in examples:
            self.rows.append((
                ex.raw_header,
                ex.peptide_id,
                ex.protein_id,
                ex.peptide_seq,
                ex.protein_seq,
                ex.label,
            ))
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        head,pi,qi,p,q,y=self.rows[i]
        return {'pep':self.store.get(pi,p)[:self.max_peptide_len],
                'prot':self.store.get(qi,q)[:self.max_protein_len],
                'label':torch.tensor(float(y if y is not None else 0)),
                'peptide_id':pi,'protein_id':qi,'raw_header':head}

def checkpoint_args(ckpt):
    args=Namespace(**ckpt['args'])
    if getattr(args,'interaction_features',None) != 'p,q,absdiff':
        raise ValueError(f"Expected p,q,absdiff, found {getattr(args,'interaction_features',None)}")
    return args

def safe_load(file):
    # Only load model checkpoints that you produced or explicitly trust.
    try: return torch.load(file,map_location='cpu',weights_only=False)
    except TypeError: return torch.load(file,map_location='cpu')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    gp=ap.add_mutually_exclusive_group(required=True)
    gp.add_argument('--run_dir',help='Completed run with fold1...fold10 checkpoints and ensemble_independent_test_metrics.csv')
    gp.add_argument('--weights',help='A single .pt training checkpoint')
    ap.add_argument('--fasta',required=True)
    ap.add_argument('--embedding_dir',required=True)
    ap.add_argument('--output_csv',required=True)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--batch_size',type=int,default=32)
    ap.add_argument('--threshold',type=float,default=None,help='Override validation-selected threshold')
    ap.add_argument('--max_peptide_len',type=int,default=None)
    ap.add_argument('--max_protein_len',type=int,default=None)
    args=ap.parse_args()
    if args.weights: paths=[Path(args.weights)]
    else:
        rd=Path(args.run_dir)
        paths=[]
        for fold_dir in sorted(
            (p for p in rd.iterdir()
             if p.is_dir()
             and p.name.startswith('fold')
             and p.name[4:].isdigit()),
            key=lambda p: int(p.name[4:])
        ):
            ckpts=list(fold_dir.glob('*.pt'))
            if len(ckpts)!=1: raise ValueError(f'Expected exactly 1 .pt file in {fold_dir}, got {len(ckpts)}')
            paths.append(ckpts[0])
        if len(paths)!=10: raise ValueError(f'Expected 10 fold checkpoints; found {len(paths)}')
    checkpoints=[safe_load(p) for p in paths]
    configs=[checkpoint_args(c) for c in checkpoints]
    for other in configs[1:]:
        for field in ['interaction_features','max_peptide_len','max_protein_len','hidden_dim','mlp_blocks']:
            if getattr(other,field)!=getattr(configs[0],field): raise ValueError(f'Incompatible fold checkpoints: {field}')
    base=configs[0]
    pep_len=args.max_peptide_len or base.max_peptide_len
    prot_len=args.max_protein_len or base.max_protein_len
    if pep_len!=base.max_peptide_len or prot_len!=base.max_protein_len:
        raise ValueError('Input truncation lengths must match training; override is not permitted')
    dataset=InferencePairs(args.fasta,args.embedding_dir,pep_len,prot_len)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=collate_batch)
    device=torch.device(args.device if args.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
    predictions=[]
    for path, ckpt,config in zip(paths,checkpoints,configs):
        model=build_model(config,int(ckpt['input_dim'])).to(device)
        model.load_state_dict(ckpt['model_state_dict'],strict=True)
        model.eval(); arr=[]
        with torch.inference_mode():
            for batch in loader:
                logits=model(batch['pep'].to(device),batch['prot'].to(device),batch['pep_mask'].to(device),batch['prot_mask'].to(device))
                arr.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        predictions.append(np.asarray(arr,dtype=float))
        print(f'Loaded {path}: {len(arr)} predictions')
    probs=np.asarray(predictions).mean(axis=0)
    if args.threshold is not None: threshold=args.threshold
    elif args.weights: threshold=float(checkpoints[0].get('threshold',0.5))
    else:
        metrics=Path(args.run_dir)/'ensemble_independent_test_metrics.csv'
        if not metrics.exists(): raise FileNotFoundError('No ensemble threshold report; specify --threshold explicitly')
        with open(metrics,newline='') as f: rows=list(csv.DictReader(f))
        rows=[r for r in rows if r.get('ensemble_threshold_strategy')=='oof_youden']
        if not rows: raise ValueError('No oof_youden threshold found; specify --threshold')
        vals={float(r['threshold']) for r in rows}
        if len(vals)!=1: raise ValueError('Different thresholds in ensemble report')
        threshold=vals.pop()
    if not (0<=threshold<=1): raise ValueError('Threshold outside [0,1]')
    out=Path(args.output_csv); out.parent.mkdir(parents=True,exist_ok=True)
    fieldnames=['record_id','peptide_id','protein_id','label','probability','prediction','threshold']+[f'probability_model{i}' for i in range(1,len(paths)+1)]
    with open(out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader()
        for i,((head,pi,qi,p,q,label),prob) in enumerate(zip(dataset.rows,probs)):
            row={'record_id':head,'peptide_id':pi,'protein_id':qi,'label':'' if label is None else label,
                 'probability':float(prob),'prediction':int(prob>=threshold),'threshold':threshold}
            row.update({f'probability_model{j+1}':float(predictions[j][i]) for j in range(len(paths))})
            w.writerow(row)
    print(f'Wrote {len(dataset)} predictions to {out}; threshold={threshold:.6f}')
    labels=[r[-1] for r in dataset.rows]
    if all(lab is not None for lab in labels) and len(set(labels))==2:
        metrics=compute_metrics(labels,probs,threshold)
        print('Labeled input metrics:',json.dumps(metrics,indent=2))
        out.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
