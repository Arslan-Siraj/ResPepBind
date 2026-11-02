#!/usr/bin/env python3
"""Offline validation of the actual 10 saved checkpoints (no ProtT5 download)."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from respepbind.predict import find_checkpoints,read_threshold,torch_load_trusted
from respepbind.model import build_model
from types import SimpleNamespace


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',type=Path,default=Path.home()/'.cache'/'respepbind'/'models'/'seed2026')
    args=ap.parse_args()
    paths=find_checkpoints(args.run_dir)
    threshold=read_threshold(args.run_dir,None)
    configurations=[]
    params=None
    for i,path in enumerate(paths,1):
        checkpoint=torch_load_trusted(path)
        assert set(('args','input_dim','model_state_dict'))<=set(checkpoint),f'fold{i}: invalid checkpoint'
        cfg=SimpleNamespace(**checkpoint['args'])
        assert int(checkpoint['input_dim'])==1024
        assert getattr(cfg,'interaction_features',None)=='p,q,absdiff'
        model=build_model(cfg,int(checkpoint['input_dim']))
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        n_params=sum(p.numel() for p in model.parameters())
        if params is None: params=n_params
        assert params==n_params
        configurations.append((getattr(cfg,'variant',None),getattr(cfg,'hidden_dim',None),getattr(cfg,'mlp_blocks',None)))
        print(f'PASS fold{i:02d}: {path.name} / {n_params:,} parameters')
    assert len(set(configurations))==1,'Different architectures across folds'
    print(f'PASS all 10 real checkpoints; threshold={threshold:.12f} / model parameters={params:,}')

if __name__=='__main__':
    main()
