#!/usr/bin/env python3
"""Offline CPU smoke test: synthetic 10-fold models, sequence hash cache, FASTA and CSV."""
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from respepbind.model import build_model
from respepbind.predict import load_fasta, load_csv, EmbeddingResolver


def main():
    single = load_fasta(HERE/'examples/single_pair.fasta')
    assert len(single) == 1 and single[0].label == 1
    full = load_fasta(HERE/'examples/pairs.fasta')
    assert len(full) == 2, len(full)
    unlabeled = load_fasta(HERE/'examples/unlabeled_pair.fasta')
    assert unlabeled[0].label is None
    from_csv = load_csv(HERE/'examples/pairs.csv')
    assert len(from_csv) == 2 and from_csv[0].label == 1 and from_csv[1].label is None

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        runs = root/'Final_model'; embs = root/'embeddings'
        embs.mkdir()
        cfg = Namespace(variant='residual_interaction_mlp', hidden_dim=16, dropout=0.25, mlp_blocks=2,
                        input_dropout=0.2, block_dropout=0.25, block_dropouts='0.25,0.25',
                        head_dropout=0.15, mlp_expand_ratio=2.0, head_ratio=0.5,
                        activation='gelu', norm_type='layernorm', no_residual=False,
                        residual_scale=1.0, interaction_features='p,q,absdiff',
                        max_peptide_len=50, max_protein_len=1024, seed=2026)
        torch.manual_seed(2026)
        model = build_model(cfg, 1024)
        for fold in range(1, 11):
            folder = runs/f'fold{fold}';folder.mkdir(parents=True)
            torch.save({'args': vars(cfg), 'input_dim':1024,
                        'model_state_dict':model.state_dict()},folder/'best_model.pt')
        (runs/'ensemble_independent_test_metrics.csv').write_text('ensemble_threshold_strategy,threshold\noof_youden,0.4309651851654053\n')
        sequences = [from_csv[0].peptide, from_csv[0].protein, from_csv[1].peptide, from_csv[1].protein]
        for seq in set(sequences):
            h = hashlib.sha1(seq.encode()).hexdigest()
            torch.save(torch.ones(min(len(seq), 1024), 1024) * 0.01, embs/f'{h}.pt')
        out = root/'predictions.csv'
        cmd = [sys.executable, '-m', 'respepbind.predict', '--run-dir', str(runs),
               '--csv', str(HERE/'examples/pairs.csv'), '--embedding-dir', str(embs),
               '--existing-only', '--cache-dir', str(root/'inference_cache'), '--output-csv', str(out), '--threads', '2']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode:
            raise RuntimeError(result.stdout+'\n'+result.stderr)
        with out.open(newline='') as fh:
            rows=list(csv.DictReader(fh))
        assert len(rows)==2 and len(rows[0])==22
        assert rows[0]['true_label']=='1' and rows[1]['true_label']==''
        assert all(float(row['interaction_probability']) >= 0 and float(row['interaction_probability']) <= 1 for row in rows)
        assert all(row['peptide_embedding_source']=='existing_sequence_hash' for row in rows)
        assert not out.with_suffix('.metrics.json').exists()
        print('PASS: multi-record FASTA, labeled/unlabeled CSV, CPU 10-fold inference, threshold, output columns')
        # End-to-end one-pair input, without a FASTA or CSV file.
        single_out=root/'single.csv'
        direct = [sys.executable,'-m', 'respepbind.predict', '--run-dir',str(runs),
                  '--peptide',from_csv[0].peptide,'--protein',from_csv[0].protein,
                  '--embedding-dir',str(embs),'--existing-only',
                  '--cache-dir',str(root/'inference_cache'), '--output-csv',str(single_out),'--threads','2']
        direct_result=subprocess.run(direct,capture_output=True,text=True,env=env)
        if direct_result.returncode: raise RuntimeError(direct_result.stdout+'\n'+direct_result.stderr)
        with single_out.open(newline='') as fh:
            direct_rows=list(csv.DictReader(fh))
        assert len(direct_rows)==1 and direct_rows[0]['id']=='pair_1'
        assert direct_rows[0]['true_label']==''
        print('PASS: direct single peptide+protein pair input')
        # Mixed binary labels yield a metrics JSON (same evaluator as training).
        two_labeled=root/'two_labeled.csv'
        two_labeled.write_text('id,peptide,protein,true_label\n'+
                               f'first,{from_csv[0].peptide},{from_csv[0].protein},1\n'+
                               f'second,{from_csv[1].peptide},{from_csv[1].protein},0\n')
        labeled_out=root/'labeled.csv'
        labeled_cmd=[sys.executable,'-m', 'respepbind.predict', '--run-dir',str(runs),
                     '--csv',str(two_labeled),'--embedding-dir',str(embs),
                     '--existing-only','--cache-dir',str(root/'inference_cache'), '--output-csv',str(labeled_out),'--threads','2']
        labeled_result=subprocess.run(labeled_cmd,capture_output=True,text=True,env=env)
        if labeled_result.returncode: raise RuntimeError(labeled_result.stdout+'\n'+labeled_result.stderr)
        summary=json.loads(labeled_out.with_suffix('.metrics.json').read_text())
        assert summary['n_samples']==2 and all(k in summary for k in ('acc','f1','auc','aupr'))
        print('PASS: labeled CSV saves ACC, F1, ROC-AUC, and AP')
        # Automatic embedding generation path uses same cache without loading real ProtT5 in the test.
        resolver = EmbeddingResolver('Rostlab/prot_t5_xl_uniref50', root/'cache', None, False, True, 1022, 128, torch.device('cpu'))
        class MockProtT5:
            def embed(self, seq):
                return torch.full((len(seq),1024), 0.2)
        resolver.plm = MockProtT5()
        a, src = resolver.get('ACDEFG', 'pair_1', 'peptide', 1024)
        assert src=='new_prott5_cpu' and a.shape==(6,1024)
        resolver2=EmbeddingResolver('Rostlab/prot_t5_xl_uniref50', root/'cache', None, False, False, 1022, 128, torch.device('cpu'))
        b, src2=resolver2.get('ACDEFG', 'pair_1', 'peptide',1024)
        assert src2=='generated_cache' and torch.equal(a,b)
        print('PASS: automatic generation dispatch and persistent cache (encoder mocked; no downloads)')

if __name__=='__main__':
    main()
