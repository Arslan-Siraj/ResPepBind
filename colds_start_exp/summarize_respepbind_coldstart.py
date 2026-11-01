#!/usr/bin/env python3
"""Summarize ResPepBind cold-start held-out test metrics (stdlib only)."""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

SCENARIOS = ('novel_peptide', 'novel_protein', 'double_cold')
METRICS = ('auc', 'aupr', 'acc', 'f1')

def read_single(path):
    with path.open(newline='') as fh:
        records = list(csv.DictReader(fh))
    if len(records) != 1:
        raise ValueError(f'{path}: expected exactly one row, found {len(records)}')
    row = records[0]
    if row.get('split') != 'coldstart_test':
        raise ValueError(f'{path}: unexpected test split {row.get("split")!r}')
    result = {}
    for field in METRICS:
        if field not in row:
            raise ValueError(f'{path}: missing metric {field}; columns={list(row)}')
        v = float(row[field])
        if not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError(f'{path}: invalid {field}={v}')
        result[field] = v
    return row, result

def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('coldstart_outputs_seed2026_10fold'))
    parser.add_argument('--folds', type=int, default=10)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    manifest = root / 'splits' / 'split_manifest.csv'
    if not manifest.is_file():
        parser.error(f'Missing split manifest: {manifest}')
    with manifest.open(newline='') as fh:
        partitions = list(csv.DictReader(fh))
    if not partitions:
        parser.error('Empty manifest')
    expected = 3 * 4 * args.folds
    if len(partitions) != expected:
        print(f'WARNING: {len(partitions)} partitions; expected {expected}')
    outdir = root / 'analysis'
    outdir.mkdir(parents=True, exist_ok=True)
    allrows = []
    incomplete = []
    by_group = defaultdict(list)
    seen = set()
    for p in partitions:
        scenario, identity, fold = p['scenario'], p['identity'], int(p['fold'])
        key = (scenario, identity, fold)
        if key in seen:
            raise ValueError(f'Duplicate manifest entry: {key}')
        seen.add(key)
        tag = f'{scenario}_id{identity}_fold{fold}'
        metrics_path = root / 'trained' / tag / 'independent_test_metrics.csv'
        if p.get('valid') != '1':
            incomplete.append({'scenario':scenario,'identity':identity,'fold':fold,'issue':'invalid partition'})
            continue
        if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
            incomplete.append({'scenario':scenario,'identity':identity,'fold':fold,'issue':'missing or empty held-out test metrics'})
            continue
        try:
            raw, values = read_single(metrics_path)
        except (ValueError, OSError) as e:
            incomplete.append({'scenario':scenario,'identity':identity,'fold':fold,'issue':str(e)})
            continue
        row = {'scenario':scenario,'identity':identity,'fold':fold, **values,
               'train_n':p.get('train_n',''),'test_n':p.get('test_n',''),
               'dropped_n':p.get('dropped_n',''),
               'unclustered_pairs_excluded':p.get('unclustered_pairs_excluded',''),
               'result_n':raw.get('n',''), 'threshold':raw.get('threshold',''),
               'metrics_file':str(metrics_path)}
        if row['result_n'] and int(float(row['result_n'])) != int(p['test_n']):
            incomplete.append({'scenario':scenario,'identity':identity,'fold':fold,'issue':f'test size mismatch: {row["result_n"]} vs {p["test_n"]}'})
            continue
        allrows.append(row)
        by_group[(scenario,identity)].append(row)
    allrows.sort(key=lambda r:(SCENARIOS.index(r['scenario']),float(r['identity']),r['fold']))
    write_csv(outdir / 'coldstart_per_fold.csv', allrows,
              ['scenario','identity','fold','auc','aupr','acc','f1','train_n','test_n','dropped_n','unclustered_pairs_excluded','result_n','threshold','metrics_file'])
    write_csv(outdir / 'coldstart_missing_or_invalid.csv',incomplete,['scenario','identity','fold','issue'])
    summary=[]
    for scenario in SCENARIOS:
        for identity in ('0.6','0.7','0.8','0.9'):
            data = by_group[(scenario,identity)]
            line={'scenario':scenario,'identity':identity,'n_folds':len(data),
                  'expected_folds':args.folds,
                  'mean_test_n':round(statistics.mean(int(x['test_n']) for x in data),2) if data else '',
                  'mean_dropped_n':round(statistics.mean(int(x['dropped_n']) for x in data),2) if data else ''}
            for metric in METRICS:
                samples=[r[metric] for r in data]
                line[f'{metric}_mean']=statistics.mean(samples) if samples else ''
                line[f'{metric}_sd']=statistics.stdev(samples) if len(samples)>1 else ''
                line[f'{metric}_formatted']=(f'{line[f"{metric}_mean"]:.4f} ± {line[f"{metric}_sd"]:.4f}' if len(samples)>1 else 'INCOMPLETE')
            summary.append(line)
    fields=['scenario','identity','n_folds','expected_folds','mean_test_n','mean_dropped_n'] + [f'{m}_{suffix}' for m in METRICS for suffix in ('mean','sd','formatted')]
    write_csv(outdir/'coldstart_mean_std.csv',summary,fields)
    md=['# ResPepBind held-out cold-start test results','',f'Root: `{root}`','',f'Valid evaluations: **{len(allrows)} / {len(partitions)}**','',
        '| Scenario | Identity | Folds | AUC (mean ± SD) | AUPR (mean ± SD) | ACC (mean ± SD) | F1 (mean ± SD) |',
        '|---|---|---:|---|---|---|---|']
    for s in summary:
        md.append('| ' + ' | '.join([s['scenario'],s['identity'],str(s['n_folds']),
                                   *(s[f'{m}_formatted'] for m in METRICS)]) + ' |')
    md.extend(['','Results are calculated from `independent_test_metrics.csv` on `coldstart_test`, not internal validation.',
               'SD is the sample standard deviation across completed folds (ddof=1).',
               'Comparisons with PepLM-GNN Figure 3 are literature-level rather than paired, identical-split comparisons.',
               'Inspect `coldstart_missing_or_invalid.csv` and the split exclusion policy before publication.'])
    (outdir/'coldstart_summary.md').write_text('\n'.join(md)+'\n')
    print('\n'.join(md))
    if incomplete:
        print(f'\nWARNING: {len(incomplete)} missing/invalid evaluations. See {outdir / "coldstart_missing_or_invalid.csv"}')
    print(f'\nSaved summary files in {outdir}')

if __name__ == '__main__':
    main()


# python summarize_respepbind_coldstart.py   --root coldstart_outputs_seed2026_10fold   --folds 10