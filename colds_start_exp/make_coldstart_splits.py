#!/usr/bin/env python3
"""Generate audited CD-HIT cold-start FASTA partitions for ResPepBind.

Requires pep_resmlp_train.py in --trainer-dir to guarantee matching FASTA parser.
A strict double-cold fold retains pairs only when BOTH entity-cluster assignments
are in the test fold, and excludes cross-partition pairs from that fold.
"""
import argparse, csv, json, random, re, shutil, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path


def write_fasta(path, examples):
    with open(path, "w") as f:
        for ex in examples:
            if not ex.peptide_seq or not ex.protein_seq:
                raise ValueError(
                    f"Missing sequences in {ex.raw_header}"
                )

            f.write(
                ">" + ex.raw_header + "\n"
                + "Peptide: " + ex.peptide_seq + "\n"
                + "Protein: " + ex.protein_seq + "\n"
            )


def cluster(sequences, kind, cutoff, output, threads):
    seqs = sorted(set(sequences))
    fasta = output / f'{kind}.fa'
    with open(fasta,'w') as f:
        for i,seq in enumerate(seqs):
            f.write(f'>s{i}\n{seq}\n')
    dest = output / kind
    word = 3 if cutoff <= 0.60 else 4 if cutoff <= 0.70 else 5
    cmd = ['cd-hit','-i',str(fasta),'-o',str(dest),'-c',str(cutoff),'-n',str(word),
           '-d','0','-l','5','-M','0','-T',str(threads)]
    with open(output / f'{kind}.cdhit.log','w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    mapping = {}
    clfile = Path(str(dest)+'.clstr')
    current = None
    for line in clfile.read_text().splitlines():
        if line.startswith('>Cluster '):
            current = int(line.split()[-1])
        else:
            match = re.search(r'>s(\d+)\.\.\.',line)
            if match:
                mapping[seqs[int(match.group(1))]] = current
    missing = [seq for seq in seqs if seq not in mapping]
    if missing:
        report = output / f'{kind}.unclustered.tsv'
        with open(report, 'w') as f:
            f.write('sequence\tlength\n')
            for seq in missing:
                f.write(f'{seq}\t{len(seq)}\n')
        print(f'WARNING: CD-HIT did not cluster {len(missing)}/{len(seqs)} unique {kind} sequences. '
              f'Details: {report}; CD-HIT log: {output / (kind + ".cdhit.log")}', file=sys.stderr)
    return mapping, set(missing)


def group_assign(groups, examples, key, nfolds, seed):
    # Balances pair counts and labels, while keeping cluster groups intact.
    rng = random.Random(seed)
    items=[]
    for group_id, indices in groups.items():
        c=Counter(examples[i].label for i in indices)
        items.append((group_id, indices, c))
    rng.shuffle(items)
    items.sort(key=lambda x: len(x[1]), reverse=True)
    counts=[[0,0] for _ in range(nfolds)]
    assignment={}
    for group_id, indices, c in items:
        j=min(range(nfolds),key=lambda k:(sum(counts[k])+len(indices), abs(counts[k][0]+c[0]-counts[k][1]-c[1]), k))
        assignment[group_id]=j
        counts[j][0]+=c[0];counts[j][1]+=c[1]
    return assignment


def verify(train,test,scenario,pep_map,prot_map):
    a={(x.peptide_seq,x.protein_seq) for x in train}
    b={(x.peptide_seq,x.protein_seq) for x in test}
    assert not(a&b),'Pair identity overlap'
    if scenario in ('novel_peptide','double_cold'):
        assert not ({pep_map[x.peptide_seq] for x in train} & {pep_map[x.peptide_seq] for x in test}), 'Peptide cluster leakage'
    if scenario in ('novel_protein','double_cold'):
        assert not ({prot_map[x.protein_seq] for x in train} & {prot_map[x.protein_seq] for x in test}), 'Protein cluster leakage'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-fasta',required=True,type=Path)
    ap.add_argument('--trainer-dir',type=Path,default=Path('.'))
    ap.add_argument('--outdir',type=Path,required=True)
    ap.add_argument('--thresholds',nargs='+',type=float,default=[.6,.7,.8,.9])
    ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--seed',type=int,default=1234)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--exclude-unclustered',action='store_true',
                    help='Explicitly exclude any pairs containing sequences CD-HIT failed to cluster; report exclusions per identity threshold.')
    args=ap.parse_args()
    if not shutil.which('cd-hit'): raise SystemExit('ERROR: cd-hit is unavailable. Install it first: conda install -c bioconda cd-hit')
    if args.folds<2: raise SystemExit('--folds must be >=2')
    sys.path.insert(0,str(args.trainer_dir.resolve()))
    from pep_resmlp_train import parse_pair_fasta
    examples=parse_pair_fasta(args.train_fasta)
    if any(not x.peptide_seq or not x.protein_seq for x in examples):
        raise SystemExit('ERROR: records lack peptide/protein sequences. Expected paired FASTA lines: Peptide: ... and Protein: ...')
    if any(x.label not in (0,1) for x in examples): raise SystemExit('Invalid class labels')
    pair_labels=defaultdict(set)
    for ex in examples: pair_labels[(ex.peptide_seq,ex.protein_seq)].add(ex.label)
    contradictory=[p for p,v in pair_labels.items() if len(v)>1]
    if contradictory: raise SystemExit(f'ERROR: {len(contradictory)} identical pairs have contradictory labels; resolve before cold-start.')
    args.outdir.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    for cutoff in args.thresholds:
        root=args.outdir/f'identity_{cutoff:.1f}'
        root.mkdir(parents=True,exist_ok=True)
        pep_map, missing_pep = cluster([x.peptide_seq for x in examples], 'peptide', cutoff, root, args.threads)
        prot_map, missing_prot = cluster([x.protein_seq for x in examples], 'protein', cutoff, root, args.threads)
        missing_pairs = [x for x in examples if x.peptide_seq in missing_pep or x.protein_seq in missing_prot]
        if missing_pairs and not args.exclude_unclustered:
            raise RuntimeError(
                f'{len(missing_pairs)} pairs contain sequences not clustered by CD-HIT at identity {cutoff:.1f}. '
                f'Inspect {root}/*.unclustered.tsv and *.cdhit.log. '
                'If scientifically acceptable, rerun with --exclude-unclustered to explicitly exclude these pairs.')
        kept = [x for x in examples if x.peptide_seq in pep_map and x.protein_seq in prot_map]
        if missing_pairs:
            with open(root / 'excluded_pairs.tsv', 'w') as f:
                f.write('header\tlabel\tpeptide_length\tprotein_length\treason\n')
                for x in missing_pairs:
                    reason = 'peptide' if x.peptide_seq in missing_pep else 'protein'
                    f.write(f"{x.raw_header}\t{x.label}\t{len(x.peptide_seq)}\t{len(x.protein_seq)}\t{reason}\n")
            print(f'NOTICE: identity {cutoff:.1f}: excluding {len(missing_pairs)} pairs; see {root / "excluded_pairs.tsv"}',file=sys.stderr)
        pep_groups=defaultdict(list);prot_groups=defaultdict(list)
        for i,x in enumerate(kept):
            pep_groups[pep_map[x.peptide_seq]].append(i)
            prot_groups[prot_map[x.protein_seq]].append(i)
        pep_assign=group_assign(pep_groups,kept,'peptide',args.folds,args.seed)
        prot_assign=group_assign(prot_groups,kept,'protein',args.folds,args.seed+1000)
        for scenario in ('novel_peptide','novel_protein','double_cold'):
            for fold in range(args.folds):
                train=[];test=[];dropped=0
                for ex in kept:
                    p=pep_assign[pep_map[ex.peptide_seq]]
                    q=prot_assign[prot_map[ex.protein_seq]]
                    if scenario=='novel_peptide':
                        (test if p==fold else train).append(ex)
                    elif scenario=='novel_protein':
                        (test if q==fold else train).append(ex)
                    else:
                        if p==fold and q==fold: test.append(ex)
                        elif p!=fold and q!=fold: train.append(ex)
                        else: dropped+=1
                directory=root/scenario/f'fold{fold+1}'
                directory.mkdir(parents=True,exist_ok=True)
                valid= bool(train and test and len({x.label for x in train})==2 and len({x.label for x in test})==2)
                if valid:
                    verify(train,test,scenario,pep_map,prot_map)
                    write_fasta(directory/'train.fasta',train)
                    write_fasta(directory/'test.fasta',test)
                else:
                    print(f'WARNING: insufficient data/labels: {cutoff} {scenario} fold{fold+1}; skipped',file=sys.stderr)
                row={'identity':cutoff,'scenario':scenario,'fold':fold+1,'train_n':len(train),
                     'test_n':len(test),'dropped_n':dropped,'unclustered_pairs_excluded':len(missing_pairs),'train_pos':sum(x.label for x in train),
                     'test_pos':sum(x.label for x in test),'valid':int(valid),
                     'train_fasta':str(directory/'train.fasta') if valid else '',
                     'test_fasta':str(directory/'test.fasta') if valid else ''}
                all_rows.append(row)
                print(json.dumps(row))
    with open(args.outdir/'split_manifest.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    (args.outdir/'split_config.json').write_text(json.dumps({'input':str(args.train_fasta),'seed':args.seed,'folds':args.folds,'thresholds':args.thresholds,'examples':len(examples),'exclude_unclustered':args.exclude_unclustered,'double_cold_protocol':'independent peptide/protein cluster assignments, mixed-boundary pairs excluded per fold'},indent=2))
    print('Done:',args.outdir/'split_manifest.csv')

if __name__=='__main__': main()
