# ResPepBind cold-start benchmark

Reproduces the *types* of cold-start tests described in PepLM-GNN (PLOS Computational Biology 2026), **not necessarily the identical published fold assignments**. The paper clusters sequences separately at CD-HIT identities 0.6, 0.7, 0.8, 0.9, then evaluates new-peptide, new-protein and new-pair settings with five folds.

## Files

- `make_coldstart_splits.py`: generate 60 outer fold datasets (3 scenarios × 4 thresholds × 5 folds); assert no cluster overlaps; write a CSV manifest including class prevalence and discard counts.
- `run_coldstart.sh`: invoke split generation and train ResPepBind anew for each valid fold using the release trainer and existing pretrained embeddings.

## Install / run

```bash
conda activate ResPepBind_training  # or the conda environment that trained ResPepBind
conda install -c bioconda cd-hit
chmod +x run_coldstart.sh
CUDA_VISIBLE_DEVICES=1 ./run_coldstart.sh
```

Override the hard-coded default paths without editing script:

```bash
TRAIN_FASTA=../data/training/training.fasta EMBEDDING_DIR=/generate_embedding/embeddings \
TRAINER=pep_resmlp_train.py OUTPUT_ROOT=coldstart_output_10folds CUDA_VISIBLE_DEVICES=1 \
./run_coldstart.sh
```

## Critical methodology

The `double_cold` scenario assigns peptide clusters and protein clusters independently to folds. For each fold, test pairs require *both* clusters assigned to that fold; train pairs require neither. All mixed-boundary pairs are excluded for that fold. 

`novel_peptide` and `novel_protein` test pairs cover every pair once across their five folds. For double-cold, some pairs can be omitted from evaluation by design; inspect `test_n`, `test_pos` and `dropped_n` before drawing conclusions.

Only the outer training fold is used to select the checkpoint and classification threshold. This produces an *independent test metric for each outer fold*. `--protocol single_split` means **no ten-model ensemble** inside any outer fold. To match the published five-fold cold-start design, reporting five independently trained outer-fold test scores is appropriate; to make an ensemble for each outer test fold, implement inner training on outer-train only.

The result is a numerical comparison on equivalent *cold-start definitions*, not exact head-to-head significance testing unless PepLM-GNN is evaluated on the exact same outer folds. A separate true independence and prevalence audit should accompany publication.
