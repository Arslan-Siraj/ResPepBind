# ResPepBind

A 10-fold residual neural network ensemble for peptide-protein interaction prediction.

The official checkpoint ZIP is hosted separately as a [GitHub Release](https://github.com/Arslan-Siraj/ResPepBind/releases/tag/ResPepBind_model_v1.0.0_seed2026_10fold). **The first prediction automatically downloads all 10 fold checkpoints** into `~/.cache/respepbind/models/`. It also downloads the pretrained ProtT5 encoder from Hugging Face when missing residue embeddings are first generated. 

## Predict: input FASTA or CSV (CPU or GPU)

Install package with `pip install respepbind` from PyPI published distribution.

FASTA:

```text
>positive_1
Peptide: MLGP
Protein: MAEVPPGPSSLLPPPAPPAPAAVE

>negative_1
Peptide: ACDW
Protein: MLEVKQRL
```

CSV:

```csv
id,peptide,protein,true_label
pair1,MLGP,MAEVPPGPSSLLPPPAPPAPAAVE,1
pair2,ACDW,MLEVKQRL,
```

Run:

```bash
respepbind --fasta examples/pairs.fasta --device cpu --output-csv results/fasta.csv
respepbind --csv examples/pairs.csv --device cuda:0 --output-csv results/csv_gpu.csv
```

Or use wrappers:

```bash
./scripts/predict_cpu.sh --fasta examples/pairs.fasta --output-csv results/cpu.csv
CUDA_VISIBLE_DEVICES=0 ./scripts/predict_gpu.sh --csv examples/pairs.csv --output-csv results/gpu.csv
```



