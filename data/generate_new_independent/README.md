# pdb pair extract until Sept2022, as specified as training in published study

```bash
python extract_pdb_window_pairs.py \
  --cutoff-datetime 2022-09-30 \
  --exclude-pdb-id-file LEADS_PEP_PDB_id.csv \
  --outdir pdb_pairs_until_2022_09_30 \
  --negative-ratio 1.0 \
  --seed 42 \
  --max-workers 16
```

# pdb pair extract for new independent, as specified as training in published study

```bash
python extract_pdb_window_pairs.py   --start-datetime 2024-08-01T00:00:00Z   --cutoff-datetime 2025-12-31T23:59:59Z   --exclude-pdb-id-file "/home/ubuntu/peptide-protein-binding/LEADS_PEP_PDB_id.csv"  --outdir new_ind_pdb_pairs_24_25 --negative-ratio 1.0   --seed 42   --max-workers 16
```

# Run CDHIT stuff

```bash
python cdhit2d_filter_independent_and_make_negatives.py \
  --train-positives pdb_pairs_until_2022_09_30/positives.csv \
  --independent-positives new_ind_pdb_pairs_24_25/positives.csv \
  --outdir pdb_pairs_new_ind_cdhit80 \
  --identity 0.8 \
  --threads 16 \
  --seed 42 \
  --negative-ratio 1.0
```

# Validate and Report

```bash
python prepare_independent_csv_for_testing.py \
  --input_csv /home/ubuntu/storage/peptide-protein-binding/For_publication/Data/generate_new_independent/pdb_pairs_new_ind_cdhit80/dataset_balanced.csv \
  --output_fasta /home/ubuntu/storage/peptide-protein-binding/For_publication/Data/generate_new_independent/pdb_pairs_new_ind_cdhit80/new_ind_cdhit80_balanced.fasta \
  --summary_json /home/ubuntu/storage/peptide-protein-binding/For_publication/Data/generate_new_independent/pdb_pairs_new_ind_cdhit80/validation_summary.json \
  --train_positives_csv /home/ubuntu/storage/peptide-protein-binding/For_publication/Data/generate_new_independent/pdb_pairs_until_2022_09_30/positives.csv \
  --write_split_fastas
```

checkout: new_ind_cdhit80_balanced.fasta
validation_summary.json
positives_only.fasta
negatives_only.fasta