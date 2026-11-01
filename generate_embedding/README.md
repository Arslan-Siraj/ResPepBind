# Extract Peptide-Protein Embeddings

This guide provides the commands to extract embeddings from the training FASTA file and five independent evaluation FASTA files using `pepri_hgt_training.py`.

## Prerequisites

Run the commands from the directory containing `pepri_hgt_training.py`. Ensure that your Python environment has the dependencies required by the script and that the input FASTA files exist.

All commands write to the same embedding directory:

```text
generate_embedding/embeddings
```

## 1. Training dataset

### Training

```bash
python pepri_hgt_training.py \
  --extract_embeddings \
  --extract_fasta ../data/training/Training.fasta \
  --extract_split_name Training \
  --embedding_dir generate_embedding/embeddings \
  --max_protein_len 5000 \
  --max_peptide_len 50 \
  --prott5_chunk_size 1022 \
  --prott5_chunk_overlap 128
```

## 2. Independent evaluation datasets

### Test251

```bash
python pepri_hgt_training.py \
  --extract_embeddings \
  --extract_fasta ../data/independent/Test251.fasta \
  --extract_split_name Test251 \
  --embedding_dir generate_embedding/embeddings \
  --max_protein_len 5000 \
  --max_peptide_len 50 \
  --prott5_chunk_size 1022 \
  --prott5_chunk_overlap 128
```

### Test167

```bash
python pepri_hgt_training.py \
  --extract_embeddings \
  --extract_fasta ../data/independent/Test167.fasta \
  --extract_split_name Test167 \
  --embedding_dir generate_embedding/embeddings \
  --max_protein_len 5000 \
  --max_peptide_len 50 \
  --prott5_chunk_size 1022 \
  --prott5_chunk_overlap 128
```

### Leadspep

```bash
python pepri_hgt_training.py \
  --extract_embeddings \
  --extract_fasta ../data/independent/Leadspep.fasta \
  --extract_split_name Leadspep \
  --embedding_dir generate_embedding/embeddings \
  --max_protein_len 5000 \
  --max_peptide_len 50 \
  --prott5_chunk_size 1022 \
  --prott5_chunk_overlap 128
```

### Test1440

```bash
python pepri_hgt_training.py \
  --extract_embeddings \
  --extract_fasta ../data/independent/Test1440.fasta \
  --extract_split_name Test1440 \
  --embedding_dir generate_embedding/embeddings \
  --max_protein_len 5000 \
  --max_peptide_len 50 \
  --prott5_chunk_size 1022 \
  --prott5_chunk_overlap 128
```

### Test3664

```bash
python pepri_hgt_training.py \
  --extract_embeddings \
  --extract_fasta ../data/independent/Test3664.fasta \
  --extract_split_name Test3664 \
  --embedding_dir generate_embedding/embeddings \
  --max_protein_len 5000 \
  --max_peptide_len 50 \
  --prott5_chunk_size 1022 \
  --prott5_chunk_overlap 128
```

## 3. Parameter descriptions

| Argument | Purpose |
| --- | --- |
| `--extract_embeddings` | Activates embedding-extraction mode. |
| `--extract_fasta` | Path to the input FASTA file. |
| `--extract_split_name` | Identifier for the dataset being processed. |
| `--embedding_dir` | Directory where embeddings are saved. |
| `--max_protein_len 5000` | Sets the maximum protein sequence length to 5,000. |
| `--max_peptide_len 50` | Sets the maximum peptide sequence length to 50. |
| `--prott5_chunk_size 1022` | Sets the ProtT5 processing chunk size to 1,022 residues. |
| `--prott5_chunk_overlap 128` | Sets the overlap between consecutive ProtT5 chunks to 128 residues. |
