#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_independent_csv_for_testing.py

Validate a balanced peptide-protein independent CSV and convert it into the
FASTA-like format expected by the training/evaluation scripts:

>positive_1
Peptide: ...
Protein: ...

>negative_1
Peptide: ...
Protein: ...

The script also writes:
  - validation_summary.json
  - optional overlap report against training positives
  - optional separated positives.fasta and negatives.fasta

Example:

python prepare_independent_csv_for_testing.py \
  --input_csv /home/ubuntu/storage/peptide-protein-binding/For_publication/Data/generate_new_independent/pdb_pairs_new_ind_cdhit80/dataset_balanced.csv \
  --output_fasta /home/ubuntu/storage/peptide-protein-binding/For_publication/Data/generate_new_independent/pdb_pairs_new_ind_cdhit80/new_ind_cdhit80_balanced.fasta \
  --summary_json /home/ubuntu/storage/peptide-protein-binding/For_publication/Data/generate_new_independent/pdb_pairs_new_ind_cdhit80/validation_summary.json \
  --train_positives_csv /home/ubuntu/storage/peptide-protein-binding/For_publication/Data/generate_new_independent/pdb_pairs_until_2022_09_30/positives.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


REQUIRED_COLUMNS = [
    "peptide_sequence",
    "protein_sequence",
    "label",
]

OPTIONAL_LENGTH_COLUMNS = [
    "peptide_length",
    "protein_length",
]

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def clean_sequence(seq: object) -> str:
    return re.sub(r"[^A-Za-z]", "", str(seq).strip().upper())


def require_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def make_pair_set(df: pd.DataFrame) -> Set[Tuple[str, str]]:
    return set(zip(df["peptide_sequence"].astype(str), df["protein_sequence"].astype(str)))


def write_pair_fasta(df: pd.DataFrame, output_fasta: Path) -> None:
    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    pos_i = 0
    neg_i = 0

    with output_fasta.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            label = int(row["label"])
            peptide = clean_sequence(row["peptide_sequence"])
            protein = clean_sequence(row["protein_sequence"])

            if label == 1:
                pos_i += 1
                header = f"positive_{pos_i}"
            elif label == 0:
                neg_i += 1
                header = f"negative_{neg_i}"
            else:
                raise ValueError(f"Label must be 0 or 1, found: {label}")

            f.write(f">{header}\n")
            f.write(f"Peptide: {peptide}\n")
            f.write(f"Protein: {protein}\n\n")


def write_split_fastas(df: pd.DataFrame, outdir: Path) -> Dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)

    pos = df[df["label"] == 1].copy()
    neg = df[df["label"] == 0].copy()

    pos_fasta = outdir / "positives_only.fasta"
    neg_fasta = outdir / "negatives_only.fasta"

    write_pair_fasta(pos, pos_fasta)
    write_pair_fasta(neg, neg_fasta)

    return {
        "positives_only_fasta": str(pos_fasta),
        "negatives_only_fasta": str(neg_fasta),
    }


def validate_dataset(df: pd.DataFrame) -> Dict[str, object]:
    require_columns(df, Path("<input_csv>"))

    # Clean sequences for validation and downstream writing.
    df["peptide_sequence"] = df["peptide_sequence"].map(clean_sequence)
    df["protein_sequence"] = df["protein_sequence"].map(clean_sequence)

    # Basic counts.
    n_total = int(len(df))
    label_counts = df["label"].value_counts(dropna=False).to_dict()
    label_counts = {str(k): int(v) for k, v in label_counts.items()}

    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())

    # Missing and empty sequences.
    missing_values = {k: int(v) for k, v in df.isna().sum().to_dict().items()}
    empty_peptides = int((df["peptide_sequence"].str.len() == 0).sum())
    empty_proteins = int((df["protein_sequence"].str.len() == 0).sum())

    # Duplicate pair checks.
    pair_cols = ["peptide_sequence", "protein_sequence"]
    duplicated_pair_rows = int(df.duplicated(subset=pair_cols, keep=False).sum())
    unique_pairs = int(df[pair_cols].drop_duplicates().shape[0])

    pos_pairs = make_pair_set(df[df["label"] == 1])
    neg_pairs = make_pair_set(df[df["label"] == 0])
    pos_neg_overlap = int(len(pos_pairs & neg_pairs))

    # Length checks.
    df["peptide_len_calc"] = df["peptide_sequence"].str.len()
    df["protein_len_calc"] = df["protein_sequence"].str.len()

    peptide_length_summary = {
        "min": int(df["peptide_len_calc"].min()) if n_total else None,
        "max": int(df["peptide_len_calc"].max()) if n_total else None,
        "mean": float(df["peptide_len_calc"].mean()) if n_total else None,
        "median": float(df["peptide_len_calc"].median()) if n_total else None,
    }

    protein_length_summary = {
        "min": int(df["protein_len_calc"].min()) if n_total else None,
        "max": int(df["protein_len_calc"].max()) if n_total else None,
        "mean": float(df["protein_len_calc"].mean()) if n_total else None,
        "median": float(df["protein_len_calc"].median()) if n_total else None,
    }

    peptide_length_mismatches = None
    protein_length_mismatches = None

    if "peptide_length" in df.columns:
        peptide_length_mismatches = int((df["peptide_length"].astype(int) != df["peptide_len_calc"]).sum())

    if "protein_length" in df.columns:
        protein_length_mismatches = int((df["protein_length"].astype(int) != df["protein_len_calc"]).sum())

    # Character checks.
    unusual_peptide_rows = int(
        (~df["peptide_sequence"].map(lambda x: set(x).issubset(VALID_AA))).sum()
    )
    unusual_protein_rows = int(
        (~df["protein_sequence"].map(lambda x: set(x).issubset(VALID_AA))).sum()
    )

    # Unique sequence counts.
    n_unique_peptides = int(df["peptide_sequence"].nunique())
    n_unique_proteins = int(df["protein_sequence"].nunique())

    checks_passed = (
        n_total > 0
        and n_pos > 0
        and n_neg > 0
        and empty_peptides == 0
        and empty_proteins == 0
        and duplicated_pair_rows == 0
        and pos_neg_overlap == 0
        and unusual_peptide_rows == 0
        and unusual_protein_rows == 0
        and (peptide_length_mismatches in (None, 0))
        and (protein_length_mismatches in (None, 0))
    )

    return {
        "n_total": n_total,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "label_counts": label_counts,
        "class_balance_positive_fraction": float(n_pos / n_total) if n_total else None,
        "n_unique_peptides": n_unique_peptides,
        "n_unique_proteins": n_unique_proteins,
        "missing_values": missing_values,
        "empty_peptide_sequences": empty_peptides,
        "empty_protein_sequences": empty_proteins,
        "duplicated_pair_rows": duplicated_pair_rows,
        "unique_peptide_protein_pairs": unique_pairs,
        "positive_negative_pair_overlap": pos_neg_overlap,
        "peptide_length_summary": peptide_length_summary,
        "protein_length_summary": protein_length_summary,
        "peptide_length_mismatches": peptide_length_mismatches,
        "protein_length_mismatches": protein_length_mismatches,
        "unusual_peptide_character_rows": unusual_peptide_rows,
        "unusual_protein_character_rows": unusual_protein_rows,
        "checks_passed": checks_passed,
    }


def overlap_with_training(df: pd.DataFrame, train_positives_csv: Path) -> Dict[str, object]:
    train = pd.read_csv(train_positives_csv)

    for col in REQUIRED_COLUMNS[:2]:
        if col not in train.columns:
            raise ValueError(f"{train_positives_csv} is missing required column: {col}")

    train["peptide_sequence"] = train["peptide_sequence"].map(clean_sequence)
    train["protein_sequence"] = train["protein_sequence"].map(clean_sequence)

    if "label" in train.columns:
        train_pos = train[train["label"] == 1].copy()
    else:
        train_pos = train.copy()

    new_pos = df[df["label"] == 1].copy()
    new_all = df.copy()

    train_pos_pairs = make_pair_set(train_pos)
    new_pos_pairs = make_pair_set(new_pos)
    new_all_pairs = make_pair_set(new_all)

    exact_pos_overlap = train_pos_pairs & new_pos_pairs
    exact_all_overlap = train_pos_pairs & new_all_pairs

    return {
        "train_positives_csv": str(train_positives_csv),
        "training_positive_pairs": int(len(train_pos_pairs)),
        "new_positive_pairs": int(len(new_pos_pairs)),
        "new_all_pairs": int(len(new_all_pairs)),
        "exact_positive_pair_overlap_with_training": int(len(exact_pos_overlap)),
        "exact_any_pair_overlap_with_training_positives": int(len(exact_all_overlap)),
        "example_positive_overlaps": [
            {"peptide_sequence": p, "protein_sequence": q}
            for p, q in list(exact_pos_overlap)[:10]
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate balanced independent CSV and convert it to model-ready FASTA."
    )

    ap.add_argument("--input_csv", required=True, help="Balanced independent CSV.")
    ap.add_argument("--output_fasta", required=True, help="Output FASTA file for model testing.")
    ap.add_argument("--summary_json", default=None, help="Optional output validation summary JSON.")
    ap.add_argument("--train_positives_csv", default=None, help="Optional training positives CSV for exact overlap check.")
    ap.add_argument("--write_split_fastas", action="store_true", help="Also write positives_only.fasta and negatives_only.fasta.")
    ap.add_argument("--fail_on_warning", action="store_true", help="Exit with error if validation checks do not pass.")

    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    output_fasta = Path(args.output_fasta)
    summary_json = Path(args.summary_json) if args.summary_json else output_fasta.with_suffix(".validation_summary.json")

    df = pd.read_csv(input_csv)
    require_columns(df, input_csv)

    # Clean once and reuse the cleaned frame.
    df["peptide_sequence"] = df["peptide_sequence"].map(clean_sequence)
    df["protein_sequence"] = df["protein_sequence"].map(clean_sequence)

    summary = validate_dataset(df)
    summary["input_csv"] = str(input_csv)
    summary["output_fasta"] = str(output_fasta)

    if args.train_positives_csv:
        summary["training_overlap"] = overlap_with_training(df, Path(args.train_positives_csv))

    write_pair_fasta(df, output_fasta)

    outputs = {"combined_fasta": str(output_fasta)}
    if args.write_split_fastas:
        outputs.update(write_split_fastas(df, output_fasta.parent))
    summary["outputs"] = outputs

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))

    if args.fail_on_warning and not summary["checks_passed"]:
        raise SystemExit("Validation checks did not pass. See summary JSON.")


if __name__ == "__main__":
    main()
