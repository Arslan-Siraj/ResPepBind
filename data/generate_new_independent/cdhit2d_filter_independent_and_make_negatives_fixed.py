#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cdhit2d_filter_independent_and_make_negatives.py

Filter an independent peptide-protein positive set against a training positive
set using CD-HIT-2D at a specified identity threshold, then regenerate random
presumed negatives from the filtered independent positives.

Install CD-HIT:
    conda install -c bioconda cd-hit -y

Example:
    python cdhit2d_filter_independent_and_make_negatives.py \
      --train-positives pdb_pairs_until_2022_09_30/positives.csv \
      --independent-positives pdb_pairs_Jan23_Jul2024_T1440/positives.csv \
      --outdir pdb_pairs_Jan23_Jul2024_T1440_cdhit80 \
      --identity 0.8 \
      --threads 16 \
      --seed 42 \
      --negative-ratio 1.0

Notes:
    Peptides and proteins use separate CD-HIT minimum lengths:
      --peptide-min-length 2
      --protein-min-length 10

    This preserves short peptides while avoiding the CD-HIT error:
      Fatal Error: Too short -l, redefine it
    during protein cd-hit-2d filtering with word size 5.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd


PAIR_COLUMNS = [
    "pair_id",
    "pdb_id",
    "peptide_chain",
    "protein_chain",
    "peptide_sequence",
    "protein_sequence",
    "peptide_length",
    "protein_length",
    "label",
    "min_contact_distance_A",
    "n_contacts",
]


def require_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in PAIR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def clean_sequence(seq: str) -> str:
    return str(seq).strip().upper()


def unique_sequences(df: pd.DataFrame, column: str) -> List[str]:
    seqs = [clean_sequence(x) for x in df[column].dropna().tolist()]
    seqs = [x for x in seqs if x]
    return sorted(set(seqs))


def write_fasta(seqs: List[str], prefix: str, fasta_path: Path) -> Dict[str, str]:
    id_to_seq = {}
    with fasta_path.open("w", encoding="utf-8") as fout:
        for i, seq in enumerate(seqs, start=1):
            seq_id = f"{prefix}_{i:08d}"
            id_to_seq[seq_id] = seq
            fout.write(f">{seq_id}\n")
            fout.write(seq + "\n")
    return id_to_seq


def read_fasta_ids(fasta_path: Path) -> Set[str]:
    ids = set()
    if not fasta_path.exists():
        raise FileNotFoundError(f"Expected FASTA output not found: {fasta_path}")
    with fasta_path.open("r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            line = line.strip()
            if line.startswith(">"):
                ids.add(line[1:].split()[0])
    return ids


def run_cdhit_2d(
    train_fasta: Path,
    independent_fasta: Path,
    output_fasta: Path,
    identity: float,
    word_size: int,
    threads: int,
    memory_mb: int,
    min_length: int,
) -> None:
    executable = shutil.which("cd-hit-2d")
    if executable is None:
        raise RuntimeError(
            "cd-hit-2d was not found in PATH. Install it with:\n"
            "  conda install -c bioconda cd-hit -y\n"
        )

    cmd = [
        executable,
        "-i", str(train_fasta),
        "-i2", str(independent_fasta),
        "-o", str(output_fasta),
        "-c", str(identity),
        "-n", str(word_size),
        "-d", "0",
        "-T", str(threads),
        "-M", str(memory_mb),
        "-l", str(min_length),
    ]

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def make_random_negatives(
    positives: pd.DataFrame,
    seed: int,
    ratio: float = 1.0,
) -> pd.DataFrame:
    rng = random.Random(seed)

    if positives.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    pep_records = (
        positives[["peptide_sequence", "peptide_length"]]
        .drop_duplicates()
        .to_dict("records")
    )

    prot_records = (
        positives[["protein_sequence", "protein_length"]]
        .drop_duplicates()
        .to_dict("records")
    )

    pos_pairs = set(zip(positives.peptide_sequence, positives.protein_sequence))
    target_n = int(round(len(positives) * ratio))

    negatives = []
    seen = set()
    attempts = 0
    max_attempts = target_n * 1000 + 10000

    while len(negatives) < target_n and attempts < max_attempts:
        attempts += 1
        pep = rng.choice(pep_records)
        prot = rng.choice(prot_records)
        key = (pep["peptide_sequence"], prot["protein_sequence"])

        if key in pos_pairs or key in seen:
            continue

        seen.add(key)

        negatives.append(
            {
                "pair_id": f"neg_{len(negatives) + 1:08d}",
                "pdb_id": "NA",
                "peptide_chain": "NA",
                "protein_chain": "NA",
                "peptide_sequence": pep["peptide_sequence"],
                "protein_sequence": prot["protein_sequence"],
                "peptide_length": pep["peptide_length"],
                "protein_length": prot["protein_length"],
                "label": 0,
                "min_contact_distance_A": np.nan,
                "n_contacts": 0,
            }
        )

    if len(negatives) < target_n:
        print(f"WARNING: only generated {len(negatives)} negatives; target was {target_n}", flush=True)

    return pd.DataFrame(negatives, columns=PAIR_COLUMNS)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--train-positives", required=True, help="Training positives.csv.")
    ap.add_argument("--independent-positives", required=True, help="Independent candidate positives.csv.")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--identity", type=float, default=0.8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--memory-mb", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negative-ratio", type=float, default=1.0)

    ap.add_argument(
        "--peptide-word-size",
        type=int,
        default=2,
        help="CD-HIT word size for peptide filtering. For very short peptides, 2 is safer. If CD-HIT rejects this, try 5.",
    )

    ap.add_argument(
        "--protein-word-size",
        type=int,
        default=5,
        help="CD-HIT word size for protein filtering at 80 percent identity. Usually 5.",
    )

    ap.add_argument(
        "--min-length",
        type=int,
        default=None,
        help=(
            "Deprecated compatibility option. If provided, it sets the peptide "
            "minimum length only. Prefer --peptide-min-length and "
            "--protein-min-length."
        ),
    )

    ap.add_argument(
        "--peptide-min-length",
        type=int,
        default=2,
        help=(
            "CD-HIT minimum sequence length for peptide filtering. "
            "Default 2 preserves very short peptides."
        ),
    )

    ap.add_argument(
        "--protein-min-length",
        type=int,
        default=10,
        help=(
            "CD-HIT minimum sequence length for protein filtering. "
            "Default 10 avoids CD-HIT errors with protein word size 5."
        ),
    )

    args = ap.parse_args()

    # Backward compatibility:
    #   --min-length, if used, applies only to peptides so short peptides are preserved.
    #   Proteins use a separate safer default because cd-hit-2d rejects -l 2 with -n 5.
    peptide_min_length = args.peptide_min_length if args.min_length is None else args.min_length
    protein_min_length = args.protein_min_length

    train_path = Path(args.train_positives)
    ind_path = Path(args.independent_positives)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    workdir = outdir / "cdhit2d_work"
    workdir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path)
    independent = pd.read_csv(ind_path)

    require_columns(train, train_path)
    require_columns(independent, ind_path)

    train = train[train["label"] == 1].copy()
    independent = independent[independent["label"] == 1].copy()

    train["peptide_sequence"] = train["peptide_sequence"].map(clean_sequence)
    train["protein_sequence"] = train["protein_sequence"].map(clean_sequence)
    independent["peptide_sequence"] = independent["peptide_sequence"].map(clean_sequence)
    independent["protein_sequence"] = independent["protein_sequence"].map(clean_sequence)

    independent = independent.drop_duplicates(
        subset=["peptide_sequence", "protein_sequence"]
    ).reset_index(drop=True)

    initial_independent_pairs = len(independent)

    train_peptides = unique_sequences(train, "peptide_sequence")
    train_proteins = unique_sequences(train, "protein_sequence")
    ind_peptides = unique_sequences(independent, "peptide_sequence")
    ind_proteins = unique_sequences(independent, "protein_sequence")

    train_pep_fasta = workdir / "train_peptides.fasta"
    train_prot_fasta = workdir / "train_proteins.fasta"
    ind_pep_fasta = workdir / "independent_peptides.fasta"
    ind_prot_fasta = workdir / "independent_proteins.fasta"

    write_fasta(train_peptides, "train_pep", train_pep_fasta)
    write_fasta(train_proteins, "train_prot", train_prot_fasta)
    ind_pep_id_to_seq = write_fasta(ind_peptides, "ind_pep", ind_pep_fasta)
    ind_prot_id_to_seq = write_fasta(ind_proteins, "ind_prot", ind_prot_fasta)

    pep_pass_fasta = workdir / "independent_peptides_cdhit2d_pass.fasta"
    prot_pass_fasta = workdir / "independent_proteins_cdhit2d_pass.fasta"

    run_cdhit_2d(
        train_fasta=train_pep_fasta,
        independent_fasta=ind_pep_fasta,
        output_fasta=pep_pass_fasta,
        identity=args.identity,
        word_size=args.peptide_word_size,
        threads=args.threads,
        memory_mb=args.memory_mb,
        min_length=peptide_min_length,
    )

    run_cdhit_2d(
        train_fasta=train_prot_fasta,
        independent_fasta=ind_prot_fasta,
        output_fasta=prot_pass_fasta,
        identity=args.identity,
        word_size=args.protein_word_size,
        threads=args.threads,
        memory_mb=args.memory_mb,
        min_length=protein_min_length,
    )

    passed_pep_ids = read_fasta_ids(pep_pass_fasta)
    passed_prot_ids = read_fasta_ids(prot_pass_fasta)

    passed_peptide_sequences = {
        ind_pep_id_to_seq[x] for x in passed_pep_ids if x in ind_pep_id_to_seq
    }
    passed_protein_sequences = {
        ind_prot_id_to_seq[x] for x in passed_prot_ids if x in ind_prot_id_to_seq
    }

    independent["peptide_pass_cdhit2d"] = independent["peptide_sequence"].isin(passed_peptide_sequences)
    independent["protein_pass_cdhit2d"] = independent["protein_sequence"].isin(passed_protein_sequences)

    filtered = independent[
        independent["peptide_pass_cdhit2d"] | independent["protein_pass_cdhit2d"]
    ].copy()
    
    removed = independent[
        ~(independent["peptide_pass_cdhit2d"] | independent["protein_pass_cdhit2d"])
    ].copy()

    def removal_reason(row) -> str:
        pep_fail = not bool(row["peptide_pass_cdhit2d"])
        prot_fail = not bool(row["protein_pass_cdhit2d"])
        if pep_fail and prot_fail:
            return "peptide_and_protein_similar_to_training"
        if pep_fail:
            return "peptide_similar_to_training"
        if prot_fail:
            return "protein_similar_to_training"
        return "kept"

    if not removed.empty:
        removed["removal_reason"] = removed.apply(removal_reason, axis=1)

    helper_cols = ["peptide_pass_cdhit2d", "protein_pass_cdhit2d"]
    filtered_out = filtered.drop(columns=[c for c in helper_cols if c in filtered.columns])
    removed_out = removed.drop(columns=[c for c in helper_cols if c in removed.columns])

    positives_out = outdir / "positives.csv"
    removed_out_path = outdir / "removed_by_cdhit2d.csv"
    negatives_out = outdir / "negatives_random.csv"

    filtered_out.to_csv(positives_out, index=False)
    removed_out.to_csv(removed_out_path, index=False)

    negatives = make_random_negatives(
        filtered_out,
        seed=args.seed,
        ratio=args.negative_ratio,
    )
    negatives.to_csv(negatives_out, index=False)

    dataset = pd.concat([filtered_out, negatives], ignore_index=True)
    if not dataset.empty:
        dataset = dataset.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    dataset.to_csv(outdir / "dataset_balanced.csv", index=False)

    summary = {
        "train_positives": str(train_path),
        "independent_positives": str(ind_path),
        "outdir": str(outdir),
        "identity": args.identity,
        "threads": args.threads,
        "memory_mb": args.memory_mb,
        "peptide_word_size": args.peptide_word_size,
        "protein_word_size": args.protein_word_size,
        "deprecated_min_length": args.min_length,
        "peptide_min_length": peptide_min_length,
        "protein_min_length": protein_min_length,
        "training_positive_pairs": int(len(train)),
        "training_unique_peptides": int(len(train_peptides)),
        "training_unique_proteins": int(len(train_proteins)),
        "initial_independent_positive_pairs_after_exact_pair_dedup": int(initial_independent_pairs),
        "initial_independent_unique_peptides": int(len(ind_peptides)),
        "initial_independent_unique_proteins": int(len(ind_proteins)),
        "passed_independent_unique_peptides": int(len(passed_peptide_sequences)),
        "passed_independent_unique_proteins": int(len(passed_protein_sequences)),
        "filtered_independent_positive_pairs": int(len(filtered_out)),
        "removed_independent_positive_pairs": int(len(removed_out)),
        "negative_pairs_regenerated": int(len(negatives)),
        "total_balanced_pairs_written": int(len(dataset)),
        "negative_ratio": args.negative_ratio,
        "seed": args.seed,
        "outputs": {
            "positives": str(positives_out),
            "removed_by_cdhit2d": str(removed_out_path),
            "negatives_random": str(negatives_out),
            "dataset_balanced": str(outdir / "dataset_balanced.csv"),
            "summary": str(outdir / "summary_cdhit2d.json"),
        },
    }

    (outdir / "summary_cdhit2d.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
