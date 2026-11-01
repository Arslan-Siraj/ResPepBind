#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_pdb_window_pairs.py

Build peptide-protein positive and presumed-negative pairs from RCSB PDB
for a specified release-date or release-datetime window.

This script is designed for reproducing a PepLM-GNN-style data-preparation
protocol while keeping positives and negatives as separate files.

Main protocol
-------------
1. Search RCSB PDB entries by initial release date/time.
2. Optionally process only user-provided PDB IDs.
3. Optionally exclude benchmark/test PDB IDs before downloading/parsing.
4. Download mmCIF files only when not already available.
5. Parse amino-acid polymer chains from each structure.
6. Treat amino-acid chains of length 2-50 as peptide candidates.
7. Treat amino-acid chains of length 51-5000 as protein candidates.
8. Remove chains with more than 20% unknown/non-standard residues.
9. Label a peptide-protein chain pair as positive when at least one residue
   pair is within 5.0 Å.
10. Generate random presumed negatives by pairing peptides and proteins from
    the observed positive pools and rejecting sequence pairs already observed
    as positive.
11. Write positives and negatives separately. This script does NOT merge them.

Outputs
-------
- positives.csv
- negatives_random.csv
- contacts.csv
- pdb_ids.txt
- excluded_pdb_ids.txt, if an exclusion file is provided
- errors.csv
- summary.json

Install
-------
    pip install requests biopython pandas numpy tqdm

Example
-------
Extract 2025 structures while excluding benchmark IDs:

    python extract_pdb_window_pairs.py \
        --start-datetime 2025-01-01T00:00:00Z \
        --cutoff-datetime 2025-12-31T23:59:59Z \
        --exclude-pdb-id-file benchmark_ids.txt \
        --outdir pdb_pairs_2025 \
        --negative-ratio 1.0 \
        --seed 42 \
        --max-workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from tqdm import tqdm


SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"


STANDARD_AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    # Common modified residues mapped conservatively.
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
}


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


@dataclass
class ChainInfo:
    pdb_id: str
    chain_id: str
    sequence: str
    residues: list
    ca_coords: np.ndarray
    heavy_atom_coords: List[np.ndarray]
    unknown_ratio: float


def normalize_rcsb_datetime(value: str, end_of_day: bool = False) -> str:
    """
    Convert a date or date-time string into an RCSB-compatible UTC timestamp.

    Accepted examples:
    - 2025-01-01
    - 2025-01-01T00:00
    - 2025-01-01T00:00:00
    - 2025-01-01T00:00:00Z
    - 2025-01-01 00:00:00

    For date-only values, end_of_day controls whether the time becomes
    T00:00:00Z or T23:59:59Z.
    """
    value = str(value).strip()

    if not value:
        return ""

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value + ("T23:59:59Z" if end_of_day else "T00:00:00Z")

    value = value.replace(" ", "T")

    if value.endswith("Z"):
        return value

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", value):
        return value + ":00Z"

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", value):
        return value + "Z"

    return value


def normalize_pdb_id(value) -> Optional[str]:
    """
    Normalize PDB IDs from values such as:
    - 1abc
    - 1ABC
    - 1abc_A_B
    - 1abc.cif
    - 1abc.cif.gz
    - 1abc.pdb

    Returns a lowercase 4-character PDB ID or None.
    """
    if value is None:
        return None

    x = str(value).strip()

    if not x or x.lower() == "nan":
        return None

    x = x.split()[0]

    for suffix in [".cif.gz", ".CIF.GZ", ".cif", ".CIF", ".pdb", ".PDB"]:
        x = x.replace(suffix, "")

    x = x.split("_")[0]
    x = x.lower()

    if len(x) >= 4:
        x = x[:4]

    if len(x) != 4:
        return None

    return x


def load_pdb_ids_from_file(path: str) -> set:
    """
    Load user-specified PDB IDs.

    Supports:
    - .csv with a column named pdb_id, pdb, structure_id, id, or pdbid
    - .tsv with a column named pdb_id, pdb, structure_id, id, or pdbid
    - .txt with one PDB ID per line

    If a CSV/TSV does not contain a recognized column name, the first column
    is used.
    """
    if not path:
        return set()

    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"PDB ID file not found: {path}")

    pdb_ids = set()
    suffix = p.suffix.lower()

    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(p, sep=sep)

        if df.empty:
            return set()

        lower_cols = {c.lower(): c for c in df.columns}
        candidate_cols = ["pdb_id", "pdb", "structure_id", "id", "pdbid"]

        col = None
        for c in candidate_cols:
            if c in lower_cols:
                col = lower_cols[c]
                break

        if col is None:
            col = df.columns[0]

        for value in df[col].dropna().tolist():
            pdb_id = normalize_pdb_id(value)

            if pdb_id:
                pdb_ids.add(pdb_id)

    else:
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            pdb_id = normalize_pdb_id(line)

            if pdb_id:
                pdb_ids.add(pdb_id)

    return pdb_ids


def rcsb_search_pdb_ids(
    cutoff_datetime: str,
    start_datetime: str = "",
    rows: int = 10000,
    sleep: float = 0.15,
) -> List[str]:
    """
    Return PDB IDs from the RCSB Search API.

    If start_datetime is empty:
        initial_release_date <= cutoff_datetime

    If start_datetime is provided:
        start_datetime <= initial_release_date <= cutoff_datetime
    """
    all_ids: List[str] = []
    start = 0

    query_nodes = []

    if start_datetime:
        query_nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_accession_info.initial_release_date",
                    "operator": "greater_or_equal",
                    "value": start_datetime,
                },
            }
        )

    query_nodes.append(
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_accession_info.initial_release_date",
                "operator": "less_or_equal",
                "value": cutoff_datetime,
            },
        }
    )

    query_nodes.append(
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                "operator": "greater_or_equal",
                "value": 2,
            },
        }
    )

    while True:
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": query_nodes,
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "start": start,
                    "rows": rows,
                },
                "results_content_type": ["experimental"],
                "sort": [
                    {
                        "sort_by": "rcsb_accession_info.initial_release_date",
                        "direction": "asc",
                    }
                ],
            },
        }

        response = requests.post(SEARCH_URL, json=query, timeout=60)
        response.raise_for_status()

        data = response.json()

        batch = [
            normalize_pdb_id(x["identifier"])
            for x in data.get("result_set", [])
        ]
        batch = [x for x in batch if x]

        all_ids.extend(batch)

        total = data.get("total_count", len(all_ids))

        if len(all_ids) >= total or not batch:
            break

        start += rows
        time.sleep(sleep)

    return sorted(set(all_ids))


def find_existing_cif(pdb_id: str, cif_dir: Path) -> Optional[Path]:
    """
    Return existing CIF path if present.

    Checks:
    - cif/{pdb_id}.cif.gz
    - cif/{pdb_id}.cif
    """
    gz_path = cif_dir / f"{pdb_id}.cif.gz"
    cif_path = cif_dir / f"{pdb_id}.cif"

    if gz_path.exists() and gz_path.stat().st_size > 0:
        return gz_path

    if cif_path.exists() and cif_path.stat().st_size > 0:
        return cif_path

    return None


def download_cif(
    pdb_id: str,
    cif_dir: Path,
    overwrite: bool = False,
    retries: int = 3,
) -> Optional[Path]:
    """
    Download mmCIF file if needed.

    Reuses existing files if already present:
    - cif/{pdb_id}.cif.gz
    - cif/{pdb_id}.cif
    """
    cif_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        existing = find_existing_cif(pdb_id, cif_dir)

        if existing is not None:
            return existing

    gz_out = cif_dir / f"{pdb_id}.cif.gz"
    url = CIF_URL.format(pdb_id=pdb_id.upper())

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=60)

            if response.status_code == 200 and response.content:
                gz_out.write_bytes(response.content)
                return gz_out

        except Exception:
            pass

        time.sleep(1 + attempt)

    return None


def residue_to_one_letter(residue) -> str:
    name = residue.get_resname().strip().upper()
    return STANDARD_AA3_TO_1.get(name, "X")


def residue_coords(residue) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Prefer C-alpha for contact definition; keep heavy atoms as fallback.
    """
    ca = None

    if "CA" in residue:
        ca = np.array(residue["CA"].coord, dtype=np.float32)

    atoms = []

    for atom in residue.get_atoms():
        element = getattr(atom, "element", "").upper()

        if element == "H":
            continue

        atoms.append(np.array(atom.coord, dtype=np.float32))

    heavy = np.vstack(atoms) if atoms else np.empty((0, 3), dtype=np.float32)

    return ca, heavy


def parse_chains_from_cif(
    cif_path: Path,
    pdb_id: str,
    max_unknown_ratio: float,
) -> List[ChainInfo]:
    """
    Parse protein/peptide-like amino-acid chains from a CIF file.

    Supports:
    - compressed .cif.gz
    - uncompressed .cif
    """
    parser = MMCIFParser(QUIET=True)

    tmp_cif = None
    parse_path = cif_path

    try:
        if str(cif_path).endswith(".gz"):
            tmp_cif = cif_path.with_suffix("")

            with gzip.open(cif_path, "rb") as fin, open(tmp_cif, "wb") as fout:
                fout.write(fin.read())

            parse_path = tmp_cif

        structure = parser.get_structure(pdb_id, str(parse_path))
        model = next(structure.get_models())

        chains: List[ChainInfo] = []

        for chain in model:
            residues = []
            seq_chars = []
            ca_coords = []
            heavy_atom_coords = []

            for residue in chain:
                if not is_aa(residue, standard=False):
                    continue

                one = residue_to_one_letter(residue)
                ca, heavy = residue_coords(residue)

                if ca is None and heavy.shape[0] == 0:
                    continue

                residues.append(residue)
                seq_chars.append(one)
                ca_coords.append(
                    ca if ca is not None else np.full(3, np.nan, dtype=np.float32)
                )
                heavy_atom_coords.append(heavy)

            if not seq_chars:
                continue

            sequence = "".join(seq_chars)
            unknown_ratio = sequence.count("X") / len(sequence)

            if unknown_ratio > max_unknown_ratio:
                continue

            chains.append(
                ChainInfo(
                    pdb_id=pdb_id,
                    chain_id=chain.id,
                    sequence=sequence,
                    residues=residues,
                    ca_coords=np.vstack(ca_coords),
                    heavy_atom_coords=heavy_atom_coords,
                    unknown_ratio=unknown_ratio,
                )
            )

        return chains

    finally:
        if tmp_cif is not None:
            try:
                if tmp_cif.exists():
                    tmp_cif.unlink()
            except Exception:
                pass


def min_residue_distance(pep: ChainInfo, prot: ChainInfo, i: int, j: int) -> float:
    p_ca = pep.ca_coords[i]
    r_ca = prot.ca_coords[j]

    if not (np.isnan(p_ca).any() or np.isnan(r_ca).any()):
        return float(np.linalg.norm(p_ca - r_ca))

    p_atoms = pep.heavy_atom_coords[i]
    r_atoms = prot.heavy_atom_coords[j]

    if p_atoms.size == 0 or r_atoms.size == 0:
        return math.inf

    distances = np.linalg.norm(
        p_atoms[:, None, :] - r_atoms[None, :, :],
        axis=-1,
    )

    return float(distances.min())


def find_contacts(
    pep: ChainInfo,
    prot: ChainInfo,
    threshold: float,
) -> List[Tuple[int, int, float]]:
    contacts = []

    for i in range(len(pep.sequence)):
        for j in range(len(prot.sequence)):
            distance = min_residue_distance(pep, prot, i, j)

            if distance < threshold:
                contacts.append((i + 1, j + 1, distance))

    return contacts


def process_pdb(
    pdb_id: str,
    cif_dir: Path,
    pep_min: int,
    pep_max: int,
    prot_min: int,
    prot_max: int,
    contact_threshold: float,
    max_unknown_ratio: float,
) -> Tuple[List[dict], List[dict], Optional[str]]:
    cif = download_cif(pdb_id, cif_dir)

    if cif is None:
        return [], [], "download_failed"

    try:
        chains = parse_chains_from_cif(cif, pdb_id, max_unknown_ratio)

        peptides = [
            chain for chain in chains
            if pep_min <= len(chain.sequence) <= pep_max
        ]

        proteins = [
            chain for chain in chains
            if prot_min <= len(chain.sequence) <= prot_max
        ]

        positives = []
        contacts_rows = []

        for pep in peptides:
            for prot in proteins:
                if pep.chain_id == prot.chain_id:
                    continue

                contacts = find_contacts(
                    pep=pep,
                    prot=prot,
                    threshold=contact_threshold,
                )

                if not contacts:
                    continue

                min_distance = min(x[2] for x in contacts)
                pair_id = f"{pdb_id}_{pep.chain_id}_{prot.chain_id}"

                positives.append(
                    {
                        "pair_id": pair_id,
                        "pdb_id": pdb_id,
                        "peptide_chain": pep.chain_id,
                        "protein_chain": prot.chain_id,
                        "peptide_sequence": pep.sequence,
                        "protein_sequence": prot.sequence,
                        "peptide_length": len(pep.sequence),
                        "protein_length": len(prot.sequence),
                        "label": 1,
                        "min_contact_distance_A": round(min_distance, 3),
                        "n_contacts": len(contacts),
                    }
                )

                for peptide_residue_index, protein_residue_index, distance in contacts:
                    contacts_rows.append(
                        {
                            "pair_id": pair_id,
                            "pdb_id": pdb_id,
                            "peptide_chain": pep.chain_id,
                            "protein_chain": prot.chain_id,
                            "peptide_residue_index": peptide_residue_index,
                            "protein_residue_index": protein_residue_index,
                            "distance_A": round(distance, 3),
                        }
                    )

        return positives, contacts_rows, None

    except Exception as exc:
        return [], [], f"parse_or_contact_failed:{type(exc).__name__}:{exc}"


def make_random_negatives(
    positives: pd.DataFrame,
    seed: int,
    ratio: float = 1.0,
) -> pd.DataFrame:
    """
    Generate presumed negatives by random peptide-protein pairing.

    This follows the common protocol used in peptide-protein interaction
    benchmarks:
    - Build a peptide pool from positive samples.
    - Build a protein pool from positive samples.
    - Randomly pair one peptide with one protein.
    - Reject the pair if the peptide-protein sequence pair is known positive.
    - Continue until the requested negative-to-positive ratio is reached.

    These are presumed negatives, not experimentally verified non-binders.
    """
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

    pos_pairs = set(
        zip(
            positives.peptide_sequence,
            positives.protein_sequence,
        )
    )

    target_n = int(round(len(positives) * ratio))

    negatives = []
    seen = set()
    attempts = 0
    max_attempts = target_n * 1000 + 10000

    while len(negatives) < target_n and attempts < max_attempts:
        attempts += 1

        pep = rng.choice(pep_records)
        prot = rng.choice(prot_records)

        key = (
            pep["peptide_sequence"],
            prot["protein_sequence"],
        )

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
        print(
            f"WARNING: only generated {len(negatives)} negatives; target was {target_n}",
            file=sys.stderr,
        )

    return pd.DataFrame(negatives, columns=PAIR_COLUMNS)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--start-datetime",
        default="",
        help=(
            "Optional start release date/datetime. Examples: "
            "2025-01-01, 2025-01-01T00:00:00Z."
        ),
    )

    ap.add_argument(
        "--cutoff-datetime",
        default="2026-05-09T23:59:59Z",
        help=(
            "Include PDB entries released on or before this date/datetime. "
            "Examples: 2025-12-31, 2025-12-31T23:59:59Z."
        ),
    )

    ap.add_argument(
        "--outdir",
        default="peppi_pdb_window",
    )

    ap.add_argument("--pep-min", type=int, default=2)
    ap.add_argument("--pep-max", type=int, default=50)

    ap.add_argument("--prot-min", type=int, default=51)
    ap.add_argument("--prot-max", type=int, default=5000)

    ap.add_argument("--contact-threshold", type=float, default=5.0)
    ap.add_argument("--max-unknown-ratio", type=float, default=0.20)

    ap.add_argument("--negative-ratio", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max-workers", type=int, default=4)

    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug only: process first N PDB IDs.",
    )

    ap.add_argument(
        "--pdb-id-file",
        default="",
        help="Optional TXT/CSV/TSV file with PDB IDs; skips RCSB search.",
    )

    ap.add_argument(
        "--include-pdb-id-file",
        default="",
        help=(
            "Optional CSV/TSV/TXT file containing PDB IDs to include. "
            "If provided, only these PDB IDs are processed and RCSB date "
            "search is skipped."
        ),
    )

    ap.add_argument(
        "--exclude-pdb-id-file",
        default="",
        help=(
            "Optional CSV/TSV/TXT file containing PDB IDs to exclude before "
            "processing."
        ),
    )

    ap.add_argument(
        "--keep-sequence-duplicates",
        action="store_true",
        help=(
            "Keep duplicate peptide-protein sequence pairs in positives.csv. "
            "Default behavior removes duplicate positive sequence pairs."
        ),
    )

    args = ap.parse_args()

    start_datetime = normalize_rcsb_datetime(args.start_datetime, end_of_day=False)
    cutoff_datetime = normalize_rcsb_datetime(args.cutoff_datetime, end_of_day=True)

    if start_datetime and start_datetime > cutoff_datetime:
        raise ValueError(
            f"start-datetime must be <= cutoff-datetime, got "
            f"{start_datetime} > {cutoff_datetime}"
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cif_dir = outdir / "cif"

    if args.include_pdb_id_file:
        pdb_ids = sorted(load_pdb_ids_from_file(args.include_pdb_id_file))
        search_mode = "include_pdb_id_file"
        print(f"Using include-only PDB ID file: {args.include_pdb_id_file}")
        print(f"Included PDB IDs: {len(pdb_ids)}")

    elif args.pdb_id_file:
        pdb_ids = sorted(load_pdb_ids_from_file(args.pdb_id_file))
        search_mode = "pdb_id_file"
        print(f"Using PDB ID file: {args.pdb_id_file}")
        print(f"PDB IDs: {len(pdb_ids)}")

    else:
        search_mode = "rcsb_release_datetime"

        if start_datetime:
            print(
                f"Searching RCSB PDB entries from {start_datetime} "
                f"to {cutoff_datetime} ..."
            )
        else:
            print(f"Searching RCSB PDB entries up to {cutoff_datetime} ...")

        pdb_ids = rcsb_search_pdb_ids(
            cutoff_datetime=cutoff_datetime,
            start_datetime=start_datetime,
        )

    pdb_ids = sorted(set([x for x in pdb_ids if x]))

    exclude_pdb_ids = load_pdb_ids_from_file(args.exclude_pdb_id_file)
    removed_by_exclusion = 0

    if exclude_pdb_ids:
        before = len(pdb_ids)

        pdb_ids = [
            x for x in pdb_ids
            if x not in exclude_pdb_ids
        ]

        removed_by_exclusion = before - len(pdb_ids)

        (outdir / "excluded_pdb_ids.txt").write_text(
            "\n".join(sorted(exclude_pdb_ids)) + "\n",
            encoding="utf-8",
        )

        print(f"User-specified excluded PDB IDs: {len(exclude_pdb_ids)}")
        print(f"Removed candidate PDB entries before processing: {removed_by_exclusion}")

    if args.limit:
        pdb_ids = pdb_ids[: args.limit]

    (outdir / "pdb_ids.txt").write_text(
        "\n".join(pdb_ids) + "\n",
        encoding="utf-8",
    )

    print(f"Candidate entries after date/include/exclude/limit filtering: {len(pdb_ids)}")
    print(f"Peptide length range: {args.pep_min}-{args.pep_max}")
    print(f"Protein length range: {args.prot_min}-{args.prot_max}")
    print(f"Contact threshold: {args.contact_threshold} Angstrom")
    print(f"Max unknown ratio: {args.max_unknown_ratio}")

    positives_all: List[dict] = []
    contacts_all: List[dict] = []
    errors: List[dict] = []

    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_pdb,
                pdb_id,
                cif_dir,
                args.pep_min,
                args.pep_max,
                args.prot_min,
                args.prot_max,
                args.contact_threshold,
                args.max_unknown_ratio,
            ): pdb_id
            for pdb_id in pdb_ids
        }

        for future in tqdm(
            cf.as_completed(futures),
            total=len(futures),
            desc="Processing PDBs",
        ):
            pdb_id = futures[future]

            try:
                pos, con, err = future.result()

            except Exception as exc:
                pos, con, err = [], [], f"worker_failed:{type(exc).__name__}:{exc}"

            positives_all.extend(pos)
            contacts_all.extend(con)

            if err:
                errors.append(
                    {
                        "pdb_id": pdb_id,
                        "error": err,
                    }
                )

    if positives_all:
        positives = pd.DataFrame(positives_all)
    else:
        positives = pd.DataFrame(columns=PAIR_COLUMNS)

    raw_positive_pairs = len(positives)

    if not positives.empty:
        if not args.keep_sequence_duplicates:
            positives = positives.drop_duplicates(
                subset=[
                    "peptide_sequence",
                    "protein_sequence",
                ]
            ).reset_index(drop=True)

    positives.to_csv(
        outdir / "positives.csv",
        index=False,
    )

    contacts_df = pd.DataFrame(contacts_all)

    contacts_df.to_csv(
        outdir / "contacts.csv",
        index=False,
    )

    errors_df = pd.DataFrame(errors)

    errors_df.to_csv(
        outdir / "errors.csv",
        index=False,
    )

    negatives = make_random_negatives(
        positives,
        seed=args.seed,
        ratio=args.negative_ratio,
    )

    negatives.to_csv(
        outdir / "negatives_random.csv",
        index=False,
    )

    summary = {
        "search_mode": search_mode,
        "start_datetime": start_datetime,
        "cutoff_datetime": cutoff_datetime,
        "outdir": str(outdir),
        "pdb_id_file": args.pdb_id_file,
        "include_pdb_id_file": args.include_pdb_id_file,
        "exclude_pdb_id_file": args.exclude_pdb_id_file,
        "excluded_pdb_ids": int(len(exclude_pdb_ids)),
        "removed_candidate_entries_by_exclusion": int(removed_by_exclusion),
        "peptide_length_range": [
            args.pep_min,
            args.pep_max,
        ],
        "protein_length_range": [
            args.prot_min,
            args.prot_max,
        ],
        "contact_threshold_Angstrom": args.contact_threshold,
        "max_unknown_ratio": args.max_unknown_ratio,
        "candidate_pdb_entries_after_filtering": len(pdb_ids),
        "raw_positive_pairs_before_sequence_deduplication": int(raw_positive_pairs),
        "keep_sequence_duplicates": bool(args.keep_sequence_duplicates),
        "positive_pairs": int(len(positives)),
        "negative_pairs": int(len(negatives)),
        "contacts": int(len(contacts_df)),
        "errors": int(len(errors)),
        "negative_ratio": args.negative_ratio,
        "seed": args.seed,
        "merged_positive_negative_file_written": False,
    }

    (outdir / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
