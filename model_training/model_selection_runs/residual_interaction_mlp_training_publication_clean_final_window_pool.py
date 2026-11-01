#!/usr/bin/env python3
"""
ResidualInteractionMLP training script for peptide-protein binding prediction.

This publication-ready script keeps the full training, validation, k-fold,
ensemble, threshold-selection, and reporting workflow, but includes only the
final residual interaction MLP architecture. Other exploratory architectures
(e.g., cross-attention or gated CrossFormer variants) were removed to make the
release focused and easier to reproduce.

Main model:
  - residual_interaction_mlp: masked-mean pooled peptide/protein embeddings
    combined as [p, q, |p-q|, p*q] and classified with residual MLP blocks.

The script is intentionally defensive about FASTA and embedding formats because
published PepPI codebases often store examples and embeddings differently.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script requires scikit-learn. Install with: pip install scikit-learn"
    ) from exc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script requires matplotlib for plots. Install with: pip install matplotlib"
    ) from exc


# -----------------------------
# Reproducibility and utilities
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sanitize_id(x: str) -> str:
    x = str(x).strip()
    x = x.replace("/", "_")
    x = x.replace("\\", "_")
    return x


# -----------------------------
# FASTA parsing
# -----------------------------


@dataclass
class PairExample:
    peptide_id: str
    protein_id: str
    label: int
    peptide_seq: Optional[str] = None
    protein_seq: Optional[str] = None
    raw_header: str = ""


def clean_sequence_line(line: str) -> str:
    """Remove PepPI field prefixes and return an uppercase amino-acid string.

    The benchmark files may store each record as:
        >positive_1
        Peptide: XYVYNTRSGWRWYT
        Protein: MHHHHHH...

    This helper strips the leading Peptide:/Protein: label and removes any
    whitespace or non-letter characters.
    """
    x = line.strip()
    x = re.sub(r"^\s*(peptide|pep|protein|prot)\s*[:=]\s*", "", x, flags=re.IGNORECASE)
    x = re.sub(r"[^A-Za-z]", "", x)
    return x.upper()


def read_fasta_records(path: str | Path) -> List[Tuple[str, str]]:
    """Read FASTA records.

    For ordinary FASTA, wrapped sequence lines are concatenated. For the
    peptide--protein pair FASTA format used by several PepPI benchmarks, one
    record may contain exactly two sequence lines: peptide on the first line
    and protein on the second line. In that case, the two lines are preserved
    using a pipe separator so the downstream pair parser can recover both
    sequences.
    """
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    seq_parts: List[str] = []

    def finalize_record(h: str, parts: List[str]) -> Tuple[str, str]:
        cleaned = [x.strip() for x in parts if x.strip()]
        if len(cleaned) == 2:
            # Pair FASTA convention: peptide line + protein line.
            seq = cleaned[0] + "|" + cleaned[1]
        else:
            seq = "".join(cleaned)
        return h, seq

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append(finalize_record(header, seq_parts))
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if header is not None:
            records.append(finalize_record(header, seq_parts))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def _extract_label(header: str) -> Optional[int]:
    """Infer a binary label from a FASTA header.

    This accepts the publication-style identifiers used in your data, e.g.
    ``positive_1`` and ``negative_1``, in addition to explicit forms such as
    ``label=1`` or a final ``0/1`` token.
    """
    h = header.strip()
    h_lower = h.lower()

    # Publication-style labels such as positive_1, negative_1, pos-10, neg|10.
    if re.match(r"^(positive|pos)(?:$|[_\-\s|,;:])", h_lower):
        return 1
    if re.match(r"^(negative|neg)(?:$|[_\-\s|,;:])", h_lower):
        return 0

    patterns = [
        r"(?:^|[|,;\s])label\s*[=:]\s*([01])(?:$|[|,;\s])",
        r"(?:^|[|,;\s])y\s*[=:]\s*([01])(?:$|[|,;\s])",
        r"(?:^|[|,;\s])target\s*[=:]\s*([01])(?:$|[|,;\s])",
        r"(?:^|[|,;\s])class\s*[=:]\s*([01])(?:$|[|,;\s])",
    ]
    for pat in patterns:
        m = re.search(pat, h, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))

    # Common convention: last pipe/comma/tab/space token is 0/1.
    for sep in ["|", ",", "\t", " "]:
        toks = [t.strip() for t in h.split(sep) if t.strip()]
        if toks and toks[-1] in {"0", "1"}:
            return int(toks[-1])

    # More permissive fallback: labels appear as a token inside the header.
    if re.search(r"(?:^|[|,;\s_\-:])(positive|pos)(?:$|[|,;\s_\-:])", h_lower):
        return 1
    if re.search(r"(?:^|[|,;\s_\-:])(negative|neg)(?:$|[|,;\s_\-:])", h_lower):
        return 0

    return None


def _extract_field(header: str, names: Sequence[str]) -> Optional[str]:
    for name in names:
        pat = rf"(?:^|[|,;\s]){re.escape(name)}\s*[=:]\s*([^|,;\s]+)"
        m = re.search(pat, header, flags=re.IGNORECASE)
        if m:
            return sanitize_id(m.group(1))
    return None


def _split_sequence_pair(seq: str) -> Tuple[Optional[str], Optional[str]]:
    """Recover peptide and protein sequences from one parsed FASTA record.

    Handles both compact records such as ``PEPTIDE|PROTEIN`` and records
    preserved by ``read_fasta_records`` such as
    ``Peptide: PEPTIDE|Protein: PROTEIN``.
    """
    for sep in ["|", ";", ",", "/"]:
        if sep in seq:
            a, b = seq.split(sep, 1)
            a = clean_sequence_line(a)
            b = clean_sequence_line(b)
            if a and b:
                return a, b

    # Last-resort regex for a single string containing both labels.
    m = re.search(
        r"peptide\s*[:=]\s*([A-Za-z]+).*?protein\s*[:=]\s*([A-Za-z]+)",
        seq,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        a = clean_sequence_line(m.group(1))
        b = clean_sequence_line(m.group(2))
        if a and b:
            return a, b
    return None, None


def parse_pair_fasta(path: str | Path) -> List[PairExample]:
    """Parse flexible pair FASTA.

    Supported header examples:
      >pep=P123_pep prot=Q9XYZ label=1
      >peptide_id|protein_id|1
      >peptide_id protein_id 0

    If the sequence line contains a separator such as PEPSEQ|PROTSEQ, those sequences
    are stored and can be used as fallback embedding keys if ids are not found.
    """
    out: List[PairExample] = []
    for header, seq in read_fasta_records(path):
        label = _extract_label(header)
        if label is None:
            raise ValueError(
                f"Could not parse label from FASTA header: {header}\n"
                "Supported labels include positive_*, negative_*, label=0/1, "
                "y=0/1, target=0/1, class=0/1, or final header token 0/1."
            )

        peptide_id = _extract_field(header, ["pep", "peptide", "pep_id", "peptide_id", "p1", "seq1"])
        protein_id = _extract_field(header, ["prot", "protein", "prot_id", "protein_id", "p2", "seq2"])

        if peptide_id is None or protein_id is None:
            # Remove label-like tokens, then infer from remaining split tokens.
            cleaned = re.sub(r"(?:^|[|,;\s])(label|y|target|class)\s*[=:]\s*[01]", " ", header, flags=re.I)
            toks: List[str] = []
            for sep in ["|", ",", "\t", " "]:
                if sep in cleaned:
                    toks = [t.strip() for t in cleaned.split(sep) if t.strip() and t.strip() not in {"0", "1"}]
                    break
            if not toks:
                toks = [t.strip() for t in cleaned.split() if t.strip() and t.strip() not in {"0", "1"}]
            if len(toks) >= 2:
                peptide_id = peptide_id or sanitize_id(toks[0])
                protein_id = protein_id or sanitize_id(toks[1])
            elif len(toks) == 1:
                # Accept a single pair id and derive sequence keys later if possible.
                peptide_id = peptide_id or sanitize_id(toks[0] + "_pep")
                protein_id = protein_id or sanitize_id(toks[0] + "_prot")

        pep_seq, prot_seq = _split_sequence_pair(seq)

        if peptide_id is None or protein_id is None:
            if pep_seq and prot_seq:
                peptide_id = sanitize_id(pep_seq)
                protein_id = sanitize_id(prot_seq)
            else:
                raise ValueError(
                    f"Could not infer peptide/protein ids from header: {header}\n"
                    "Use a header such as: >pep=PEP_ID prot=PROT_ID label=1"
                )

        out.append(PairExample(peptide_id, protein_id, int(label), pep_seq, prot_seq, header))
    return out


# -----------------------------
# Embedding loading
# -----------------------------


class EmbeddingStore:
    """Lazy embedding resolver for directory or single-file stores.

    Supported forms:
      embedding_dir/
        ID.pt / ID.npy / ID.npz / ID.pkl
      embedding_dir/embeddings.pt       dict: id -> tensor/array/dict
      embedding_dir/embeddings.pkl      dict: id -> tensor/array/dict
      embedding_dir/embeddings.npz      arrays keyed by id

    Tensor shape can be [D] pooled or [L, D] residue-level. Pooled vectors are
    automatically converted to [1, D].
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Embedding path does not exist: {self.root}")
        self.cache: Dict[str, torch.Tensor] = {}
        self.global_maps: Dict[str, Dict[str, Any]] = {}
        self.file_index: Dict[str, Path] = {}
        self._build_index()

    def _build_index(self) -> None:
        if self.root.is_file():
            self._load_global_file(self.root)
            return
        exts = {".pt", ".pth", ".npy", ".npz", ".pkl", ".pickle"}
        for p in self.root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            stem = sanitize_id(p.stem)
            self.file_index[stem] = p
            self.file_index[p.name] = p
            if p.name.lower() in {"embeddings.pt", "embeddings.pth", "embeddings.pkl", "embeddings.pickle", "embeddings.npz"}:
                try:
                    self._load_global_file(p)
                except Exception as exc:
                    print(f"[warning] Could not preload {p}: {exc}", file=sys.stderr)

    def _load_global_file(self, p: Path) -> None:
        key = str(p.resolve())
        if key in self.global_maps:
            return
        suffix = p.suffix.lower()
        obj: Any
        if suffix in {".pt", ".pth"}:
            obj = torch.load(p, map_location="cpu")
        elif suffix in {".pkl", ".pickle"}:
            with open(p, "rb") as f:
                obj = pickle.load(f)
        elif suffix == ".npz":
            npz = np.load(p, allow_pickle=True)
            obj = {k: npz[k] for k in npz.files}
        else:
            return
        if isinstance(obj, dict):
            # Some stores wrap embeddings under a key.
            for wrapper_key in ["embeddings", "representations", "data"]:
                if wrapper_key in obj and isinstance(obj[wrapper_key], dict):
                    obj = obj[wrapper_key]
                    break
            self.global_maps[key] = obj

    @staticmethod
    def _unwrap(obj: Any) -> Any:
        if isinstance(obj, dict):
            for k in ["embedding", "emb", "representations", "mean_representations", "tensor", "x"]:
                if k in obj:
                    val = obj[k]
                    if isinstance(val, dict):
                        # ESM often stores {layer: tensor}; use highest numeric key.
                        keys = sorted(val.keys(), key=lambda z: int(z) if str(z).isdigit() else str(z))
                        return val[keys[-1]]
                    return val
            # Fall back to the first tensor-like value.
            for val in obj.values():
                if isinstance(val, (torch.Tensor, np.ndarray, list, tuple)):
                    return val
        return obj

    @staticmethod
    def _to_tensor(obj: Any) -> torch.Tensor:
        obj = EmbeddingStore._unwrap(obj)
        if isinstance(obj, torch.Tensor):
            t = obj.detach().cpu().float()
        elif isinstance(obj, np.ndarray):
            if obj.dtype == object and obj.shape == ():
                obj = obj.item()
                return EmbeddingStore._to_tensor(obj)
            t = torch.tensor(obj, dtype=torch.float32)
        elif isinstance(obj, (list, tuple)):
            t = torch.tensor(obj, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported embedding object type: {type(obj)}")
        if t.ndim == 1:
            t = t.unsqueeze(0)
        if t.ndim > 2:
            t = t.reshape(-1, t.shape[-1])
        if t.ndim != 2:
            raise ValueError(f"Embedding must be [D] or [L,D], got shape {tuple(t.shape)}")
        return t

    def _load_file(self, p: Path) -> torch.Tensor:
        suffix = p.suffix.lower()
        if suffix in {".pt", ".pth"}:
            obj = torch.load(p, map_location="cpu")
        elif suffix == ".npy":
            obj = np.load(p, allow_pickle=True)
        elif suffix == ".npz":
            npz = np.load(p, allow_pickle=True)
            if len(npz.files) == 1:
                obj = npz[npz.files[0]]
            else:
                # Prefer common keys; otherwise stack/first array.
                for k in ["embedding", "emb", "x", "arr_0"]:
                    if k in npz.files:
                        obj = npz[k]
                        break
                else:
                    obj = npz[npz.files[0]]
        elif suffix in {".pkl", ".pickle"}:
            with open(p, "rb") as f:
                obj = pickle.load(f)
        else:
            raise ValueError(f"Unsupported embedding file suffix: {p.suffix}")
        return self._to_tensor(obj)

    def get(self, identifier: str, fallback_seq: Optional[str] = None) -> torch.Tensor:
        base_identifier = sanitize_id(identifier)
        candidates = [base_identifier]
        if fallback_seq:
            clean_fallback = clean_sequence_line(fallback_seq)
            candidates.extend([sanitize_id(fallback_seq), sanitize_id(clean_fallback), clean_fallback])
            # Common hashed sequence filename/key conventions used when raw
            # protein sequences are too long for filenames. These are cheap to
            # test and make the loader compatible with many embedding pipelines.
            if clean_fallback:
                import hashlib
                md5 = hashlib.md5(clean_fallback.encode("utf-8")).hexdigest()
                sha1 = hashlib.sha1(clean_fallback.encode("utf-8")).hexdigest()
                sha256 = hashlib.sha256(clean_fallback.encode("utf-8")).hexdigest()
                candidates.extend([md5, sha1, sha256, f"seq_{md5}", f"seq_{sha1}", f"seq_{sha256}"])

        # Common alternatives for pair FASTA records with headers such as positive_1.
        # Different embedding pipelines may save the two sequences as positive_1_pep,
        # positive_1_peptide, positive_1_prot, or positive_1_protein.
        alt_ids = []
        if base_identifier.endswith("_pep"):
            root = base_identifier[:-4]
            alt_ids.extend([root + "_peptide", root + "|peptide", root + ":peptide", root + "_Peptide"])
        if base_identifier.endswith("_prot"):
            root = base_identifier[:-5]
            alt_ids.extend([root + "_protein", root + "|protein", root + ":protein", root + "_Protein"])
        candidates.extend(alt_ids)

        for cid in list(candidates):
            # Common ESM/ProtT5 filename forms.
            candidates.extend([cid.replace("|", "_"), cid.replace(":", "_"), cid.split()[0]])
        seen: set[str] = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]

        for cid in candidates:
            if cid in self.cache:
                return self.cache[cid]

        # Check global dicts first.
        for cmap in self.global_maps.values():
            for cid in candidates:
                if cid in cmap:
                    t = self._to_tensor(cmap[cid])
                    self.cache[cid] = t
                    return t

        # Check direct file index.  Avoid constructing/testing filesystem paths
        # from full protein sequences: a sequence can be hundreds or thousands of
        # characters long and will exceed the operating-system filename limit.
        suffixes = [".pt", ".pth", ".npy", ".npz", ".pkl", ".pickle"]

        def _safe_as_filename(x: str, suffix: str = "") -> bool:
            if not x:
                return False
            # Do not allow path separators or path traversal in generated names.
            if "/" in x or "\\" in x or x in {".", ".."}:
                return False
            # Most Linux filesystems allow 255 bytes per path component.
            # Keep margin for suffix and multibyte characters.
            try:
                return len((x + suffix).encode("utf-8")) <= 240
            except Exception:
                return False

        for cid in candidates:
            if cid in self.file_index:
                t = self._load_file(self.file_index[cid])
                self.cache[cid] = t
                return t
            for suf in suffixes:
                if not _safe_as_filename(cid, suf):
                    continue
                p = self.root / f"{cid}{suf}"
                if p.exists():
                    t = self._load_file(p)
                    self.cache[cid] = t
                    return t

        raise FileNotFoundError(
            "Could not find embedding for id candidates: " + ", ".join(candidates[:10]) +
            f" in {self.root}. Check FASTA ids and embedding filenames."
        )


# -----------------------------
# Dataset and collation
# -----------------------------


def window_mean_pool_sequence(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Pool a residue-level embedding sequence into non-overlapping windows.

    For a protein with length L and window_size W, this returns one vector per
    consecutive window: [1:W], [W+1:2W], ..., including the final shorter
    window. The model's existing masked_mean over these window vectors then
    computes q = mean(window-level masked-mean vectors). This avoids discarding
    residues beyond max_protein_len while keeping tensors compact for batching.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected [L,D] embedding tensor, got shape {tuple(x.shape)}")
    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if x.shape[0] == 0:
        raise ValueError("Cannot pool an empty embedding sequence")
    windows = []
    for start in range(0, x.shape[0], window_size):
        w = x[start:start + window_size]
        windows.append(w.mean(dim=0))
    return torch.stack(windows, dim=0)


class PepPIDataset(Dataset):
    def __init__(
        self,
        fasta_path: str | Path,
        embedding_dir: str | Path,
        max_peptide_len: int = 256,
        max_protein_len: int = 1024,
        protein_pooling_mode: str = "truncate",
    ):
        self.fasta_path = Path(fasta_path)
        self.examples = parse_pair_fasta(self.fasta_path)
        self.store = EmbeddingStore(embedding_dir)
        self.max_peptide_len = int(max_peptide_len)
        self.max_protein_len = int(max_protein_len)
        self.protein_pooling_mode = str(protein_pooling_mode).lower().strip()
        if self.protein_pooling_mode not in {"truncate", "window_mean"}:
            raise ValueError(
                "protein_pooling_mode must be 'truncate' or 'window_mean', "
                f"got {protein_pooling_mode!r}"
            )
        self.labels = np.array([ex.label for ex in self.examples], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.examples)

    def _prepare_protein_embedding(self, prot: torch.Tensor) -> torch.Tensor:
        if self.protein_pooling_mode == "truncate":
            return prot[: self.max_protein_len]
        if self.protein_pooling_mode == "window_mean":
            return window_mean_pool_sequence(prot, self.max_protein_len)
        raise ValueError(f"Unknown protein_pooling_mode: {self.protein_pooling_mode}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        pep = self.store.get(ex.peptide_id, ex.peptide_seq)[: self.max_peptide_len]
        prot = self.store.get(ex.protein_id, ex.protein_seq)
        prot = self._prepare_protein_embedding(prot)
        return {
            "pep": pep,
            "prot": prot,
            "label": torch.tensor(float(ex.label), dtype=torch.float32),
            "peptide_id": ex.peptide_id,
            "protein_id": ex.protein_id,
            "raw_header": ex.raw_header,
        }


def _pad_2d(items: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(x.shape[0] for x in items)
    dim = items[0].shape[1]
    out = torch.zeros(len(items), max_len, dim, dtype=torch.float32)
    mask = torch.ones(len(items), max_len, dtype=torch.bool)  # True means padded for PyTorch attention.
    for i, x in enumerate(items):
        l = x.shape[0]
        out[i, :l] = x
        mask[i, :l] = False
    return out, mask


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    pep, pep_mask = _pad_2d([b["pep"] for b in batch])
    prot, prot_mask = _pad_2d([b["prot"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return {
        "pep": pep,
        "prot": prot,
        "pep_mask": pep_mask,
        "prot_mask": prot_mask,
        "label": labels,
        "peptide_id": [b["peptide_id"] for b in batch],
        "protein_id": [b["protein_id"] for b in batch],
        "raw_header": [b["raw_header"] for b in batch],
    }


# -----------------------------
# Models
# -----------------------------



def masked_mean(x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    valid = (~pad_mask).float().unsqueeze(-1)
    denom = valid.sum(dim=1).clamp_min(1.0)
    return (x * valid).sum(dim=1) / denom


def parse_interaction_features(value: str) -> List[str]:
    """Parse selected interaction-vector components.

    Allowed features:
      p        peptide pooled vector
      q        protein pooled vector
      absdiff  absolute difference |p-q|
      product  element-wise product p*q
      sum      p+q
      diff     p-q

    The default final model is: p,q,absdiff,product.
    """
    aliases = {
        "pep": "p", "peptide": "p",
        "prot": "q", "protein": "q",
        "|p-q|": "absdiff", "abs": "absdiff", "abs_diff": "absdiff",
        "p*q": "product", "mul": "product", "hadamard": "product",
        "p+q": "sum", "plus": "sum",
        "p-q": "diff", "minus": "diff",
    }
    allowed = {"p", "q", "absdiff", "product", "sum", "diff"}
    feats: List[str] = []
    for raw in str(value or "").split(","):
        key = raw.strip().lower()
        if not key:
            continue
        key = aliases.get(key, key)
        if key not in allowed:
            raise ValueError(f"Unknown interaction feature {raw!r}. Allowed: p,q,absdiff,product,sum,diff")
        if key not in feats:
            feats.append(key)
    if not feats:
        raise ValueError("At least one interaction feature must be selected.")
    return feats


def build_interaction_vector(p: torch.Tensor, q: torch.Tensor, features: Sequence[str]) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for feature in features:
        if feature == "p":
            parts.append(p)
        elif feature == "q":
            parts.append(q)
        elif feature == "absdiff":
            parts.append(torch.abs(p - q))
        elif feature == "product":
            parts.append(p * q)
        elif feature == "sum":
            parts.append(p + q)
        elif feature == "diff":
            parts.append(p - q)
        else:
            raise ValueError(f"Unknown interaction feature: {feature}")
    return torch.cat(parts, dim=-1)



def build_activation(name: str) -> nn.Module:
    """Return an activation layer from a short command-line name."""
    name = str(name).lower().strip()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unknown activation: {name}")


def build_norm(norm_type: str, dim: int) -> nn.Module:
    """Return a normalization layer. Identity disables normalization."""
    norm_type = str(norm_type).lower().strip()
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "batchnorm":
        return nn.BatchNorm1d(dim)
    if norm_type in {"none", "identity", "no_norm"}:
        return nn.Identity()
    raise ValueError(f"Unknown normalization type: {norm_type}")


def parse_float_list(value: str, expected_len: int, fallback: float) -> List[float]:
    """Parse comma-separated per-block values.

    If value is empty, returns [fallback] * expected_len.
    If one value is supplied, it is repeated for all blocks.
    If expected_len values are supplied, they are used block-by-block.
    """
    value = str(value or "").strip()
    if not value:
        return [float(fallback)] * int(expected_len)
    vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(vals) == 1:
        return vals * int(expected_len)
    if len(vals) != int(expected_len):
        raise ValueError(
            f"Expected either 1 value or {expected_len} comma-separated values, got {len(vals)}: {value}"
        )
    return vals


class ResidualBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float,
        expand_ratio: float = 2.0,
        activation: str = "gelu",
        norm_type: str = "layernorm",
        use_residual: bool = True,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        hidden_dim = max(1, int(round(dim * float(expand_ratio))))
        self.use_residual = bool(use_residual)
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            build_norm(norm_type, dim),
            nn.Linear(dim, hidden_dim),
            build_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        if self.use_residual:
            return x + self.residual_scale * z
        return z


class InteractionClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_blocks: int,
        input_dropout: float,
        block_dropouts: Sequence[float],
        head_dropout: float,
        expand_ratio: float = 2.0,
        head_ratio: float = 0.5,
        activation: str = "gelu",
        norm_type: str = "layernorm",
        use_residual: bool = True,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            build_activation(activation),
            nn.Dropout(float(input_dropout)),
        ]

        block_dropouts = list(block_dropouts)
        if len(block_dropouts) != int(num_blocks):
            raise ValueError(f"block_dropouts length {len(block_dropouts)} != num_blocks {num_blocks}")

        for block_dropout in block_dropouts:
            layers.append(
                ResidualBlock(
                    dim=hidden_dim,
                    dropout=float(block_dropout),
                    expand_ratio=expand_ratio,
                    activation=activation,
                    norm_type=norm_type,
                    use_residual=use_residual,
                    residual_scale=residual_scale,
                )
            )

        head_dim = max(1, int(round(hidden_dim * float(head_ratio))))
        layers.extend([
            build_norm(norm_type, hidden_dim),
            nn.Linear(hidden_dim, head_dim),
            build_activation(activation),
            nn.Dropout(float(head_dropout)),
            nn.Linear(head_dim, 1),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ResidualInteractionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        mlp_blocks: int = 3,
        input_dropout: Optional[float] = None,
        block_dropout: Optional[float] = None,
        block_dropouts: str = "",
        head_dropout: Optional[float] = None,
        mlp_expand_ratio: float = 2.0,
        head_ratio: float = 0.5,
        activation: str = "gelu",
        norm_type: str = "layernorm",
        use_residual: bool = True,
        residual_scale: float = 1.0,
        interaction_features: str = "p,q,absdiff,product",
    ):
        super().__init__()
        selected_features = parse_interaction_features(interaction_features)
        input_dropout = float(dropout if input_dropout is None else input_dropout)
        block_dropout = float(dropout if block_dropout is None else block_dropout)
        head_dropout = float(dropout if head_dropout is None else head_dropout)
        per_block_dropouts = parse_float_list(block_dropouts, mlp_blocks, block_dropout)

        self.config = {
            "hidden_dim": int(hidden_dim),
            "mlp_blocks": int(mlp_blocks),
            "dropout": float(dropout),
            "input_dropout": float(input_dropout),
            "block_dropout": float(block_dropout),
            "block_dropouts": per_block_dropouts,
            "head_dropout": float(head_dropout),
            "mlp_expand_ratio": float(mlp_expand_ratio),
            "head_ratio": float(head_ratio),
            "activation": str(activation),
            "norm_type": str(norm_type),
            "use_residual": bool(use_residual),
            "residual_scale": float(residual_scale),
            "interaction_features": selected_features,
            "interaction_dim": int(input_dim * len(selected_features)),
        }
        self.interaction_features = selected_features

        self.classifier = InteractionClassifier(
            in_dim=input_dim * len(selected_features),
            hidden_dim=hidden_dim,
            num_blocks=mlp_blocks,
            input_dropout=input_dropout,
            block_dropouts=per_block_dropouts,
            head_dropout=head_dropout,
            expand_ratio=mlp_expand_ratio,
            head_ratio=head_ratio,
            activation=activation,
            norm_type=norm_type,
            use_residual=use_residual,
            residual_scale=residual_scale,
        )

    def forward(self, pep: torch.Tensor, prot: torch.Tensor, pep_mask: torch.Tensor, prot_mask: torch.Tensor) -> torch.Tensor:
        p = masked_mean(pep, pep_mask)
        q = masked_mean(prot, prot_mask)
        pair = build_interaction_vector(p, q, self.interaction_features)
        return self.classifier(pair)



# -----------------------------
# Training and evaluation
# -----------------------------


def infer_input_dim(dataset: Dataset) -> int:
    item = dataset[0]
    pep_dim = int(item["pep"].shape[-1])
    prot_dim = int(item["prot"].shape[-1])
    if pep_dim != prot_dim:
        raise ValueError(f"Peptide/protein embedding dimensions differ: {pep_dim} vs {prot_dim}")
    return pep_dim


def build_model(args: argparse.Namespace, input_dim: int) -> nn.Module:
    if args.variant != "residual_interaction_mlp":
        raise ValueError(
            "This publication script contains only the final residual_interaction_mlp model. "
            f"Received variant={args.variant!r}."
        )
    return ResidualInteractionMLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        mlp_blocks=args.mlp_blocks,
        input_dropout=args.input_dropout,
        block_dropout=args.block_dropout,
        block_dropouts=args.block_dropouts,
        head_dropout=args.head_dropout,
        mlp_expand_ratio=args.mlp_expand_ratio,
        head_ratio=args.head_ratio,
        activation=args.activation,
        norm_type=args.norm_type,
        use_residual=(not args.no_residual),
        residual_scale=args.residual_scale,
        interaction_features=args.interaction_features,
    )


def compute_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    """Compute the metrics reported by the publication workflow.

     is intentionally not reported in this release to keep the output focused
    on ACC, F1, AUC, and AUPR.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    pred = (p >= threshold).astype(int)
    metrics: Dict[str, float] = {}
    metrics["acc"] = float(accuracy_score(y, pred))
    metrics["f1"] = float(f1_score(y, pred, zero_division=0))
    try:
        metrics["auc"] = float(roc_auc_score(y, p))
    except Exception:
        metrics["auc"] = float("nan")
    try:
        metrics["aupr"] = float(average_precision_score(y, p))
    except Exception:
        metrics["aupr"] = float("nan")
    metrics["threshold"] = float(threshold)
    return metrics


def find_best_threshold(y_true: Sequence[int], y_prob: Sequence[float], objective: str = "f1") -> float:
    """Select a validation-derived classification threshold.

    Supported objectives:
      - f1: maximize F1
      - acc: maximize accuracy
      - youden: maximize sensitivity + specificity - 1

    AUC and AUPR are threshold-free ranking metrics. Therefore, there is no
    mathematically valid threshold that "maximizes AUPR". If objective is
    passed as ``aupr`` or ``oof_aupr``, this function falls back to F1 and
    reports a warning.
    """
    objective = str(objective).lower().strip()
    if objective in {"aupr", "oof_aupr", "ap", "average_precision"}:
        print(
            "[warning] AUPR is threshold-free; using F1 to choose the threshold instead.",
            file=sys.stderr,
        )
        objective = "f1"

    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    finite = np.isfinite(p)
    y, p = y[finite], p[finite]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5

    dense_grid = np.linspace(0.001, 0.999, 999)
    observed = np.unique(p)
    quantiles = np.unique(np.quantile(p, np.linspace(0.001, 0.999, 999)))
    thresholds = np.unique(np.concatenate([dense_grid, observed, quantiles, np.array([0.5])]))

    best_t, best_score = 0.5, -float("inf")
    best_acc_at_t = -1.0
    for t in thresholds:
        pred = (p >= t).astype(int)
        acc = float(accuracy_score(y, pred))

        if objective == "acc":
            score = acc
        elif objective in {"youden", "j", "balanced_acc"}:
            tp = float(((y == 1) & (pred == 1)).sum())
            tn = float(((y == 0) & (pred == 0)).sum())
            fp = float(((y == 0) & (pred == 1)).sum())
            fn = float(((y == 1) & (pred == 0)).sum())
            sens = tp / max(tp + fn, 1.0)
            spec = tn / max(tn + fp, 1.0)
            score = sens + spec - 1.0
        else:
            score = float(f1_score(y, pred, zero_division=0))

        is_better = score > best_score
        if np.isclose(score, best_score):
            if acc > best_acc_at_t:
                is_better = True
            elif np.isclose(acc, best_acc_at_t) and abs(float(t) - 0.5) < abs(best_t - 0.5):
                is_better = True

        if is_better:
            best_score = float(score)
            best_acc_at_t = float(acc)
            best_t = float(t)
    return best_t


def make_loader(dataset: Dataset, indices: Optional[Sequence[int]], batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    ds = dataset if indices is None else Subset(dataset, list(indices))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=collate_batch)


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: Optional[torch.optim.Optimizer], device: torch.device, criterion: nn.Module, grad_clip: float) -> Tuple[float, np.ndarray, np.ndarray, List[Tuple[str, str, str]]]:
    train = optimizer is not None
    model.train(train)
    losses: List[float] = []
    all_y: List[float] = []
    all_p: List[float] = []
    all_ids: List[Tuple[str, str, str]] = []
    for batch in loader:
        pep = batch["pep"].to(device, non_blocking=True)
        prot = batch["prot"].to(device, non_blocking=True)
        pep_mask = batch["pep_mask"].to(device, non_blocking=True)
        prot_mask = batch["prot_mask"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits = model(pep, prot, pep_mask, prot_mask)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        losses.append(float(loss.detach().cpu().item()))
        all_y.extend(y.detach().cpu().numpy().tolist())
        all_p.extend(prob.tolist())
        all_ids.extend(list(zip(batch["peptide_id"], batch["protein_id"], batch["raw_header"])))
    return float(np.mean(losses)) if losses else 0.0, np.asarray(all_y), np.asarray(all_p), all_ids


def save_predictions(path: Path, y: np.ndarray, p: np.ndarray, ids: List[Tuple[str, str, str]], threshold: float) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["peptide_id", "protein_id", "label", "probability", "prediction", "raw_header"])
        for (pep_id, prot_id, raw), yy, pp in zip(ids, y, p):
            writer.writerow([pep_id, prot_id, int(yy), float(pp), int(pp >= threshold), raw])


def plot_history(history: List[Dict[str, Any]], out_dir: Path, fold_name: str = "split") -> None:
    """Create only the per-fold metric curves needed for publication.

    The function writes train-vs-validation curves for AUC, AUPR, ACC, and F1.
    Loss and threshold plots are intentionally not generated in each fold folder
    to keep the output directory compact.
    """
    if not history:
        return
    ensure_dir(out_dir)
    epochs = [int(h["epoch"]) for h in history]

    metric_pairs = [
        ("auc", "train_auc", "val_auc", "AUC"),
        ("aupr", "train_aupr", "val_aupr", "AUPR"),
        ("acc", "train_acc", "val_acc", "Accuracy"),
        ("f1", "train_f1", "val_f1", "F1"),
    ]
    for suffix, train_key, val_key, label in metric_pairs:
        if train_key not in history[0] and val_key not in history[0]:
            continue
        plt.figure(figsize=(8, 5))
        if train_key in history[0]:
            plt.plot(epochs, [h.get(train_key, np.nan) for h in history], label=f"train_{suffix}")
        if val_key in history[0]:
            plt.plot(epochs, [h.get(val_key, np.nan) for h in history], label=f"validation_{suffix}")
        plt.xlabel("Epoch")
        plt.ylabel(label)
        plt.title(f"{fold_name} train/validation {label}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"training_curves_train_validation_{suffix}.png", dpi=200)
        plt.close()


def save_history_outputs(history: List[Dict[str, Any]], out_dir: Path, fold_name: str = "split") -> None:
    """Persist epoch history and update plots.

    This is called after every epoch, so a partial history is preserved if a
    run is interrupted before early stopping or final evaluation.
    """
    ensure_dir(out_dir)
    write_metrics_csv(out_dir / "history.csv", history)
    plot_history(history, out_dir, fold_name=fold_name)


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out: Dict[str, Any] = {}
            for k, v in row.items():
                if v is None or v == "":
                    out[k] = v
                    continue
                try:
                    if k in {"epoch", "is_best"}:
                        out[k] = int(float(v))
                    else:
                        out[k] = float(v)
                except Exception:
                    out[k] = v
            rows.append(out)
    return rows


def aggregate_fold_histories(output_dir: Path, num_folds: int) -> None:
    """Aggregate per-fold history files and write run-level metric plots.

    Written files:
      - all_folds_history.csv
      - all_folds_history_mean.csv
      - all_folds_history_std.csv
      - all_folds_train_validation_auc.png
      - all_folds_train_validation_aupr.png
      - all_folds_train_validation_acc.png
      - all_folds_train_validation_f1.png
    """
    output_dir = ensure_dir(output_dir)
    all_rows: List[Dict[str, Any]] = []
    for fold in range(1, num_folds + 1):
        fold_name = f"fold{fold}"
        hist_path = output_dir / fold_name / "history.csv"
        for row in _read_csv_rows(hist_path):
            row["fold"] = fold_name
            all_rows.append(row)

    if not all_rows:
        return

    write_metrics_csv(output_dir / "all_folds_history.csv", all_rows)

    numeric_keys = [
        "train_loss", "val_loss",
        "train_acc", "train_auc", "train_aupr", "train_f1",
        "val_acc", "val_auc", "val_aupr", "val_f1",
        "val_threshold", "generalization_gap", "gap_wait", "best_val_loss_seen",
    ]
    epochs = sorted({int(r["epoch"]) for r in all_rows if "epoch" in r})
    mean_rows: List[Dict[str, Any]] = []
    std_rows: List[Dict[str, Any]] = []

    for epoch in epochs:
        erows = [r for r in all_rows if int(r.get("epoch", -1)) == epoch]
        mrow: Dict[str, Any] = {"epoch": epoch, "n_folds_at_epoch": len(erows)}
        srow: Dict[str, Any] = {"epoch": epoch, "n_folds_at_epoch": len(erows)}
        for key in numeric_keys:
            vals = []
            for r in erows:
                try:
                    val = float(r[key])
                except Exception:
                    continue
                if np.isfinite(val):
                    vals.append(val)
            if vals:
                arr = np.asarray(vals, dtype=float)
                mrow[key] = float(arr.mean())
                srow[key] = float(arr.std(ddof=0))
        mean_rows.append(mrow)
        std_rows.append(srow)

    write_metrics_csv(output_dir / "all_folds_history_mean.csv", mean_rows)
    write_metrics_csv(output_dir / "all_folds_history_std.csv", std_rows)

    # Combined metric figures. Each plot shows all fold trajectories lightly and
    # the epoch-wise mean trajectory more prominently.
    folds = sorted({str(r["fold"]) for r in all_rows if "fold" in r})
    metric_pairs = [
        ("auc", "train_auc", "val_auc", "AUC"),
        ("aupr", "train_aupr", "val_aupr", "AUPR"),
        ("acc", "train_acc", "val_acc", "Accuracy"),
        ("f1", "train_f1", "val_f1", "F1"),
    ]
    for suffix, train_key, val_key, label in metric_pairs:
        if not any(train_key in r or val_key in r for r in all_rows):
            continue
        plt.figure(figsize=(9, 6))
        for fold in folds:
            frows = sorted([r for r in all_rows if str(r.get("fold")) == fold], key=lambda x: int(x.get("epoch", 0)))
            fx = [int(r["epoch"]) for r in frows]
            if train_key in frows[0]:
                plt.plot(fx, [r.get(train_key, np.nan) for r in frows], alpha=0.20, linewidth=1)
            if val_key in frows[0]:
                plt.plot(fx, [r.get(val_key, np.nan) for r in frows], alpha=0.20, linewidth=1)

        mx = [int(r["epoch"]) for r in mean_rows]
        if any(train_key in r for r in mean_rows):
            plt.plot(mx, [r.get(train_key, np.nan) for r in mean_rows], linewidth=3, label=f"mean_train_{suffix}")
        if any(val_key in r for r in mean_rows):
            plt.plot(mx, [r.get(val_key, np.nan) for r in mean_rows], linewidth=3, label=f"mean_validation_{suffix}")
        plt.xlabel("Epoch")
        plt.ylabel(label)
        plt.title(f"All folds train/validation {label}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"all_folds_train_validation_{suffix}.png", dpi=220)
        plt.close()


def plot_curves(y: np.ndarray, p: np.ndarray, out_prefix: Path, title: str) -> None:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(np.unique(y)) < 2:
        return
    precision, recall, _ = precision_recall_curve(y, p)
    aupr = average_precision_score(y, p)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} PR curve (AUPR={aupr:.4f})")
    plt.tight_layout()
    plt.savefig(str(out_prefix) + "_pr_curve.png", dpi=180)
    plt.close()

    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{title} ROC curve (AUC={auc:.4f})")
    plt.tight_layout()
    plt.savefig(str(out_prefix) + "_roc_curve.png", dpi=180)
    plt.close()



def plot_combined_independent_curves(
    y: np.ndarray,
    fold_probs: Sequence[np.ndarray],
    mean_prob: np.ndarray,
    output_dir: Path,
    split_name: str,
) -> None:
    """Write combined ROC and PR figures for one independent dataset.

    The plot contains one curve per fold plus the ensemble-mean probability
    curve. These figures replace per-fold independent-test ROC/PR figures.
    """
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return

    safe_split = str(split_name).replace("/", "_").replace(" ", "_")

    # ROC
    plt.figure(figsize=(9, 6))
    for i, p in enumerate(fold_probs, start=1):
        p = np.asarray(p).astype(float)
        try:
            fpr, tpr, _ = roc_curve(y, p)
            auc = roc_auc_score(y, p)
            plt.plot(fpr, tpr, alpha=0.25, linewidth=1, label=f"fold {i} AUC={auc:.2f}")
        except Exception:
            continue
    fpr, tpr, _ = roc_curve(y, mean_prob)
    auc = roc_auc_score(y, mean_prob)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2, label="Random")
    plt.plot(fpr, tpr, linewidth=3, label=f"ensemble AUC={auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{split_name} ROC curve")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"combined_{safe_split}_roc_curve.png", dpi=220)
    plt.close()

    # PR
    plt.figure(figsize=(9, 6))
    baseline = float(np.mean(y))
    for i, p in enumerate(fold_probs, start=1):
        p = np.asarray(p).astype(float)
        try:
            precision, recall, _ = precision_recall_curve(y, p)
            aupr = average_precision_score(y, p)
            plt.plot(recall, precision, alpha=0.25, linewidth=1, label=f"fold {i} AUPR={aupr:.2f}")
        except Exception:
            continue
    precision, recall, _ = precision_recall_curve(y, mean_prob)
    aupr = average_precision_score(y, mean_prob)
    plt.axhline(baseline, linestyle="--", linewidth=2, label=f"Random={baseline:.2f}")
    plt.plot(recall, precision, linewidth=3, label=f"ensemble AUPR={aupr:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{split_name} precision-recall curve")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"combined_{safe_split}_pr_curve.png", dpi=220)
    plt.close()


def write_metrics_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_json(path: Path, rows: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def write_report_table(base_path: Path, rows: List[Dict[str, Any]], report_format: str = "csv") -> None:
    """Write a report in CSV, JSON, or both.

    ``base_path`` should be passed without suffix, for example
    ``output_dir / "independent_test_metrics"``.
    """
    if not rows:
        return
    report_format = str(report_format).lower()
    if report_format in {"csv", "both"}:
        write_metrics_csv(base_path.with_suffix(".csv"), rows)
    if report_format in {"json", "both"}:
        write_metrics_json(base_path.with_suffix(".json"), rows)


def monitor_value(metrics: Dict[str, float], monitor: str, loss: float) -> float:
    if monitor == "loss":
        return -loss
    return float(metrics.get(monitor, float("nan")))


def train_one_split(
    args: argparse.Namespace,
    dataset: PepPIDataset,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    fold_dir: Path,
    fold_name: str,
) -> Tuple[Path, Dict[str, float], float, int]:
    ensure_dir(fold_dir)
    input_dim = infer_input_dim(dataset)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = build_model(args, input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = make_loader(dataset, train_idx, args.batch_size, True, args.num_workers)
    val_loader = make_loader(dataset, val_idx, args.batch_size, False, args.num_workers)

    best_score = -float("inf")
    best_metrics: Dict[str, float] = {}
    best_epoch = -1
    wait = 0
    history: List[Dict[str, Any]] = []
    ckpt_path = fold_dir / args.checkpoint

    gap_wait = 0
    best_val_loss_seen = float("inf")
    stopped_by_gap = False

    print(f"[{now()}] Starting {fold_name} on {device}; train={len(train_idx)} val={len(val_idx)} input_dim={input_dim}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_y, train_p, _ = run_epoch(model, train_loader, optimizer, device, criterion, args.grad_clip)
        val_loss, val_y, val_p, val_ids = run_epoch(model, val_loader, None, device, criterion, args.grad_clip)
        val_threshold = find_best_threshold(val_y, val_p, args.threshold_objective) if args.tune_threshold else 0.5
        val_metrics = compute_metrics(val_y, val_p, val_threshold)
        train_metrics = compute_metrics(train_y, train_p, 0.5)
        score = monitor_value(val_metrics, args.monitor, val_loss)

        best_val_loss_seen = min(best_val_loss_seen, float(val_loss))
        generalization_gap = float(val_loss) - float(train_loss)
        gap_stop_active = (
            args.max_generalization_gap > 0
            and epoch >= args.min_epochs_before_gap_stop
        )
        val_loss_is_bad = float(val_loss) > best_val_loss_seen + args.gap_val_loss_tolerance
        gap_is_large = generalization_gap > args.max_generalization_gap

        if gap_stop_active and gap_is_large and val_loss_is_bad:
            gap_wait += 1
        else:
            gap_wait = 0

        row: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_metrics["acc"],
            "train_auc": train_metrics["auc"],
            "train_aupr": train_metrics["aupr"],
            "train_f1": train_metrics["f1"],
            "val_acc": val_metrics["acc"],
            "val_auc": val_metrics["auc"],
            "val_aupr": val_metrics["aupr"],
            "val_f1": val_metrics["f1"],
            "val_threshold": val_threshold,
            "generalization_gap": generalization_gap,
            "gap_wait": gap_wait,
            "best_val_loss_seen": best_val_loss_seen,
            "gap_stop_active": int(gap_stop_active),
            "gap_is_large": int(gap_is_large),
            "gap_val_loss_is_bad": int(val_loss_is_bad),
            "is_best": 0,
        }
        print(
            f"[{now()}] {fold_name} epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_aupr={val_metrics['aupr']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} thr={val_threshold:.3f} "
            f"gap={generalization_gap:.4f} gap_wait={gap_wait}"
        )
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_metrics = dict(val_metrics)
            best_metrics["val_loss"] = val_loss
            best_epoch = epoch
            wait = 0
            # Keep exactly one row marked as the current best epoch.
            for prev in history:
                prev["is_best"] = 0
            row["is_best"] = 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "input_dim": input_dim,
                    "best_epoch": best_epoch,
                    "best_metrics": best_metrics,
                    "threshold": val_threshold,
                },
                ckpt_path,
            )
            if args.save_oof_predictions:
                save_predictions(fold_dir / "val_predictions_best.csv", val_y, val_p, val_ids, val_threshold)
        else:
            wait += 1

        history.append(row)
        # Save after every epoch so interrupted runs still keep useful curves/CSV.
        save_history_outputs(history, fold_dir, fold_name=fold_name)

        if wait >= args.patience:
            print(f"[{now()}] Early stopping {fold_name} at epoch {epoch}; best_epoch={best_epoch}")
            break

        if gap_stop_active and gap_wait >= args.gap_patience:
            stopped_by_gap = True
            print(
                f"[{now()}] Generalization-gap stopping {fold_name} at epoch {epoch}; "
                f"gap={generalization_gap:.4f}, "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"best_val_loss_seen={best_val_loss_seen:.4f}, "
                f"best_epoch={best_epoch}",
                flush=True,
            )
            break

    save_history_outputs(history, fold_dir, fold_name=fold_name)
    best_summary = [{
        "best_epoch": best_epoch,
        "stopped_by_generalization_gap": int(stopped_by_gap),
        "max_generalization_gap": args.max_generalization_gap,
        "gap_patience": args.gap_patience,
        "min_epochs_before_gap_stop": args.min_epochs_before_gap_stop,
        "gap_val_loss_tolerance": args.gap_val_loss_tolerance,
        **best_metrics,
    }]
    write_report_table(fold_dir / "best_model_summary", best_summary, args.report_format)
    return ckpt_path, best_metrics, float(best_metrics.get("threshold", 0.5)), best_epoch


def load_model_from_checkpoint(args: argparse.Namespace, ckpt_path: Path, device: torch.device) -> Tuple[nn.Module, float]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    input_dim = int(ckpt["input_dim"])
    model = build_model(args, input_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    threshold = float(ckpt.get("threshold", ckpt.get("best_metrics", {}).get("threshold", 0.5)))
    return model, threshold


def evaluate_dataset(args: argparse.Namespace, ckpt_path: Path, fasta_path: str, split_name: str, out_dir: Path, fold_name: str) -> Dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    test_ds = PepPIDataset(
        fasta_path,
        args.embedding_dir,
        args.max_peptide_len,
        args.max_protein_len,
        args.protein_pooling_mode,
    )
    loader = make_loader(test_ds, None, args.batch_size, False, args.num_workers)
    model, threshold = load_model_from_checkpoint(args, ckpt_path, device)
    criterion = nn.BCEWithLogitsLoss()
    loss, y, p, ids = run_epoch(model, loader, None, device, criterion, args.grad_clip)
    metrics = compute_metrics(y, p, threshold)
    metrics.update({"split": split_name, "fold": fold_name, "loss": loss, "n": len(test_ds)})
    prefix = out_dir / f"{fold_name}_{split_name}"
    save_predictions(Path(str(prefix) + "_predictions.csv"), y, p, ids, threshold)
    if args.report_format in {"json", "both"}:
        with open(Path(str(prefix) + "_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    print(
        f"[{now()}] Eval {fold_name}/{split_name}: "
        f"ACC={metrics['acc']:.4f} F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f} AUPR={metrics['aupr']:.4f} n={len(test_ds)}"
    )
    return metrics


def make_single_split(labels: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    idx = np.arange(len(labels))
    train_idx, val_idx = next(splitter.split(idx, labels))
    return train_idx, val_idx


def aggregate_fold_metrics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    out: List[Dict[str, Any]] = []
    splits = sorted(set(str(r["split"]) for r in rows))
    metric_keys = ["acc", "f1", "auc", "aupr", "loss"]
    for split in splits:
        sr = [r for r in rows if str(r["split"]) == split]
        row: Dict[str, Any] = {"split": split, "fold": "mean±std", "n_folds": len(sr)}
        for k in metric_keys:
            vals = np.array([float(r[k]) for r in sr if k in r and np.isfinite(float(r[k]))])
            if len(vals):
                row[f"{k}_mean"] = float(vals.mean())
                row[f"{k}_std"] = float(vals.std(ddof=0))
        out.append(row)
    return out



# -----------------------------
# K-fold probability ensemble
# -----------------------------


def _read_prediction_csv(path: Path) -> Dict[str, List[Any]]:
    """Read a prediction CSV written by save_predictions()."""
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    return {
        "peptide_id": [r.get("peptide_id", "") for r in rows],
        "protein_id": [r.get("protein_id", "") for r in rows],
        "raw_header": [r.get("raw_header", "") for r in rows],
        "label": np.asarray([int(float(r["label"])) for r in rows], dtype=int),
        "probability": np.asarray([float(r["probability"]) for r in rows], dtype=float),
    }


def _binary_log_loss_from_prob(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    y = np.asarray(y_true).astype(float)
    p = np.asarray(y_prob).astype(float)
    eps = 1e-7
    p = np.clip(p, eps, 1.0 - eps)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _collect_fold_thresholds(output_dir: Path, num_folds: int) -> List[float]:
    thresholds: List[float] = []
    for fold in range(1, num_folds + 1):
        fold_dir = output_dir / f"fold{fold}"

        # Preferred when report_format=json/both.
        summary_json = fold_dir / "best_model_summary.json"
        if summary_json.exists():
            try:
                with open(summary_json, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, list) and obj and "threshold" in obj[0]:
                    thresholds.append(float(obj[0]["threshold"]))
                    continue
            except Exception as exc:
                print(f"[warning] Could not read threshold from {summary_json}: {exc}", file=sys.stderr)

        # Preferred when report_format=csv.
        summary_csv = fold_dir / "best_model_summary.csv"
        if summary_csv.exists():
            try:
                rows = _read_csv_rows(summary_csv)
                if rows and "threshold" in rows[0]:
                    thresholds.append(float(rows[0]["threshold"]))
                    continue
            except Exception as exc:
                print(f"[warning] Could not read threshold from {summary_csv}: {exc}", file=sys.stderr)

        # Backward compatibility with older scripts.
        old_json = fold_dir / "best_validation_metrics.json"
        if old_json.exists():
            try:
                with open(old_json, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if "threshold" in obj:
                    thresholds.append(float(obj["threshold"]))
            except Exception as exc:
                print(f"[warning] Could not read threshold from {old_json}: {exc}", file=sys.stderr)
    return thresholds

def _collect_oof_validation_predictions(output_dir: Path, num_folds: int) -> Tuple[np.ndarray, np.ndarray]:
    """Collect out-of-fold validation predictions from fold val_predictions_best.csv files."""
    all_y: List[int] = []
    all_p: List[float] = []
    for fold in range(1, num_folds + 1):
        path = output_dir / f"fold{fold}" / "val_predictions_best.csv"
        if not path.exists():
            print(f"[warning] Missing OOF validation prediction file: {path}", file=sys.stderr)
            continue
        pred = _read_prediction_csv(path)
        all_y.extend(pred["label"].astype(int).tolist())
        all_p.extend(pred["probability"].astype(float).tolist())
    return np.asarray(all_y, dtype=int), np.asarray(all_p, dtype=float)


def _choose_ensemble_threshold(args: argparse.Namespace, output_dir: Path, strategy: str) -> float:
    """Choose one threshold for ensemble independent-test predictions.

    This function never uses independent test labels. AUC/AUPR are always
    computed from probabilities and are therefore unchanged by this threshold.
    """
    strategy = str(strategy).strip()
    if strategy == "fixed_0_5":
        return 0.5
    if strategy == "custom":
        return float(args.ensemble_custom_threshold)

    fold_thresholds = _collect_fold_thresholds(output_dir, args.num_folds)
    if strategy == "median_fold_threshold":
        if fold_thresholds:
            return float(np.median(np.asarray(fold_thresholds, dtype=float)))
        print("[warning] No fold thresholds found; falling back to 0.5", file=sys.stderr)
        return 0.5
    if strategy == "mean_fold_threshold":
        if fold_thresholds:
            return float(np.mean(np.asarray(fold_thresholds, dtype=float)))
        print("[warning] No fold thresholds found; falling back to 0.5", file=sys.stderr)
        return 0.5

    if strategy in {"oof_f1", "oof_acc", "oof_youden", "oof_aupr"}:
        y, p = _collect_oof_validation_predictions(output_dir, args.num_folds)
        if len(y) and len(np.unique(y)) > 1:
            objective_map = {
                "oof_f1": "f1",
                "oof_acc": "acc",
                "oof_youden": "youden",
                # AUPR is threshold-free; this falls back to F1 with a warning.
                "oof_aupr": "aupr",
            }
            return find_best_threshold(y, p, objective=objective_map[strategy])
        print("[warning] Could not collect OOF validation predictions; falling back to median fold threshold or 0.5", file=sys.stderr)
        if fold_thresholds:
            return float(np.median(np.asarray(fold_thresholds, dtype=float)))
        return 0.5

    raise ValueError(f"Unknown ensemble threshold strategy: {strategy}")


def _parse_ensemble_strategies(args: argparse.Namespace) -> List[str]:
    if getattr(args, "ensemble_threshold_strategies", None):
        raw = [x.strip() for x in args.ensemble_threshold_strategies.split(",") if x.strip()]
    else:
        raw = [args.ensemble_threshold_strategy]
    if len(raw) == 1 and raw[0] == "all":
        raw = ["median_fold_threshold", "mean_fold_threshold", "oof_f1", "oof_acc", "oof_youden", "fixed_0_5"]
    # preserve order while removing duplicates
    seen: set[str] = set()
    out: List[str] = []
    for x in raw:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def evaluate_kfold_ensemble(args: argparse.Namespace, output_dir: Path) -> List[Dict[str, Any]]:
    """Average fold probabilities and compute ensemble metrics.

    When multiple threshold strategies are requested, AUC/AUPR will be identical
    for a given split because they are threshold-free; ACC/F1 will differ.
    """
    if args.protocol != "kfold" or not args.test_split_names:
        return []

    strategies = _parse_ensemble_strategies(args)
    rows: List[Dict[str, Any]] = []

    # Load all fold predictions once per split.
    split_cache: Dict[str, Tuple[Dict[str, List[Any]], np.ndarray, np.ndarray, List[np.ndarray]]] = {}
    for split_name in args.test_split_names:
        fold_preds: List[Dict[str, List[Any]]] = []
        for fold in range(1, args.num_folds + 1):
            path = output_dir / f"fold{fold}_{split_name}_predictions.csv"
            fold_preds.append(_read_prediction_csv(path))

        ref = fold_preds[0]
        y = ref["label"].astype(int)
        n = len(y)
        probs: List[np.ndarray] = []
        for i, pred in enumerate(fold_preds, start=1):
            if len(pred["label"]) != n:
                raise ValueError(f"Fold {i} prediction count differs for {split_name}")
            if not np.array_equal(pred["label"].astype(int), y):
                raise ValueError(f"Fold {i} labels differ for {split_name}; cannot ensemble safely")
            if pred["raw_header"] != ref["raw_header"]:
                raise ValueError(
                    f"Fold {i} raw_header order differs for {split_name}; "
                    "rerun with deterministic test ordering or merge by unique ids."
                )
            probs.append(pred["probability"].astype(float))
        prob_mean = np.mean(np.vstack(probs), axis=0)
        split_cache[split_name] = (ref, y, prob_mean, probs)
        plot_combined_independent_curves(y, probs, prob_mean, output_dir, split_name)

    for strategy in strategies:
        threshold = _choose_ensemble_threshold(args, output_dir, strategy)
        print(f"[{now()}] K-fold ensemble threshold={threshold:.6f} strategy={strategy}", flush=True)

        safe_strategy = strategy.replace("/", "_")
        for split_name in args.test_split_names:
            ref, y, prob_mean, probs = split_cache[split_name]
            n = len(y)
            metrics = compute_metrics(y, prob_mean, threshold)
            metrics.update({
                "split": split_name,
                "fold": "ensemble_mean_probability",
                "loss": _binary_log_loss_from_prob(y, prob_mean),
                "n": int(n),
                "threshold": float(threshold),
                "ensemble_threshold_strategy": strategy,
                "n_folds": int(args.num_folds),
            })
            rows.append(metrics)

            if args.save_ensemble_predictions:
                out_pred = output_dir / f"ensemble_{safe_strategy}_{split_name}_predictions.csv"
                with open(out_pred, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    header = ["peptide_id", "protein_id", "label", "probability", "prediction", "raw_header"]
                    header.extend([f"probability_fold{fold}" for fold in range(1, args.num_folds + 1)])
                    writer.writerow(header)
                    for idx in range(n):
                        writer.writerow([
                            ref["peptide_id"][idx],
                            ref["protein_id"][idx],
                            int(y[idx]),
                            float(prob_mean[idx]),
                            int(prob_mean[idx] >= threshold),
                            ref["raw_header"][idx],
                            *[float(probs[fold - 1][idx]) for fold in range(1, args.num_folds + 1)],
                        ])

            print(
                f"[{now()}] Ensemble/{split_name}/{strategy}: "
                f"ACC={metrics['acc']:.4f} F1={metrics['f1']:.4f} "
                f"AUC={metrics['auc']:.4f} AUPR={metrics['aupr']:.4f} "
                f"threshold={threshold:.4f} n={n}",
                flush=True,
            )

    write_report_table(output_dir / "ensemble_independent_test_metrics", rows, args.report_format)
    return rows

def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def write_model_summary(args: argparse.Namespace, output_dir: Path, input_dim: int) -> None:
    """Write one run-level model summary in the main output directory."""
    device = torch.device("cpu")
    model = build_model(args, input_dim).to(device)
    total_params, trainable_params = count_trainable_parameters(model)

    summary = {
        "model": "ResidualInteractionMLP",
        "variant": args.variant,
        "interaction_features": args.interaction_features,
        "input_representation": args.interaction_features,
        "pooling": "masked_mean",
        "input_dim": int(input_dim),
        "interaction_dim": int(input_dim * len(parse_interaction_features(args.interaction_features))),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "hidden_dim": args.hidden_dim,
        "mlp_blocks": args.mlp_blocks,
        "mlp_expand_ratio": args.mlp_expand_ratio,
        "head_ratio": args.head_ratio,
        "activation": args.activation,
        "norm_type": args.norm_type,
        "residual_enabled": not args.no_residual,
        "residual_scale": args.residual_scale,
        "dropout_global_fallback": args.dropout,
        "input_dropout": args.input_dropout if args.input_dropout is not None else args.dropout,
        "block_dropout": args.block_dropout if args.block_dropout is not None else args.dropout,
        "block_dropouts": args.block_dropouts,
        "head_dropout": args.head_dropout if args.head_dropout is not None else args.dropout,
        "protocol": args.protocol,
        "num_folds": args.num_folds,
        "monitor": args.monitor,
        "threshold_objective": args.threshold_objective,
        "tune_threshold": args.tune_threshold,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "max_generalization_gap": args.max_generalization_gap,
        "gap_patience": args.gap_patience,
        "min_epochs_before_gap_stop": args.min_epochs_before_gap_stop,
        "gap_val_loss_tolerance": args.gap_val_loss_tolerance,
        "max_peptide_len": args.max_peptide_len,
        "max_protein_len": args.max_protein_len,
        "protein_pooling_mode": args.protein_pooling_mode,
        "ensemble_kfold": args.ensemble_kfold,
        "ensemble_threshold_strategy": args.ensemble_threshold_strategy,
        "ensemble_threshold_strategies": args.ensemble_threshold_strategies,
        "save_oof_predictions": args.save_oof_predictions,
    }

    with open(output_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_metrics_csv(output_dir / "model_summary.csv", [summary])

    with open(output_dir / "model_summary.txt", "w", encoding="utf-8") as f:
        f.write("ResidualInteractionMLP model summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Input embedding dimension: {input_dim}\n")
        selected_features = parse_interaction_features(args.interaction_features)
        f.write(f"Interaction features: {','.join(selected_features)}\n")
        f.write(f"Interaction vector dimension: {input_dim * len(selected_features)}\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n\n")
        f.write(str(model))
        f.write("\n")

def copy_run_script(output_dir: Path) -> None:
    # The bash wrapper also copies itself. This protects direct Python runs.
    try:
        src = Path(sys.argv[0]).resolve()
        if src.exists():
            shutil.copy2(src, output_dir / src.name)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="ResidualInteractionMLP peptide-protein interaction training")
    parser.add_argument("--train_fasta", required=True)
    parser.add_argument("--embedding_dir", required=True)
    parser.add_argument("--train_split_name", default="train")
    parser.add_argument("--protocol", choices=["single_split", "kfold"], default="single_split")
    parser.add_argument("--internal_val_fraction", type=float, default=0.1)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--variant", choices=["residual_interaction_mlp"], default="residual_interaction_mlp")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--mlp_blocks", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument(
        "--input_dropout",
        type=float,
        default=None,
        help="Dropout after the first input projection. If omitted, uses --dropout.",
    )
    parser.add_argument(
        "--block_dropout",
        type=float,
        default=None,
        help="Default dropout inside each residual MLP block. If omitted, uses --dropout.",
    )
    parser.add_argument(
        "--block_dropouts",
        default="",
        help=(
            "Optional comma-separated per-block dropout values. "
            "Example for 3 blocks: --block_dropouts 0.20,0.25,0.30. "
            "Overrides --block_dropout."
        ),
    )
    parser.add_argument(
        "--head_dropout",
        type=float,
        default=None,
        help="Dropout in the final classifier head. If omitted, uses --dropout.",
    )
    parser.add_argument(
        "--mlp_expand_ratio",
        type=float,
        default=2.0,
        help="Expansion ratio inside each residual block. Default 2.0 means hidden_dim -> 2*hidden_dim -> hidden_dim.",
    )
    parser.add_argument(
        "--head_ratio",
        type=float,
        default=0.5,
        help="Classifier-head hidden ratio. Default 0.5 means hidden_dim -> hidden_dim/2 -> 1.",
    )
    parser.add_argument(
        "--activation",
        choices=["gelu", "relu", "silu", "leaky_relu", "elu"],
        default="gelu",
        help="Activation function used in the projection, residual blocks, and classifier head.",
    )
    parser.add_argument(
        "--norm_type",
        choices=["layernorm", "batchnorm", "none"],
        default="layernorm",
        help="Normalization used inside residual blocks and before the classifier head.",
    )
    parser.add_argument(
        "--no_residual",
        action="store_true",
        help="Disable residual skip connections inside MLP blocks for ablation.",
    )
    parser.add_argument(
        "--residual_scale",
        type=float,
        default=1.0,
        help="Scale applied to each residual block update before adding to the skip connection.",
    )
    parser.add_argument(
        "--interaction_features",
        default="p,q,absdiff,product",
        help=(
            "Comma-separated interaction features before the MLP. "
            "Allowed: p,q,absdiff,product,sum,diff. "
            "Default reproduces the selected model: p,q,absdiff,product."
        ),
    )
    parser.add_argument("--monitor", choices=["aupr", "auc", "f1", "acc", "loss"], default="aupr")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--checkpoint", default="best_residual_interaction_mlp.pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_fastas", nargs="*", default=[])
    parser.add_argument("--test_split_names", nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_peptide_len", type=int, default=256)
    parser.add_argument("--max_protein_len", type=int, default=1024)
    parser.add_argument(
        "--protein_pooling_mode",
        choices=["truncate", "window_mean"],
        default="truncate",
        help=(
            "How to handle protein embeddings longer than --max_protein_len. "
            "'truncate' preserves the original behavior and keeps only the first "
            "max_protein_len residues. 'window_mean' splits the full protein into "
            "non-overlapping max_protein_len windows, mean-pools each window, and "
            "lets the model mean-pool the resulting window vectors."
        ),
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--max_generalization_gap",
        type=float,
        default=-1.0,
        help=(
            "Optional overfitting stop. If >0, stop when val_loss - train_loss "
            "exceeds this value for --gap_patience consecutive epochs after "
            "--min_epochs_before_gap_stop, provided val_loss is also worse than "
            "the best validation loss by --gap_val_loss_tolerance."
        ),
    )
    parser.add_argument(
        "--gap_patience",
        type=int,
        default=6,
        help="Number of consecutive large-gap epochs required for generalization-gap stopping.",
    )
    parser.add_argument(
        "--min_epochs_before_gap_stop",
        type=int,
        default=15,
        help="Earliest epoch at which generalization-gap stopping may activate.",
    )
    parser.add_argument(
        "--gap_val_loss_tolerance",
        type=float,
        default=0.03,
        help="Require val_loss to exceed best_val_loss_seen by this margin before counting a large-gap epoch.",
    )
    parser.add_argument(
        "--ensemble_kfold",
        action="store_true",
        help=(
            "After k-fold training/evaluation, average independent-test probabilities "
            "across folds and compute ensemble metrics. This is only used with --protocol kfold."
        ),
    )
    parser.add_argument(
        "--ensemble_threshold_strategy",
        choices=["median_fold_threshold", "mean_fold_threshold", "oof_f1", "oof_acc", "oof_youden", "oof_aupr", "fixed_0_5", "custom", "all"],
        default="median_fold_threshold",
        help=(
            "Threshold used for ensemble ACC/F1. Strategies use validation-derived "
            "thresholds or out-of-fold validation predictions, not independent test labels."
        ),
    )
    parser.add_argument(
        "--ensemble_custom_threshold",
        type=float,
        default=0.5,
        help="Custom threshold used only with --ensemble_threshold_strategy custom.",
    )
    parser.add_argument(
        "--ensemble_threshold_strategies",
        default="",
        help=(
            "Optional comma-separated list of ensemble threshold strategies to evaluate in one run, "
            "for example: median_fold_threshold,oof_f1,oof_acc,oof_youden. Use 'all' for common strategies. "
            "AUC/AUPR are threshold-free and will be identical across strategies for the same probabilities."
        ),
    )
    parser.add_argument(
        "--save_ensemble_predictions",
        action="store_true",
        help="Save ensemble prediction CSV files. Disabled by default to reduce output clutter.",
    )
    parser.add_argument(
        "--save_oof_predictions",
        action="store_true",
        help=(
            "Save fold-level out-of-fold validation prediction files. Disabled by default "
            "to avoid val_predictions_best.csv files. Required for oof_* threshold strategies."
        ),
    )
    parser.add_argument(
        "--save_ensemble_curves",
        action="store_true",
        help="Save ensemble PR/ROC curve PNGs. Disabled by default to reduce output clutter.",
    )
    parser.add_argument(
        "--report_format",
        choices=["csv", "json", "both"],
        default="csv",
        help="Write metric/report tables as CSV, JSON, or both. Default: csv only.",
    )
    parser.add_argument(
        "--fig_format",
        choices=["png", "pdf"],
        default="png",
        help=(
            "Compatibility option accepted by publication launcher scripts. "
            "This script currently writes training and evaluation figures using the internal plotting routines."
        ),
    )
    parser.add_argument(
        "--merge_all_independent_predictions",
        action="store_true",
        help=(
            "Compatibility option accepted by publication launcher scripts. "
            "When enabled, downstream ensemble outputs remain available from the standard prediction CSV files."
        ),
    )
    parser.add_argument(
        "--ensemble_only",
        action="store_true",
        help=(
            "Do not train models. Recompute ensemble metrics from existing fold prediction files "
            "inside --output_dir. Useful for trying multiple threshold strategies without rerunning k-fold training."
        ),
    )
    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument(
        "--tune_threshold",
        dest="tune_threshold",
        action="store_true",
        default=True,
        help="Tune classification threshold on validation set. This is the default and matches the dynamic F1-threshold strategy reported by PepLM-GNN.",
    )
    threshold_group.add_argument(
        "--fixed_threshold",
        dest="tune_threshold",
        action="store_false",
        help="Disable validation threshold tuning and use a fixed threshold of 0.5.",
    )
    parser.add_argument("--threshold_objective", choices=["f1", "acc", "youden"], default="f1")
    args = parser.parse_args()

    if args.test_fastas and len(args.test_fastas) != len(args.test_split_names):
        raise ValueError("--test_fastas and --test_split_names must have the same length")

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    copy_run_script(output_dir)
    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    if args.ensemble_only:
        if args.protocol != "kfold":
            raise ValueError("--ensemble_only requires --protocol kfold")
        if not args.ensemble_kfold:
            args.ensemble_kfold = True
        evaluate_kfold_ensemble(args, output_dir)
        print(f"[{now()}] Done. Ensemble-only outputs saved to {output_dir}")
        return

    dataset = PepPIDataset(
        args.train_fasta,
        args.embedding_dir,
        args.max_peptide_len,
        args.max_protein_len,
        args.protein_pooling_mode,
    )
    labels = dataset.labels
    print(f"[{now()}] Loaded train dataset: n={len(dataset)} positives={int(labels.sum())} negatives={int(len(labels)-labels.sum())}")

    input_dim_for_summary = infer_input_dim(dataset)
    write_model_summary(args, output_dir, input_dim=input_dim_for_summary)

    all_eval_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []

    if args.protocol == "single_split":
        train_idx, val_idx = make_single_split(labels, args.internal_val_fraction, args.seed)
        fold_dir = ensure_dir(output_dir / "single_split")
        ckpt, val_metrics, threshold, best_epoch = train_one_split(args, dataset, train_idx, val_idx, fold_dir, "single_split")
        val_rows.append({"fold": "single_split", "best_epoch": best_epoch, **{f"val_{k}": v for k, v in val_metrics.items()}})
        for fasta, name in zip(args.test_fastas, args.test_split_names):
            all_eval_rows.append(evaluate_dataset(args, ckpt, fasta, name, output_dir, "single_split"))

    else:
        skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(labels)), labels), start=1):
            fold_name = f"fold{fold}"
            fold_dir = ensure_dir(output_dir / fold_name)
            # Different seed per fold while keeping reproducibility.
            set_seed(args.seed + fold)
            ckpt, val_metrics, threshold, best_epoch = train_one_split(args, dataset, train_idx, val_idx, fold_dir, fold_name)
            val_rows.append({"fold": fold_name, "best_epoch": best_epoch, **{f"val_{k}": v for k, v in val_metrics.items()}})
            for fasta, name in zip(args.test_fastas, args.test_split_names):
                all_eval_rows.append(evaluate_dataset(args, ckpt, fasta, name, output_dir, fold_name))

    if args.protocol == "kfold":
        aggregate_fold_histories(output_dir, args.num_folds)
        if args.ensemble_kfold:
            evaluate_kfold_ensemble(args, output_dir)

    write_report_table(output_dir / "validation_summary", val_rows, args.report_format)
    write_report_table(output_dir / "independent_test_metrics", all_eval_rows, args.report_format)
    agg = aggregate_fold_metrics(all_eval_rows)
    if agg:
        write_report_table(output_dir / "independent_test_metrics_mean_std", agg, args.report_format)

    print(f"[{now()}] Done. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
