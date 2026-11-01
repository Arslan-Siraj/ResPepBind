#!/usr/bin/env python3
"""
Generate residue-level embeddings for peptide-protein pairs

This script expects FASTA-like files with records of the form:

>positive_1
Peptide: XYVYNTRSGWRWYT
Protein: MHHHHHHSSGVDLGTENLYFQS...

>negative_1
Peptide: ...
Protein: ...

The script can either:
  1. train from precomputed ProtT5 residue embeddings, or
  2. train from deterministic amino-acid embeddings for debugging.

For real experiments, use ProtT5 residue embeddings.

Expected embedding directory layout:

embeddings/
    train/
        positive_1_peptide.pt   # Tensor [peptide_len, 1024]
        positive_1_protein.pt   # Tensor [protein_len, 1024]
        ...
    independent/
        ...

Example training:

python pepri_hgt_training.py \
    --train_fasta train.fasta \
    --val_fasta independent.fasta \
    --embedding_dir embeddings \
    --use_precomputed_embeddings \
    --epochs 100 \
    --batch_size 8 \
    --hidden_dim 256 \
    --num_layers 3 \
    --top_k 8 \
    --lr 1e-4
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Reproducibility
# -----------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -----------------------------
# FASTA parser
# -----------------------------


@dataclass
class PairRecord:
    pair_id: str
    peptide: str
    protein: str
    label: int


def parse_pair_fasta(path: str | Path) -> List[PairRecord]:
    path = Path(path)
    records: List[PairRecord] = []

    current_id: Optional[str] = None
    peptide: Optional[str] = None
    protein: Optional[str] = None

    def flush() -> None:
        nonlocal current_id, peptide, protein
        if current_id is None:
            return
        if peptide is None or protein is None:
            raise ValueError(f"Record {current_id} is missing Peptide or Protein line.")

        lower_id = current_id.lower()
        if lower_id.startswith("positive") or "positive" in lower_id:
            label = 1
        elif lower_id.startswith("negative") or "negative" in lower_id:
            label = 0
        else:
            raise ValueError(
                f"Cannot infer label from header '{current_id}'. Use headers containing positive or negative."
            )

        records.append(
            PairRecord(
                pair_id=current_id,
                peptide=peptide.strip().replace(" ", ""),
                protein=protein.strip().replace(" ", ""),
                label=label,
            )
        )
        current_id = None
        peptide = None
        protein = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                current_id = line[1:].strip()
            elif line.lower().startswith("peptide:"):
                peptide = line.split(":", 1)[1].strip()
            elif line.lower().startswith("protein:"):
                protein = line.split(":", 1)[1].strip()
            else:
                # Allows wrapped sequence continuation if needed.
                if protein is not None:
                    protein += line.strip()
                elif peptide is not None:
                    peptide += line.strip()
                else:
                    raise ValueError(f"Unexpected line before Peptide/Protein field: {line}")
    flush()
    return records


# -----------------------------
# Optional debugging encoder
# -----------------------------


AA = "ACDEFGHIKLMNPQRSTVWYXBUZO"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}


class DebugResidueEncoder(nn.Module):
    """
    Small deterministic amino-acid embedding encoder for code testing only.
    Replace this with precomputed ProtT5 embeddings for experiments.
    """

    def __init__(self, embedding_dim: int = 1024):
        super().__init__()
        self.embedding = nn.Embedding(len(AA), embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, sequence: str) -> torch.Tensor:
        ids = [AA_TO_IDX.get(ch.upper(), AA_TO_IDX["X"]) for ch in sequence]
        idx = torch.tensor(ids, dtype=torch.long)
        return self.embedding(idx)


# -----------------------------
# ProtT5 residue embedding extraction
# -----------------------------


def sanitize_protein_sequence(seq: str) -> str:
    """
    ProtT5 expects amino acids separated by spaces. Rare or ambiguous residues are
    mapped to X, following the usual ProtTrans preprocessing convention.
    """
    seq = seq.upper().replace(" ", "")
    seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
    return seq


@torch.no_grad()
def extract_prott5_embeddings_for_fasta(
    fasta_path: str | Path,
    output_dir: str | Path,
    split_name: str,
    model_name: str = "Rostlab/prot_t5_xl_uniref50",
    half_precision: bool = True,
    max_protein_len: Optional[int] = None,
    max_peptide_len: Optional[int] = None,
    prott5_chunk_size: int = 1022,
    prott5_chunk_overlap: int = 128,
) -> None:
    """
    Extract residue-level ProtT5 embeddings and save them as:

        output_dir/split_name/{pair_id}_peptide.pt
        output_dir/split_name/{pair_id}_protein.pt

    Each saved tensor has shape [sequence_length, 1024].

    This function requires:
        pip install transformers sentencepiece tqdm

    A GPU is strongly recommended because ProtT5-XL is large.
    """
    try:
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError as exc:
        raise ImportError(
            "ProtT5 extraction requires transformers and sentencepiece. Install with:\n"
            "pip install transformers sentencepiece tqdm"
        ) from exc

    records = parse_pair_fasta(fasta_path)
    save_dir = Path(output_dir) / split_name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ProtT5 model: {model_name}")
    print(f"Embedding device: {device}")

    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    if half_precision and device.type == "cuda":
        model.half()

    def embed_short_sequence(seq: str) -> torch.Tensor:
        seq = sanitize_protein_sequence(seq)
        spaced = " ".join(list(seq))
        encoded = tokenizer(
            spaced,
            add_special_tokens=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[0]

        # Remove the final special token. The remaining rows correspond to residues.
        residue_emb = hidden[: len(seq)].detach().float().cpu()
        return residue_emb

    def embed_one_sequence(seq: str) -> torch.Tensor:
        """
        Embed a complete sequence. Long proteins are embedded with overlapping
        windows and overlapping residue embeddings are averaged.

        This is more appropriate than truncating proteins at 1024 residues for
        this dataset, where the maximum protein length is above 4000 and the
        intended upper bound is approximately 5000 residues.
        """
        seq = sanitize_protein_sequence(seq)
        length = len(seq)

        if length <= prott5_chunk_size:
            return embed_short_sequence(seq)

        if prott5_chunk_overlap >= prott5_chunk_size:
            raise ValueError("prott5_chunk_overlap must be smaller than prott5_chunk_size.")

        step = prott5_chunk_size - prott5_chunk_overlap
        emb_sum: Optional[torch.Tensor] = None
        emb_count = torch.zeros(length, 1, dtype=torch.float32)

        starts = list(range(0, length, step))
        if starts[-1] + prott5_chunk_size < length:
            starts.append(max(0, length - prott5_chunk_size))

        for start in starts:
            end = min(start + prott5_chunk_size, length)
            chunk = seq[start:end]
            chunk_emb = embed_short_sequence(chunk)

            if emb_sum is None:
                emb_sum = torch.zeros(length, chunk_emb.size(1), dtype=torch.float32)

            emb_sum[start:end] += chunk_emb[: end - start]
            emb_count[start:end] += 1.0

            if end == length:
                break

        assert emb_sum is not None
        return emb_sum / emb_count.clamp_min(1.0)

    for record in tqdm(records, desc=f"Extracting ProtT5 embeddings for {split_name}"):
        pep_path = save_dir / f"{record.pair_id}_peptide.pt"
        prot_path = save_dir / f"{record.pair_id}_protein.pt"

        if pep_path.exists() and prot_path.exists():
            continue

        peptide_seq = record.peptide[:max_peptide_len] if max_peptide_len is not None else record.peptide
        protein_seq = record.protein[:max_protein_len] if max_protein_len is not None else record.protein

        peptide_emb = embed_one_sequence(peptide_seq)
        protein_emb = embed_one_sequence(protein_seq)

        torch.save(peptide_emb, pep_path)
        torch.save(protein_emb, prot_path)

    print(f"Saved ProtT5 embeddings to: {save_dir}")


# -----------------------------
# Dataset and collate
# -----------------------------


class PepProteinDataset(Dataset):
    def __init__(
        self,
        fasta_path: Optional[str | Path] = None,
        records: Optional[List[PairRecord]] = None,
        embedding_dir: Optional[str | Path] = None,
        split_name: Optional[str] = None,
        use_precomputed_embeddings: bool = True,
        max_protein_len: Optional[int] = 1024,
        max_peptide_len: Optional[int] = 128,
    ):
        if records is not None:
            self.records = records
        elif fasta_path is not None:
            self.records = parse_pair_fasta(fasta_path)
        else:
            raise ValueError("Either fasta_path or records must be provided.")
        self.embedding_dir = Path(embedding_dir) if embedding_dir is not None else None
        self.split_name = split_name
        self.use_precomputed_embeddings = use_precomputed_embeddings
        self.max_protein_len = max_protein_len
        self.max_peptide_len = max_peptide_len
        self.debug_encoder = None if use_precomputed_embeddings else DebugResidueEncoder(1024)

    def __len__(self) -> int:
        return len(self.records)

    def _embedding_path(self, pair_id: str, molecule: str) -> Path:
        if self.embedding_dir is None:
            raise ValueError("embedding_dir must be provided when use_precomputed_embeddings=True.")
        if self.split_name is not None:
            return self.embedding_dir / self.split_name / f"{pair_id}_{molecule}.pt"
        return self.embedding_dir / f"{pair_id}_{molecule}.pt"

    def _load_embeddings(self, record: PairRecord) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_precomputed_embeddings:
            pep_path = self._embedding_path(record.pair_id, "peptide")
            prot_path = self._embedding_path(record.pair_id, "protein")
            if not pep_path.exists():
                raise FileNotFoundError(f"Missing peptide embedding: {pep_path}")
            if not prot_path.exists():
                raise FileNotFoundError(f"Missing protein embedding: {prot_path}")
            peptide_emb = torch.load(pep_path, map_location="cpu").float()
            protein_emb = torch.load(prot_path, map_location="cpu").float()
        else:
            assert self.debug_encoder is not None
            peptide_emb = self.debug_encoder(record.peptide).float()
            protein_emb = self.debug_encoder(record.protein).float()

        if self.max_peptide_len is not None:
            peptide_emb = peptide_emb[: self.max_peptide_len]
        if self.max_protein_len is not None:
            protein_emb = protein_emb[: self.max_protein_len]
        return peptide_emb, protein_emb

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int]:
        record = self.records[idx]
        peptide_emb, protein_emb = self._load_embeddings(record)
        return {
            "pair_id": record.pair_id,
            "peptide_emb": peptide_emb,
            "protein_emb": protein_emb,
            "label": torch.tensor(record.label, dtype=torch.float32),
            "peptide_len": peptide_emb.shape[0],
            "protein_len": protein_emb.shape[0],
        }


def collate_pairs(batch: List[Dict[str, torch.Tensor | str | int]]) -> Dict[str, object]:
    return {
        "pair_ids": [x["pair_id"] for x in batch],
        "peptide_embs": [x["peptide_emb"] for x in batch],
        "protein_embs": [x["protein_emb"] for x in batch],
        "labels": torch.stack([x["label"] for x in batch]),
    }


def make_internal_train_val_split(
    fasta_path: str | Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[PairRecord], List[PairRecord]]:
    """
    Create a stratified train/validation split from the training FASTA file.

    The independent test FASTA files must not be used for early stopping or model
    selection. This function keeps the positive/negative ratio approximately
    identical in the internal training and validation subsets.
    """
    records = parse_pair_fasta(fasta_path)
    labels = [record.label for record in records]
    indices = np.arange(len(records))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
        shuffle=True,
    )

    train_records = [records[int(i)] for i in train_idx]
    val_records = [records[int(i)] for i in val_idx]
    return train_records, val_records


# -----------------------------
# Model components
# -----------------------------


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RelationAttentionLayer(nn.Module):
    """
    Lightweight heterogeneous relation-aware attention layer.

    For each target node, messages are aggregated from typed incoming edges.
    This avoids a torch-geometric dependency while still implementing relation-specific
    graph attention.
    """

    def __init__(self, hidden_dim: int, num_relations: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_relations)])
        self.v = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_relations)])
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        x: [N, D]
        edge_index: [2, E], source -> target
        edge_type: [E]
        """
        n_nodes = x.size(0)
        if edge_index.numel() == 0:
            return self.norm2(x + self.dropout(self.ffn(self.norm1(x))))

        src, dst = edge_index[0], edge_index[1]
        q_all = self.q(x).view(n_nodes, self.num_heads, self.head_dim)
        aggregated = torch.zeros_like(q_all)

        # Relation-specific attention. This loop is small because the number of relation types is small.
        for rel in range(self.num_relations):
            mask = edge_type == rel
            if not torch.any(mask):
                continue
            rel_src = src[mask]
            rel_dst = dst[mask]

            q = q_all[rel_dst]
            k = self.k[rel](x[rel_src]).view(-1, self.num_heads, self.head_dim)
            v = self.v[rel](x[rel_src]).view(-1, self.num_heads, self.head_dim)

            score = (q * k).sum(dim=-1) * self.scale  # [E_rel, H]

            # Stable per-target softmax implemented without torch_scatter.
            attn = torch.zeros_like(score)
            unique_dst = torch.unique(rel_dst)
            for node in unique_dst:
                local = rel_dst == node
                attn[local] = F.softmax(score[local], dim=0)

            msg = attn.unsqueeze(-1) * v
            aggregated.index_add_(0, rel_dst, msg)

        aggregated = aggregated.reshape(n_nodes, self.hidden_dim)
        x = self.norm1(x + self.dropout(self.out(aggregated)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class CrossAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, residue_nodes: torch.Tensor, condition_node: torch.Tensor) -> torch.Tensor:
        if residue_nodes.size(0) == 0:
            return condition_node
        q = self.query(condition_node).unsqueeze(0)
        k = self.key(residue_nodes)
        logits = self.score(torch.tanh(k + q)).squeeze(-1)
        weights = F.softmax(logits, dim=0)
        return torch.sum(weights.unsqueeze(-1) * residue_nodes, dim=0)


class PepRIHGT(nn.Module):
    """
    Residue-level dynamic interaction graph model.

    Node layout for each graph:
        [peptide residues] [protein residues] [peptide_global] [protein_global]
    """

    # Relation IDs
    PEP_SEQ = 0
    PROT_SEQ = 1
    PEP_GLOBAL = 2
    PROT_GLOBAL = 3
    PEP_TO_PROT = 4
    PROT_TO_PEP = 5
    SELF = 6
    NUM_RELATIONS = 7

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        top_k: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k = top_k

        self.peptide_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.protein_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.peptide_global_token = nn.Parameter(torch.zeros(hidden_dim))
        self.protein_global_token = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.peptide_global_token, std=0.02)
        nn.init.normal_(self.protein_global_token, std=0.02)

        self.contact_q = nn.Linear(hidden_dim, hidden_dim)
        self.contact_k = nn.Linear(hidden_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [
                RelationAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_relations=self.NUM_RELATIONS,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.pep_pool = CrossAttentionPooling(hidden_dim)
        self.prot_pool = CrossAttentionPooling(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _add_edge(self, edges: List[Tuple[int, int]], types: List[int], src: int, dst: int, rel: int) -> None:
        edges.append((src, dst))
        types.append(rel)

    def _build_graph(self, pep: torch.Tensor, prot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
        device = pep.device
        m, n = pep.size(0), prot.size(0)
        pep_global_idx = m + n
        prot_global_idx = m + n + 1

        nodes = torch.cat(
            [
                pep,
                prot,
                self.peptide_global_token.to(device).view(1, -1),
                self.protein_global_token.to(device).view(1, -1),
            ],
            dim=0,
        )

        edges: List[Tuple[int, int]] = []
        types: List[int] = []

        # Self-loops.
        for i in range(m + n + 2):
            self._add_edge(edges, types, i, i, self.SELF)

        # Intra-peptide sequential edges.
        for i in range(max(0, m - 1)):
            self._add_edge(edges, types, i, i + 1, self.PEP_SEQ)
            self._add_edge(edges, types, i + 1, i, self.PEP_SEQ)

        # Intra-protein sequential edges.
        prot_offset = m
        for j in range(max(0, n - 1)):
            a = prot_offset + j
            b = prot_offset + j + 1
            self._add_edge(edges, types, a, b, self.PROT_SEQ)
            self._add_edge(edges, types, b, a, self.PROT_SEQ)

        # Global edges.
        for i in range(m):
            self._add_edge(edges, types, i, pep_global_idx, self.PEP_GLOBAL)
            self._add_edge(edges, types, pep_global_idx, i, self.PEP_GLOBAL)
        for j in range(n):
            node = prot_offset + j
            self._add_edge(edges, types, node, prot_global_idx, self.PROT_GLOBAL)
            self._add_edge(edges, types, prot_global_idx, node, self.PROT_GLOBAL)

        # Dynamic peptide-protein candidate contact edges.
        q = self.contact_q(pep)
        k = self.contact_k(prot)
        contact_scores = torch.matmul(q, k.t()) / math.sqrt(self.hidden_dim)  # [m, n]
        k_keep = min(self.top_k, n)
        if m > 0 and n > 0 and k_keep > 0:
            top_idx = torch.topk(contact_scores, k=k_keep, dim=1).indices
            for i in range(m):
                for j in top_idx[i].tolist():
                    protein_node = prot_offset + j
                    self._add_edge(edges, types, i, protein_node, self.PEP_TO_PROT)
                    self._add_edge(edges, types, protein_node, i, self.PROT_TO_PEP)

        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        edge_type = torch.tensor(types, dtype=torch.long, device=device)
        index_info = {
            "m": m,
            "n": n,
            "pep_global_idx": pep_global_idx,
            "prot_global_idx": prot_global_idx,
        }
        return nodes, edge_index, edge_type, index_info

    def encode_pair(self, peptide_emb: torch.Tensor, protein_emb: torch.Tensor) -> torch.Tensor:
        pep = self.peptide_proj(peptide_emb)
        prot = self.protein_proj(protein_emb)
        x, edge_index, edge_type, info = self._build_graph(pep, prot)

        for layer in self.layers:
            x = layer(x, edge_index, edge_type)

        m = info["m"]
        n = info["n"]
        pep_nodes = x[:m]
        prot_nodes = x[m : m + n]
        pep_global = x[info["pep_global_idx"]]
        prot_global = x[info["prot_global_idx"]]

        z_pep = self.pep_pool(pep_nodes, prot_global)
        z_prot = self.prot_pool(prot_nodes, pep_global)
        z_pair = torch.cat([z_pep, z_prot, z_pep * z_prot, torch.abs(z_pep - z_prot)], dim=-1)
        return z_pair

    def forward(self, peptide_embs: List[torch.Tensor], protein_embs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_reprs = []
        for peptide_emb, protein_emb in zip(peptide_embs, protein_embs):
            pair_reprs.append(self.encode_pair(peptide_emb, protein_emb))
        pair_reprs = torch.stack(pair_reprs, dim=0)
        logits = self.classifier(pair_reprs).squeeze(-1)
        return logits, pair_reprs


# -----------------------------
# Losses and metrics
# -----------------------------


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.size(0) < 2:
            return torch.tensor(0.0, device=features.device)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(features.device)
        features = F.normalize(features, dim=1)
        logits = torch.div(torch.matmul(features, features.t()), self.temperature)
        logits = logits - torch.max(logits, dim=1, keepdim=True).values.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=features.device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positives_per_sample = mask.sum(dim=1)
        valid = positives_per_sample > 0
        if not torch.any(valid):
            return torch.tensor(0.0, device=features.device)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positives_per_sample + 1e-12)
        loss = -mean_log_prob_pos[valid].mean()
        return loss


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out: Dict[str, float] = {}

    out["acc"] = accuracy_score(y_true, y_pred)
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["recall_sensitivity"] = recall_score(y_true, y_pred, zero_division=0)
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)
    out["mcc"] = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0

    try:
        out["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        out["auc"] = float("nan")

    try:
        out["aupr"] = average_precision_score(y_true, y_prob)
    except ValueError:
        out["aupr"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["specificity"] = tn / (tn + fp + 1e-12)
    out["tn"] = float(tn)
    out["fp"] = float(fp)
    out["fn"] = float(fn)
    out["tp"] = float(tp)
    return out


def format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    keys = ["loss", "auc", "aupr", "acc", "f1", "mcc", "precision", "recall_sensitivity", "specificity"]
    parts = []
    for k in keys:
        if k in metrics:
            value = metrics[k]
            if isinstance(value, float):
                parts.append(f"{prefix}_{k}={value:.4f}")
            else:
                parts.append(f"{prefix}_{k}={value}")
    return " | ".join(parts)


def write_independent_report(results: Dict[str, Dict[str, float]], output_path: str | Path) -> None:
    """Write independent-test metrics to a tab-separated text report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    main_metrics = ["auc", "acc", "f1", "aupr", "mcc"]
    extended_metrics = [
        "loss",
        "auc",
        "aupr",
        "acc",
        "f1",
        "mcc",
        "precision",
        "recall_sensitivity",
        "specificity",
        "tp",
        "tn",
        "fp",
        "fn",
    ]

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Independent test results")
        handle.write("========================")
        handle.write("Main metrics: AUC, ACC, F1, AUPR, MCC")
        handle.write("dataset	" + "	".join(main_metrics) + " ")

        for dataset_name, metrics in results.items():
            row = [dataset_name]
            row.extend(f"{metrics.get(metric, float('nan')):.6f}" for metric in main_metrics)
            handle.write("	".join(row) + " ")

        handle.write("Extended metrics")
        handle.write("----------------")
        handle.write("dataset	" + "	".join(extended_metrics) + " ")

        for dataset_name, metrics in results.items():
            row = [dataset_name]
            row.extend(f"{metrics.get(metric, float('nan')):.6f}" for metric in extended_metrics)
            handle.write("	".join(row) + " ")


def plot_independent_results(results: Dict[str, Dict[str, float]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_to_plot = ["auc", "acc", "f1", "aupr", "mcc"]
    dataset_names = list(results.keys())

    x = np.arange(len(dataset_names))
    width = 0.15

    plt.figure(figsize=(max(10, len(dataset_names) * 1.8), 6))
    for idx, metric in enumerate(metrics_to_plot):
        values = [results[name].get(metric, np.nan) for name in dataset_names]
        offset = (idx - (len(metrics_to_plot) - 1) / 2) * width
        plt.bar(x + offset, values, width, label=metric.upper())

    plt.axhline(0.0, linewidth=0.8)
    plt.xticks(x, dataset_names, rotation=20, ha="right")
    plt.ylabel("Metric value")
    plt.ylim(-1.0, 1.0)
    plt.title("Independent test performance")
    plt.legend(ncol=len(metrics_to_plot), loc="upper center", bbox_to_anchor=(0.5, 1.15))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# -----------------------------
# Train/evaluate loops
# -----------------------------


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        "pair_ids": batch["pair_ids"],
        "peptide_embs": [x.to(device) for x in batch["peptide_embs"]],
        "protein_embs": [x.to(device) for x in batch["protein_embs"]],
        "labels": batch["labels"].to(device),
    }


def run_epoch(
    model: PepRIHGT,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    contrastive_loss_fn: Optional[SupervisedContrastiveLoss] = None,
    contrastive_weight: float = 0.0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    bce_loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_examples = 0
    all_probs: List[float] = []
    all_labels: List[int] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"]

        with torch.set_grad_enabled(is_train):
            logits, pair_reprs = model(batch["peptide_embs"], batch["protein_embs"])
            bce_loss = bce_loss_fn(logits, labels)
            loss = bce_loss

            if contrastive_loss_fn is not None and contrastive_weight > 0:
                supcon = contrastive_loss_fn(pair_reprs, labels.long())
                loss = loss + contrastive_weight * supcon

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics


def evaluate_independent_sets(
    model: PepRIHGT,
    fasta_paths: List[str],
    split_names: Optional[List[str]],
    embedding_dir: str | Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_protein_len: Optional[int],
    max_peptide_len: Optional[int],
) -> Dict[str, Dict[str, float]]:
    if split_names is None or len(split_names) == 0:
        split_names = [Path(path).stem for path in fasta_paths]
    if len(split_names) != len(fasta_paths):
        raise ValueError("--test_split_names must have the same length as --test_fastas.")

    results: Dict[str, Dict[str, float]] = {}
    for fasta_path, split_name in zip(fasta_paths, split_names):
        dataset = PepProteinDataset(
            fasta_path=fasta_path,
            embedding_dir=embedding_dir,
            split_name=split_name,
            use_precomputed_embeddings=True,
            max_protein_len=max_protein_len,
            max_peptide_len=max_peptide_len,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_pairs,
            pin_memory=torch.cuda.is_available(),
        )
        metrics = run_epoch(model=model, loader=loader, device=device, optimizer=None)
        results[Path(fasta_path).stem] = metrics
        print(f"Independent | {Path(fasta_path).stem} | {format_metrics('test', metrics)}")
    return results


class EarlyStopping:
    def __init__(self, patience: int = 15, mode: str = "max", min_delta: float = 1e-5):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
            return True

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fasta", type=str, default=None)
    parser.add_argument("--val_fasta", type=str, default=None)
    parser.add_argument("--internal_val_fraction", type=float, default=0.1)
    parser.add_argument("--use_internal_split", action="store_true")
    parser.add_argument("--embedding_dir", type=str, default=None)
    parser.add_argument("--train_split_name", type=str, default="train")
    parser.add_argument("--val_split_name", type=str, default="independent")
    parser.add_argument("--use_precomputed_embeddings", action="store_true")
    parser.add_argument("--max_protein_len", type=int, default=5000)
    parser.add_argument("--max_peptide_len", type=int, default=128)
    parser.add_argument("--input_dim", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--contrastive_weight", type=float, default=0.05)
    parser.add_argument("--contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--monitor", type=str, default="aupr", choices=["aupr", "auc", "mcc", "f1", "loss"])
    parser.add_argument("--checkpoint", type=str, default="best_pepri_hgt.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint_to_eval", type=str, default=None)
    parser.add_argument("--test_fastas", nargs="+", default=None)
    parser.add_argument("--test_split_names", nargs="+", default=None)
    parser.add_argument("--results_report", type=str, default="independent_results.txt")
    parser.add_argument("--results_plot", type=str, default="independent_metrics.png")

    # ProtT5 extraction mode. Use this before training to create residue-level .pt files.
    parser.add_argument("--extract_embeddings", action="store_true")
    parser.add_argument("--extract_fasta", type=str, default=None)
    parser.add_argument("--extract_split_name", type=str, default="train")
    parser.add_argument("--prott5_model_name", type=str, default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--prott5_chunk_size", type=int, default=1022)
    parser.add_argument("--prott5_chunk_overlap", type=int, default=128)
    parser.add_argument("--no_half_precision", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.extract_embeddings:
        if args.extract_fasta is None:
            raise ValueError("Please provide --extract_fasta when using --extract_embeddings.")
        if args.embedding_dir is None:
            raise ValueError("Please provide --embedding_dir when using --extract_embeddings.")
        extract_prott5_embeddings_for_fasta(
            fasta_path=args.extract_fasta,
            output_dir=args.embedding_dir,
            split_name=args.extract_split_name,
            model_name=args.prott5_model_name,
            half_precision=not args.no_half_precision,
            max_protein_len=args.max_protein_len,
            max_peptide_len=args.max_peptide_len,
            prott5_chunk_size=args.prott5_chunk_size,
            prott5_chunk_overlap=args.prott5_chunk_overlap,
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.eval_only:
        if args.test_fastas is None:
            raise ValueError("Please provide --test_fastas when using --eval_only.")
        if args.embedding_dir is None:
            raise ValueError("Please provide --embedding_dir when using --eval_only.")

        model = PepRIHGT(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            top_k=args.top_k,
            dropout=args.dropout,
        ).to(device)

        checkpoint_path = args.checkpoint_to_eval or args.checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        independent_results = evaluate_independent_sets(
            model=model,
            fasta_paths=args.test_fastas,
            split_names=args.test_split_names,
            embedding_dir=args.embedding_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_protein_len=args.max_protein_len,
            max_peptide_len=args.max_peptide_len,
        )
        write_independent_report(independent_results, args.results_report)
        plot_independent_results(independent_results, args.results_plot)
        print(f"Saved independent report: {args.results_report}")
        print(f"Saved independent plot: {args.results_plot}")
        return

    if args.train_fasta is None:
        raise ValueError(
            "Training requires --train_fasta. Independent FASTA files should be passed only with --test_fastas."
        )

    if args.use_internal_split or args.val_fasta is None:
        train_records, val_records = make_internal_train_val_split(
            fasta_path=args.train_fasta,
            val_fraction=args.internal_val_fraction,
            seed=args.seed,
        )
        train_dataset = PepProteinDataset(
            records=train_records,
            embedding_dir=args.embedding_dir,
            split_name=args.train_split_name,
            use_precomputed_embeddings=args.use_precomputed_embeddings,
            max_protein_len=args.max_protein_len,
            max_peptide_len=args.max_peptide_len,
        )
        val_dataset = PepProteinDataset(
            records=val_records,
            embedding_dir=args.embedding_dir,
            split_name=args.train_split_name,
            use_precomputed_embeddings=args.use_precomputed_embeddings,
            max_protein_len=args.max_protein_len,
            max_peptide_len=args.max_peptide_len,
        )
        print(
            f"Using internal stratified validation split from training FASTA: "
            f"val_fraction={args.internal_val_fraction:.3f}"
        )
    else:
        train_dataset = PepProteinDataset(
            fasta_path=args.train_fasta,
            embedding_dir=args.embedding_dir,
            split_name=args.train_split_name,
            use_precomputed_embeddings=args.use_precomputed_embeddings,
            max_protein_len=args.max_protein_len,
            max_peptide_len=args.max_peptide_len,
        )
        val_dataset = PepProteinDataset(
            fasta_path=args.val_fasta,
            embedding_dir=args.embedding_dir,
            split_name=args.val_split_name,
            use_precomputed_embeddings=args.use_precomputed_embeddings,
            max_protein_len=args.max_protein_len,
            max_peptide_len=args.max_peptide_len,
        )
        print("Using external validation FASTA. Do not use independent test sets here for final reporting.")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_pairs,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pairs,
        pin_memory=torch.cuda.is_available(),
    )

    model = PepRIHGT(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        top_k=args.top_k,
        dropout=args.dropout,
    ).to(device)

    if args.eval_only:
        checkpoint_path = args.checkpoint_to_eval or args.checkpoint
        if args.test_fastas is None:
            raise ValueError("Please provide --test_fastas when using --eval_only.")
        if args.embedding_dir is None:
            raise ValueError("Please provide --embedding_dir when using --eval_only.")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        independent_results = evaluate_independent_sets(
            model=model,
            fasta_paths=args.test_fastas,
            split_names=args.test_split_names,
            embedding_dir=args.embedding_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_protein_len=args.max_protein_len,
            max_peptide_len=args.max_peptide_len,
        )
        write_independent_report(independent_results, args.results_report)
        plot_independent_results(independent_results, args.results_plot)
        print(f"Saved independent report: {args.results_report}")
        print(f"Saved independent plot: {args.results_plot}")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if args.monitor == "loss" else "max",
        factor=0.5,
        patience=5,
    )
    contrastive_loss_fn = SupervisedContrastiveLoss(temperature=args.contrastive_temperature)

    early = EarlyStopping(
        patience=args.patience,
        mode="min" if args.monitor == "loss" else "max",
    )

    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            contrastive_loss_fn=contrastive_loss_fn,
            contrastive_weight=args.contrastive_weight,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            contrastive_loss_fn=None,
            contrastive_weight=0.0,
        )

        monitor_score = val_metrics[args.monitor]
        scheduler.step(monitor_score)
        improved = early.step(monitor_score)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | lr={lr_now:.2e} | "
            f"{format_metrics('train', train_metrics)} | "
            f"{format_metrics('val', val_metrics)}"
        )

        if improved:
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
            }
            torch.save(checkpoint, args.checkpoint)
            print(f"  Saved best checkpoint to {args.checkpoint} using val_{args.monitor}={monitor_score:.4f}")

        if early.should_stop:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
            break

    print(f"Training finished. Best epoch: {best_epoch}. Best checkpoint: {args.checkpoint}")

    if args.test_fastas is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        independent_results = evaluate_independent_sets(
            model=model,
            fasta_paths=args.test_fastas,
            split_names=args.test_split_names,
            embedding_dir=args.embedding_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_protein_len=args.max_protein_len,
            max_peptide_len=args.max_peptide_len,
        )
        write_independent_report(independent_results, args.results_report)
        plot_independent_results(independent_results, args.results_plot)
        print(f"Saved independent report: {args.results_report}")
        print(f"Saved independent plot: {args.results_plot}")


if __name__ == "__main__":
    main()
