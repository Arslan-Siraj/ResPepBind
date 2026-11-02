"""Inference-only architecture extracted from the finalized ResPepBind training code.

NO training loop, validation splitting, optimizer, or plotting code is included.
The model structure and PyTorch state_dict keys match the original source.
Only load checkpoints/embeddings you trust.
"""
from __future__ import annotations
import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score


def _load_tensor_file_safely(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        raise RuntimeError("This release requires PyTorch supporting weights_only=True")


def _untrusted_legacy_pickle_disabled():
    raise RuntimeError("Pickle-based embedding files are not supported by the public inference package")

def sanitize_id(x: str) -> str:
    x = str(x).strip()
    x = x.replace("/", "_")
    x = x.replace("\\", "_")
    return x


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
        exts = {".pt", ".pth", ".npy", ".npz", }
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
            obj = _load_tensor_file_safely(p)
        elif suffix in {}:
            with open(p, "rb") as f:
                obj = _untrusted_legacy_pickle_disabled()
        elif suffix == ".npz":
            npz = np.load(p, allow_pickle=False)
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
            obj = _load_tensor_file_safely(p)
        elif suffix == ".npy":
            obj = np.load(p, allow_pickle=False)
        elif suffix == ".npz":
            npz = np.load(p, allow_pickle=False)
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
        elif suffix in {}:
            with open(p, "rb") as f:
                obj = _untrusted_legacy_pickle_disabled()
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
        suffixes = [".pt", ".pth", ".npy", ".npz", ]

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
