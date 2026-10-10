from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score


SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
COMMON_PATH = SCRIPT_DIR / "semi_inductive_common_v8.txt"
OURS_ROOT = SCRIPT_DIR.parent
DEFAULT_COLDSTARTCPI_ROOT = Path("/mnt/ColdstartCPI")
DEFAULT_DATASET_DIR = OURS_ROOT.parent / "datasets"
DEFAULT_FEATURE_PATH = OURS_ROOT / "features" / "v1_features.pt"
DEFAULT_SPLIT_DIR = DEFAULT_DATASET_DIR
DEFAULT_RESULTS_ROOT = OURS_ROOT / "results"
FEATURE_MANIFEST_NAME = "v1_feature_manifest.json"
REFERENCE_LOCAL707_TRAINING_SHA256 = (
    "82804f0810bfcdbb72b0e6af2237d132bc83471b55939d92aa6b07689beef95f"
)
REFERENCE_LOCAL707_COMMON_SHA256 = (
    "fe86fefc52da14d537a608e0693d3341d0aed3599b0cddb85d28903a85335d3b"
)
EVALUATION_CACHE_PREDECESSOR_TRAINING_SHA256S = frozenset()
EVALUATION_CACHE_SCHEMA_VERSION = 1
EVALUATION_CACHE_SCORING_REVISION = "max_ratio_shared_resumable_probability_v1"
EVALUATION_SCORE_STREAM_FILES = {
    "val_positive": "val_positive.npy",
    "val_negative": "val_negative.npy",
    "test_positive": "test_positive.npy",
    "test_negative": "test_negative.npy",
}

ABLATION_SPECS = {
    "full": {
        "experiment": "ours_biosnap_fullpair_compound_cold_v1",
        "description": (
            "Current local707 full Ours/TIDE-DTI model on a deterministic BioSNAP "
            "full_pair.csv five-fold compound-cold rebuild."
        ),
    },
}


def load_module(path: Path, module_name: str):
    loader = importlib.machinery.SourceFileLoader(module_name, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if not COMMON_PATH.exists():
    raise FileNotFoundError(COMMON_PATH)
C = load_module(COMMON_PATH, "coldstartcpi_guided_common_v8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Current local707 Ours/TIDE-DTI architecture and training protocol on "
            "the BioSNAP full_pair.csv five-fold compound-cold rebuild."
        )
    )
    parser.add_argument("--ablation", default="full", choices=sorted(ABLATION_SPECS))
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument(
        "--hierarchical-feature-path",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
        help=(
            "BioSNAP stores fine-grained and hierarchical drug tensors in one bundle; "
            "the formal run therefore points both interfaces to v1_features.pt."
        ),
    )
    parser.add_argument("--split-cache-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional exact output directory. By default, results/<ablation experiment> is used.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument(
        "--evaluation-cache-chunk-size",
        type=int,
        default=512,
        help=(
            "Number of maximum-ratio evaluation pairs committed per resumable cache update. "
            "The effective value is aligned to --score-batch-size."
        ),
    )
    parser.add_argument(
        "--restart-evaluation-cache",
        action="store_true",
        help=(
            "Archive only an existing fold evaluation score cache and rebuild it; "
            "training checkpoints are preserved."
        ),
    )
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--validation-frequency", type=int, default=5)
    parser.add_argument("--hard-neg-pool", type=int, default=60000)
    parser.add_argument("--train-neg-ratio", type=float, default=2.0)
    parser.add_argument("--hard-to-random-ratio", type=float, default=2.0)
    parser.add_argument("--hard-neg-update-freq", type=int, default=10)
    parser.add_argument("--kl-lambda", type=float, default=1e-4)
    parser.add_argument("--pos-conf-lambda", type=float, default=0.05)
    parser.add_argument("--pos-weight", type=float, default=1.5)
    parser.add_argument("--bedroc-alpha", type=float, default=80.5)
    parser.add_argument(
        "--ef-fractions", nargs="*", type=float, default=[0.005, 0.01, 0.02, 0.05]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratios", nargs="*", type=int, default=[1, 2, 5, 10, 20, 50, 100])
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an incomplete fold from its last atomically saved epoch.",
    )
    parser.add_argument(
        "--restart-incomplete",
        action="store_true",
        help="Discard an incomplete epoch checkpoint and restart that fold from epoch 1.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate feature provenance and all rebuilt BioSNAP folds without training.",
    )
    parser.add_argument(
        "--allow-nonformal-settings",
        action="store_true",
        help="Permit separately labelled smoke/sensitivity settings; never use for formal results.",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


class FeatureStore:
    def __init__(self, bundle: dict, hierarchical: dict) -> None:
        self.drug_global = bundle["drug_global"].float().contiguous()
        self.drug_local = bundle["drug_local"].contiguous()
        self.drug_mask = bundle["drug_padding_mask"].bool().contiguous()
        self.protein_global = bundle["protein_global"].float().contiguous()
        self.protein_local = bundle["protein_local"].contiguous()
        self.protein_mask = bundle["protein_padding_mask"].bool().contiguous()
        self.ecfp = hierarchical["ecfp"].float().contiguous()
        self.descriptors_raw = hierarchical["descriptors"].float().contiguous()
        self.functional_group_ids = hierarchical["functional_group_ids"].long().contiguous()
        self.descriptor_mean = torch.zeros(self.descriptors_raw.size(1), dtype=torch.float32)
        self.descriptor_std = torch.ones(self.descriptors_raw.size(1), dtype=torch.float32)

    def fit_descriptor_scaler(self, train_drugs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.descriptors_raw[train_drugs.detach().cpu().long()]
        self.descriptor_mean = values.mean(dim=0)
        self.descriptor_std = values.std(dim=0, unbiased=False).clamp_min(1e-6)
        return self.descriptor_mean.clone(), self.descriptor_std.clone()

    def pair_batch(self, edge_index: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, ...]:
        cpu_edges = edge_index.detach().cpu().long()
        drug_index = cpu_edges[0]
        protein_index = cpu_edges[1]
        descriptors = (
            self.descriptors_raw[drug_index] - self.descriptor_mean
        ) / self.descriptor_std
        functional_group_ids = self.functional_group_ids[drug_index]
        return (
            self.drug_global[drug_index].to(device, non_blocking=True),
            self.drug_local[drug_index].to(device, dtype=torch.float32, non_blocking=True),
            self.drug_mask[drug_index].to(device, non_blocking=True),
            self.ecfp[drug_index].to(device, non_blocking=True),
            descriptors.to(device, non_blocking=True),
            functional_group_ids.to(device, non_blocking=True),
            functional_group_ids.eq(0).to(device, non_blocking=True),
            self.protein_global[protein_index].to(device, non_blocking=True),
            self.protein_local[protein_index].to(device, dtype=torch.float32, non_blocking=True),
            self.protein_mask[protein_index].to(device, non_blocking=True),
        )


class FeatureAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InducedFitEvidentialModel(nn.Module):
    def __init__(
        self,
        drug_dim: int,
        protein_dim: int,
        descriptor_dim: int,
        functional_group_vocab: int,
        drug_tokens: int,
        functional_group_tokens: int,
        protein_tokens: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("--hidden-dim must be divisible by --num-heads")
        self.drug_tokens = int(drug_tokens)
        self.functional_group_tokens = int(functional_group_tokens)
        self.protein_tokens = int(protein_tokens)
        self.drug_global_adapter = FeatureAdapter(drug_dim, hidden_dim, dropout)
        self.ecfp_adapter = nn.Sequential(
            nn.Linear(2048, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.descriptor_adapter = FeatureAdapter(descriptor_dim, hidden_dim, dropout)
        self.functional_group_embedding = nn.Embedding(
            functional_group_vocab, hidden_dim, padding_idx=0
        )
        self.functional_group_norm = nn.LayerNorm(hidden_dim)
        self.drug_feature_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 3),
            nn.Sigmoid(),
        )
        self.drug_feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.drug_local_adapter = FeatureAdapter(drug_dim, hidden_dim, dropout)
        self.protein_global_adapter = FeatureAdapter(protein_dim, hidden_dim, dropout)
        self.protein_local_adapter = FeatureAdapter(protein_dim, hidden_dim, dropout)
        self.type_embedding = nn.Embedding(5, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.joint_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.interaction = C.DeepGatedInteraction(hidden_dim)
        pair_dim = hidden_dim * 3
        self.decoder = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.evidence_generator = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Softplus(),
        )

        type_ids = torch.cat(
            [
                torch.zeros(1, dtype=torch.long),
                torch.ones(self.drug_tokens, dtype=torch.long),
                torch.full((self.functional_group_tokens,), 2, dtype=torch.long),
                torch.full((1,), 3, dtype=torch.long),
                torch.full((self.protein_tokens,), 4, dtype=torch.long),
            ]
        )
        self.register_buffer("type_ids", type_ids, persistent=False)

    def forward(
        self,
        drug_global: torch.Tensor,
        drug_local: torch.Tensor,
        drug_mask: torch.Tensor,
        ecfp: torch.Tensor,
        descriptors: torch.Tensor,
        functional_group_ids: torch.Tensor,
        functional_group_mask: torch.Tensor,
        protein_global: torch.Tensor,
        protein_local: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group_tokens = self.functional_group_norm(
            self.functional_group_embedding(functional_group_ids)
        )
        drug_sources = torch.cat(
            [
                self.drug_global_adapter(drug_global),
                self.ecfp_adapter(ecfp),
                self.descriptor_adapter(descriptors),
            ],
            dim=-1,
        )
        gated_sources = drug_sources * self.drug_feature_gate(drug_sources)
        drug_global_token = self.drug_feature_fusion(gated_sources).unsqueeze(1)
        drug_local_tokens = self.drug_local_adapter(drug_local)
        protein_global_token = self.protein_global_adapter(protein_global).unsqueeze(1)
        protein_local_tokens = self.protein_local_adapter(protein_local)
        tokens = torch.cat(
            [
                drug_global_token,
                drug_local_tokens,
                group_tokens,
                protein_global_token,
                protein_local_tokens,
            ],
            dim=1,
        )
        tokens = tokens + self.type_embedding(self.type_ids).unsqueeze(0)
        global_mask = torch.zeros((tokens.size(0), 1), dtype=torch.bool, device=tokens.device)
        padding_mask = torch.cat(
            [global_mask, drug_mask, functional_group_mask, global_mask, protein_mask], dim=1
        )
        encoded = self.final_norm(self.joint_encoder(tokens, src_key_padding_mask=padding_mask))

        drug_pair = encoded[:, 0]
        protein_pair = encoded[:, 1 + self.drug_tokens + self.functional_group_tokens]
        interaction = self.interaction(drug_pair, protein_pair)
        pair = torch.cat([drug_pair, protein_pair, interaction], dim=-1)
        logits = self.decoder(pair).squeeze(-1)
        alpha = self.evidence_generator(pair) + 1.0
        return logits, alpha


def normalized_ids(values) -> list[str]:
    return [str(value) for value in values]


def validate_inputs(feature_bundle: dict, hierarchical_bundle: dict) -> None:
    required = [
        "drug_global", "drug_local", "drug_padding_mask", "protein_global", "protein_local",
        "protein_padding_mask", "drug_ids", "protein_ids", "metadata",
    ]
    missing = [key for key in required if key not in feature_bundle]
    if missing:
        raise KeyError(f"Fine-grained feature bundle is missing keys: {missing}")
    hierarchical_required = [
        "ecfp", "descriptors", "functional_group_ids", "drug_ids", "metadata",
    ]
    hierarchical_missing = [
        key for key in hierarchical_required if key not in hierarchical_bundle
    ]
    if hierarchical_missing:
        raise KeyError(f"Hierarchical feature bundle is missing keys: {hierarchical_missing}")
    drug_ids = normalized_ids(feature_bundle["drug_ids"])
    protein_ids = normalized_ids(feature_bundle["protein_ids"])
    hierarchical_drug_ids = normalized_ids(hierarchical_bundle["drug_ids"])
    if len(drug_ids) != len(set(drug_ids)):
        raise ValueError("Fine-grained feature bundle contains duplicate drug IDs")
    if len(protein_ids) != len(set(protein_ids)):
        raise ValueError("Fine-grained feature bundle contains duplicate protein IDs")
    if hierarchical_drug_ids != drug_ids:
        raise ValueError(
            "Hierarchical and fine-grained drug feature rows use different entity orders"
        )
    metadata = feature_bundle["metadata"]
    if metadata.get("dataset") != "BioSNAP":
        raise ValueError(
            "Feature bundle is not BioSNAP: "
            f"dataset={metadata.get('dataset')!r}"
        )
    if metadata.get("uses_dti_labels") is not False:
        raise ValueError("Feature provenance does not prove label-free extraction")
    protein_metadata = metadata.get("protein", {})
    protein_source = str(protein_metadata.get("source", ""))
    protein_model = str(protein_metadata.get("model", "")).rstrip("/")
    uses_author_protbert = "ProtTransBertBFD1024" in protein_source
    uses_direct_protbert = (
        protein_source == "transformers"
        and protein_model.casefold() == "rostlab/prot_bert_bfd"
    )
    if not (uses_author_protbert or uses_direct_protbert):
        raise ValueError(
            "BioSNAP Ours requires either the ColdstartCPI author "
            "ProtTransBertBFD1024 pickle or a direct Rostlab/prot_bert_bfd Transformers "
            "extraction; bundle reports "
            f"source={protein_source!r}, model={protein_model!r}"
        )
    if hierarchical_bundle["metadata"].get("uses_dti_labels") is not False:
        raise ValueError("Hierarchical feature provenance does not prove label-free extraction")
    if normalized_ids(hierarchical_bundle.get("protein_ids", protein_ids)) != protein_ids:
        raise ValueError(
            "Hierarchical and fine-grained protein feature rows use different entity orders"
        )
    if int(feature_bundle["drug_global"].size(1)) != 300:
        raise ValueError("BioSNAP Ours requires 300-D Mol2Vec drug features")
    if int(feature_bundle["protein_global"].size(1)) != 1024:
        raise ValueError("BioSNAP Ours requires 1024-D ProtBert-BFD protein features")
    if int(feature_bundle["drug_local"].size(1)) != 100:
        raise ValueError("BioSNAP Ours requires 100 Mol2Vec local drug tokens")
    if int(feature_bundle["protein_local"].size(1)) != 32:
        raise ValueError("BioSNAP Ours requires 32 grouped protein-local tokens")
    if int(hierarchical_bundle["functional_group_ids"].size(1)) != 32:
        raise ValueError("BioSNAP Ours requires 32 functional-group token slots")
    if tuple(hierarchical_bundle["ecfp"].shape) != (len(drug_ids), 2048):
        raise ValueError("Expected ECFP shape [num_drugs, 2048]")
    drug_row_keys = ["drug_global", "drug_local", "drug_padding_mask"]
    protein_row_keys = ["protein_global", "protein_local", "protein_padding_mask"]
    hierarchical_row_keys = ["ecfp", "descriptors", "functional_group_ids"]
    for key in drug_row_keys:
        if int(feature_bundle[key].size(0)) != len(drug_ids):
            raise ValueError(f"{key} row count does not match drug_ids")
    for key in protein_row_keys:
        if int(feature_bundle[key].size(0)) != len(protein_ids):
            raise ValueError(f"{key} row count does not match protein_ids")
    for key in hierarchical_row_keys:
        if int(hierarchical_bundle[key].size(0)) != len(drug_ids):
            raise ValueError(f"{key} row count does not match drug_ids")
    if feature_bundle["drug_local"].ndim != 3 or feature_bundle["protein_local"].ndim != 3:
        raise ValueError("Local feature tensors must be three-dimensional")
    if feature_bundle["drug_padding_mask"].shape != feature_bundle["drug_local"].shape[:2]:
        raise ValueError("drug_padding_mask shape does not match drug_local tokens")
    if feature_bundle["protein_padding_mask"].shape != feature_bundle["protein_local"].shape[:2]:
        raise ValueError("protein_padding_mask shape does not match protein_local tokens")


def score_edges_batched(
    model: InducedFitEvidentialModel,
    feature_store: FeatureStore,
    edge_index: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for start in range(0, edge_index.size(1), batch_size):
            batch_edges = edge_index[:, start : start + batch_size]
            batch = feature_store.pair_batch(batch_edges, device)
            logits, _ = model(*batch)
            outputs.append(torch.sigmoid(logits))
    if not outputs:
        return torch.empty(0, device=device)
    return torch.cat(outputs)


def validation_aupr(
    model: InducedFitEvidentialModel,
    feature_store: FeatureStore,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    probabilities = score_edges_batched(model, feature_store, edge_index, device, batch_size)
    return float(average_precision_score(labels.cpu().numpy(), probabilities.cpu().numpy()))


def bedroc_score(labels: np.ndarray, scores: np.ndarray, alpha: float) -> float:
    try:
        from rdkit.ML.Scoring.Scoring import CalcBEDROC
    except ImportError as exc:
        raise ImportError("BEDROC evaluation requires RDKit") from exc
    order = np.argsort(-scores, kind="mergesort")
    ranked = [[float(scores[index]), int(labels[index])] for index in order]
    return float(CalcBEDROC(ranked, 1, alpha))


def enrichment_factor(
    labels: np.ndarray, scores: np.ndarray, fraction: float
) -> tuple[float, int, int]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("EF fractions must be in (0, 1]")
    total = int(labels.size)
    total_positives = int(labels.sum())
    if total == 0 or total_positives == 0:
        return float("nan"), 0, 0
    top_n = max(1, int(math.ceil(total * fraction)))
    top_indices = np.argsort(-scores, kind="mergesort")[:top_n]
    hits = int(labels[top_indices].sum())
    value = (hits / top_n) / (total_positives / total)
    return float(value), hits, top_n


def save_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def evaluation_cache_dir(save_dir: Path, fold: int) -> Path:
    return save_dir / f"evaluation_score_cache_fold{fold}"


def archive_evaluation_cache(cache_dir: Path, reason: str) -> Path | None:
    if not cache_dir.exists():
        return None
    archive_root = cache_dir.parent / "_archived_evaluation_caches"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = archive_root / f"{cache_dir.name}_{stamp}"
    cache_dir.replace(destination)
    print(f"Archived evaluation cache ({reason}) -> {destination}")
    return destination


def evaluation_stream_edges(split: dict, max_ratio: int) -> dict[str, torch.Tensor]:
    val_positive = split["pos_val"].cpu()
    test_positive = split["test_pos"].cpu()
    val_negative_count = int(val_positive.size(1) * max_ratio)
    test_negative_count = int(test_positive.size(1) * max_ratio)
    if val_negative_count > split["val_neg_pool"].size(1):
        raise ValueError(
            f"val_neg_pool has {split['val_neg_pool'].size(1)} negatives, "
            f"need {val_negative_count} for 1:{max_ratio}"
        )
    if test_negative_count > split["test_neg_pool"].size(1):
        raise ValueError(
            f"test_neg_pool has {split['test_neg_pool'].size(1)} negatives, "
            f"need {test_negative_count} for 1:{max_ratio}"
        )
    return {
        "val_positive": val_positive,
        "val_negative": split["val_neg_pool"][:, :val_negative_count].cpu(),
        "test_positive": test_positive,
        "test_negative": split["test_neg_pool"][:, :test_negative_count].cpu(),
    }


def evaluation_cache_signature(
    fold: int,
    model: InducedFitEvidentialModel,
    split_file: Path,
    streams: dict[str, torch.Tensor],
    args: argparse.Namespace,
    artifact_sha256: dict[str, str],
) -> dict:
    return {
        "schema_version": EVALUATION_CACHE_SCHEMA_VERSION,
        "scoring_revision": EVALUATION_CACHE_SCORING_REVISION,
        "experiment": ABLATION_SPECS[args.ablation]["experiment"],
        "fold": fold,
        "model_state_sha256": tensor_state_sha256(model.state_dict()),
        "feature_path": str(args.feature_path.resolve()),
        "feature_sha256": artifact_sha256["feature"],
        "hierarchical_feature_path": str(args.hierarchical_feature_path.resolve()),
        "hierarchical_feature_sha256": artifact_sha256["hierarchical_feature"],
        "feature_manifest_sha256": artifact_sha256["feature_manifest"],
        "split_path": str(split_file.resolve()),
        "split_sha256": file_sha256(split_file),
        "score_batch_size": args.score_batch_size,
        "maximum_ratio": max(args.eval_ratios),
        "stream_counts": {name: int(edges.size(1)) for name, edges in streams.items()},
    }


def initialize_evaluation_cache(
    cache_dir: Path,
    signature: dict,
    restart: bool,
) -> tuple[dict, dict[str, np.memmap], Path]:
    metadata_path = cache_dir / "metadata.json"
    if restart:
        archive_evaluation_cache(cache_dir, "--restart-evaluation-cache")

    if cache_dir.exists():
        if not metadata_path.exists():
            raise RuntimeError(
                f"Evaluation cache has no metadata: {cache_dir}. "
                "Use --restart-evaluation-cache to archive and rebuild it."
            )
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Cannot read evaluation cache metadata: {metadata_path}. "
                "Use --restart-evaluation-cache to archive and rebuild it."
            ) from error
        if metadata.get("signature") != signature:
            raise RuntimeError(
                f"Evaluation cache does not match the current split/model: {cache_dir}. "
                "Use --restart-evaluation-cache to archive and rebuild only evaluation scores."
            )
    else:
        cache_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "signature": signature,
            "completed": {name: 0 for name in EVALUATION_SCORE_STREAM_FILES},
            "complete": False,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        for name, filename in EVALUATION_SCORE_STREAM_FILES.items():
            count = int(signature["stream_counts"][name])
            values = np.lib.format.open_memmap(
                cache_dir / filename,
                mode="w+",
                dtype=np.float32,
                shape=(count,),
            )
            values[:] = np.nan
            values.flush()
            del values
        save_json(metadata, metadata_path)

    arrays: dict[str, np.memmap] = {}
    for name, filename in EVALUATION_SCORE_STREAM_FILES.items():
        path = cache_dir / filename
        expected_count = int(signature["stream_counts"][name])
        if not path.exists():
            raise RuntimeError(
                f"Evaluation cache stream is missing: {path}. "
                "Use --restart-evaluation-cache to archive and rebuild it."
            )
        values = np.load(path, mmap_mode="r+")
        if values.dtype != np.dtype(np.float32) or values.shape != (expected_count,):
            raise RuntimeError(
                f"Evaluation cache stream shape/dtype mismatch: {path}; "
                f"expected float32[{expected_count}], got {values.dtype}{values.shape}. "
                "Use --restart-evaluation-cache to rebuild it."
            )
        completed = int(metadata.get("completed", {}).get(name, -1))
        if completed < 0 or completed > expected_count:
            raise RuntimeError(f"Invalid completed count for {name}: {completed}/{expected_count}")
        completed_values = np.asarray(values[:completed])
        if completed and (
            not np.isfinite(completed_values).all()
            or completed_values.min() < -1e-6
            or completed_values.max() > 1.0 + 1e-6
        ):
            raise RuntimeError(
                f"Completed evaluation cache prefix is invalid for {name}: {path}. "
                "Use --restart-evaluation-cache to rebuild it."
            )
        arrays[name] = values
    return metadata, arrays, metadata_path


def score_evaluation_streams_resumable(
    fold: int,
    model: InducedFitEvidentialModel,
    feature_store: FeatureStore,
    split: dict,
    split_file: Path,
    save_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
    artifact_sha256: dict[str, str],
) -> dict[str, np.ndarray]:
    max_ratio = max(args.eval_ratios)
    streams = evaluation_stream_edges(split, max_ratio)
    signature = evaluation_cache_signature(
        fold,
        model,
        split_file,
        streams,
        args,
        artifact_sha256,
    )
    cache_dir = evaluation_cache_dir(save_dir, fold)
    metadata, arrays, metadata_path = initialize_evaluation_cache(
        cache_dir,
        signature,
        args.restart_evaluation_cache,
    )

    requested_chunk = max(args.score_batch_size, args.evaluation_cache_chunk_size)
    chunk_size = max(
        args.score_batch_size,
        (requested_chunk // args.score_batch_size) * args.score_batch_size,
    )
    if chunk_size != args.evaluation_cache_chunk_size:
        print(
            f"Fold {fold}: evaluation cache chunk adjusted from "
            f"{args.evaluation_cache_chunk_size} to {chunk_size} to preserve score-batch boundaries"
        )

    resumed = {name: int(metadata["completed"][name]) for name in EVALUATION_SCORE_STREAM_FILES}
    if any(resumed.values()):
        summary = ", ".join(
            f"{name}={resumed[name]}/{streams[name].size(1)}"
            for name in EVALUATION_SCORE_STREAM_FILES
        )
        print(f"Fold {fold}: resumed evaluation score cache [{summary}]")

    for name in EVALUATION_SCORE_STREAM_FILES:
        edges = streams[name]
        total = int(edges.size(1))
        start = int(metadata["completed"][name])
        while start < total:
            end = min(start + chunk_size, total)
            probabilities = (
                score_edges_batched(
                    model,
                    feature_store,
                    edges[:, start:end],
                    device,
                    args.score_batch_size,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            if probabilities.shape != (end - start,):
                raise RuntimeError(
                    f"Evaluation scorer returned {probabilities.shape} for {name}[{start}:{end}]"
                )
            if (
                not np.isfinite(probabilities).all()
                or probabilities.min() < -1e-6
                or probabilities.max() > 1.0 + 1e-6
            ):
                raise RuntimeError(f"Evaluation scorer produced invalid probabilities for {name}")
            arrays[name][start:end] = np.clip(probabilities, 0.0, 1.0)
            arrays[name].flush()
            metadata["completed"][name] = end
            metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")
            save_json(metadata, metadata_path)
            start = end
            print(
                f"Fold {fold}: evaluation cache {name} "
                f"{end}/{total} ({100.0 * end / max(total, 1):.1f}%) saved"
            )

    metadata["complete"] = True
    metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_json(metadata, metadata_path)
    scores = {name: np.asarray(values).copy() for name, values in arrays.items()}
    arrays.clear()
    print(
        f"Fold {fold}: complete 1:{max_ratio} evaluation score cache ready; "
        "all ratios/AUPR/AUC/BEDROC/EF will reuse it"
    )
    return scores


def evaluate_ratios_from_cached_scores(
    scores: dict[str, np.ndarray],
    eval_ratios: list[int],
    bedroc_alpha: float,
    ef_fractions: list[float],
) -> pd.DataFrame:
    val_positive = scores["val_positive"]
    val_negative_all = scores["val_negative"]
    test_positive = scores["test_positive"]
    test_negative_all = scores["test_negative"]
    rows: list[dict] = []
    for ratio in eval_ratios:
        val_negative_count = int(val_positive.size * ratio)
        test_negative_count = int(test_positive.size * ratio)
        if val_negative_count > val_negative_all.size:
            raise ValueError(
                f"Cached validation negatives cannot support 1:{ratio}: "
                f"{val_negative_all.size} < {val_negative_count}"
            )
        if test_negative_count > test_negative_all.size:
            raise ValueError(
                f"Cached test negatives cannot support 1:{ratio}: "
                f"{test_negative_all.size} < {test_negative_count}"
            )
        val_negative = val_negative_all[:val_negative_count]
        test_negative = test_negative_all[:test_negative_count]
        val_probabilities = np.concatenate([val_positive, val_negative])
        val_labels = np.concatenate(
            [np.ones(val_positive.size), np.zeros(val_negative_count)]
        )
        test_probabilities = np.concatenate([test_positive, test_negative])
        test_labels = np.concatenate(
            [np.ones(test_positive.size), np.zeros(test_negative_count)]
        )
        threshold = C.find_best_threshold_fbeta(
            val_labels,
            val_probabilities,
            beta=0.5,
        )
        metrics = C.compute_metrics(
            test_labels,
            test_probabilities,
            threshold=threshold,
        )
        metrics["Ratio"] = f"1:{ratio}"
        metrics["Opt_Thresh"] = threshold
        metrics[f"BEDROC_alpha_{bedroc_alpha:g}"] = bedroc_score(
            test_labels,
            test_probabilities,
            bedroc_alpha,
        )
        for fraction in ef_fractions:
            ef, hits, top_n = enrichment_factor(
                test_labels,
                test_probabilities,
                fraction,
            )
            label = f"{fraction * 100:g}pct"
            metrics[f"EF_{label}"] = ef
            metrics[f"EF_hits_{label}"] = hits
            metrics[f"EF_topN_{label}"] = top_n
        rows.append(metrics)
        print(
            f"  Ratio 1:{ratio:<4} | Thr={threshold:.3f} | "
            f"AUPR={metrics['AUPR']:.4f} | AUC={metrics['AUC']:.4f} | "
            f"BEDROC={metrics[f'BEDROC_alpha_{bedroc_alpha:g}']:.4f}"
        )
    return pd.DataFrame(rows)


def load_fold_files(split_dir: Path, selected_folds: list[int] | None) -> list[Path]:
    paths = sorted(split_dir.glob("split_fold*.pt"), key=C.fold_number_from_path)
    if not paths:
        raise FileNotFoundError(f"No split_fold*.pt files found in {split_dir}")
    if selected_folds:
        wanted = set(selected_folds)
        paths = [path for path in paths if C.fold_number_from_path(path) in wanted]
        missing = wanted - {C.fold_number_from_path(path) for path in paths}
        if missing:
            raise ValueError(f"Requested folds are missing from split cache: {sorted(missing)}")
    return paths


def reconstruct_global_positive_set(split_files: list[Path]) -> set[tuple[int, int]]:
    """Read the canonical BioSNAP global-positive tensor and require five-fold agreement."""
    global_positive_set: set[tuple[int, int]] | None = None
    for split_file in split_files:
        split = torch_load(split_file)
        if "global_pos_edges" not in split:
            raise KeyError(f"{split_file} is not a V1 split: missing global_pos_edges")
        current = C.edge_set(split["global_pos_edges"])
        if not current:
            raise ValueError(f"{split_file} contains an empty global positive set")
        if global_positive_set is None:
            global_positive_set = current
        elif current != global_positive_set:
            raise ValueError("BioSNAP folds disagree on global_pos_edges")
    if global_positive_set is None:
        raise ValueError("No V1 split files were loaded")
    return global_positive_set


def validate_crossfold_compound_assignment(split_files: list[Path]) -> None:
    drug_universe: set[int] | None = None
    test_sets: list[set[int]] = []
    for split_file in split_files:
        split = torch_load(split_file)
        train = set(map(int, split["train_drugs"].tolist()))
        validation = set(map(int, split["val_drugs"].tolist()))
        test = set(map(int, split["test_drugs"].tolist()))
        current_universe = train | validation | test
        if drug_universe is None:
            drug_universe = current_universe
        elif current_universe != drug_universe:
            raise ValueError("BioSNAP folds disagree on the full_pair drug universe")
        if any(test & previous for previous in test_sets):
            raise ValueError("BioSNAP rebuilt test-drug folds overlap")
        test_sets.append(test)
    if drug_universe is None or set().union(*test_sets) != drug_universe:
        raise ValueError("BioSNAP test folds do not cover every full_pair drug exactly once")


def validate_split_feature_indices(split: dict, num_drugs: int, num_proteins: int, fold: int) -> None:
    edge_keys = [
        "pos_tr", "pos_val", "test_pos", "neg_tr", "neg_val_es",
        "val_neg_pool", "test_neg_pool", "global_pos_edges",
    ]
    for key in edge_keys:
        edges = split[key]
        if edges.ndim != 2 or edges.size(0) != 2:
            raise ValueError(f"Fold {fold} {key} must have shape [2, N]")
        if edges.numel() == 0:
            continue
        if int(edges[0].min()) < 0 or int(edges[0].max()) >= num_drugs:
            raise ValueError(f"Fold {fold} {key} contains a drug index outside feature rows")
        if int(edges[1].min()) < 0 or int(edges[1].max()) >= num_proteins:
            raise ValueError(f"Fold {fold} {key} contains a protein index outside feature rows")


def require_edge_scope(
    name: str,
    edges: torch.Tensor,
    drugs: set[int],
    proteins: set[int],
) -> None:
    if edges.ndim != 2 or edges.size(0) != 2:
        raise ValueError(f"{name} must have shape [2,N], got {tuple(edges.shape)}")
    bad = [
        (int(drug), int(protein))
        for drug, protein in zip(edges[0].tolist(), edges[1].tolist())
        if int(drug) not in drugs or int(protein) not in proteins
    ]
    if bad:
        raise ValueError(f"{name} contains {len(bad)} edges outside its V1 entity scope")


def validate_biosnap_split(
    split: dict,
    fold: int,
    feature_bundle: dict,
    global_pos_set: set[tuple[int, int]],
    max_ratio: int,
) -> None:
    required = [
        "schema_version", "dataset", "fold", "protocol", "pos_tr", "pos_val", "test_pos", "neg_tr",
        "neg_val_es", "val_neg_pool", "test_neg_pool", "train_drugs",
        "val_drugs", "test_drugs", "train_proteins", "evaluation_proteins",
        "global_pos_edges", "drug_ids_sha256", "protein_ids_sha256",
    ]
    missing = [key for key in required if key not in split]
    if missing:
        raise KeyError(f"split_fold{fold} missing V1 keys: {missing}")
    if split["dataset"] != "BioSNAP" or int(split["fold"]) != fold:
        raise ValueError(f"split_fold{fold} has incorrect dataset/fold metadata")
    if int(split["schema_version"]) < 2 or "full_pair.csv" not in str(split["protocol"]):
        raise ValueError(f"split_fold{fold} is not the rebuilt full_pair protocol")
    if bool(split.get("strict_protein_known", False)):
        raise ValueError(
            f"split_fold{fold} unexpectedly declares strict_protein_known=true; "
            "the locked BioSNAP protocol uses false"
        )
    metadata = feature_bundle["metadata"]
    if str(split["drug_ids_sha256"]) != str(metadata.get("drug_ids_sha256")):
        raise ValueError(f"split_fold{fold} drug entity order differs from the feature bundle")
    if str(split["protein_ids_sha256"]) != str(metadata.get("protein_ids_sha256")):
        raise ValueError(f"split_fold{fold} protein entity order differs from the feature bundle")

    train_drugs = set(map(int, split["train_drugs"].tolist()))
    val_drugs = set(map(int, split["val_drugs"].tolist()))
    test_drugs = set(map(int, split["test_drugs"].tolist()))
    train_proteins = set(map(int, split["train_proteins"].tolist()))
    evaluation_proteins = set(map(int, split["evaluation_proteins"].tolist()))
    if train_drugs & val_drugs or train_drugs & test_drugs or val_drugs & test_drugs:
        raise ValueError(f"split_fold{fold} rebuilt train/validation/test drugs overlap")
    if not train_proteins.issubset(evaluation_proteins):
        raise ValueError(f"split_fold{fold} train proteins are not an evaluation subset")

    require_edge_scope("pos_tr", split["pos_tr"], train_drugs, train_proteins)
    require_edge_scope("neg_tr", split["neg_tr"], train_drugs, train_proteins)
    require_edge_scope("pos_val", split["pos_val"], val_drugs, evaluation_proteins)
    require_edge_scope("neg_val_es", split["neg_val_es"], val_drugs, evaluation_proteins)
    require_edge_scope("val_neg_pool", split["val_neg_pool"], val_drugs, evaluation_proteins)
    require_edge_scope("test_pos", split["test_pos"], test_drugs, evaluation_proteins)
    require_edge_scope("test_neg_pool", split["test_neg_pool"], test_drugs, evaluation_proteins)

    for name in ("pos_tr", "pos_val", "test_pos"):
        if split[name].size(1) == 0:
            raise ValueError(f"split_fold{fold} {name} is empty")
    for name in ("neg_tr", "neg_val_es", "val_neg_pool", "test_neg_pool"):
        pairs = C.edge_set(split[name])
        if len(pairs) != split[name].size(1):
            raise ValueError(f"split_fold{fold} {name} contains duplicates")
        if pairs & global_pos_set:
            raise ValueError(f"split_fold{fold} {name} collides with known positives")
    if C.edge_set(split["val_neg_pool"]) & C.edge_set(split["test_neg_pool"]):
        raise ValueError(f"split_fold{fold} validation/test negative pools overlap")
    expected_train_neg = int(split["pos_tr"].size(1) * 2)
    if int(split["neg_tr"].size(1)) != expected_train_neg:
        raise ValueError(
            f"split_fold{fold} initial training ratio is not 1:2: "
            f"pos={split['pos_tr'].size(1)}, neg={split['neg_tr'].size(1)}"
        )
    if split["val_neg_pool"].size(1) < split["pos_val"].size(1) * max_ratio:
        raise ValueError(f"split_fold{fold} validation pool cannot support 1:{max_ratio}")
    if split["test_neg_pool"].size(1) < split["test_pos"].size(1) * max_ratio:
        raise ValueError(f"split_fold{fold} test pool cannot support 1:{max_ratio}")


def audit_split_sources(split_file: Path, dataset_dir: Path, feature_sha256: str) -> dict:
    fold = C.fold_number_from_path(split_file)
    manifest_path = split_file.with_name(f"split_fold{fold}_manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("dataset") != "BioSNAP" or int(manifest.get("fold", -1)) != fold:
        raise ValueError(f"Incorrect V1 split manifest identity: {manifest_path}")
    expected_feature_sha256 = manifest.get("source", {}).get("feature_sha256")
    if expected_feature_sha256 != feature_sha256:
        raise ValueError(
            f"V1 split fold {fold} was built against a different feature package: "
            f"expected={expected_feature_sha256}, actual={feature_sha256}"
        )
    protocol = manifest.get("protocol", {})
    if (
        protocol.get("outer_split_source") != "full_pair.csv"
        or protocol.get("author_split_pair_assignments_reused") is not False
        or protocol.get("train_validation_test_drug_sets_are_disjoint") is not True
    ):
        raise ValueError(f"Fold {fold} is not the locked full_pair compound-cold rebuild")
    full_pair_path = dataset_dir.resolve() / "full_pair.csv"
    if not full_pair_path.is_file():
        raise FileNotFoundError(full_pair_path)
    actual_full_pair_sha256 = file_sha256(full_pair_path)
    if manifest.get("source", {}).get("full_pair_sha256") != actual_full_pair_sha256:
        raise ValueError(f"Full-pair source hash mismatch for fold {fold}: {full_pair_path}")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "full_pair_sha256": actual_full_pair_sha256,
        "test_drug_ids_sha256": protocol.get("test_drug_ids_sha256"),
    }


def collect_saved_results(save_dir: Path) -> tuple[list[dict], list[pd.DataFrame]]:
    records = []
    ratios = []
    for record_path in sorted(save_dir.glob("fold*_record.json")):
        with record_path.open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    for ratio_path in sorted(save_dir.glob("fold*_ratios.csv")):
        ratios.append(pd.read_csv(ratio_path))
    return records, ratios


def save_extended_ratio_summary(
    save_dir: Path, ratio_dfs: list[pd.DataFrame], experiment_name: str
) -> None:
    all_ratios = pd.concat(ratio_dfs, ignore_index=True)
    numeric_columns = [
        column
        for column in all_ratios.select_dtypes(include=[np.number]).columns
        if column != "Fold"
    ]
    summary = all_ratios.groupby("Ratio", sort=False)[numeric_columns].agg(["mean", "std"])
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    summary.reset_index().to_csv(
        save_dir / f"{experiment_name}_extended_summary.csv", index=False
    )


def dynamic_negative_counts(
    positive_count: int,
    train_neg_ratio: float,
    hard_to_random_ratio: float,
    use_hard_negatives: bool,
) -> tuple[int, int, int]:
    total_negative_count = int(round(positive_count * train_neg_ratio))
    if total_negative_count < 1:
        raise ValueError("The ratio-based dynamic negative count must be positive")
    if not use_hard_negatives:
        return total_negative_count, 0, total_negative_count
    random_negative_count = max(
        1, int(round(total_negative_count / (hard_to_random_ratio + 1.0)))
    )
    hard_negative_count = total_negative_count - random_negative_count
    return total_negative_count, hard_negative_count, random_negative_count


def atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def cpu_clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_clone_tree(item) for item in value)
    return copy.deepcopy(value)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def build_resume_signature(
    args: argparse.Namespace,
    fold: int,
    split_file: Path,
    effective_kl_lambda: float,
    effective_pos_conf_lambda: float,
    artifact_sha256: dict[str, str],
) -> dict:
    return {
        "schema_version": 2,
        "dataset": "BioSNAP",
        "outer_split": (
            "full_pair.csv deterministic five-fold compound-cold rebuild; disjoint "
            "train/validation/test drugs; strict_protein_known=false"
        ),
        "ablation": args.ablation,
        "fold": fold,
        "seed": args.seed,
        "feature_path": str(args.feature_path.resolve()),
        "hierarchical_feature_path": str(args.hierarchical_feature_path.resolve()),
        "split_path": str(split_file.resolve()),
        "feature_sha256": artifact_sha256["feature"],
        "hierarchical_feature_sha256": artifact_sha256["hierarchical_feature"],
        "feature_manifest_sha256": artifact_sha256["feature_manifest"],
        "training_code_sha256": artifact_sha256["training_code"],
        "common_code_sha256": artifact_sha256["common_code"],
        "split_sha256": file_sha256(split_file),
        "split_manifest_sha256": file_sha256(
            split_file.with_name(f"split_fold{fold}_manifest.json")
        ),
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "score_batch_size": args.score_batch_size,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "validation_frequency": args.validation_frequency,
        "hard_neg_pool": args.hard_neg_pool,
        "train_neg_ratio": args.train_neg_ratio,
        "hard_to_random_ratio": args.hard_to_random_ratio,
        "hard_neg_update_freq": args.hard_neg_update_freq,
        "effective_kl_lambda": effective_kl_lambda,
        "effective_pos_conf_lambda": effective_pos_conf_lambda,
        "pos_weight": args.pos_weight,
    }


def validate_resume_signature(saved: dict, current: dict, path: Path) -> None:
    if saved == current:
        return
    differences = {
        key: {"saved": saved.get(key), "current": current.get(key)}
        for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
    }
    evaluation_cache_upgrade_resume = (
        set(differences) == {"training_code_sha256"}
        and saved.get("training_code_sha256")
        in EVALUATION_CACHE_PREDECESSOR_TRAINING_SHA256S
    )
    if evaluation_cache_upgrade_resume:
        print(
            "EVALUATION-CACHE resume: accepting the exact pre-cache runner; "
            "all scientific training settings match the saved checkpoint."
        )
        return
    raise ValueError(
        f"Epoch checkpoint settings do not match the current command: {path}\n"
        + json.dumps(differences, indent=2, ensure_ascii=True)
    )


FORMAL_SETTINGS = {
    "ablation": "full",
    "hidden_dim": 128,
    "num_heads": 4,
    "num_layers": 1,
    "dropout": 0.1,
    "batch_size": 32,
    "score_batch_size": 64,
    "evaluation_cache_chunk_size": 512,
    "num_epochs": 100,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "patience": 15,
    "validation_frequency": 5,
    "hard_neg_pool": 60000,
    "train_neg_ratio": 2.0,
    "hard_to_random_ratio": 2.0,
    "hard_neg_update_freq": 10,
    "kl_lambda": 1e-4,
    "pos_conf_lambda": 0.05,
    "pos_weight": 1.5,
    "bedroc_alpha": 80.5,
    "ef_fractions": [0.005, 0.01, 0.02, 0.05],
    "seed": 42,
    "eval_ratios": [1, 2, 5, 10, 20, 50, 100],
}


def validate_arguments(args: argparse.Namespace) -> None:
    if args.resume and args.restart_incomplete:
        raise ValueError("--resume and --restart-incomplete cannot be used together")
    if args.batch_size < 1 or args.score_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.evaluation_cache_chunk_size < 1:
        raise ValueError("Evaluation cache chunk size must be positive")
    if args.validation_frequency < 1 or args.hard_neg_update_freq < 1:
        raise ValueError("Validation and hard-negative update frequencies must be positive")
    if args.train_neg_ratio <= 0 or args.hard_to_random_ratio <= 0:
        raise ValueError("Negative-sampling ratios must be positive")
    if not args.ef_fractions or not args.eval_ratios:
        raise ValueError("Evaluation ratios and EF fractions cannot be empty")
    if args.folds is not None:
        invalid = sorted(set(args.folds) - set(range(5)))
        if invalid:
            raise ValueError(f"BioSNAP formal folds are 0..4; invalid={invalid}")
        if len(args.folds) != len(set(args.folds)):
            raise ValueError("--folds contains duplicates")

    mismatches = {
        key: {"expected": expected, "actual": getattr(args, key)}
        for key, expected in FORMAL_SETTINGS.items()
        if getattr(args, key) != expected
    }
    if args.feature_path.resolve() != args.hierarchical_feature_path.resolve():
        mismatches["shared_feature_bundle"] = {
            "expected": str(args.feature_path.resolve()),
            "actual": str(args.hierarchical_feature_path.resolve()),
        }
    if mismatches and not args.allow_nonformal_settings:
        raise ValueError(
            "Formal BioSNAP Ours must preserve the locked local707-matched settings; "
            f"mismatches={mismatches}. Use --allow-nonformal-settings only for a separately "
            "labelled smoke/sensitivity run."
        )
    if mismatches:
        print("WARNING: nonformal settings enabled: " + json.dumps(mismatches, ensure_ascii=True))


def main() -> None:
    args = parse_args()
    validate_arguments(args)
    ablation_spec = ABLATION_SPECS[args.ablation]
    experiment_name = ablation_spec["experiment"]
    if args.save_dir is None:
        args.save_dir = DEFAULT_RESULTS_ROOT / experiment_name
    for path in [
        args.feature_path,
        args.hierarchical_feature_path,
        args.split_cache_dir,
        args.dataset_dir,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    C.set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    feature_bundle = torch_load(args.feature_path)
    hierarchical_bundle = (
        feature_bundle
        if args.feature_path.resolve() == args.hierarchical_feature_path.resolve()
        else torch_load(args.hierarchical_feature_path)
    )
    validate_inputs(feature_bundle, hierarchical_bundle)
    feature_manifest_path = args.feature_path.with_name(FEATURE_MANIFEST_NAME)
    if not feature_manifest_path.exists():
        raise FileNotFoundError(
            f"Feature manifest is required for provenance validation: {feature_manifest_path}"
        )
    with feature_manifest_path.open("r", encoding="utf-8") as handle:
        feature_manifest = json.load(handle)
    if feature_manifest.get("dataset") != "BioSNAP":
        raise ValueError("Feature manifest is not BioSNAP")
    for key in ("uses_dti_labels", "drug_ids_sha256", "protein_ids_sha256"):
        if feature_manifest.get(key) != feature_bundle["metadata"].get(key):
            raise ValueError(f"Feature bundle and manifest disagree on {key}")
    training_code_path = Path(__file__).resolve()
    artifact_sha256 = {
        "feature": file_sha256(args.feature_path),
        "hierarchical_feature": file_sha256(args.hierarchical_feature_path),
        "feature_manifest": file_sha256(feature_manifest_path),
        "training_code": file_sha256(training_code_path),
        "common_code": file_sha256(COMMON_PATH),
    }
    if artifact_sha256["common_code"] != REFERENCE_LOCAL707_COMMON_SHA256:
        raise ValueError(
            "Bundled semi_inductive_common_v8.txt differs from the current local707 Ours helper: "
            f"expected={REFERENCE_LOCAL707_COMMON_SHA256}, "
            f"actual={artifact_sha256['common_code']}"
        )
    feature_store = FeatureStore(feature_bundle, hierarchical_bundle)
    all_split_files = load_fold_files(args.split_cache_dir, None)
    all_fold_ids = [C.fold_number_from_path(path) for path in all_split_files]
    if all_fold_ids != [0, 1, 2, 3, 4]:
        raise ValueError(f"Expected exact BioSNAP folds [0,1,2,3,4], got {all_fold_ids}")
    global_pos_set = reconstruct_global_positive_set(all_split_files)
    validate_crossfold_compound_assignment(all_split_files)
    split_files = load_fold_files(args.split_cache_dir, args.folds)
    effective_kl_lambda = args.kl_lambda
    effective_pos_conf_lambda = args.pos_conf_lambda

    split_source_audits: dict[str, dict] = {}
    for split_file in all_split_files:
        fold = C.fold_number_from_path(split_file)
        split = torch_load(split_file)
        validate_split_feature_indices(
            split,
            num_drugs=len(feature_bundle["drug_ids"]),
            num_proteins=len(feature_bundle["protein_ids"]),
            fold=fold,
        )
        validate_biosnap_split(
            split, fold, feature_bundle, global_pos_set, max(args.eval_ratios)
        )
        split_source_audits[str(fold)] = audit_split_sources(
            split_file, args.dataset_dir, artifact_sha256["feature"]
        )
        dynamic_total, _, _ = dynamic_negative_counts(
            positive_count=int(split["pos_tr"].size(1)),
            train_neg_ratio=args.train_neg_ratio,
            hard_to_random_ratio=args.hard_to_random_ratio,
            use_hard_negatives=True,
        )
        forbidden_for_training = (
            global_pos_set
            | C.edge_set(split["val_neg_pool"])
            | C.edge_set(split["test_neg_pool"])
        )
        available = C.count_available_negative_pairs(
            split["train_drugs"], split["train_proteins"], forbidden_for_training
        )
        if min(args.hard_neg_pool, available) <= dynamic_total:
            raise ValueError(
                f"Fold {fold} hard-negative pool cannot preserve exact 1:2 refresh: "
                f"pool_cap={args.hard_neg_pool}, available={available}, required={dynamic_total}"
            )

    drug_dim = int(feature_store.drug_global.size(1))
    protein_dim = int(feature_store.protein_global.size(1))
    descriptor_dim = int(feature_store.descriptors_raw.size(1))
    functional_group_vocab = int(feature_store.functional_group_ids.max()) + 1
    drug_tokens = int(feature_store.drug_local.size(1))
    functional_group_tokens = int(feature_store.functional_group_ids.size(1))
    protein_tokens = int(feature_store.protein_local.size(1))
    print(f"Device: {device}")
    print(
        "Protocol: BioSNAP full_pair.csv deterministic five-fold compound-cold rebuild; "
        "train/validation/test drugs are mutually disjoint; "
        "strict_protein_known=false"
    )
    print(f"Model: {args.ablation} | {ablation_spec['description']}")
    print("Training: weak-balanced labels + evidential loss + dual validation")
    print(
        f"Effective controls: KL lambda={effective_kl_lambda:g}; "
        f"positive-confidence lambda={effective_pos_conf_lambda:g}"
    )
    print(
        f"Dynamic negative policy: positives/negatives=1:{args.train_neg_ratio:g}; "
        f"hard/random={args.hard_to_random_ratio:g}:1 within the negative set"
    )
    print("Selection: drug-cold early-stop validation; ratio-specific validation selects thresholds")
    print("Relation graph: not loaded; positive-pair audit uses canonical V1 global_pos_edges")
    print(
        "Feature access: Mol2Vec global/local + ECFP + descriptors + functional groups; "
        "ProtBert-BFD global/grouped-residue tokens"
    )
    print(
        f"Pair tokens: 1 + {drug_tokens} + {functional_group_tokens} + 1 + {protein_tokens}"
    )
    print(f"BEDROC alpha={args.bedroc_alpha:g}; EF fractions={args.ef_fractions}")
    print(
        f"Evaluation: score maximum 1:{max(args.eval_ratios)} val/test streams once; "
        f"atomic cache every {args.evaluation_cache_chunk_size} pairs; "
        "all ratios/AUPR/AUC/BEDROC/EF reuse the saved scores"
    )
    print(f"Feature SHA256: {artifact_sha256['feature']}")
    print(f"Common helper SHA256: {artifact_sha256['common_code']}")
    print(f"Reference local707 training SHA256: {REFERENCE_LOCAL707_TRAINING_SHA256}")
    print(f"Folds: {[C.fold_number_from_path(path) for path in split_files]}")
    if args.audit_only:
        print("Audit complete: feature/entity order, five rebuilt folds, source hashes, edge scopes, "
              "nested 1:100 pools, initial 1:2 labels, and dynamic-pool capacity all passed.")
        return

    for split_file in split_files:
        fold = C.fold_number_from_path(split_file)
        checkpoint_path = args.save_dir / f"best_model_fold{fold}.pt"
        fold_ratio_path = args.save_dir / f"fold{fold}_ratios.csv"
        fold_record_path = args.save_dir / f"fold{fold}_record.json"
        epoch_checkpoint_path = args.save_dir / f"epoch_checkpoint_fold{fold}.pt"
        if args.resume and checkpoint_path.exists() and fold_ratio_path.exists() and fold_record_path.exists():
            with fold_record_path.open("r", encoding="utf-8") as handle:
                completed_record = json.load(handle)
            completed_hashes = {
                "feature": completed_record.get("FeatureSHA256"),
                "hierarchical_feature": completed_record.get("HierarchicalFeatureSHA256"),
                "feature_manifest": completed_record.get("FeatureManifestSHA256"),
                "training_code": completed_record.get("TrainingCodeSHA256"),
                "common_code": completed_record.get("CommonCodeSHA256"),
            }
            if completed_hashes != artifact_sha256:
                completed_differences = {
                    key: {
                        "saved": completed_hashes.get(key),
                        "current": artifact_sha256.get(key),
                    }
                    for key in sorted(set(completed_hashes) | set(artifact_sha256))
                    if completed_hashes.get(key) != artifact_sha256.get(key)
                }
                evaluation_cache_upgrade_complete = (
                    set(completed_differences) == {"training_code"}
                    and completed_hashes.get("training_code")
                    in EVALUATION_CACHE_PREDECESSOR_TRAINING_SHA256S
                )
                if evaluation_cache_upgrade_complete:
                    print(
                        f"Fold {fold}: accepting completed outputs from the exact pre-cache "
                        "runner; metric definitions are unchanged."
                    )
                else:
                    raise ValueError(
                        f"Fold {fold} complete outputs belong to different/legacy features or "
                        "scientific code; archive the old result directory before using "
                        "--resume.\n"
                        + json.dumps(
                            {"saved": completed_hashes, "current": artifact_sha256},
                            indent=2,
                            ensure_ascii=True,
                        )
                    )
            if epoch_checkpoint_path.exists():
                epoch_checkpoint_path.unlink()
            print(f"Fold {fold}: complete outputs found, skipping due to --resume")
            continue

        C.set_seed(args.seed + fold)
        print(f"\n=== Hierarchical features + full v8 framework fold {fold} ===")
        split = torch_load(split_file)
        validate_split_feature_indices(
            split,
            num_drugs=len(feature_bundle["drug_ids"]),
            num_proteins=len(feature_bundle["protein_ids"]),
            fold=fold,
        )
        validate_biosnap_split(
            split, fold, feature_bundle, global_pos_set, max(args.eval_ratios)
        )
        pos_tr = split["pos_tr"].cpu()
        pos_val = split["pos_val"].cpu()
        test_pos = split["test_pos"].cpu()
        neg_tr = split["neg_tr"].cpu()
        neg_val = split["neg_val_es"].cpu()
        val_neg_pool = split["val_neg_pool"].cpu()
        test_neg_pool = split["test_neg_pool"].cpu()
        train_drugs = split["train_drugs"].cpu()
        train_proteins = split["train_proteins"].cpu()
        descriptor_mean, descriptor_std = feature_store.fit_descriptor_scaler(train_drugs)
        dynamic_negative_total, fold_hard_negative_count, fold_random_negative_count = (
            dynamic_negative_counts(
                positive_count=int(pos_tr.size(1)),
                train_neg_ratio=args.train_neg_ratio,
                hard_to_random_ratio=args.hard_to_random_ratio,
                use_hard_negatives=args.ablation != "no_hard_negative",
            )
        )

        train_edges = torch.cat([pos_tr, neg_tr], dim=1)
        train_labels = torch.cat([torch.ones(pos_tr.size(1)), torch.zeros(neg_tr.size(1))])
        print(
            f"Fold {fold} initial weak-balanced set: pos/neg="
            f"{pos_tr.size(1)}/{neg_tr.size(1)}; dynamic negative target="
            f"{dynamic_negative_total} "
            f"({fold_hard_negative_count} hard + {fold_random_negative_count} random)"
        )
        val_edges = torch.cat([pos_val, neg_val], dim=1)
        val_labels = torch.cat([torch.ones(pos_val.size(1)), torch.zeros(neg_val.size(1))])
        forbidden_all = global_pos_set | C.edge_set(val_neg_pool) | C.edge_set(test_neg_pool)

        model = InducedFitEvidentialModel(
            drug_dim=drug_dim,
            protein_dim=protein_dim,
            descriptor_dim=descriptor_dim,
            functional_group_vocab=functional_group_vocab,
            drug_tokens=drug_tokens,
            functional_group_tokens=functional_group_tokens,
            protein_tokens=protein_tokens,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val_aupr = -1.0
        best_state = None
        best_epoch = 0
        patience_count = 0
        start_epoch = 1
        training_finished = False
        resumed_from_epoch = None
        resume_signature = build_resume_signature(
            args,
            fold,
            split_file,
            effective_kl_lambda,
            effective_pos_conf_lambda,
            artifact_sha256,
        )

        if epoch_checkpoint_path.exists():
            if args.restart_incomplete:
                epoch_checkpoint_path.unlink()
                temporary = epoch_checkpoint_path.with_suffix(epoch_checkpoint_path.suffix + ".tmp")
                if temporary.exists():
                    temporary.unlink()
                print(f"Fold {fold}: discarded incomplete epoch checkpoint")
            elif not args.resume:
                raise FileExistsError(
                    f"Incomplete epoch checkpoint exists: {epoch_checkpoint_path}. "
                    "Use --resume to continue or --restart-incomplete to restart this fold."
                )
            else:
                progress = torch_load(epoch_checkpoint_path)
                validate_resume_signature(
                    progress.get("resume_signature", {}), resume_signature, epoch_checkpoint_path
                )
                model.load_state_dict(progress["model_state"])
                optimizer.load_state_dict(progress["optimizer_state"])
                move_optimizer_state_to_device(optimizer, device)
                train_edges = progress["train_edges"].cpu()
                train_labels = progress["train_labels"].cpu()
                best_val_aupr = float(progress["best_val_aupr"])
                best_state = progress.get("best_state")
                best_epoch = int(progress["best_epoch"])
                patience_count = int(progress["patience_count"])
                last_completed_epoch = int(progress["last_completed_epoch"])
                start_epoch = last_completed_epoch + 1
                resumed_from_epoch = last_completed_epoch
                training_finished = bool(progress.get("training_finished", False))
                restore_rng_state(progress["rng_state"])
                print(
                    f"Fold {fold}: resumed after epoch {last_completed_epoch}; "
                    f"next epoch={start_epoch}, best epoch={best_epoch}, "
                    f"best val AUPR={best_val_aupr:.4f}, patience={patience_count}"
                )

        epoch_iterator = range(start_epoch, args.num_epochs + 1) if not training_finished else []
        for epoch in epoch_iterator:
            model.train()
            permutation = torch.randperm(train_edges.size(1))
            epoch_loss = 0.0
            seen = 0
            for start in range(0, train_edges.size(1), args.batch_size):
                selected = permutation[start : start + args.batch_size]
                batch_edges = train_edges[:, selected]
                batch_labels = train_labels[selected].to(device)
                pair_features = feature_store.pair_batch(batch_edges, device)
                optimizer.zero_grad(set_to_none=True)
                logits, alpha = model(*pair_features)
                loss = C.evidential_loss(
                    logits,
                    alpha,
                    batch_labels,
                    epoch,
                    args.num_epochs,
                    args.pos_weight,
                    effective_kl_lambda,
                    effective_pos_conf_lambda,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                count = int(batch_labels.numel())
                epoch_loss += float(loss.detach()) * count
                seen += count

            refresh_due = epoch % args.hard_neg_update_freq == 0
            if args.ablation == "no_dynamic":
                refresh_due = epoch == args.hard_neg_update_freq
            if refresh_due:
                model.eval()

                def score_for_mining(edges: torch.Tensor) -> torch.Tensor:
                    return score_edges_batched(model, feature_store, edges, device, args.score_batch_size)

                dynamic_edges, dynamic_labels = C.build_dynamic_train_edges(
                    pos_tr=pos_tr.to(device),
                    train_drugs=train_drugs,
                    train_proteins=train_proteins,
                    forbidden_pairs=forbidden_all,
                    score_edges_fn=score_for_mining,
                    device=device,
                    seed=args.seed + fold * 10000 + epoch,
                    hard_neg_pool=args.hard_neg_pool,
                    random_neg_num=fold_random_negative_count,
                    hard_neg_num=fold_hard_negative_count,
                )
                train_edges = dynamic_edges.cpu()
                train_labels = dynamic_labels.cpu()
                print(
                    f"Fold {fold} epoch {epoch}: "
                    f"{'fixed static set' if args.ablation == 'no_dynamic' else 'refreshed negatives'}; "
                    f"hard/random={fold_hard_negative_count}/{fold_random_negative_count}; "
                    f"train pos/neg={pos_tr.size(1)}/{int((train_labels == 0).sum())}"
                )

            should_stop = False
            if epoch % args.validation_frequency == 0:
                model.eval()
                val_aupr = validation_aupr(
                    model, feature_store, val_edges, val_labels, device, args.score_batch_size
                )
                print(
                    f"Fold {fold} epoch {epoch:03d} | loss={epoch_loss / max(seen, 1):.5f} "
                    f"| cold-val AUPR={val_aupr:.4f}"
                )
                if val_aupr > best_val_aupr:
                    best_val_aupr = val_aupr
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                    }
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= args.patience:
                        print(f"Fold {fold}: early stop at epoch {epoch}")
                        should_stop = True

            training_finished = should_stop or epoch == args.num_epochs
            epoch_progress = {
                "schema_version": 1,
                "resume_signature": resume_signature,
                "fold": fold,
                "last_completed_epoch": epoch,
                "training_finished": training_finished,
                "model_state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "optimizer_state": cpu_clone_tree(optimizer.state_dict()),
                "train_edges": train_edges.cpu().clone(),
                "train_labels": train_labels.cpu().clone(),
                "best_val_aupr": best_val_aupr,
                "best_state": cpu_clone_tree(best_state) if best_state is not None else None,
                "best_epoch": best_epoch,
                "patience_count": patience_count,
                "rng_state": capture_rng_state(),
                "last_epoch_loss": epoch_loss / max(seen, 1),
            }
            atomic_torch_save(epoch_progress, epoch_checkpoint_path)
            if epoch == start_epoch or epoch % 10 == 0 or training_finished:
                print(f"Fold {fold}: epoch checkpoint saved after epoch {epoch}")
            if should_stop:
                break

        if best_state is None:
            raise RuntimeError(f"Fold {fold} produced no valid checkpoint")
        model.load_state_dict(best_state)
        checkpoint = {
            "model_state": best_state,
            "fold": fold,
            "best_epoch": best_epoch,
            "best_val_aupr": best_val_aupr,
            "model_config": {
                "drug_dim": drug_dim,
                "protein_dim": protein_dim,
                "descriptor_dim": descriptor_dim,
                "functional_group_vocab": functional_group_vocab,
                "drug_tokens": drug_tokens,
                "functional_group_tokens": functional_group_tokens,
                "protein_tokens": protein_tokens,
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
            },
            "feature_path": str(args.feature_path.resolve()),
            "hierarchical_feature_path": str(args.hierarchical_feature_path.resolve()),
            "feature_manifest_path": str(feature_manifest_path.resolve()),
            "artifact_sha256": artifact_sha256,
            "reference_local707_training_sha256": REFERENCE_LOCAL707_TRAINING_SHA256,
            "split_source_audit": split_source_audits[str(fold)],
            "descriptor_mean": descriptor_mean,
            "descriptor_std": descriptor_std,
            "split_path": str(split_file.resolve()),
            "ablation": args.ablation,
            "effective_kl_lambda": effective_kl_lambda,
            "effective_pos_conf_lambda": effective_pos_conf_lambda,
            "train_negative_ratio": args.train_neg_ratio,
            "dynamic_negative_total": dynamic_negative_total,
            "dynamic_hard_negative_count": fold_hard_negative_count,
            "dynamic_random_negative_count": fold_random_negative_count,
            "uses_repeated_negative_refresh": args.ablation != "no_dynamic",
            "uses_score_based_hard_negatives": args.ablation != "no_hard_negative",
            "resumed_from_epoch": resumed_from_epoch,
        }
        atomic_torch_save(checkpoint, checkpoint_path)

        model.eval()
        evaluation_scores = score_evaluation_streams_resumable(
            fold,
            model,
            feature_store,
            split,
            split_file,
            args.save_dir,
            device,
            args,
            artifact_sha256,
        )
        fold_ratio_df = evaluate_ratios_from_cached_scores(
            evaluation_scores,
            args.eval_ratios,
            args.bedroc_alpha,
            args.ef_fractions,
        )
        fold_ratio_df.insert(0, "Fold", fold)
        fold_ratio_df.to_csv(fold_ratio_path, index=False)
        ratio100 = fold_ratio_df.loc[fold_ratio_df["Ratio"] == "1:100"]
        record = {
            "Fold": fold,
            "BestEpoch": best_epoch,
            "ValAUPR_ES": best_val_aupr,
            "AUPR_1_100": float(ratio100["AUPR"].iloc[0]) if not ratio100.empty else None,
            "Checkpoint": str(checkpoint_path),
            "FeatureSHA256": artifact_sha256["feature"],
            "HierarchicalFeatureSHA256": artifact_sha256["hierarchical_feature"],
            "FeatureManifestSHA256": artifact_sha256["feature_manifest"],
            "TrainingCodeSHA256": artifact_sha256["training_code"],
            "CommonCodeSHA256": artifact_sha256["common_code"],
            "ReferenceLocal707TrainingSHA256": REFERENCE_LOCAL707_TRAINING_SHA256,
            "SplitSHA256": file_sha256(split_file),
            "SplitManifestSHA256": split_source_audits[str(fold)]["manifest_sha256"],
            "ResumedFromEpoch": resumed_from_epoch,
            "UsesEpochCheckpointResume": True,
            "TrainDrugs": int(split["train_drugs"].numel()),
            "ValDrugs": int(split["val_drugs"].numel()),
            "TestDrugs": int(split["test_drugs"].numel()),
            "TrainPos": int(pos_tr.size(1)),
            "InitialTrainNeg": int(neg_tr.size(1)),
            "DynamicTrainNeg": dynamic_negative_total,
            "DynamicHardNeg": fold_hard_negative_count,
            "DynamicRandomNeg": fold_random_negative_count,
            "ValPos": int(pos_val.size(1)),
            "TestPos": int(test_pos.size(1)),
            "UsesRelationGraph": False,
            "UsesFineGrainedJointTransformer": True,
            "UsesRepeatedNegativeRefresh": args.ablation != "no_dynamic",
            "UsesScoreBasedHardNegatives": args.ablation != "no_hard_negative",
            "UsesKLRegularizer": args.ablation != "no_kl",
            "UsesEvidentialHead": True,
            "UsesPositiveConfidenceConstraint": args.ablation != "no_kl",
            "UsesDualValidation": True,
            "UsesHierarchicalDrugFeatures": True,
            "UsesSharedMaximumRatioScoreCache": True,
            "EvaluationScoreCacheResumable": True,
            "EvaluationScoreCacheChunkSize": args.evaluation_cache_chunk_size,
            "EvaluationMaximumRatio": max(args.eval_ratios),
            "BEDROC_1_100": (
                float(ratio100[f"BEDROC_alpha_{args.bedroc_alpha:g}"].iloc[0])
                if not ratio100.empty
                else None
            ),
        }
        save_json(record, fold_record_path)
        if epoch_checkpoint_path.exists():
            epoch_checkpoint_path.unlink()
        print(f"Fold {fold} saved: {checkpoint_path}")

    cv_rows, ratio_dfs = collect_saved_results(args.save_dir)
    if not cv_rows or not ratio_dfs:
        raise RuntimeError("No complete fold result files were found")
    C.save_result_tables(args.save_dir, experiment_name, cv_rows, ratio_dfs)
    save_extended_ratio_summary(args.save_dir, ratio_dfs, experiment_name)
    run_manifest = {
        "experiment": experiment_name,
        "ablation": args.ablation,
        "ablation_description": ablation_spec["description"],
        "dataset": "BioSNAP",
        "baseline_training_code_sha256": REFERENCE_LOCAL707_TRAINING_SHA256,
        "baseline_common_code_sha256": REFERENCE_LOCAL707_COMMON_SHA256,
        "changed_scope": [
            (
                "dataset/entity order: local707 707/1489 -> BioSNAP feature package "
                f"{len(feature_bundle['drug_ids'])}/{len(feature_bundle['protein_ids'])}"
            ),
            (
                "outer split: local707 ten-fold strict drug-cold/protein-known -> BioSNAP "
                "full_pair.csv deterministic five-fold compound-cold rebuild with mutually "
                "disjoint train/validation/test drugs; strict_protein_known=false"
            ),
            (
                "hard-negative candidate pool: 10000 -> 60000 so exact 1:2 refresh remains "
                "available on BioSNAP"
            ),
        ],
        "preserved_framework": [
            "current local707 166-token Ours/TIDE-DTI architecture",
            "weak-balanced training",
            "drug-cold early-stop validation",
            "ratio-specific threshold validation",
            "Joint Transformer pair conditioning",
            "DeepGatedInteraction and evidential prediction heads",
            "hierarchical drug and fine-grained protein feature model",
        ],
        "uses_relation_graph": False,
        "epoch_checkpoint_resume": True,
        "epoch_checkpoint_pattern": "epoch_checkpoint_fold{fold}.pt",
        "epoch_checkpoint_atomic_write": True,
        "evaluation_execution": {
            "scoring_revision": EVALUATION_CACHE_SCORING_REVISION,
            "maximum_ratio": max(args.eval_ratios),
            "shared_ordered_prefix_scores": True,
            "resumable_score_cache": True,
            "cache_directory_pattern": "evaluation_score_cache_fold{fold}",
            "cache_chunk_size": args.evaluation_cache_chunk_size,
            "shared_metrics": ["AUPR", "AUC", "BEDROC", "EF"],
        },
        "positive_pair_audit_source": (
            "full_pair positives plus auxiliary split-CSV positives used only as a "
            "collision blacklist, shared by all rebuilt folds"
        ),
        "effective_kl_lambda": effective_kl_lambda,
        "effective_pos_conf_lambda": effective_pos_conf_lambda,
        "dynamic_negative_count_policy": "round(train_positive_count * train_negative_ratio)",
        "train_negative_ratio": args.train_neg_ratio,
        "dynamic_hard_to_random_ratio": args.hard_to_random_ratio,
        "uses_repeated_negative_refresh": args.ablation != "no_dynamic",
        "uses_score_based_hard_negatives": args.ablation != "no_hard_negative",
        "bedroc_alpha": args.bedroc_alpha,
        "ef_fractions": args.ef_fractions,
        "feature_path": str(args.feature_path.resolve()),
        "feature_sha256": artifact_sha256["feature"],
        "hierarchical_feature_path": str(args.hierarchical_feature_path.resolve()),
        "hierarchical_feature_sha256": artifact_sha256["hierarchical_feature"],
        "feature_manifest_path": str(feature_manifest_path.resolve()),
        "feature_manifest_sha256": artifact_sha256["feature_manifest"],
        "training_code": str(training_code_path),
        "training_code_sha256": artifact_sha256["training_code"],
        "common_code": str(COMMON_PATH.resolve()),
        "common_code_sha256": artifact_sha256["common_code"],
        "protein_feature": feature_manifest["protein"],
        "split_cache_dir": str(args.split_cache_dir.resolve()),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "split_source_audits": split_source_audits,
        "strict_protein_known": False,
    }
    with (args.save_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
