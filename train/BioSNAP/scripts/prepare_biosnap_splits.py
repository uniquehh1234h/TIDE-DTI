from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
OURS_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_DIR = OURS_ROOT.parent / "datasets"
DEFAULT_FEATURE_PATH = OURS_ROOT / "features" / "v1_features.pt"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR
EVAL_RATIOS = (1, 2, 5, 10, 20, 50, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild deterministic five-fold BioSNAP compound-cold partitions directly "
            "from full_pair.csv and construct leakage-audited nested negative pools."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--cold-val-fraction", type=float, default=0.10)
    parser.add_argument("--train-neg-ratio", type=int, default=2)
    parser.add_argument("--early-stop-neg-ratio", type=int, default=2)
    parser.add_argument("--eval-ratios", nargs="*", type=int, default=list(EVAL_RATIOS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_dump(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_pair_csv(path: Path) -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(path)
    required = ["head", "tail", "label"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame = frame[required].copy()
    frame["head"] = frame["head"].astype(str)
    frame["tail"] = frame["tail"].astype(str)
    frame["label"] = frame["label"].astype(int)
    invalid = sorted(set(frame["label"]) - {0, 1})
    if invalid:
        raise ValueError(f"{path} has non-binary labels: {invalid}")
    conflicts = frame.groupby(["head", "tail"])["label"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(f"{path} has conflicting labels for {len(conflicts)} pairs")
    original_count = len(frame)
    frame = frame.drop_duplicates(["head", "tail"], keep="first").reset_index(drop=True)
    return frame, original_count - len(frame)


def choose_cold_validation_drugs(
    development: pd.DataFrame,
    fraction: float,
    seed: int,
) -> set[str]:
    if not 0.0 < fraction < 0.5:
        raise ValueError("Cold validation fraction must be in (0, 0.5)")
    positive_counts = (
        development.loc[development["label"] == 1].groupby("head").size().to_dict()
    )
    positive_drugs = sorted(positive_counts)
    if len(positive_drugs) < 2:
        raise ValueError("Not enough positive drugs to rebuild a cold validation set")
    rng = np.random.default_rng(seed)
    order = np.asarray(positive_drugs, dtype=object)[rng.permutation(len(positive_drugs))]
    target = max(1, int(round(sum(positive_counts.values()) * fraction)))
    selected: set[str] = set()
    selected_positive_count = 0
    for drug in order.tolist():
        if len(selected) >= len(positive_drugs) - 1:
            break
        selected.add(str(drug))
        selected_positive_count += int(positive_counts[str(drug)])
        if selected_positive_count >= target:
            break
    if not selected or selected == set(positive_drugs):
        raise RuntimeError("Failed to rebuild a non-empty drug-cold validation set")
    return selected


def assign_compound_cold_test_folds(
    full_pair: pd.DataFrame,
    num_folds: int,
    seed: int,
) -> tuple[list[set[str]], dict]:
    """Greedily balance positive edges, negative edges, and drug counts across folds."""
    if num_folds < 2:
        raise ValueError("At least two compound-cold folds are required")
    counts = (
        full_pair.groupby(["head", "label"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
    )
    if len(counts) < num_folds:
        raise ValueError("There are fewer drugs than requested compound-cold folds")
    rows = [
        (str(drug), int(values[1]), int(values[0]))
        for drug, values in counts.iterrows()
    ]
    rng = np.random.default_rng(seed)
    tie_order = {drug: rank for rank, drug in enumerate(rng.permutation([row[0] for row in rows]))}
    rows.sort(
        key=lambda row: (
            -(row[1] + row[2]),
            -row[1],
            -row[2],
            tie_order[row[0]],
        )
    )
    target_positive = max(1.0, sum(row[1] for row in rows) / num_folds)
    target_negative = max(1.0, sum(row[2] for row in rows) / num_folds)
    target_drugs = len(rows) / num_folds
    folds: list[set[str]] = [set() for _ in range(num_folds)]
    positive_load = [0] * num_folds
    negative_load = [0] * num_folds
    fold_priority = rng.permutation(num_folds).tolist()
    priority_rank = {fold: rank for rank, fold in enumerate(fold_priority)}
    for drug, positive_count, negative_count in rows:
        def assignment_score(fold: int) -> tuple[float, int]:
            score = (
                ((positive_load[fold] + positive_count) / target_positive) ** 2
                + ((negative_load[fold] + negative_count) / target_negative) ** 2
                + 0.10 * ((len(folds[fold]) + 1) / target_drugs) ** 2
            )
            return score, priority_rank[fold]

        selected_fold = min(range(num_folds), key=assignment_score)
        folds[selected_fold].add(drug)
        positive_load[selected_fold] += positive_count
        negative_load[selected_fold] += negative_count
    if set().union(*folds) != set(counts.index.astype(str)):
        raise RuntimeError("Compound-cold test assignment lost one or more drugs")
    if any(folds[i] & folds[j] for i in range(num_folds) for j in range(i + 1, num_folds)):
        raise RuntimeError("Compound-cold test folds overlap")
    return folds, {
        "algorithm": "deterministic_greedy_balance_positive_negative_drug_count_v1",
        "seed": seed,
        "num_folds": num_folds,
        "fold_drug_counts": [len(values) for values in folds],
        "fold_positive_counts": positive_load,
        "fold_negative_counts": negative_load,
    }


def frame_pairs(
    frame: pd.DataFrame,
    drug_to_index: dict[str, int],
    protein_to_index: dict[str, int],
    label: int,
) -> list[tuple[int, int]]:
    selected = frame.loc[frame["label"] == label, ["head", "tail"]]
    missing_drugs = sorted(set(selected["head"]) - set(drug_to_index))
    missing_proteins = sorted(set(selected["tail"]) - set(protein_to_index))
    if missing_drugs or missing_proteins:
        raise ValueError(
            "Official pair IDs are absent from the feature package: "
            f"drugs={missing_drugs[:5]}, proteins={missing_proteins[:5]}"
        )
    return [
        (drug_to_index[drug], protein_to_index[protein])
        for drug, protein in selected.itertuples(index=False, name=None)
    ]


def entity_indices(values: Iterable[str], mapping: dict[str, int], name: str) -> set[int]:
    values = set(map(str, values))
    missing = sorted(values - set(mapping))
    if missing:
        raise ValueError(f"{name} contains IDs absent from features: {missing[:5]}")
    return {mapping[value] for value in values}


def edge_tensor(pairs: Iterable[tuple[int, int]]) -> torch.Tensor:
    pairs = list(pairs)
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def edge_set(edges: torch.Tensor) -> set[tuple[int, int]]:
    edges = edges.detach().cpu().long()
    if edges.numel() == 0:
        return set()
    return set(zip(edges[0].tolist(), edges[1].tolist()))


def sample_negative_edges(
    drug_ids: set[int],
    protein_ids: set[int],
    forbidden_pairs: set[tuple[int, int]],
    count: int,
    seed: int,
) -> list[tuple[int, int]]:
    drugs = np.asarray(sorted(drug_ids), dtype=np.int64)
    proteins = np.asarray(sorted(protein_ids), dtype=np.int64)
    if count <= 0:
        return []
    if not len(drugs) or not len(proteins):
        raise ValueError("Cannot sample negatives from an empty entity scope")
    blocked = sum(1 for pair in forbidden_pairs if pair[0] in drug_ids and pair[1] in protein_ids)
    available = len(drugs) * len(proteins) - blocked
    if count > available:
        raise ValueError(f"Need {count} negative pairs but only {available} are available")
    rng = np.random.default_rng(seed)
    selected: set[tuple[int, int]] = set()
    batch_size = min(max(count * 2, 1024), 200000)
    max_attempts = max(10000, count * 100)
    attempts = 0
    while len(selected) < count and attempts < max_attempts:
        sampled_drugs = rng.choice(drugs, size=batch_size, replace=True)
        sampled_proteins = rng.choice(proteins, size=batch_size, replace=True)
        for drug, protein in zip(sampled_drugs.tolist(), sampled_proteins.tolist()):
            pair = (int(drug), int(protein))
            if pair not in forbidden_pairs and pair not in selected:
                selected.add(pair)
                if len(selected) == count:
                    break
        attempts += batch_size
    if len(selected) < count:
        candidates = [(int(drug), int(protein)) for drug in drugs for protein in proteins]
        rng.shuffle(candidates)
        for pair in candidates:
            if pair not in forbidden_pairs and pair not in selected:
                selected.add(pair)
                if len(selected) == count:
                    break
    pairs = sorted(selected)
    rng.shuffle(pairs)
    return pairs


def build_global_positive_set(
    dataset_dir: Path,
    drug_to_index: dict[str, int],
    protein_to_index: dict[str, int],
) -> tuple[set[tuple[int, int]], dict]:
    paths = [dataset_dir / "full_pair.csv"]
    auxiliary_dir = dataset_dir / "split"
    if auxiliary_dir.is_dir():
        paths.extend(sorted(auxiliary_dir.glob("*.csv")))
    frames: list[pd.DataFrame] = []
    source_counts: dict[str, dict] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        frame, duplicates = read_pair_csv(path)
        frames.append(frame)
        source_counts[str(path.resolve())] = {
            "rows_after_deduplication": len(frame),
            "duplicate_rows_removed": duplicates,
            "sha256": sha256_file(path),
        }
    combined = pd.concat(frames, ignore_index=True)
    conflicts = combined.groupby(["head", "tail"])["label"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(f"BioSNAP sources contain {len(conflicts)} cross-file label conflicts")
    combined = combined.drop_duplicates(["head", "tail"], keep="first")
    outside_feature_order = combined.loc[
        ~combined["head"].isin(drug_to_index) | ~combined["tail"].isin(protein_to_index)
    ]
    combined_in_feature_order = combined.loc[
        combined["head"].isin(drug_to_index) & combined["tail"].isin(protein_to_index)
    ].reset_index(drop=True)
    positives = frame_pairs(
        combined_in_feature_order, drug_to_index, protein_to_index, label=1
    )
    return set(positives), {
        "unique_pairs_across_all_sources": int(len(combined)),
        "pairs_outside_feature_entity_order_ignored_for_collision_audit": int(
            len(outside_feature_order)
        ),
        "positives_outside_feature_entity_order_ignored_for_collision_audit": int(
            outside_feature_order["label"].sum()
        ),
        "global_known_positive": int(len(set(positives))),
        "outer_split_assignment_source": str((dataset_dir / "full_pair.csv").resolve()),
        "auxiliary_split_csvs_used_only_for_positive_collision_audit": max(
            0, len(paths) - 1
        ),
        "source_files": source_counts,
    }


def make_negative_pool(
    explicit_pairs: Iterable[tuple[int, int]],
    drug_ids: set[int],
    protein_ids: set[int],
    global_positive_pairs: set[tuple[int, int]],
    count: int,
    seed: int,
    additionally_forbidden: set[tuple[int, int]] | None = None,
) -> tuple[torch.Tensor, dict]:
    additionally_forbidden = set() if additionally_forbidden is None else set(additionally_forbidden)
    explicit = sorted(
        {
            pair
            for pair in explicit_pairs
            if pair[0] in drug_ids
            and pair[1] in protein_ids
            and pair not in global_positive_pairs
            and pair not in additionally_forbidden
        }
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(explicit)
    author_used = explicit[:count]
    sampled: list[tuple[int, int]] = []
    if len(author_used) < count:
        forbidden = global_positive_pairs | additionally_forbidden | set(author_used)
        sampled = sample_negative_edges(
            drug_ids, protein_ids, forbidden, count - len(author_used), seed + 1
        )
    pairs = author_used + sampled
    if len(pairs) != count or len(set(pairs)) != count:
        raise RuntimeError("Negative-pool construction failed to return the requested unique pairs")
    return edge_tensor(pairs), {
        "requested": count,
        "author_labeled_negatives": len(author_used),
        "sampled_unlabeled_as_negative": len(sampled),
        "ordering": "seed-shuffled author negatives first, then sampled unlabeled pairs",
    }


def require_scope(
    name: str,
    edges: torch.Tensor,
    drugs: set[int],
    proteins: set[int],
) -> None:
    bad = [pair for pair in edge_set(edges) if pair[0] not in drugs or pair[1] not in proteins]
    if bad:
        raise ValueError(f"{name} contains {len(bad)} pairs outside its rebuilt entity scope")


def validate_split(split: dict, global_positive_pairs: set[tuple[int, int]], max_ratio: int) -> None:
    train_drugs = set(map(int, split["train_drugs"].tolist()))
    val_drugs = set(map(int, split["val_drugs"].tolist()))
    test_drugs = set(map(int, split["test_drugs"].tolist()))
    train_proteins = set(map(int, split["train_proteins"].tolist()))
    evaluation_proteins = set(map(int, split["evaluation_proteins"].tolist()))
    if train_drugs & val_drugs or train_drugs & test_drugs or val_drugs & test_drugs:
        raise ValueError("BioSNAP rebuilt train/validation/test drug partitions overlap")
    if not train_proteins.issubset(evaluation_proteins):
        raise ValueError("Training proteins are not a subset of evaluation proteins")
    require_scope("pos_tr", split["pos_tr"], train_drugs, train_proteins)
    require_scope("neg_tr", split["neg_tr"], train_drugs, train_proteins)
    require_scope("pos_val", split["pos_val"], val_drugs, evaluation_proteins)
    require_scope("neg_val_es", split["neg_val_es"], val_drugs, evaluation_proteins)
    require_scope("val_neg_pool", split["val_neg_pool"], val_drugs, evaluation_proteins)
    require_scope("test_pos", split["test_pos"], test_drugs, evaluation_proteins)
    require_scope("test_neg_pool", split["test_neg_pool"], test_drugs, evaluation_proteins)
    positive_names = ("pos_tr", "pos_val", "test_pos")
    positive_sets = []
    for name in positive_names:
        pairs = edge_set(split[name])
        if not pairs or len(pairs) != split[name].size(1):
            raise ValueError(f"{name} is empty or contains duplicates")
        positive_sets.append(pairs)
    if any(positive_sets[i] & positive_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Rebuilt positive train/validation/test pair assignments overlap")
    negative_sets: dict[str, set[tuple[int, int]]] = {}
    for name in ("neg_tr", "neg_val_es", "val_neg_pool", "test_neg_pool"):
        pairs = edge_set(split[name])
        if len(pairs) != split[name].size(1):
            raise ValueError(f"{name} contains duplicate pairs")
        if pairs & global_positive_pairs:
            raise ValueError(f"{name} collides with a globally known positive")
        negative_sets[name] = pairs
    if negative_sets["val_neg_pool"] & negative_sets["test_neg_pool"]:
        raise ValueError("Validation and test negative pools overlap")
    if split["neg_tr"].size(1) != split["pos_tr"].size(1) * 2:
        raise ValueError("Initial training negatives do not preserve the formal 1:2 ratio")
    if split["neg_val_es"].size(1) != split["pos_val"].size(1) * 2:
        raise ValueError("Early-stop validation negatives do not preserve the formal 1:2 ratio")
    if split["val_neg_pool"].size(1) < split["pos_val"].size(1) * max_ratio:
        raise ValueError("Validation negative pool cannot support the maximum ratio")
    if split["test_neg_pool"].size(1) < split["test_pos"].size(1) * max_ratio:
        raise ValueError("Test negative pool cannot support the maximum ratio")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    feature_path = args.feature_path.resolve()
    output_dir = args.output_dir.resolve()
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Missing BioSNAP feature package: {feature_path}. Run the launcher prepare action first."
        )
    full_pair_path = dataset_dir / "full_pair.csv"
    if not full_pair_path.is_file():
        raise FileNotFoundError(full_pair_path)
    if sorted(set(args.folds)) != sorted(args.folds) or any(fold not in range(5) for fold in args.folds):
        raise ValueError("--folds must contain unique fold IDs from 0 through 4")
    if args.train_neg_ratio != 2 or args.early_stop_neg_ratio != 2:
        raise ValueError("Formal BioSNAP Ours split preparation requires 1:2 train/early-stop negatives")
    if not args.eval_ratios or min(args.eval_ratios) < 1:
        raise ValueError("Evaluation ratios must be positive")
    max_ratio = max(args.eval_ratios)

    feature_bundle = torch_load(feature_path)
    metadata = feature_bundle.get("metadata", {})
    if metadata.get("dataset") != "BioSNAP":
        raise ValueError(f"Feature package dataset is not BioSNAP: {metadata.get('dataset')!r}")
    if metadata.get("uses_dti_labels") is not False:
        raise ValueError("Feature package does not prove label-free extraction")
    drug_ids = [str(value) for value in feature_bundle["drug_ids"]]
    protein_ids = [str(value) for value in feature_bundle["protein_ids"]]
    if len(drug_ids) != len(set(drug_ids)) or len(protein_ids) != len(set(protein_ids)):
        raise ValueError("Feature package contains duplicate entity IDs")
    if metadata.get("drug_ids_sha256") != sha256_strings(drug_ids):
        raise ValueError("Feature package drug ID order hash is invalid")
    if metadata.get("protein_ids_sha256") != sha256_strings(protein_ids):
        raise ValueError("Feature package protein ID order hash is invalid")
    drug_to_index = {value: index for index, value in enumerate(drug_ids)}
    protein_to_index = {value: index for index, value in enumerate(protein_ids)}
    full_pair, full_pair_duplicates = read_pair_csv(full_pair_path)
    # Fail before assigning folds if full_pair contains an entity absent from features.
    entity_indices(full_pair["head"], drug_to_index, "full_pair.csv drugs")
    entity_indices(full_pair["tail"], protein_to_index, "full_pair.csv proteins")
    test_drug_folds, test_assignment_audit = assign_compound_cold_test_folds(
        full_pair, num_folds=5, seed=args.seed
    )
    global_positive_pairs, global_audit = build_global_positive_set(
        dataset_dir, drug_to_index, protein_to_index
    )
    feature_sha256 = sha256_file(feature_path)
    full_pair_sha256 = sha256_file(full_pair_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[dict] = []

    for fold in args.folds:
        output_path = output_dir / f"split_fold{fold}.pt"
        manifest_path = output_dir / f"split_fold{fold}_manifest.json"
        if output_path.exists() and not args.force:
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing_manifest = json.load(handle)
            expected_protocol = existing_manifest.get("protocol", {})
            expected_source = existing_manifest.get("source", {})
            if (
                int(existing_manifest.get("schema_version", 0)) < 2
                or expected_protocol.get("outer_split_source") != "full_pair.csv"
                or expected_protocol.get("author_split_pair_assignments_reused") is not False
                or expected_source.get("full_pair_sha256") != full_pair_sha256
                or expected_source.get("feature_sha256") != feature_sha256
            ):
                raise ValueError(
                    f"Fold {fold} cache was not built by the current full_pair compound-cold "
                    "protocol. Re-run prepare with --force before formal training."
                )
            split = torch_load(output_path)
            validate_split(split, global_positive_pairs, max_ratio)
            manifests.append(existing_manifest)
            print(f"Fold {fold} already exists and passed audit: {output_path}")
            continue

        test_drug_names = test_drug_folds[fold]
        development = full_pair.loc[~full_pair["head"].isin(test_drug_names)].reset_index(drop=True)
        rebuilt_test = full_pair.loc[full_pair["head"].isin(test_drug_names)].reset_index(drop=True)
        validation_drug_names = choose_cold_validation_drugs(
            development, args.cold_val_fraction, args.seed + fold * 1000
        )
        rebuilt_validation = development.loc[
            development["head"].isin(validation_drug_names)
        ].reset_index(drop=True)
        rebuilt_train = development.loc[
            ~development["head"].isin(validation_drug_names)
        ].reset_index(drop=True)

        train_drugs = entity_indices(rebuilt_train["head"], drug_to_index, "rebuilt training split")
        val_drugs = entity_indices(
            rebuilt_validation["head"], drug_to_index, "rebuilt validation split"
        )
        test_drugs = entity_indices(rebuilt_test["head"], drug_to_index, "rebuilt test split")
        if train_drugs & val_drugs or train_drugs & test_drugs or val_drugs & test_drugs:
            raise ValueError(f"Fold {fold} rebuilt train/validation/test drug sets overlap")
        train_proteins = entity_indices(
            rebuilt_train["tail"], protein_to_index, "rebuilt training split"
        )
        evaluation_proteins = set(range(len(protein_ids)))
        train_positive_pairs = frame_pairs(rebuilt_train, drug_to_index, protein_to_index, 1)
        val_positive_pairs = frame_pairs(rebuilt_validation, drug_to_index, protein_to_index, 1)
        test_positive_pairs = frame_pairs(rebuilt_test, drug_to_index, protein_to_index, 1)
        train_negative_explicit = frame_pairs(rebuilt_train, drug_to_index, protein_to_index, 0)
        val_negative_explicit = frame_pairs(
            rebuilt_validation, drug_to_index, protein_to_index, 0
        )
        test_negative_explicit = frame_pairs(rebuilt_test, drug_to_index, protein_to_index, 0)

        test_explicit_set = set(test_negative_explicit)
        validation_pool, validation_stats = make_negative_pool(
            val_negative_explicit,
            val_drugs,
            evaluation_proteins,
            global_positive_pairs,
            len(val_positive_pairs) * max_ratio,
            args.seed + fold * 1000 + 23,
            additionally_forbidden=test_explicit_set,
        )
        validation_pool_set = edge_set(validation_pool)
        test_pool, test_stats = make_negative_pool(
            test_negative_explicit,
            test_drugs,
            evaluation_proteins,
            global_positive_pairs,
            len(test_positive_pairs) * max_ratio,
            args.seed + fold * 1000 + 37,
            additionally_forbidden=validation_pool_set | set(val_negative_explicit),
        )
        train_forbidden = validation_pool_set | edge_set(test_pool)
        train_negative, train_stats = make_negative_pool(
            train_negative_explicit,
            train_drugs,
            train_proteins,
            global_positive_pairs,
            len(train_positive_pairs) * args.train_neg_ratio,
            args.seed + fold * 1000 + 11,
            additionally_forbidden=train_forbidden,
        )
        early_stop_count = len(val_positive_pairs) * args.early_stop_neg_ratio
        split = {
            "schema_version": 2,
            "dataset": "BioSNAP",
            "fold": fold,
            "protocol": (
                "BioSNAP full_pair.csv deterministic five-fold compound-cold; each drug is "
                "assigned to exactly one test fold; 10% positive-mass whole-drug validation "
                "is rebuilt from the remaining development drugs"
            ),
            "strict_protein_known": False,
            "pos_tr": edge_tensor(train_positive_pairs),
            "pos_val": edge_tensor(val_positive_pairs),
            "test_pos": edge_tensor(test_positive_pairs),
            "neg_tr": train_negative,
            "neg_val_es": validation_pool[:, :early_stop_count].clone(),
            "val_neg_pool": validation_pool,
            "test_neg_pool": test_pool,
            "train_drugs": torch.tensor(sorted(train_drugs), dtype=torch.long),
            "val_drugs": torch.tensor(sorted(val_drugs), dtype=torch.long),
            "test_drugs": torch.tensor(sorted(test_drugs), dtype=torch.long),
            "train_proteins": torch.tensor(sorted(train_proteins), dtype=torch.long),
            "evaluation_proteins": torch.tensor(sorted(evaluation_proteins), dtype=torch.long),
            "global_pos_edges": edge_tensor(sorted(global_positive_pairs)),
            "drug_ids_sha256": sha256_strings(drug_ids),
            "protein_ids_sha256": sha256_strings(protein_ids),
        }
        validate_split(split, global_positive_pairs, max_ratio)
        manifest = {
            "schema_version": 2,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "BioSNAP",
            "fold": fold,
            "source": {
                "dataset_dir": str(dataset_dir),
                "full_pair": str(full_pair_path.resolve()),
                "full_pair_sha256": full_pair_sha256,
                "feature_bundle": str(feature_path),
                "feature_sha256": feature_sha256,
            },
            "protocol": {
                "outer_split_source": "full_pair.csv",
                "author_split_pair_assignments_reused": False,
                "test_assignment": test_assignment_audit,
                "test_drug_ids_sha256": sha256_strings(sorted(test_drug_names)),
                "each_full_pair_drug_is_test_in_exactly_one_fold": True,
                "rebuilt_cold_validation_fraction": args.cold_val_fraction,
                "train_validation_test_drug_sets_are_disjoint": True,
                "strict_protein_known": False,
                "auxiliary_split_csv_positive_labels_used_only_for_collision_audit": True,
                "extra_negative_semantics": "unlabeled compound-protein pairs treated as negatives",
                "negative_pools_are_nested": True,
                "validation_and_test_negative_pools_are_disjoint": True,
                "evaluation_ratios": args.eval_ratios,
            },
            "counts": {
                "feature_drugs": len(drug_ids),
                "feature_proteins": len(protein_ids),
                "train_drugs": len(train_drugs),
                "validation_drugs": len(val_drugs),
                "test_drugs": len(test_drugs),
                "train_proteins": len(train_proteins),
                "evaluation_proteins": len(evaluation_proteins),
                "train_positive": len(train_positive_pairs),
                "validation_positive": len(val_positive_pairs),
                "test_positive": len(test_positive_pairs),
                "train_explicit_negative": len(train_negative_explicit),
                "validation_explicit_negative": len(val_negative_explicit),
                "test_explicit_negative": len(test_negative_explicit),
                "full_pair_duplicate_rows_removed": full_pair_duplicates,
                "global_known_positive": len(global_positive_pairs),
            },
            "negative_sources": {
                "train": train_stats,
                "validation": validation_stats,
                "test": test_stats,
            },
            "global_source_audit": global_audit,
        }
        atomic_torch_save(split, output_path)
        atomic_json_dump(manifest, manifest_path)
        manifests.append(manifest)
        print(
            f"Fold {fold}: positives train/val/test="
            f"{len(train_positive_pairs)}/{len(val_positive_pairs)}/{len(test_positive_pairs)}; "
            f"drugs={len(train_drugs)}/{len(val_drugs)}/{len(test_drugs)}; "
            f"negative pools={validation_pool.size(1)}/{test_pool.size(1)}"
        )

    atomic_json_dump(
        {
            "dataset": "BioSNAP",
            "protocol": "full_pair.csv deterministic five-fold compound-cold rebuild",
            "test_assignment": test_assignment_audit,
            "folds": manifests,
            "warning": (
                "Ratios above available author negatives add unlabeled drug-protein pairs as "
                "negatives; this is the Ours implicit-negative evaluation protocol."
            ),
        },
        output_dir / "split_manifest_all_folds.json",
    )
    print(f"Saved and audited BioSNAP split cache: {output_dir}")


if __name__ == "__main__":
    main()
