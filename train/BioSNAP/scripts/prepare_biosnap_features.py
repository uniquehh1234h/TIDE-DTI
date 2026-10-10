from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


DATASETS = ("BindingDB_AIBind", "BioSNAP", "BindingDB")


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


def read_entity_file(path: Path) -> tuple[list[str], dict[str, str]]:
    """Read ColdstartCPI's ID plus optional raw-sequence/SMILES entity file."""
    entity_ids: list[str] = []
    raw_values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            pieces = raw.rstrip("\n\r").split(maxsplit=1)
            if not pieces:
                continue
            entity_id = pieces[0]
            if entity_id in raw_values:
                raise ValueError(f"Duplicate entity ID at {path}:{line_number}: {entity_id}")
            entity_ids.append(entity_id)
            raw_values[entity_id] = pieces[1].strip() if len(pieces) == 2 else ""
    return entity_ids, raw_values


FUNCTIONAL_GROUP_SMARTS = [
    ("hydroxyl", "[OX2H]"),
    ("phenol", "[OX2H][c]"),
    ("ether", "[OD2]([#6])[#6]"),
    ("carbonyl", "[CX3]=[OX1]"),
    ("carboxylic_acid", "[CX3](=O)[OX2H1]"),
    ("ester", "[CX3](=O)[OX2][#6]"),
    ("amide", "[NX3][CX3](=[OX1])"),
    ("primary_amine", "[NX3;H2][#6]"),
    ("secondary_amine", "[NX3;H1]([#6])[#6]"),
    ("tertiary_amine", "[NX3;H0]([#6])([#6])[#6]"),
    ("quaternary_ammonium", "[NX4+]"),
    ("nitrile", "[CX2]#N"),
    ("nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("thiol", "[SX2H]"),
    ("thioether", "[SX2]([#6])[#6]"),
    ("sulfoxide", "[SX3](=O)([#6])[#6]"),
    ("sulfone", "[SX4](=O)(=O)([#6])[#6]"),
    ("sulfonamide", "[SX4](=O)(=O)[NX3]"),
    ("phosphate", "[PX4](=O)(O)(O)O"),
    ("halogen", "[F,Cl,Br,I]"),
    ("aromatic_carbon", "[c]"),
    ("aromatic_nitrogen", "[n]"),
    ("heteroaromatic_ring", "[a;!#6]"),
    ("aliphatic_ring", "[R;!a]"),
    ("guanidine", "NC(=N)N"),
    ("urea", "NC(=O)N"),
    ("boronic_acid", "B(O)O"),
    ("epoxide", "[OX2r3]1[CX4r3][CX4r3]1"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare label-free V1 features for ColdstartCPI benchmark datasets."
    )
    parser.add_argument("--root", type=Path, default=Path("/mnt/ColdstartCPI"))
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing drug_list.txt and protein_list.txt. When omitted, "
            "the legacy <root>/Datasets/<dataset>/feature directory is used."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mol2vec-model", type=Path, default=None)
    parser.add_argument("--protein-model", default="Rostlab/prot_bert_bfd")
    parser.add_argument(
        "--protein-source",
        choices=("auto", "author_pickle", "transformers"),
        default="auto",
        help="Reuse the author's residue pickle when present, otherwise run Transformers.",
    )
    parser.add_argument("--drug-max-tokens", type=int, default=100)
    parser.add_argument("--protein-groups", type=int, default=32)
    parser.add_argument("--max-functional-group-tokens", type=int, default=32)
    parser.add_argument("--max-residues-per-chunk", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_word_vectors(path: Path):
    try:
        from gensim.models import KeyedVectors, Word2Vec
    except ImportError as exc:
        raise ImportError("Mol2Vec extraction requires gensim") from exc
    errors: list[str] = []
    for loader in (Word2Vec.load, KeyedVectors.load):
        try:
            value = loader(str(path))
            return value.wv if hasattr(value, "wv") else value
        except Exception as error:
            errors.append(repr(error))
    raise RuntimeError(f"Cannot load Mol2Vec weights {path}:\n" + "\n".join(errors))


def vector_contains(vectors, key: str) -> bool:
    if hasattr(vectors, "key_to_index"):
        return key in vectors.key_to_index
    if hasattr(vectors, "vocab"):
        return key in vectors.vocab
    try:
        vectors[key]
        return True
    except KeyError:
        return False


def mol2alt_sentence(molecule, radius: int = 1) -> list[str]:
    from rdkit.Chem import AllChem

    radii = list(range(radius + 1))
    information: dict[int, tuple[tuple[int, int], ...]] = {}
    AllChem.GetMorganFingerprint(molecule, radius, bitInfo=information)
    by_atom = {
        atom.GetIdx(): {current_radius: None for current_radius in radii}
        for atom in molecule.GetAtoms()
    }
    for identifier, occurrences in information.items():
        for atom_index, current_radius in occurrences:
            by_atom[atom_index][current_radius] = identifier
    return [
        str(by_atom[atom_index][current_radius])
        for atom_index in by_atom
        for current_radius in radii
        if by_atom[atom_index][current_radius] is not None
    ]


def prepare_drug_features(
    drug_ids: list[str],
    smiles: dict[str, str],
    model_path: Path,
    max_tokens: int,
    max_group_tokens: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    try:
        from rdkit import Chem, DataStructs, RDLogger, rdBase
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
    except ImportError as exc:
        raise ImportError("Drug feature extraction requires RDKit") from exc

    vectors = load_word_vectors(model_path)
    vector_size = int(getattr(vectors, "vector_size", 0))
    if vector_size != 300:
        raise ValueError(f"Expected 300-dimensional Mol2Vec weights, got {vector_size}")
    unknown_key = "UNK" if vector_contains(vectors, "UNK") else None
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    compiled_groups = []
    for name, smarts in FUNCTIONAL_GROUP_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise ValueError(f"Invalid SMARTS for {name}: {smarts}")
        compiled_groups.append((name, pattern))

    count = len(drug_ids)
    global_features = torch.zeros((count, vector_size), dtype=torch.float32)
    local_features = torch.zeros((count, max_tokens, vector_size), dtype=torch.float16)
    local_mask = torch.ones((count, max_tokens), dtype=torch.bool)
    ecfp = torch.zeros((count, 2048), dtype=torch.float32)
    descriptors = torch.zeros((count, 9), dtype=torch.float32)
    group_ids = torch.zeros((count, max_group_tokens), dtype=torch.long)
    original_token_counts: list[int] = []
    truncated_drugs = 0
    oov_tokens = 0
    canonical_smiles: list[str] = []

    RDLogger.DisableLog("rdApp.warning")
    try:
        for row, drug_id in enumerate(drug_ids):
            molecule = Chem.MolFromSmiles(smiles[drug_id])
            if molecule is None:
                raise ValueError(f"Invalid SMILES for {drug_id}: {smiles[drug_id]!r}")
            canonical_smiles.append(Chem.MolToSmiles(molecule, canonical=True))
            token_vectors: list[np.ndarray] = []
            for token in mol2alt_sentence(molecule, radius=1):
                key = token if vector_contains(vectors, token) else unknown_key
                if key != token:
                    oov_tokens += 1
                if key is not None:
                    token_vectors.append(np.asarray(vectors[key], dtype=np.float32))
            if not token_vectors:
                raise ValueError(f"Mol2Vec produced no token vectors for {drug_id}")
            matrix = np.stack(token_vectors).astype(np.float32, copy=False)
            global_features[row] = torch.from_numpy(matrix.mean(axis=0))
            used = min(matrix.shape[0], max_tokens)
            local_features[row, :used] = torch.from_numpy(matrix[:used]).half()
            local_mask[row, :used] = False
            original_token_counts.append(int(matrix.shape[0]))
            truncated_drugs += int(matrix.shape[0] > max_tokens)

            fingerprint = fingerprint_generator.GetFingerprint(molecule)
            array = np.zeros(2048, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fingerprint, array)
            ecfp[row] = torch.from_numpy(array)
            descriptors[row] = torch.tensor(
                [
                    Descriptors.MolWt(molecule),
                    Crippen.MolLogP(molecule),
                    Descriptors.TPSA(molecule),
                    Lipinski.NumHDonors(molecule),
                    Lipinski.NumHAcceptors(molecule),
                    Lipinski.NumRotatableBonds(molecule),
                    Lipinski.RingCount(molecule),
                    Descriptors.FractionCSP3(molecule),
                    Descriptors.HeavyAtomCount(molecule),
                ],
                dtype=torch.float32,
            )
            tokens: list[int] = []
            for group_index, (_, pattern) in enumerate(compiled_groups, start=1):
                matches = len(molecule.GetSubstructMatches(pattern, uniquify=True))
                tokens.extend([group_index] * matches)
            if not tokens:
                tokens = [len(compiled_groups) + 1]
            group_used = min(len(tokens), max_group_tokens)
            group_ids[row, :group_used] = torch.tensor(tokens[:group_used], dtype=torch.long)
            if (row + 1) % 100 == 0 or row + 1 == count:
                print(f"Drug features {row + 1}/{count}")
    finally:
        RDLogger.EnableLog("rdApp.warning")

    if not torch.isfinite(descriptors).all():
        raise ValueError("Drug descriptors contain NaN or infinity")
    metadata = {
        "mol2vec_model": str(model_path.resolve()),
        "mol2vec_sha256": sha256_file(model_path),
        "local_policy": f"first {max_tokens} radius-1 alternating Mol2Vec tokens",
        "global_policy": "mean over all recognized Mol2Vec tokens",
        "token_count_min": min(original_token_counts),
        "token_count_max": max(original_token_counts),
        "token_count_mean": float(np.mean(original_token_counts)),
        "truncated_drugs": truncated_drugs,
        "oov_tokens": oov_tokens,
        "canonical_smiles_sha256": sha256_strings(canonical_smiles),
        "rdkit_version": rdBase.rdkitVersion,
        "ecfp": {"radius": 2, "bits": 2048},
        "functional_groups": [name for name, _ in compiled_groups] + ["other"],
    }
    return {
        "drug_global": global_features,
        "drug_local": local_features,
        "drug_padding_mask": local_mask,
        "ecfp": ecfp,
        "descriptors": descriptors,
        "functional_group_ids": group_ids,
    }, metadata


def clean_sequence(sequence: str) -> str:
    cleaned = re.sub(r"[^A-Z]", "", sequence.upper())
    return re.sub(r"[UZOB]", "X", cleaned)


def group_residue_matrix(matrix: np.ndarray, groups: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(f"Invalid residue matrix shape: {matrix.shape}")
    length, dimension = matrix.shape
    group_ids = np.minimum(np.arange(length, dtype=np.int64) * groups // length, groups - 1)
    sums = np.zeros((groups, dimension), dtype=np.float32)
    counts = np.zeros(groups, dtype=np.float32)
    np.add.at(sums, group_ids, matrix.astype(np.float32, copy=False))
    np.add.at(counts, group_ids, 1.0)
    mask = counts == 0
    local = sums / np.maximum(counts[:, None], 1.0)
    global_feature = sums.sum(axis=0) / max(float(counts.sum()), 1.0)
    return global_feature, local, mask


def protein_features_from_author_pickle(
    protein_ids: list[str], path: Path, groups: int
) -> tuple[dict[str, torch.Tensor], dict]:
    with path.open("rb") as handle:
        source = pickle.load(handle)
    missing = [protein_id for protein_id in protein_ids if protein_id not in source]
    if missing:
        raise ValueError(f"Author protein pickle misses {len(missing)} IDs: {missing[:10]}")
    first_value = source[protein_ids[0]]
    first_matrix = np.asarray(first_value[1], dtype=np.float32)
    dimension = int(first_matrix.shape[1])
    global_features = torch.zeros((len(protein_ids), dimension), dtype=torch.float32)
    local_features = torch.zeros((len(protein_ids), groups, dimension), dtype=torch.float16)
    masks = torch.ones((len(protein_ids), groups), dtype=torch.bool)
    lengths: list[int] = []
    for row, protein_id in enumerate(protein_ids):
        value = source[protein_id]
        matrix = np.asarray(value[1], dtype=np.float32)
        global_feature, local, mask = group_residue_matrix(matrix, groups)
        global_features[row] = torch.from_numpy(global_feature)
        local_features[row] = torch.from_numpy(local).half()
        masks[row] = torch.from_numpy(mask)
        lengths.append(int(matrix.shape[0]))
        if (row + 1) % 100 == 0 or row + 1 == len(protein_ids):
            print(f"Converted author protein features {row + 1}/{len(protein_ids)}")
    return {
        "protein_global": global_features,
        "protein_local": local_features,
        "protein_padding_mask": masks,
    }, {
        "source": "ColdstartCPI author aas_ProtTransBertBFD1024.pkl",
        "source_path": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "groups": groups,
        "dimension": dimension,
        "sequence_length_min": min(lengths),
        "sequence_length_max": max(lengths),
        "sequence_length_mean": float(np.mean(lengths)),
    }


def load_protein_transformer(model_name: str, local_files_only: bool):
    try:
        from transformers import BertConfig, BertModel, BertTokenizer
    except ImportError as exc:
        raise ImportError("Protein extraction requires transformers") from exc
    tokenizer = BertTokenizer.from_pretrained(
        model_name,
        do_lower_case=False,
        local_files_only=local_files_only,
    )
    config = BertConfig.from_pretrained(model_name, local_files_only=local_files_only)
    model = BertModel.from_pretrained(
        model_name,
        config=config,
        add_pooling_layer=False,
        local_files_only=local_files_only,
    )
    return tokenizer, model


def encode_protein(
    sequence: str,
    tokenizer,
    model,
    device: torch.device,
    fp16: bool,
    groups: int,
    max_residues: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dimension = int(model.config.hidden_size)
    sums = torch.zeros((groups, dimension), dtype=torch.float32)
    counts = torch.zeros(groups, dtype=torch.float32)
    for start in range(0, len(sequence), max_residues):
        chunk = sequence[start : start + max_residues]
        encoded = tokenizer(
            " ".join(chunk),
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"
            ):
                hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        residue = hidden[0, 1 : len(chunk) + 1].detach().float().cpu()
        if residue.size(0) != len(chunk):
            raise ValueError(
                f"ProtBert token mismatch: residues={len(chunk)}, embeddings={residue.size(0)}"
            )
        absolute = torch.arange(start, start + len(chunk), dtype=torch.long)
        group_ids = torch.clamp(absolute * groups // len(sequence), max=groups - 1)
        sums.index_add_(0, group_ids, residue)
        counts.index_add_(0, group_ids, torch.ones(len(chunk)))
    mask = counts.eq(0)
    local = sums / counts.clamp_min(1).unsqueeze(1)
    global_feature = sums.sum(dim=0) / counts.sum().clamp_min(1)
    return global_feature, local.half(), mask


def protein_features_from_transformers(
    protein_ids: list[str],
    sequences: dict[str, str],
    model_name: str,
    groups: int,
    max_residues: int,
    device_name: str,
    fp16: bool,
    local_files_only: bool,
    cache_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict]:
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    tokenizer, model = load_protein_transformer(model_name, local_files_only)
    model = model.to(device).eval()
    if fp16 and device.type == "cuda":
        model.half()
    dimension = int(model.config.hidden_size)
    globals_out = torch.zeros((len(protein_ids), dimension), dtype=torch.float32)
    locals_out = torch.zeros((len(protein_ids), groups, dimension), dtype=torch.float16)
    masks_out = torch.ones((len(protein_ids), groups), dtype=torch.bool)
    lengths: list[int] = []
    model_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    model_cache = cache_dir / f"{model_key}_groups{groups}"
    model_cache.mkdir(parents=True, exist_ok=True)
    for row, protein_id in enumerate(protein_ids):
        sequence = clean_sequence(sequences[protein_id])
        if not sequence:
            raise ValueError(f"Empty protein sequence for {protein_id}")
        sequence_hash = sha256_strings([sequence])
        cache_path = model_cache / f"{sequence_hash}.pt"
        cached = torch_load(cache_path) if cache_path.exists() else None
        if not isinstance(cached, dict) or cached.get("model") != model_name or cached.get("groups") != groups:
            global_feature, local, mask = encode_protein(
                sequence, tokenizer, model, device, fp16, groups, max_residues
            )
            cached = {
                "model": model_name,
                "groups": groups,
                "sequence_sha256": sequence_hash,
                "global": global_feature,
                "local": local,
                "mask": mask,
            }
            atomic_torch_save(cached, cache_path)
        globals_out[row] = cached["global"].float()
        locals_out[row] = cached["local"].half()
        masks_out[row] = cached["mask"].bool()
        lengths.append(len(sequence))
        if (row + 1) % 20 == 0 or row + 1 == len(protein_ids):
            print(f"Protein features {row + 1}/{len(protein_ids)}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "protein_global": globals_out,
        "protein_local": locals_out,
        "protein_padding_mask": masks_out,
    }, {
        "source": "transformers",
        "model": model_name,
        "groups": groups,
        "dimension": dimension,
        "fp16_inference": bool(fp16 and device.type == "cuda"),
        "sequence_length_min": min(lengths),
        "sequence_length_max": max(lengths),
        "sequence_length_mean": float(np.mean(lengths)),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    legacy_dataset_dir = root / "Datasets" / args.dataset
    legacy_feature_dir = legacy_dataset_dir / "feature"
    entity_dir = (args.dataset_dir or legacy_feature_dir).resolve()
    output_dir = (args.output_dir or root / "V1" / "features" / args.dataset).resolve()
    output_path = output_dir / "v1_features.pt"
    manifest_path = output_dir / "v1_feature_manifest.json"
    if output_path.exists() and not args.force:
        print(f"Feature bundle already exists: {output_path}")
        print("Use --force only for an intentional rebuild.")
        return
    drug_file = entity_dir / "drug_list.txt"
    protein_file = entity_dir / "protein_list.txt"
    mol2vec_model = (
        args.mol2vec_model or root / "Feature_generation" / "Mol2Vec" / "model_300dim.pkl"
    ).resolve()
    for path in (drug_file, protein_file, mol2vec_model):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.drug_max_tokens < 1 or args.protein_groups < 1:
        raise ValueError("Token/group counts must be positive")

    drug_ids, smiles = read_entity_file(drug_file)
    protein_ids, raw_sequences = read_entity_file(protein_file)
    sequences = {protein_id: clean_sequence(value) for protein_id, value in raw_sequences.items()}
    print(f"Dataset: {args.dataset} | drugs={len(drug_ids)} | proteins={len(protein_ids)}")
    drug_tensors, drug_metadata = prepare_drug_features(
        drug_ids,
        smiles,
        mol2vec_model,
        args.drug_max_tokens,
        args.max_functional_group_tokens,
    )

    author_pickle_candidates = [
        entity_dir / "aas_ProtTransBertBFD1024.pkl",
        legacy_feature_dir / "aas_ProtTransBertBFD1024.pkl",
    ]
    author_pickle = next(
        (path.resolve() for path in author_pickle_candidates if path.is_file()),
        author_pickle_candidates[0].resolve(),
    )
    use_author = args.protein_source == "author_pickle" or (
        args.protein_source == "auto" and author_pickle.exists()
    )
    if args.protein_source == "author_pickle" and not author_pickle.exists():
        raise FileNotFoundError(author_pickle)
    if use_author:
        protein_tensors, protein_metadata = protein_features_from_author_pickle(
            protein_ids, author_pickle, args.protein_groups
        )
    else:
        protein_tensors, protein_metadata = protein_features_from_transformers(
            protein_ids=protein_ids,
            sequences=sequences,
            model_name=args.protein_model,
            groups=args.protein_groups,
            max_residues=args.max_residues_per_chunk,
            device_name=args.device,
            fp16=args.fp16,
            local_files_only=args.local_files_only,
            cache_dir=root / "V1" / "cache" / "protein",
        )

    metadata = {
        "schema_version": 1,
        "experiment": "ColdstartCPI_three_datasets_V1",
        "dataset": args.dataset,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "uses_dti_labels": False,
        "entity_source": {
            "dataset_dir": str(entity_dir),
            "drug_list": str(drug_file.resolve()),
            "drug_list_sha256": sha256_file(drug_file),
            "protein_list": str(protein_file.resolve()),
            "protein_list_sha256": sha256_file(protein_file),
        },
        "drug_ids_sha256": sha256_strings(drug_ids),
        "protein_ids_sha256": sha256_strings(protein_ids),
        "protein_sequences_sha256": sha256_strings([sequences[value] for value in protein_ids]),
        "drug": drug_metadata,
        "protein": protein_metadata,
    }
    bundle = {
        "drug_ids": drug_ids,
        "protein_ids": protein_ids,
        **drug_tensors,
        **protein_tensors,
        "metadata": metadata,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(bundle, output_path)
    atomic_json_dump(metadata, manifest_path)
    check = torch_load(output_path)
    if check["drug_global"].shape[0] != len(drug_ids):
        raise RuntimeError("Saved drug feature row count is incorrect")
    if check["protein_global"].shape[0] != len(protein_ids):
        raise RuntimeError("Saved protein feature row count is incorrect")
    print(f"Saved: {output_path}")
    print(
        f"Drug global/local: {tuple(check['drug_global'].shape)} / {tuple(check['drug_local'].shape)}"
    )
    print(
        f"Protein global/local: {tuple(check['protein_global'].shape)} / "
        f"{tuple(check['protein_local'].shape)}"
    )


if __name__ == "__main__":
    main()
