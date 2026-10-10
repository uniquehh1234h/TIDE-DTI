# TIDE-DTI

TIDE-DTI is a drug–target interaction (DTI) prediction framework designed for compound-cold and semi-inductive evaluation. This repository provides the model and training code, curated dataset splits, split provenance manifests, and fold-specific pretrained checkpoints for **BindingDB_AIBind**, **BioSNAP**, and **DrugBank**.

> The current release focuses on reproducible data partitions and trained models. Please read the [feature requirements](#feature-requirements) before starting a new training run.

## Framework overview

![TIDE-DTI framework](assets/overframework.png)

The complete TIDE-DTI framework contains:

- multi-view drug features: global and local Mol2Vec representations, 2,048-bit ECFP, nine physicochemical descriptors, and functional-group tokens;
- global and grouped residue-level protein representations based on ProtBert-BFD features;
- learnable feature adapters and modality/type embeddings;
- a joint Transformer that conditions drug and protein tokens on one another;
- `DeepGatedInteraction` for explicit drug–protein interaction modeling;
- a scalar interaction logit together with evidential `alpha` outputs;
- weighted BCE and evidential confidence regularization;
- dynamic mixed hard-negative training;
- drug-cold early stopping, validation-based threshold selection, and evaluation under multiple positive-to-negative ratios.

The joint Transformer and `DeepGatedInteraction` are components of the complete framework rather than standalone replacements for it.

## Repository contents

```text
TIDE-DTI/
├── assets/
│   └── overframework.png
├── Datasets and Splits/
│   ├── BindingDB_AIBind/
│   │   ├── full_pair.csv
│   │   ├── split_fold0.pt ... split_fold4.pt
│   │   ├── split_fold0_manifest.json ... split_fold4_manifest.json
│   │   └── split_manifest_all_folds.json
│   ├── BioSNAP/
│   │   ├── drug_list.txt
│   │   ├── protein_list.txt
│   │   ├── full_pair.csv
│   │   ├── split_fold0.pt ... split_fold4.pt
│   │   ├── split_fold0_manifest.json ... split_fold4_manifest.json
│   │   └── split_manifest_all_folds.json
│   └── DrugBank/
│       ├── Dataset_drug_protein.pkl
│       ├── drugs_smiles.json
│       ├── protein_sequences.csv
│       └── split_fold1.pt ... split_fold10.pt
└── train/
    ├── BindingDB_AIBind/
    │   ├── best_model_fold0.pt ... best_model_fold4.pt
    │   ├── requirements.txt
    │   └── scripts/
    ├── BioSNAP/
    │   ├── best_model_fold0.pt ... best_model_fold4.pt
    │   ├── requirements.txt
    │   └── scripts/
    └── DrugBank/
        ├── best_model_fold1.pt ... best_model_fold10.pt
        └── scripts/
```

Files named `semi_inductive_common_v8.txt` contain shared Python source that is loaded dynamically by the training programs. They are support modules and should not be run directly. The DrugBank training and feature-preparation programs also use a `.txt` suffix but can be executed with Python.

## Datasets and evaluation protocols

| Dataset | Entities in the provided data | Splits | Evaluation setting |
|---|---:|---:|---|
| BindingDB_AIBind | 6,788 drugs; 4,472 proteins; 50,312 rows in `full_pair.csv` | 5 folds (`0–4`) | Author S2 compound-cold test sets; cold validation is rebuilt from training drugs |
| BioSNAP | 4,501 drugs; 2,181 proteins; 27,438 rows in `full_pair.csv` | 5 folds (`0–4`) | Deterministic five-fold drug-cold assignment with disjoint train/validation/test drug sets |
| DrugBank | 707 drugs; 1,489 proteins | 10 folds (`1–10`) | Strict semi-inductive drug-cold evaluation with known proteins |

For BindingDB_AIBind and BioSNAP, the JSON manifests record source hashes, entity counts, positive counts, negative-sampling semantics, and fold construction details. These manifests are the authoritative description of the released splits.

The formal evaluation ratios are:

```text
1:1, 1:2, 1:5, 1:10, 1:20, 1:50, and 1:100
```

Unlabelled drug–protein pairs used by the split builders are treated as negatives. Validation and test negative pools are nested so that results at different ratios can reuse a consistent ordered pool.

## Installation

Model checkpoints are stored with Git LFS. Install Git LFS before cloning or pulling the repository:

```bash
git lfs install
git clone https://github.com/uniquehh1234h/TIDE-DTI.git
cd TIDE-DTI
git lfs pull
```

Create a Python environment and install the supplied dependencies. Python 3.9 or newer is required by the type annotations used in the scripts.

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r train/BioSNAP/requirements.txt
```

The BioSNAP requirements file is the most complete supplied environment list. Install the PyTorch build that matches your CUDA version before installing PyTorch Geometric when using a GPU.

## Feature requirements

The serialized split files and pretrained fold checkpoints are included, but the large precomputed feature bundles used by the training scripts are **not included in this release**.

Training expects the following feature artifacts:

| Dataset | Required feature files |
|---|---|
| BindingDB_AIBind | `v1_features.pt` and the adjacent `v1_feature_manifest.json` |
| BioSNAP | `v1_features.pt` and the adjacent `v1_feature_manifest.json` |
| DrugBank | `v1_finegrained_features.pt`, `v1_hierarchical_drug_features.pt`, and `v1_aligned_feature_manifest.json` |

The feature pipeline additionally requires a 300-dimensional Mol2Vec model. Protein features are obtained from `Rostlab/prot_bert_bfd` or from compatible author-provided residue embeddings.

### Generate BioSNAP features

BioSNAP includes the drug SMILES and protein sequences needed by the provided feature script. Run the following command from the repository root, replacing the Mol2Vec path:

```bash
python train/BioSNAP/scripts/prepare_biosnap_features.py \
  --root . \
  --dataset BioSNAP \
  --dataset-dir "Datasets and Splits/BioSNAP" \
  --output-dir "features/BioSNAP" \
  --mol2vec-model "/path/to/model_300dim.pkl" \
  --protein-source transformers \
  --device cuda \
  --fp16
```

Use `--device cpu` and omit `--fp16` when CUDA is unavailable. The released BioSNAP splits can be used directly; split regeneration is optional:

```bash
python train/BioSNAP/scripts/prepare_biosnap_splits.py \
  --dataset-dir "Datasets and Splits/BioSNAP" \
  --feature-path "features/BioSNAP/v1_features.pt" \
  --output-dir "Datasets and Splits/BioSNAP"
```

BindingDB_AIBind and DrugBank feature reconstruction depends on additional source assets referenced by their preparation scripts. Supply compatible feature bundles explicitly when running those datasets.

## Training

All commands below are intended to be run from the repository root.

### BindingDB_AIBind

```bash
python train/BindingDB_AIBind/scripts/train_ours_bindingdb_aibind.py \
  --feature-path "/path/to/BindingDB_AIBind/v1_features.pt" \
  --hierarchical-feature-path "/path/to/BindingDB_AIBind/v1_features.pt" \
  --split-cache-dir "Datasets and Splits/BindingDB_AIBind" \
  --dataset-dir "Datasets and Splits/BindingDB_AIBind" \
  --save-dir "outputs/BindingDB_AIBind" \
  --device cuda
```

To train only selected folds, add—for example—`--folds 0 1`. Use `--resume` to continue an interrupted fold from its atomic epoch checkpoint.

### BioSNAP

```bash
python train/BioSNAP/scripts/train_ours_biosnap.py \
  --feature-path "features/BioSNAP/v1_features.pt" \
  --hierarchical-feature-path "features/BioSNAP/v1_features.pt" \
  --split-cache-dir "Datasets and Splits/BioSNAP" \
  --dataset-dir "Datasets and Splits/BioSNAP" \
  --save-dir "outputs/BioSNAP" \
  --device cuda
```

### DrugBank

Although the main DrugBank program has a `.txt` suffix, it is a Python script:

```bash
python train/DrugBank/scripts/train_hierarchical_features_ablation_v1.txt \
  --ablation full \
  --feature-path "/path/to/DrugBank/v1_finegrained_features.pt" \
  --hierarchical-feature-path "/path/to/DrugBank/v1_hierarchical_drug_features.pt" \
  --split-cache-dir "Datasets and Splits/DrugBank" \
  --save-dir "outputs/DrugBank" \
  --device cuda
```

The DrugBank runner supports the following controlled settings:

- `full`: complete TIDE-DTI framework;
- `no_kl`: removes both evidential confidence regularizers;
- `no_dynamic`: performs the first mixed hard-negative refresh and then freezes it;
- `no_hard_negative`: replaces score-mined hard negatives with random negatives.

## Evaluation and outputs

Training uses cold-validation AUPR for early stopping. Ratio-specific decision thresholds are selected on validation data, and the shared evaluation code reports:

- Accuracy, Precision, Recall, F1, and MCC;
- ROC-AUC and AUPR;
- BEDROC;
- enrichment factors at the configured early-retrieval fractions.

A completed run writes fold checkpoints and result tables to the selected `--save-dir`. The BindingDB_AIBind and BioSNAP runners also support resumable maximum-ratio score caches so interrupted large evaluations can continue without rescoring completed chunks.

## Pretrained checkpoints

The repository provides:

- five BindingDB_AIBind checkpoints (`fold0–fold4`);
- five BioSNAP checkpoints (`fold0–fold4`);
- ten DrugBank checkpoints (`fold1–fold10`).

Each checkpoint stores at least the model state, model configuration, fold identity, selected epoch, validation AUPR, feature provenance, split path, and training-protocol metadata. A checkpoint can be inspected with:

```python
import torch

checkpoint = torch.load(
    "train/BioSNAP/best_model_fold0.pt",
    map_location="cpu",
    weights_only=False,
)

print(checkpoint["fold"])
print(checkpoint["model_config"])
print(checkpoint["best_val_aupr"])
```

## Reproducibility notes

- Use the released split files instead of generating a new random split when comparing models.
- Keep the model-specific input feature package and outer split fixed when comparing an original model with the complete TIDE-DTI framework.
- The JSON manifests and checkpoint provenance fields contain hashes that can be used to audit the data and feature artifacts.
- Formal runs use seed `42`, a training negative ratio of `1:2`, and validation-based early stopping unless explicitly labelled as a sensitivity or smoke-test setting.

## Repository

GitHub: <https://github.com/uniquehh1234h/TIDE-DTI>

