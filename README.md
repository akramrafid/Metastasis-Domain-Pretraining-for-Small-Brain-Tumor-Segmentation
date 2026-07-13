# Metastasis-Domain Pretraining for Small Brain Tumor Segmentation

This repository contains the official codebase for:
**"Metastasis-Domain Pretraining for Small Brain Tumor Segmentation: Closing the Specificity Gap in Limited-Data Settings"**

An industry-grade, reproducible research codebase isolating domain pretraining (glioma vs metastasis) as the single variable in self-supervised learning (SSL) to close the specificity gap in small brain metastasis cohorts.

---

## 1. Directory Structure

```
brain-mets-domain-pretrain/
├── configs/                 # Hydra configuration files
│   ├── base.yaml            # Main hyperparameters, seed, W&B settings
│   ├── data/                # Dataset-specific modality and mask configurations
│   ├── model/               # Swin UNETR architecture parameters
│   └── experiment/          # Experiment overrides for Arm 0, 1, 2, and eval
├── src/                     # Source library modules
│   ├── data/                # Preprocessing, Dataset, Dataloader factory
│   ├── models/              # Swin UNETR wrapper and SSL dual-head network
│   ├── training/            # SSL Pretraining, Fine-tuning loops, losses
│   └── evaluation/          # Sliding-window cohort evaluator and Wilcoxon test
├── scripts/                 # Executable entrypoints
├── tests/                   # Pytest test suite
├── docker/                  # Dockerfile for complete replication
├── environment.yaml         # Conda environment definition
├── dvc.yaml                 # DVC pipeline stages definition
└── README.md
```

---

## 2. Environment Setup

### Option A: Conda Environment (Local)
Create the environment using the provided `environment.yaml`:
```bash
conda env create -f environment.yaml
conda activate brain-mets-domain-pretrain
```

### Option B: Docker Container
Build the reproducible Docker image:
```bash
docker build -t brain-mets-pretrain -f docker/Dockerfile .
```
Run container interactively:
```bash
docker run --gpus all -it -v /path/to/data:/app/data brain-mets-pretrain bash
```

---

## 3. Reproduction Workflow

### Step 1: Data Exploration & Validation
Inventory all cohorts and check NIfTI headers for shape/spacing consistency:
```bash
python -m scripts.explore_data
```

### Step 2: Modality Preprocessing
Resample all cohorts to 1mm³ isotropic, align to RAS orientation, skull-strip, and Z-score normalize:
```bash
python -m scripts.preprocess_all
```
*For a quick test run on 2 cases per cohort, use the `--limit` flag:*
```bash
python -m scripts.preprocess_all --limit 2
```

### Step 3: Run Experiment Arms

#### Arm 1: Glioma pretraining baseline (mismatched domain)
Pretrain Swin UNETR on glioma (BraTS-GLI) and fine-tune on metastasis:
```bash
python -m scripts.run_arm1 --config-name=base experiment=arm1_glioma_pretrain_baseline
```

#### Arm 2: Metastasis pretraining proposed (matched domain)
Pretrain Swin UNETR on metastasis (Pretreat-MetsToBrain) and fine-tune on metastasis:
```bash
python -m scripts.run_arm2 --config-name=base experiment=arm2_metastasis_pretrain_proposed
```

#### Arm 0: Loss-based baseline
Fine-tune the glioma-pretrained model using the Sensitivity-Specificity Tradeoff loss (Huang et al., 2022) to evaluate a loss-engineering fix:
```bash
python -m scripts.run_arm0 --config-name=base experiment=arm0_glioma_pretrain_lossfix
```

### Step 4: Run External Evaluations
Evaluate the three fine-tuned model checkpoints on the metastasis test split, Stanford BrainMetShare, and UCSF-BMSR:
```bash
python -m scripts.run_external_eval --config-name=base experiment=external_eval
```

### Step 5: Statistical Significance & Paper Tables
Execute Wilcoxon signed-rank significance tests comparing patient-wise specificity/Dice across arms and export LaTeX / CSV tables:
```bash
python -m scripts.generate_paper_tables
```
The resulting LaTeX table can be copied directly into the paper, and the CSV file is saved under `outputs/results/paper_results_table.csv`.

---

## 4. Testing
Run the complete unit test suite validating preprocessing, modality zero-filling, custom losses, and evaluation metrics:
```bash
pytest
```
