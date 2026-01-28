# Sobolev-Regularized Score Difference Estimator

This repository contains code for three experiments demonstrating Sobolev-regularized score difference estimation.

---

## Environment Setup

### Install Dependencies

We recommend using a conda environment. The project has been tested with Python 3.8+ and CUDA 11.8.

#### Option 1: Using requirements.txt (Recommended)

```bash
# Create a new conda environment (optional)
conda create -n ecg_env python=3.8
conda activate ecg_env

# Install PyTorch with CUDA support (adjust CUDA version as needed)
# For CUDA 11.8:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

#### Option 2: Using the provided ecg_env environment

If you have access to the original `ecg_env` conda environment, you can activate it directly:

```bash
conda activate ecg_env
```

The `requirements.txt` file contains all necessary dependencies extracted from the `ecg_env` environment configuration.

### Key Dependencies

- **PyTorch** (>=2.0.0) - Deep learning framework
- **NumPy, SciPy** - Scientific computing
- **scikit-learn, scikit-image** - Machine learning utilities
- **POT** - Python Optimal Transport library
- **wfdb** - ECG data processing
- **fastai** - Deep learning utilities
- **einops** - Tensor operations
- **matplotlib** - Visualization
- **pandas, h5py** - Data processing

---

## Experiments Overview

| Experiment | Description | Directory |
|------------|-------------|-----------|
| 1. Simulation | 2D simulation comparing Baseline, Sobolev, and Local Linear methods | Root |
| 2. WGF Domain Adaptation | Wasserstein Gradient Flow for domain adaptation | `gradest2/` |
| 3. ECG Transfer Learning | Transfer learning for diffusion models on ECG data | `SSSD-ECG-main/` |

---

## Experiment 1: Simulation

Compare score difference estimation methods (Baseline, Baseline+Sobolev, Local Linear) on 2D distributions.

### Run

```bash
python sim_score_diff_eval.py
```

### Output

Results are saved in `sim_exp_result/`:
- `distributions_*.png` - Source and target distribution visualizations
- `error_vs_samples_*.png` - Estimation error vs sample size plots
- `results_comparison.csv` - Numerical results table

### Distribution Types
- **Rotated Ridge**: Gaussian mixture with rotated covariance
- **Orthogonal GMM**: Gaussian mixture with orthogonal means
- **Bounded**: Uniform source to cosine-modulated target

---

## Experiment 2: WGF Domain Adaptation

Wasserstein Gradient Flow for domain adaptation on Caltech-Office dataset.

### Run

```bash
cd gradest2
python demo_caltechoffice.py
```

---

## Experiment 3: ECG Transfer Learning

Transfer learning for diffusion models on ECG datasets (PTB-XL → ICBEB).

### Step 1: Download Data

```bash
cd SSSD-ECG-main/src/sssd/Diffusion_RL/ecg_ptbxl_benchmarking
bash get_datasets.sh
```

This downloads:
- **PTB-XL**: Large ECG dataset (source domain)
- **ICBEB**: China Physiological Signal Challenge dataset (target domain)

### Step 2: Prepare Pretrained Models

#### 2.1 Train Pretrained Diffusion Model

```bash
cd SSSD-ECG-main/src/sssd
python train.py --config config/config_SSSD_ECG.json
```

#### 2.2 Train Domain Classifier (DRE)

```bash
cd SSSD-ECG-main/src/sssd
python train_DRE.py
```

### Step 3: Train Guidance Network

#### Option A: Train with Sobolev Regularization (Recommended)

```bash
cd SSSD-ECG-main/src/sssd
python train_density_ratio_net.py \
    --use_sobolev \
    --device cuda:0 \
    --total_iters 1000 \
    --experiment_name my_exp
```

#### Option B: Train Naive Classifier

```bash
python train_density_ratio_net.py \
    --device cuda:0 \
    --total_iters 1000 \
    --experiment_name my_exp
```

#### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_sobolev` | False | Enable Sobolev regularization |
| `--no_decay` | False | Keep Sobolev λ constant (default: decay) |
| `--lambda_grad` | 5.0 | Sobolev penalty weight |
| `--device` | cuda:1 | GPU device |
| `--total_iters` | 1000 | Training iterations |

### Step 4: Generate Samples

```bash
cd SSSD-ECG-main/src/sssd
CUDA_VISIBLE_DEVICES=0 python inference_guided_ptbxl.py \
    --use_sobolev \
    --experiment_name my_exp
```

### Step 5: Evaluate Results

#### Downstream Classifier Performance

```bash
python test_classifiers.py \
    --experiment_name my_exp \
    --use_sobolev
```

#### FID Score

```bash
python evaluate_fid_coverage.py \
    --synthetic_data output/guided_my_exp/ch256_T200_betaT0.02/gen_data_clf_sobd.npy
```

### Complete Pipeline Example

```bash
cd SSSD-ECG-main/src/sssd

# 1. Train Sobolev-regularized classifier
python train_density_ratio_net.py --use_sobolev --device cuda:0 --experiment_name exp1

# 2. Generate samples
CUDA_VISIBLE_DEVICES=0 python inference_guided_ptbxl.py --use_sobolev --experiment_name exp1

# 3. Evaluate
python test_classifiers.py --experiment_name exp1 --use_sobolev
python evaluate_fid_coverage.py --synthetic_data output/guided_exp1/ch256_T200_betaT0.02/gen_data_clf_sobd.npy
```

---
