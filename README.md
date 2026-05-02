# MedMNIST3D Active Learning

This repository implements **Passive Learning** and **Active Learning** strategies on MedMNIST3D datasets, with support for:

* Random / Passive sampling
* Entropy-based acquisition (MC Dropout)
* BALD acquisition
* Gaussian Dropout (custom hook)
* Bernoulli MC Dropout (standard MC dropout)

The implementation is based on ResNet backbones with 3D convolution variants.

---

## Installation

### 1. Create environment

```bash
conda create -n medmnist_al python=3.10 -y
conda activate medmnist_al
```

### 2. Install dependencies

```bash
pip install torch torchvision torchaudio
pip install medmnist matplotlib numpy
pip install acsconv
```

---

## How to Run

### Basic Command

```bash
python main.py \
  --data_flag organmnist3d \
  --output_root ./output \
  --strategy passive \
  --model_flag resnet18 \
  --batch_size 32 \
  --max_epochs 5 \
  --samples_per_round 10 \
  --initial_size 200 \
  --gpu_ids 0 \
  --download
```

---

## Supported Strategies

### 1. Passive / Random Learning

```bash
--strategy passive
```

No uncertainty estimation is used.

---

### 2. Entropy-based Active Learning (Gaussian Dropout)

```bash
--strategy entropy \
--dropout_type gaussian \
--gaussian_drop_prob 0.2 \
--mc_samples 20
```

---

### 3. BALD (Bayesian Active Learning)

```bash
--strategy bald \
--dropout_type gaussian \
--gaussian_drop_prob 0.2 \
--mc_samples 20
```

---

### 4. Bernoulli MC Dropout

```bash
--strategy entropy \
--dropout_type bernoulli_mc \
--bernoulli_mc_drop_prob 0.2 \
--mc_samples 20
```

---

## Important Arguments

| Argument                   | Description                            |
| -------------------------- | -------------------------------------- |
| `--data_flag`              | MedMNIST dataset (e.g. `organmnist3d`) |
| `--strategy`               | passive / random / entropy / bald      |
| `--model_flag`             | resnet18 / resnet50                    |
| `--samples_per_round`      | number of samples added each round     |
| `--initial_size`           | initial labeled set size               |
| `--max_epochs`             | training epochs per round              |
| `--dropout_type`           | gaussian / bernoulli_mc / none         |
| `--gaussian_drop_prob`     | Gaussian dropout strength              |
| `--bernoulli_mc_drop_prob` | MC dropout probability                 |
| `--mc_samples`             | number of stochastic forward passes    |
| `--conv`                   | ACSConv / Conv2_5d / Conv3d            |

---
