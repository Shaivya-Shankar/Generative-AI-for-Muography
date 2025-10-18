
---

## 🧠 Generative AI for Muography

This repository contains **Jupyter notebooks**, **source code**, **documentation**, and **presentation materials** explaining the full implementation.

---

### 🎯 Project Overview

**Muon Propagation Simulation with Generative Adversarial Networks**

This project is part of a **Master’s thesis** focused on developing a **surrogate model** for muon propagation in matter using **machine learning**.
The primary goal is to create a **high-speed alternative** to computationally expensive **Monte Carlo simulations** (e.g., Geant4) by training a **conditional Generative Adversarial Network (GAN)** to learn the complex, stochastic physics of muon–matter interactions.

---

### 💡 Motivation

**Muography** is a powerful imaging technique that uses cosmic-ray muons to map the density of large-scale structures.
A major bottleneck in this field is the **significant computational cost and time** required to generate millions of simulated particle tracks for high-resolution studies.

This project aims to overcome this challenge by developing a **GAN-based surrogate model** capable of generating **physically realistic muon interaction steps** in a fraction of the time — serving as a **high-speed complement** to traditional simulation methods.

---

### 🔑 Key Features

* **Conditional GAN Architecture**
  An MLP-based conditional GAN built in **PyTorch** that learns to generate the three key outcomes of a muon interaction step:

  * Momentum loss (−ΔP)
  * Scattering angle (Δθ)
  * Step length (Δr)
    Conditioned on the muon’s energy and the material’s properties.

* **Advanced Data Processing**
  A custom `GANDataset` class for loading Geant4 data and applying a **two-pipeline normalization strategy** to ensure training stability.

* **Hyperparameter Optimization**
  Systematic search for optimal architecture and optimizer parameters using the **Optuna** framework.

* **Advanced GAN Variants**
  Implementation and comparison of multiple architectures, including:

  * Vanilla GAN (Goodfellow et al., 2014)
  * Wasserstein GAN (WGAN)

* **Evaluation Framework**
  A suite of callback tools for in-depth evaluation, including:

  * 1D/2D histograms
  * Pair plots
  * Cumulative Distribution Functions (CDFs)
    for comparing generated and ground-truth data.

---

### ⚙️ Environment Setup

This project uses **Conda** for environment management.

#### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Generative-AI-for-Muography
```

#### 2. Create the conda environment

Choose the appropriate file based on your hardware:

**CPU:**

```bash
conda env create -n ENVNAME --file env.yml
```

**GPU:**

```bash
conda env create -n ENVNAME --file env_gpu.yml
```

> **Note:** The GPU environment assumes **CUDA 11.8**.
> Check your CUDA version with:
>
> ```bash
> nvidia-smi
> ```
>
> and update the `.yml` file if needed.

#### 3. Activate the environment

```bash
conda activate ENVNAME
```

---

### 📁 Project Structure

```
Generative-AI-for-Muography/
├── data/         # Input Geant4 .root files (not included)
├── notebooks/    # Jupyter notebooks for data exploration, training, and evaluation
├── src/          # Core Python scripts and class definitions
├── env.yml       # Conda environment (CPU)
├── env_gpu.yml   # Conda environment (GPU)
└── README.md
```

---

### 🚀 Workflow and Usage

The project workflow is organized into Jupyter notebooks within the `notebooks/` directory.
For a full end-to-end run, execute the notebooks in the recommended order listed there.

Before running any notebook, add the `src` directory to your Python path:

```python
import os
import sys
src_path = os.path.abspath('../src/')
if src_path not in sys.path:
    sys.path.append(src_path)
```

---

