Generative-AI-for-Muography

This repository contains Jupyter notebooks, source code, comprehensive documentation, and presentation materials explaining the implementation.

Muon Propagation Simulation with Generative Adversarial Networks

This project is a Master's thesis focused on the development of a surrogate model for muon propagation in matter using machine learning. The primary goal is to create a high-speed alternative to computationally expensive Monte Carlo simulations (like Geant4) by training a conditional Generative Adversarial Network (GAN) to learn the complex, stochastic physics of muon-matter interactions.

Motivation

Muography is a powerful imaging technique that uses cosmic-ray muons to map the density of large-scale structures. A major bottleneck in this field is the significant computational cost and time required to generate the millions of simulated particle tracks needed for a high-resolution study. This project aims to address this challenge by developing a GAN that can generate physically realistic interaction steps in a fraction of the time, serving as a high-speed complement to traditional simulation methods.

Key Features

Conditional GAN Architecture: An MLP-based conditional GAN built in PyTorch that learns to generate the three key outcomes of a muon interaction step: momentum loss (-ΔP), scattering angle (Δθ), and step length (Δr), conditioned on the muon's energy and the material's properties.

Advanced Data Processing: A custom GANDataset class that handles loading of Geant4 data and applies a specialized two-pipeline normalization strategy to ensure training stability.

Hyperparameter Optimization: Systematic hyperparameter search for the GAN architecture and optimizers using the Optuna framework.

Advanced GAN Architectures: Implementation and comparison of multiple GAN variants, including the original vanilla GAN (2014) and the Wasserstein GAN (WGAN), to address common training challenges.

Evaluation Framework: A suite of callback tools for in-depth model evaluation, including the generation of 1D/2D histograms, pair plots, and Cumulative Distribution Functions (CDFs) to compare generated data against the ground truth.

Environment Setup

This project uses conda for environment management.

Clone the repository:

git clone <your-repository-url>
cd Generative-AI-for-Muography


Create the conda environment. Choose the appropriate file based on your hardware.

CPU:

conda env create -n ENVNAME --file env.yml


GPU:

conda env create -n ENVNAME --file env_gpu.yml


Note: The GPU environment assumes CUDA version 11.8. Please check your CUDA version (e.g., using nvidia-smi) and adapt the .yml file if necessary.

Activate the environment:

conda activate ENVNAME


Project Structure

data/: Directory for storing the input Geant4 .root files (not included in the repo).

notebooks/: Contains Jupyter notebooks for data exploration, model training, and evaluation.

src/: Contains all the core Python scripts and class definitions.

Workflow and Usage

The project is organized into a series of Jupyter notebooks in the notebooks/ directory. For a complete workflow, it is recommended to run them in the following order:

Note: Before running any notebook, you must add the src directory to the system path by including the following lines at the top of the notebook:

import os
import sys
src_path = os.path.abspath('../src/')
if src_path not in sys.path:
    sys.path.append(src_path)
