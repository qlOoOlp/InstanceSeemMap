# Installation Guide
---
- This venv is configured to run the following packages:
  - VLMaps
  - SEEM
## General Environment
- Linux System
- CUDA enabled GPU with Memory > 8GB (Evaluation)
- CUDA enabled GPU with Memory > 12GB (Training)

## Installation
### Create venv
```sh
conda create -n vlmaps_comb python=3.8
```
### Dependencies
```sh
conda install mpi4py
pip install -r assets/requirements/requirements.txt
pip install -r assets/requirements/requirements_git.txt
pip install torchshow
```