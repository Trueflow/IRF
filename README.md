# IRF-main
Unity ML-Agents-based Multi-Agent Reinforcement Learning Tool

## Overview

This repository provides a multi-agent reinforcement learning framework built on a **pre-built Unity ML-Agents environment**.

Since the Unity environment has already been built, users do **not** need to install the Unity project or Unity-side ML-Agents package separately in order to run experiments. Only the Python-side dependencies are required.

This public repository includes:

- the Unity ML-Agents-based simulation environment
- the implementation of the IRF method
- redistributable baseline implementations such as COMA and CDS
- scripts required for training and evaluation
- project configuration and reproducibility metadata

Certain baseline implementations derived from third-party or privately shared sources are **not included in the public repository** due to redistribution, licensing, or permission constraints.

---

## 1. Environment Setup

### 1.1. Python Version

This project is designed for **Python 3.10.12**.

The project root includes a `.python-version` file with the following content:

```text
3.10.12
```

### 1.2. Virtual Environment Setup with `uv`

Open a terminal such as Windows Command Prompt, PowerShell, or Anaconda Prompt, and move to the project root directory.

```bash
cd (IRF-main folder path)
uv venv --python 3.10.12
```

### 1.3. Install Base Dependencies

Install the base dependencies defined in `pyproject.toml`:

```bash
uv sync
```

The base dependencies include:

- `mlagents==1.1.0`
- `mlagents-envs==1.1.0`
- `python-box==7.3.2`

### 1.4. Install PyTorch

PyTorch should be installed according to the user's hardware environment.
Choose **one** of the following options.

#### Option 1. CPU-only

```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
```

#### Option 2. NVIDIA GPU (CUDA 11.8)

This option is recommended for reproducing the original setup.

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
```

#### Option 3. NVIDIA GPU (CUDA 12.1)

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
```

> Note: Install **only one** PyTorch option.

---

## 2. Running the Code

### 2.1. Run `main.py`

After the environment setup is complete, run:

```bash
uv run python src/main.py
```

This command runs the default **MARL(IRF) vs. RSA** setting.

### 2.2. Available Frameworks and Publicly Distributed Algorithms

The following frameworks and algorithms are available in the **public repository**.

#### Frameworks

- `marl`  
  Used for general multi-agent environments.

- `rsa`  
  An agent that selects actions uniformly at random.  
  It is used as the opponent of the learning algorithm.
  - `rsa` should be assigned only to the Red team.

#### Publicly distributed algorithms

- `irf`: Intrinsic Reward Function in MARL, based on Learning Individual Intrinsic Reward (LIIR)
- `coma`: Counterfactual Multi-Agent Policy Gradients (COMA)
- `cds`: Celebrating Diversity in Shared Multi-Agent Reinforcement Learning (CDS)

#### Algorithms excluded from public distribution

The following baseline implementations were used in experiments but are **not included in the public repository**:

- `poca`: MA-POCA-related code is excluded from public release because redistribution permission for the local implementation has not been confirmed
- `emc`: EMC-related code is excluded from public release due to redistribution and licensing considerations

Trained model checkpoints and related experimental artifacts for all evaluated algorithms are provided separately as described below.

---

## 3. Optional Command-Line Arguments

The following command-line arguments are optional.

- `--workerid`  
  Required when running multiple prompts or instances simultaneously.  
  Set this value between `0` and `10`.  
  Default: `0`

- `--graphic`  
  Set this to `True` if you want to visually observe the training process.  
  Default: `False`

Example:

```bash
uv run python src/main.py --workerid 2 --graphic False
```

---

## 4. Notes

- This repository uses a **pre-built Unity environment**, so Unity Editor installation is not required for running experiments.
- However, the Python-side ML-Agents packages must remain compatible with the built environment.
- Some baseline implementations used during the study are intentionally excluded from public distribution because their redistribution status is restricted or unclear.
- The recommended configuration for reproduction is:

  - Python `3.10.12`
  - `mlagents==1.1.0`
  - `mlagents-envs==1.1.0`
  - `python-box==7.3.2`
  - PyTorch `2.2.1` with CUDA `11.8`

---

## 5. Experimental Artifacts and Reproducibility

The experimental artifacts supporting this project are available separately.

They include:

- trained model checkpoints for all evaluated algorithms
- hyperparameter configuration files
- TensorBoard logs
- other result artifacts used for analysis and reporting

These materials are intended to support reproducibility even where some third-party-derived baseline source code is not publicly redistributed in this repository.

---

## 6. Repository Structure

A typical public project structure is as follows:

```text
IRF-main/
├─ src/
│  └─ main.py
├─ pyproject.toml
├─ .python-version
├─ LICENSE
├─ THIRD_PARTY_NOTICES.md
├─ README.md
└─ ...
```

If your actual entry point is different from `src/main.py`, update the execution command accordingly.

---

## 7. Third-Party Code and Redistribution Policy

This repository contains a mixture of:

- original project-authored code
- modified third-party open-source code that can be redistributed
- third-party-derived code that is excluded from public release

For provenance, attribution, and handling policy, see:

- `THIRD_PARTY_NOTICES.md`

Users should consult that file for the scope of included third-party code and the public distribution status of excluded baseline components.
