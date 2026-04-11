# Third-Party Notices

This repository contains code that is partially derived from third-party open-source projects.
Original ownership and license rights remain with the respective upstream authors and projects.

Repository maintainer: **Seohyeon Jang**

## 1. Original repository-authored components

Portions of this repository are original works authored by Seohyeon Jang for a Unity-based multi-agent reinforcement learning environment, including but not limited to:

- Unity environment integration logic
- experiment orchestration and evaluation code
- project-specific environment handling
- configuration and reproducibility metadata prepared for this repository

These original components are distributed under the repository-level license described in the root `LICENSE` file, unless otherwise noted.

## 2. LIIR-derived components

- Upstream repository: `https://github.com/yalidu/liir`
- Upstream project: **LIIR: Learning Individual Intrinsic Reward in Multi-Agent Reinforcement Learning**
- Upstream license: **MIT**
- Typical local scope:
  - IRF / LIIR-related learner code
  - IRF / LIIR-related module code
- Modification status:
  - Adapted for a Unity-based multi-agent reinforcement learning environment
  - Environment interface, execution flow, training pipeline, and input/output handling were modified

### Suggested file header

```python
# Adapted from: https://github.com/yalidu/liir
# Original project: LIIR: Learning Individual Intrinsic Reward in Multi-Agent Reinforcement Learning
# Upstream license: MIT
# Modified by Seohyeon Jang for a Unity-based multi-agent reinforcement learning environment.
```

## 3. PyMARL / COMA-derived components

- Upstream repository: `https://github.com/oxwhirl/pymarl`
- Upstream project: **PyMARL**
- Upstream license: **Apache License 2.0**
- Typical local scope:
  - COMA-related components
  - PyMARL-derived framework or training components, where applicable
- Modification status:
  - Adapted for Unity environment integration
  - Local experiment control flow and evaluation workflow may have been modified

### Suggested file header

```python
# Adapted from: https://github.com/oxwhirl/pymarl
# Original project: PyMARL
# Upstream license: Apache-2.0
# Modified by Seohyeon Jang for Unity environment integration and local training/evaluation workflow.
```

## 4. CDS-derived components

- Upstream repository: `https://github.com/lich14/CDS`
- Upstream project: **CDS**
- Upstream license: **Apache License 2.0**
- Typical local scope:
  - CDS baseline-related components
- Modification status:
  - Adapted for the local Unity-based environment and experiment setup

### Suggested file header

```python
# Adapted from: https://github.com/lich14/CDS
# Original project: CDS
# Upstream license: Apache-2.0
# Modified by Seohyeon Jang for a Unity-based multi-agent reinforcement learning environment.
```

## 5. EMC-related components

- Upstream repository: `https://github.com/kikojay/EMC`
- Upstream project: **EMC**
- License status:
  - Repository-level license status was not clearly verified at the time this repository was organized
- Handling policy in this repository:
  - EMC-specific files are excluded from the claimed original copyright scope unless separately verified
  - If retained locally, they should be treated as third-party code pending further license confirmation

## 6. Repository handling policy

This repository contains three types of code:

1. **Original code**
   - Files written directly by Seohyeon Jang

2. **Modified third-party code**
   - Files adapted from upstream open-source projects for use in the present Unity-based MARL environment

3. **Excluded or separately handled third-party code**
   - Files whose provenance or license status is unclear
   - Files that remain too close to upstream originals to be reasonably claimed as original authorship

## 7. Registration-oriented handling guidance

The following practical classification is recommended for documentation and registration preparation:

### Included as original or substantially modified material
- Original Unity-environment integration code
- Original experiment orchestration and evaluation code
- Substantially modified IRF/LIIR-related files
- Substantially modified COMA/PyMARL-related files
- Substantially modified CDS-related files
- Project-authored metadata files such as:
  - `pyproject.toml`
  - `.python-version`

### Included only with clear attribution
- Files directly adapted from LIIR, PyMARL, or CDS
- Files that still preserve recognizable upstream structure or logic

### Excluded or handled separately
- Files with unclear license provenance
- EMC-specific files unless separately verified
- Files that are nearly identical to upstream originals
- Virtual environments, caches, logs, checkpoints, and generated artifacts

## 8. Non-core generated or environment files

The following are generally outside the claimed original copyright scope:

- `.venv/`
- `__pycache__/`
- `*.pyc`
- training logs
- checkpoints
- generated result artifacts
- Unity executable build outputs, unless separately documented for distribution purposes
- dependency lockfiles such as `uv.lock`, when treated only as reproducibility metadata rather than core authored code

## 9. Practical note

This file is intended to document provenance, attribution, and handling policy for mixed-origin code in this repository.
Where upstream license obligations apply, those obligations remain in effect for the relevant files and components.
