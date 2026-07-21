# DPT-GNN: Transition-Aware Dual-Phase Graph Transformer for Gravimetric Capacity Prediction of Lithium-Ion Battery Cathodes

[![Paper](https://img.shields.io/badge/Paper-NXMATE--D--26--05704-blue)](https://doi.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official code repository for the paper:

> **"Transition-Aware Dual-Phase Graph Transformer for Gravimetric Capacity Prediction of Lithium-Ion Battery Cathodes"**

## Overview

DPT-GNN is a shared-attention graph Siamese Transformer that predicts the gravimetric capacity of lithium-ion battery cathode materials by jointly encoding the charged and discharged crystal structures of each voltage step. Rather than treating capacity as a single-structure regression, the model frames it as a **transition-aware learning problem**, capturing the structural evolution between two electrochemical phases.

### Key Design Features

- **Dual-CIF Siamese Encoder** — A shared 4-layer, 8-head distance-biased Graph Transformer encodes both charged and discharged crystal structures into a common latent space.
- **Three-Term Relation Block** — Models inter-phase structural transitions through signed difference, element-wise product, and absolute difference, projecting 774 → 256 dimensions.
- **Numeric Feature Branch** — Eight physicochemical descriptors (voltage, stability, stoichiometry) are fused with the structural embeddings.
- **Huber–Pearson Composite Loss** — Combines robust regression with correlation-preserving optimisation.

### Results

| Split | MAE (mAh g⁻¹) | RMSE (mAh g⁻¹) | MAPE (%) | R² |
|:---|:---|:---|:---|:---|
| Train (2,314) | 1.543 | 4.699 | 1.55 | 0.9959 |
| Validation (330) | 1.402 | 2.875 | 1.48 | 0.9971 |
| Test (695) | **1.761** | 7.835 | **1.83** | **0.9866** |

DPT-GNN achieves a **73.3% reduction** in test MAE compared to the reported mCGCNN baseline (6.6 → 1.761 mAh g⁻¹).

## Repository Contents

```
DPT-GNN-Repo/
├── README.md                   # This file
├── DPT_GNN_Model.ipynb         # Complete training and evaluation pipeline
└── data/                       # Dataset (see Dataset section below)
```

## ⚠️ Naming Conventions

During development, the model and features used internal laboratory naming conventions. To guarantee exact reproducibility of the reported weights and results, **no functional code has been altered**. The table below maps the code names to the submitted manuscript terminology.

### Model Name

| Published Manuscript | Source Code |
|:---|:---|
| **DPT-GNN** (Dual-Phase Transition Graph Neural Network) | `BattGTXv7` |

### Feature Name Mapping

The eight physicochemical descriptors in `CFG.NUM_FEATURES` map to the paper's Table 1 as follows:

| # | Code Variable (`CFG.NUM_FEATURES`) | Paper Name (Table 1) | Preprocessing |
|:---|:---|:---|:---|
| 1 | `max_delta_volume` | Maximum volume change (ΔV) | Clipped [0, 10], log1p |
| 2 | `average_voltage` | Average voltage (V) | Clipped [−2, 6] |
| 3 | `stability_charge` | Charged-state hull energy (Eₕ,c) | Clipped [0, 0.2], log1p |
| 4 | `delta_stability` | Hull energy asymmetry (ΔEₕ) | Clipped [−1, 1] |
| 5 | `delta_fracA` | Working-ion fraction change (Δx) | Robust scaling |
| 6 | `fracA_charge` | Li site fraction, charged (xc) | Robust scaling |
| 7 | `fracA_discharge` | Li site fraction, discharged (xd) | Robust scaling |
| 8 | `total_steps_in_battery` | Voltage-step count (Nsteps) | Linear normalisation to [0, 1] |

### Architecture Hyperparameters

| Hyperparameter | Value | Code Reference |
|:---|:---|:---|
| Hidden dimension | 256 | `CFG.HIDDEN` |
| Attention heads / head dim | 8 / 32 | `CFG.HEADS`, `CFG.HEAD_DIM` |
| Transformer layers | 4 | `CFG.LAYERS` |
| Atom feature dimension | 16 | `CFG.ATOM_DIM` |
| FFN inner dimension | 1024 (4 × 256) | `H * 4` in `MultiHeadGraphTransformer` |
| Dropout | 0.15 (head/numeric); 0.075 (FFN) | `CFG.DROPOUT` |
| Total trainable parameters | 3,657,362 | Verified programmatically |

## Dataset

The dataset comprises **3,341 discrete voltage plateaus** extracted from the [Materials Project Battery Explorer](https://next-gen.materialsproject.org/batteries) using the MP-API. Each data point consists of:

- A **charged-state CIF** and a **discharged-state CIF** (crystal structure files)
- **8 physicochemical descriptors** (see table above)
- The **gravimetric capacity** target (mAh g⁻¹)

The data is split at the **battery-ID level** (not at the data-point level) using `GroupShuffleSplit` to prevent data leakage between structurally related voltage steps:

| Split | Samples | Unique Batteries |
|:---|:---|:---|
| Train | 2,314 (69.3%) | — |
| Validation | 330 (9.9%) | — |
| Test | 695 (20.8%) | 252 |

## Usage

The entire pipeline is contained in `DPT_GNN_Model.ipynb`, designed for **Google Colab with GPU**.

### Quick Start

1. Clone this repository:
   ```bash
   git clone https://github.com/Raza786R/DPT-GNN.git
   ```
2. Open `DPT_GNN_Model.ipynb` in Google Colab.
3. Run all cells sequentially from top to bottom.

### Dependencies

- Python 3.10+
- PyTorch 2.x (with CUDA)
- Pymatgen
- SHAP
- scikit-learn
- matplotlib, seaborn

## Citation

If you use this code or methodology, please cite:

```
@article{dptgnn2026,
  title={Transition-Aware Dual-Phase Graph Transformer for Gravimetric Capacity Prediction of Lithium-Ion Battery Cathodes},
  year={2026},
  note={Under review}
}
```

## License

This project is licensed under the MIT License.
