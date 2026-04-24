# CSA-Swin-OSCC-Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-orange)
![Notebook](https://img.shields.io/badge/Notebook-dental(1).ipynb-informational?logo=jupyter)

**Automated Oral Squamous Cell Carcinoma Detection via Cross-Scale Attention Swin Transformer**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Proposed Solution](#proposed-solution)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Ablation Study](#ablation-study)
- [Explainability](#explainability)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Contributions](#key-contributions)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)

---

## Overview

This project presents a deep learning-based framework for the **automated classification of histopathological images** of the oral cavity into two classes:

- **Normal Epithelium**
- **Oral Squamous Cell Carcinoma (OSCC)**

Early detection of OSCC is critical for improving patient survival rates. However, manual diagnosis is time-consuming, subjective, and heavily dependent on expert interpretation. To address these challenges, this work proposes a novel multi-scale transformer-based architecture: **CSA-Swin (Cross-Scale Attention Swin Transformer)**.

> **Notebook:** [`dental(1).ipynb`](./dental(1).ipynb) — All code, training, evaluation, and explainability is contained within this single notebook.

---

## Problem Statement

- Manual OSCC diagnosis from histopathology slides is **labour-intensive** and requires specialized expertise.
- Existing deep learning methods often rely on **single-scale inputs**, missing critical multi-resolution tissue patterns.
- High **inter-observer variability** among pathologists leads to inconsistent diagnoses.
- There is a pressing need for **automated, interpretable, and high-accuracy** diagnostic tools.

---

## Proposed Solution

We propose **CSA-Swin (Cross-Scale Attention Swin Transformer)**, which addresses the above limitations through:

| Feature | Description |
|---|---|
| Dual-scale input | Images at `100×` and `400×` magnifications |
| Parallel Swin branches | Hierarchical feature extraction per scale |
| Cross-attention fusion | Discriminative multi-resolution feature integration |
| Explainability | Grad-CAM and occlusion sensitivity maps |

---

## Architecture

```
Input (100×) ──────────────────────────────────────────────────┐
     │                                                          │
     ▼                                                          │
┌──────────────────────┐                                       │
│  Swin Transformer    │ ── Hierarchical Feature Maps (Scale 1) │
│  Branch A  (100×)    │                        │              │
└──────────────────────┘                        ▼              │
                                      ┌──────────────────┐     │
                                      │  Cross-Attention │     │
                                      │  Fusion Module   │     │
                                      └──────────────────┘     │
                                                │              │
┌──────────────────────┐                        ▲              │
│  Swin Transformer    │ ── Hierarchical Feature Maps (Scale 2) │
│  Branch B  (400×)    │                                       │
└──────────────────────┘                                       │
     ▲                                                          │
     │                                                          │
Input (400×) ──────────────────────────────────────────────────┘

                                      │
                                      ▼
                          ┌───────────────────────┐
                          │  Classification Head  │
                          │  (Normal  vs  OSCC)   │
                          └───────────────────────┘
```

**Core Components:**

- **Dual-Branch Swin Transformer** — Extracts hierarchical patch-based features independently for each magnification scale.
- **Cross-Attention Fusion Module** — Enables interaction between features from 100× (global structure) and 400× (fine cellular details) branches.
- **Classification Head** — Fully connected layers with softmax output for binary classification.

---

## Dataset

| Property | Detail |
|---|---|
| Total Images | 1,200+ |
| Magnifications | 100× and 400× |
| Classes | Normal Epithelium, OSCC |
| Source | Publicly available histopathological dataset |
| Format | RGB images (`.jpg` / `.png`) |

---

## Results

### Classification Performance

| Metric | Value |
|---|---|
| Overall Accuracy | ~95–96% |
| AUC (ROC) | ~0.94+ |
| Normal Class Accuracy | 100% |
| OSCC Detection Recall | ~94% |

### Evaluation Outputs

The following are generated and visualized within [`dental(1).ipynb`](./dental(1).ipynb):

- Confusion Matrix
- ROC Curve & AUC
- Precision-Recall Curve
- Per-Class Accuracy Report
- Grad-CAM Heatmaps
- Occlusion Sensitivity Maps
- Correct & Misclassified Sample Visualizations

---

## Ablation Study

| Model Variant | Accuracy | AUC |
|---|---|---|
| Single-scale (100× only) | ~0.85 | ~0.88 |
| Single-scale (400× only) | ~0.87 | ~0.89 |
| Dual-scale (no cross-attention) | ~0.89 | ~0.91 |
| **CSA-Swin (Proposed)** | **~0.95–0.96** | **~0.94+** |

The ablation study confirms that both the dual-scale input strategy and the cross-attention fusion module contribute significantly to overall performance.

---

## Explainability

### Grad-CAM (Gradient-weighted Class Activation Mapping)
- Highlights discriminative regions in histopathological images that drive model predictions.
- Visualizes attention at both 100× and 400× scales.

### Occlusion Sensitivity Analysis
- Systematically occludes patches of the input image.
- Measures prediction sensitivity to identify critical regions for OSCC classification.

Both methods confirm that the model focuses on **clinically relevant cellular and tissue regions**, enhancing interpretability and clinical trust.

---

## Project Structure

```
CSA-Swin-OSCC-Detection/
│
├── dental(1).ipynb      # Complete pipeline: data loading, training,
│                        # evaluation, and explainability
│
└── README.md
```

> All model definitions, training loops, evaluation metrics, and visualizations are self-contained within `dental(1).ipynb`.

---

## Installation

### Prerequisites

- Python 3.12+
- CUDA-enabled GPU (recommended)
- Jupyter Notebook or JupyterLab

### Clone the Repository

```bash
git clone https://github.com/MUKILAN0608/CSA-Swin-OSCC-Detection.git
cd CSA-Swin-OSCC-Detection
```

### Install Dependencies

```bash
pip install torch torchvision timm numpy opencv-python matplotlib scikit-learn grad-cam jupyter
```

---

## Usage

Open and run the notebook end-to-end:

```bash
jupyter notebook "dental(1).ipynb"
```

The notebook is organized into the following sequential sections:

| Section | Description |
|---|---|
| 1. Data Loading & Preprocessing | Load 100× and 400× images, apply augmentations |
| 2. Model Definition | CSA-Swin architecture (dual-branch + cross-attention) |
| 3. Training | Training loop with validation and checkpointing |
| 4. Evaluation | Metrics, confusion matrix, ROC, precision-recall |
| 5. Explainability | Grad-CAM and occlusion sensitivity visualizations |
| 6. Ablation Study | Comparison across model variants |

---

## Key Contributions

1. **Novel Multi-Scale Architecture** — CSA-Swin leverages both 100× and 400× histopathological inputs simultaneously, capturing complementary global and local tissue features.
2. **Cross-Attention Fusion** — A dedicated fusion module enables dynamic, learned interaction between multi-resolution feature maps.
3. **High Classification Performance** — Achieves ~95–96% accuracy and AUC >0.94, outperforming standard single-scale and baseline deep learning models.
4. **Clinical Interpretability** — Grad-CAM and occlusion sensitivity analyses provide visual evidence of clinically meaningful attention.
5. **Comprehensive Evaluation** — Includes ablation study, per-class analysis, precision-recall curves, and misclassification analysis.

---

## Future Work

- [ ] Deployment as a clinical decision support system
- [ ] Real-time whole-slide image (WSI) inference pipeline
- [ ] Extension to multi-class oral lesion classification
- [ ] Integration with federated learning for multi-hospital training
- [ ] Validation on external clinical datasets for generalizability


---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
Made for advancing early detection of Oral Squamous Cell Carcinoma through interpretable deep learning.
</div>
