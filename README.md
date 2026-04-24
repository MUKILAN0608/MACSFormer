<div align="center">

```
 ██████╗███████╗ █████╗       ███████╗██╗    ██╗██╗███╗   ██╗
██╔════╝██╔════╝██╔══██╗      ██╔════╝██║    ██║██║████╗  ██║
██║     ███████╗███████║█████╗███████╗██║ █╗ ██║██║██╔██╗ ██║
██║     ╚════██║██╔══██║╚════╝╚════██║██║███╗██║██║██║╚██╗██║
╚██████╗███████║██║  ██║      ███████║╚███╔███╔╝██║██║ ╚████║
 ╚═════╝╚══════╝╚═╝  ╚═╝      ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝  ╚═══╝
```

### 🔬 Cross-Scale Attention Swin Transformer for OSCC Detection

<br/>

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Jupyter](https://img.shields.io/badge/Notebook-dental(1).ipynb-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](./dental(1).ipynb)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-8B5CF6?style=for-the-badge)]()
[![GitHub](https://img.shields.io/badge/GitHub-MUKILAN0608-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MUKILAN0608/CSA-Swin-OSCC-Detection)

<br/>

> *"Empowering early cancer detection through multi-scale vision transformers"*

<br/>

</div>

---

## 🗂️ Table of Contents

| | Section |
|---|---|
| 🔭 | [Overview](#-overview) |
| ⚠️ | [Problem Statement](#-problem-statement) |
| 💡 | [Proposed Solution](#-proposed-solution) |
| 🏗️ | [Architecture](#-architecture) |
| 🗄️ | [Dataset](#-dataset) |
| 📊 | [Results](#-results) |
| 🧪 | [Ablation Study](#-ablation-study) |
| 🔍 | [Explainability](#-explainability) |
| 📁 | [Project Structure](#-project-structure) |
| ⚙️ | [Installation](#-installation) |
| ▶️ | [Usage](#-usage) |
| 🏆 | [Key Contributions](#-key-contributions) |
| 🚀 | [Future Work](#-future-work) |
| 📝 | [Citation](#-citation) |

---

## 🔭 Overview

<div align="center">

```
┌─────────────────────────────────────────────────────────┐
│   🦷  Histopathological Image  →  🤖  CSA-Swin  →  🎯  Diagnosis   │
│                                                         │
│         Normal Epithelium  ✅   or   OSCC  🔴           │
└─────────────────────────────────────────────────────────┘
```

</div>

This project presents a **deep learning-based framework** for the automated classification of oral cavity histopathological images. Early detection of **Oral Squamous Cell Carcinoma (OSCC)** dramatically improves patient survival rates — yet manual diagnosis remains time-consuming, subjective, and expert-dependent.

We solve this with **CSA-Swin**, a novel *Cross-Scale Attention Swin Transformer* that sees what humans might miss.

> 📓 **Everything lives in one file:** [`dental(1).ipynb`](./dental(1).ipynb) — training, evaluation, explainability, all in one place.

---

## ⚠️ Problem Statement

```
❌  Manual diagnosis  →  Slow · Subjective · Expert-only
❌  Single-scale CNNs →  Miss multi-resolution tissue patterns
❌  Black-box models  →  Clinically untrustworthy
❌  Inter-observer variability → Inconsistent outcomes
```

There is a pressing clinical need for a system that is **fast**, **accurate**, **multi-scale**, and **interpretable**.

---

## 💡 Proposed Solution

<div align="center">

### ✨ Introducing **CSA-Swin** ✨
### *Cross-Scale Attention Swin Transformer*

</div>

| 🔑 Feature | 📋 Description |
|:---:|:---|
| 🔭 **Dual-Scale Input** | Captures images at both `100×` and `400×` magnifications |
| 🌳 **Parallel Swin Branches** | Independent hierarchical feature extraction per scale |
| 🔗 **Cross-Attention Fusion** | Dynamically integrates global structure + fine cellular detail |
| 🎨 **Grad-CAM Explainability** | Visual attention maps for clinical validation |
| 🎯 **High Accuracy** | ~95–96% accuracy, AUC >0.94 |

---

## 🏗️ Architecture

```
                    ╔══════════════════════════════════════════╗
                    ║         CSA-Swin Architecture            ║
                    ╚══════════════════════════════════════════╝

  🔬 Input (100×)                           🔬 Input (400×)
       │                                          │
       ▼                                          ▼
┌─────────────────┐                    ┌─────────────────┐
│  🌲 Swin Branch │                    │  🌲 Swin Branch │
│     (100×)      │                    │     (400×)      │
│                 │                    │                 │
│ Global Tissue   │                    │  Fine Cellular  │
│   Structure     │                    │    Details      │
└────────┬────────┘                    └────────┬────────┘
         │   Feature Maps (Scale 1)             │  Feature Maps (Scale 2)
         └──────────────┐  ┌───────────────────┘
                        ▼  ▼
               ╔═══════════════════╗
               ║  🔗 Cross-Scale  ║
               ║ Attention Fusion  ║
               ║     Module        ║
               ╚════════╤══════════╝
                        │  Fused Multi-Resolution Features
                        ▼
               ╔═══════════════════╗
               ║  🎯 Classifier   ║
               ║   Normal / OSCC   ║
               ╚═══════════════════╝
```

**🧩 Core Components:**

- 🌲 **Dual-Branch Swin Transformer** — Patch-based hierarchical features from each magnification scale independently
- 🔗 **Cross-Attention Fusion Module** — Learned cross-scale interaction between global (100×) and local (400×) features
- 🎯 **Classification Head** — Fully connected layers with softmax for binary prediction

---

## 🗄️ Dataset

<div align="center">

| 📌 Property | 📋 Detail |
|:---:|:---:|
| 🖼️ Total Images | **1,200+** |
| 🔭 Magnifications | **100× and 400×** |
| 🏷️ Classes | **Normal Epithelium · OSCC** |
| 🌐 Source | Publicly available histopathological dataset |
| 📂 Format | RGB images `.jpg` / `.png` |

</div>

---

## 📊 Results

### 🏆 Classification Performance

<div align="center">

| 📈 Metric | 🎯 Value |
|:---:|:---:|
| ✅ Overall Accuracy | **~95–96%** |
| 📉 AUC (ROC) | **~0.94+** |
| 🟢 Normal Class Accuracy | **100%** |
| 🔴 OSCC Detection Recall | **~94%** |

</div>

### 📋 Evaluation Outputs

All of the following are generated directly inside [`dental(1).ipynb`](./dental(1).ipynb):

```
📊  Confusion Matrix              🔵  ROC Curve & AUC
📈  Precision-Recall Curve        📋  Per-Class Accuracy Report
🔥  Grad-CAM Heatmaps             🌡️  Occlusion Sensitivity Maps
✅  Correct Predictions           ❌  Misclassified Sample Analysis
```

---

## 🧪 Ablation Study

<div align="center">

| 🧬 Model Variant | 🎯 Accuracy | 📈 AUC |
|:---|:---:|:---:|
| 📷 Single-scale (100× only) | ~0.85 | ~0.88 |
| 📷 Single-scale (400× only) | ~0.87 | ~0.89 |
| 🔭 Dual-scale (no cross-attention) | ~0.89 | ~0.91 |
| 🏆 **CSA-Swin (Proposed)** | **~0.95–0.96** | **~0.94+** |

</div>

```
Performance Gain:
  Single-scale  ████████████░░░░  ~85%
  Dual-scale    ██████████████░░  ~89%
  CSA-Swin      ███████████████▉  ~95–96%  ← PROPOSED
```

> 🔑 Both **dual-scale inputs** and **cross-attention fusion** are essential — the ablation confirms each component contributes meaningfully.

---

## 🔍 Explainability

### 🔥 Grad-CAM *(Gradient-weighted Class Activation Mapping)*

```
  Input Image  →  Swin Features  →  Gradient Maps  →  🔥 Heatmap Overlay
```
- Pinpoints discriminative tissue regions driving each prediction
- Visualized at both 100× and 400× scales for complete coverage

### 🌡️ Occlusion Sensitivity Analysis

```
  Input Image  →  Patch Occlusion  →  Prediction Drop  →  🌡️ Sensitivity Map
```
- Systematically masks patches and measures prediction sensitivity
- Reveals which spatial regions are truly critical for classification

> ✅ Both methods confirm the model attends to **clinically meaningful cellular structures** — not noise or artifacts.

---

## 📁 Project Structure

```
🗂️  CSA-Swin-OSCC-Detection/
│
├── 📓  dental(1).ipynb     ← THE ENTIRE PROJECT LIVES HERE
│                              ├─ Data Loading & Preprocessing
│                              ├─ CSA-Swin Model Definition
│                              ├─ Training & Validation Loop
│                              ├─ Evaluation & Metrics
│                              ├─ Grad-CAM Explainability
│                              └─ Ablation Study
│
└── 📄  README.md
```

> 💡 **No scripts, no modules, no subfolders** — the complete pipeline is self-contained within a single Jupyter notebook.

---

## ⚙️ Installation

### 📋 Prerequisites

```
🐍  Python 3.12+
⚡  CUDA-enabled GPU  (recommended)
📓  Jupyter Notebook or JupyterLab
```

### 📥 Clone the Repository

```bash
git clone https://github.com/MUKILAN0608/CSA-Swin-OSCC-Detection.git
cd CSA-Swin-OSCC-Detection
```

### 📦 Install Dependencies

```bash
pip install torch torchvision timm numpy opencv-python matplotlib scikit-learn grad-cam jupyter
```

---

## ▶️ Usage

### 🚀 Launch the Notebook

```bash
jupyter notebook "dental(1).ipynb"
```

### 📖 Notebook Sections

| 🔢 Step | 📌 Section | 📋 What Happens |
|:---:|:---|:---|
| 1️⃣ | Data Loading & Preprocessing | Load 100× and 400× images, augmentations |
| 2️⃣ | Model Definition | Build CSA-Swin dual-branch + cross-attention |
| 3️⃣ | Training | Train with validation, checkpointing |
| 4️⃣ | Evaluation | Metrics, confusion matrix, ROC, P-R curve |
| 5️⃣ | Explainability | Grad-CAM & occlusion sensitivity maps |
| 6️⃣ | Ablation Study | Compare single-scale vs dual-scale vs CSA-Swin |

---

## 🏆 Key Contributions

```
🥇  Novel multi-scale transformer architecture (CSA-Swin)
🥈  Cross-attention fusion for 100× + 400× feature integration
🥉  ~95–96% accuracy and AUC >0.94 on OSCC classification
🎖️  Clinically interpretable via Grad-CAM & occlusion sensitivity
🎖️  Full ablation study validating each architectural component
```

---

## 🚀 Future Work

- [ ] 🏥 **Clinical Deployment** — Decision support system for pathology labs
- [ ] ⚡ **Real-Time WSI Inference** — Whole-slide image streaming pipeline
- [ ] 🏷️ **Multi-Class Extension** — Classify multiple oral lesion subtypes
- [ ] 🌐 **Federated Learning** — Multi-hospital privacy-preserving training
- [ ] 🔬 **External Validation** — Generalizability on independent clinical datasets
- [ ] 🤖 **Vision-Language Models** — Automated pathology report generation



---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — feel free to use, modify, and distribute with attribution.

---

<div align="center">

```
🔬 Built with passion for early cancer detection 🔬
```

[![GitHub Stars](https://img.shields.io/github/stars/MUKILAN0608/CSA-Swin-OSCC-Detection?style=for-the-badge&color=FFD700&logo=github)](https://github.com/MUKILAN0608/CSA-Swin-OSCC-Detection)
[![GitHub Forks](https://img.shields.io/github/forks/MUKILAN0608/CSA-Swin-OSCC-Detection?style=for-the-badge&color=4ECDC4&logo=github)](https://github.com/MUKILAN0608/CSA-Swin-OSCC-Detection)

<br/>

*If this project helped you, please consider giving it a ⭐ on GitHub!*

</div>
