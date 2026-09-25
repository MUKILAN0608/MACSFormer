# MACSFormer for OSCC Detection

Dual-magnification histopathology classification for distinguishing normal oral epithelium from oral squamous cell carcinoma (OSCC).

This repository contains a notebook-based research implementation of **MACSFormer**, a dual-branch transformer that processes paired **100x** and **400x** microscopy images. The two views are fused through morphology-aware and cross-scale modules before binary classification.

> Research software notice: this project is intended for experimentation and evaluation. It is not a medical device and must not be used as a substitute for diagnosis by a qualified pathologist.

## Contents

- [What the notebook does](#what-the-notebook-does)
- [Model](#model)
- [Dataset and pairing](#dataset-and-pairing)
- [Evaluation](#evaluation)
- [Requirements](#requirements)
- [Running the notebook](#running-the-notebook)
- [Outputs](#outputs)
- [Repository layout](#repository-layout)
- [Reproducibility notes](#reproducibility-notes)
- [Limitations](#limitations)
- [License](#license)

## What the notebook does

[`oscc_completed.ipynb`](oscc_completed.ipynb) contains the end-to-end workflow:

1. Select a CUDA device when available.
2. Mount Google Drive and extract `OSCC_DS1.zip`.
3. Discover and pair matching 100x and 400x images.
4. Split cases into training and held-out test sets with class stratification.
5. Apply image transforms and create PyTorch data loaders.
6. Build and train MACSFormer++ with focal loss, AdamW, mixed precision, gradient accumulation, checkpointing, and early stopping.
7. Evaluate predictions with classification and calibration metrics.
8. Generate Grad-CAM and occlusion-sensitivity explanations.
9. Run deletion/insertion explanation evaluation.
10. Train and evaluate ablations without CSMM, AMWG, or CSTE, and profile model cost.

The notebook is a research log as well as an implementation. It contains repeated exploratory and evaluation cells, so cells should be run in logical sections rather than blindly using **Run All**.

## Model

MACSFormer++ uses two parallel image branches:

```text
100x image ─┐
            ├─ hierarchical patch embedding
            ├─ morphology descriptors (AMWG)
            ├─ Swin-style window attention stages
            ├─ cross-stage morphology memory (CSMM)
            ├─ cross-scale token exchange (CSTE)
            └─ adaptive magnification routing (AMR) ─┐
                                                     ├─ classifier
400x image ─┘                                        ┘
```

Main components implemented in the notebook:

- **HierarchicalPatchEmbedding**: convolutional feature extraction before tokenization.
- **AdaptiveMorphologyWindowGenerator (AMWG)**: produces local morphology descriptors from token grids.
- **ModifiedSwinBlock**: window attention with morphology-aware gating and residual refinement.
- **CrossStageMorphologyMemory (CSMM)**: carries morphology information between stages.
- **CrossScaleTokenExchange (CSTE)**: exchanges descriptors between the 100x and 400x branches.
- **AdaptiveMagnificationRouter (AMR)**: learns per-token weights for fusing both magnifications.
- **FocalLoss**: handles the binary classification objective with optional label smoothing.

## Dataset and pairing

The notebook expects a dataset archive at:

```text
/content/drive/MyDrive/OSCC_DS1.zip
```

After extraction, the expected directory structure is equivalent to:

```text
dataset/
└── Histopathological images of oral squamous cell car/
    ├── First set/
    │   ├── 400X normal epithelium/
    │   └── 400x OSCC/
    └── Second set/
        ├── 100x normal epithelium/
        └── 100X OSCC/
```

Images are paired by the case identifier and image number in filenames such as:

```text
Case 10. 37.jpg
```

Only pairs present at both magnifications are retained. The notebook rejects duplicate pair keys, checks that files exist, and verifies that the 100x and 400x members of every pair refer to the same case and image number.

### Split strategy

The split is performed at the **case level**, not the image level. All image pairs from one case remain in the same subset, which prevents near-duplicate images from the same case appearing in both training and test data. The configured split is stratified 80/20 with seed `42`.

## Evaluation

The notebook includes:

- accuracy, precision, recall/sensitivity, specificity, and F1 score;
- ROC curves and ROC-AUC;
- confusion matrices and classification reports;
- expected calibration error (ECE) and Brier score;
- per-class accuracy;
- Grad-CAM overlays for both magnifications;
- patch-based occlusion sensitivity maps;
- deletion/insertion curves and AUC summaries;
- qualitative visualizations of correct and incorrect predictions;
- ablation comparisons for CSMM, AMWG, and CSTE;
- parameter, FLOPs, memory, latency, and throughput profiling.

The notebook contains cells referring to saved checkpoints such as `best_model_finetuned.pth`, `best_model_no_csmm.pth`, `best_model_no_amwg.pth`, and `best_model_no_cste.pth`. These files are generated artifacts and are not included in this repository.

## Requirements

Recommended environment:

- Python 3.10+
- PyTorch and torchvision
- CUDA-enabled GPU with a compatible PyTorch build
- Jupyter Notebook, JupyterLab, or Google Colab

The notebook uses the following Python packages:

```bash
pip install torch torchvision timm numpy pandas pillow opencv-python \
    matplotlib seaborn scikit-learn scipy tqdm thop gdown huggingface_hub
```

Install a PyTorch build appropriate for the local CUDA version by following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/) before installing the remaining packages.

## Running the notebook

### Google Colab

1. Open `oscc_completed.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Upload `OSCC_DS1.zip` to Google Drive at `My Drive/OSCC_DS1.zip`.
4. Run the setup, extraction, path configuration, pairing, and loader cells first.
5. Confirm the pairing and case-leakage checks pass.
6. Define the model and run the training cell.
7. Run evaluation and explainability cells after a checkpoint has been created.
8. Run ablation and profiling cells only after the required baseline variables and checkpoints exist.

### Local Jupyter

The notebook contains Colab-specific cells such as `google.colab.drive.mount` and `/content` paths. For local execution, replace those cells with local dataset paths and remove or skip the Drive mount. Then launch it with:

```bash
jupyter lab oscc_completed.ipynb
```

The notebook does not currently provide a standalone training script or a `requirements.txt` file, so the notebook cell order and variable state are part of the current execution contract.

## Outputs

Depending on the cells executed, the workflow can create:

```text
best_model_finetuned.pth
best_model_no_csmm.pth
best_model_no_amwg.pth
best_model_no_cste.pth
prediction_results.csv
MACSFormer_ablation_complete.csv
MACSFormer_ablation_baseline_delta.csv
MACSFormer_component_importance.csv
explainability_figures_fixed/
explainability_evaluation/
```

The explainability evaluation writes raw curve data, sample-level results, aggregate tables, figures, and representative sample visualizations under `explainability_evaluation/`.

## Repository layout

```text
.
├── LICENSE
├── README.md
├── oscc_100x.ipynb
├── oscc_400x.ipynb
└── oscc_completed.ipynb
```

`oscc_completed.ipynb` is the consolidated notebook described by this README. The 100x and 400x notebooks are additional notebook artifacts in the repository and may contain earlier or specialized experiments.

## Reproducibility notes

- The configured random seed is `42` for Python, NumPy, and PyTorch.
- GPU availability changes memory use, runtime, mixed-precision behavior, and profiling results.
- Data-loader worker settings may need adjustment for Windows or constrained environments.
- Training and evaluation are stateful notebook operations; re-running cells out of order can overwrite models, loaders, metrics, or helper functions.
- Reported metrics should be regenerated from a fresh run with a documented dataset split and checkpoint. The notebook file itself has not been validated as a clean, linear run in this repository state.
- Results from the held-out test set should be reported separately from cross-validation results. The notebook includes both workflows.

## Limitations

- The dataset path and directory names are currently hard-coded for the Colab/Google Drive layout.
- Dataset metadata, subject-level provenance, and external validation are not managed by a separate data pipeline.
- The repository does not include the image dataset or trained checkpoint files.
- The notebook contains duplicate and exploratory cells that would benefit from consolidation into reusable Python modules and a configuration file.
- Performance on this dataset does not establish clinical validity or generalization to other laboratories, scanners, staining protocols, or patient populations.

## License

This project is distributed under the [MIT License](LICENSE).

## Citation

No formal publication citation is included in the repository yet. When using this work, cite the repository and identify the exact notebook version, dataset split, checkpoint, and evaluation protocol used.
