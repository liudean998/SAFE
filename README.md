<div align="left">

# SAFE

**Structure-Assisted Feature Enhancement for End-to-End Adsorption-Energy Prediction**

[Getting Started](#getting-started) ·  [Training & Inference](#training-and-inference) · [Datasets](#datasets) ·[Models](#models)

---

## Overview

**SAFE** is a framework designed to improve **adsorption energy prediction for the Initial Structure to Relaxed Energy (IS2RE) task**.

SAFE focuses on **structure-assisted feature enhancement**, using structural information to improve the feature representations of IS2RE adsorption-energy prediction models.

The codebase uses **PyTorch**, **PyTorch Geometric**, and the `ocpmodels` framework. It can serve as a starting point for experiments on atomistic graph neural networks, catalyst–adsorbate systems, and related materials.


## Repository Structure

```text
SAFE/
├── configs/               # Experiment and model configurations
├── ocpmodels/             # Models, trainers, tasks, and utilities
├── scripts/               # Preprocessing and experiment utilities
├── data_analyse/          # Data analysis code
├── main.py                # Main command-line entry point
├── environment.yml        # Conda environment specification
├── requirements.txt       # Python dependencies
└── LICENSE.md             # License
```

## Getting Started

### 1. Prerequisites

We recommend using a **Linux** environment with **Conda or Mamba** for dependency management. GPU-accelerated training requires an NVIDIA GPU with a compatible CUDA driver.

The required software environment is specified in `environment.yml`:

```text
Python: 3.9
PyTorch: 1.13.1
CUDA: 11.6
```

### 2. Clone the repository

```bash
git clone https://github.com/liudean998/SAFE.git
cd SAFE
```

### 3. Create the environment

The following uses the repository's `environment.yml`:

```bash
conda env create -f environment.yml
conda activate ocp-models
python -m pip install -e .
```

**Verify the installation:**

```bash
python -c "import torch; import torch_geometric; import ocpmodels; print('PyTorch:', torch.__version__)"
python main.py --help
```

## Training and Inference

SAFE supports both **training from scratch** and **inference using pretrained models**.

The datasets used in this study are described in the [Datasets](#datasets) section, and pretrained model checkpoints are provided in the [Models](#models) section.

### Train from Scratch

To train a model from scratch, specify the corresponding YAML configuration file:

```bash
python main.py \
  --mode train \
  --config "$CONFIG" 
```

Checkpoints and logs are saved to the output directories specified by the experiment configuration.

### Inference with a Pretrained Model

To perform inference using a pretrained checkpoint:

```bash
python main.py \
  --mode predict \
  --config "$CONFIG" \
  --checkpoint /path/to/checkpoint.pt
```

The input dataset and prediction output path are determined by the selected configuration.

## Datasets

The datasets used in this work are available from:

- **HetCataX:** https://hetcatax.com
- **Zenodo:** https://doi.org/10.5281/zenodo.20842824

| Dataset             | Description                                                  | Download                                                     |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **OC20-SurfAds**    | OC20-derived dataset retaining only surface–adsorbate structures | [download](https://zenodo.org/records/20842824/files/OC20-SurfAds.zip?download=1) |
| **OC20-SurfAds-AF** | Filtered subset of OC20-SurfAds used in this work            | [download](https://zenodo.org/records/20842824/files/OC20-SurfAds-AF.zip?download=1) |
| **SAAs-SurfAds**    | SAAs-derived dataset retaining only surface–adsorbate structures | [download](https://zenodo.org/records/20842824/files/CO_CuSAAS_SurfAds.zip?download=1) |

The datasets used in this study were constructed and processed based on the following publicly available source datasets:

- **OC20:** https://fair-chem.github.io/oc20/
- **SAAs:** https://zenodo.org/records/10679574

For reproducing the experiments reported in this work, we recommend using the processed datasets provided through **HetCataX** or **Zenodo**, as these correspond to the data preparation used in the SAFE experiments.

## Models

The table below lists the main models trained on the **OC20-SurfAds** and **OC20-SurfAds-AF** datasets. These pretrained models can be downloaded directly and used for inference or further experiments.

| Model                 | Dataset         | MAE (ID) | Download                                                                          |
| --------------------- | --------------- | -------- |-----------------------------------------------------------------------------------|
| **GemNet-OC**         | OC20-SurfAds    | 0.5743   | [download](https://zenodo.org/records/23294701/files/gemnet_oc_see0.pt?download=1) |
| **GemNet-OC-SAFE**    | OC20-SurfAds    | 0.4878   | [download](https://zenodo.org/records/23294701/files/gemnet_oc_safe_seed0.pt?download=1) |
| **GemNet-OC**         | OC20-SurfAds-AF | 0.4323   | [download](https://zenodo.org/records/23294701/files/gemnet_oc_af.pt?download=1) |
| **GemNet-OC-SAFE**    | OC20-SurfAds-AF | 0.3973   | [download](https://zenodo.org/records/23294701/files/gemnet_oc_safe_af.pt?download=1)                                                                                  |
| **EquiformerV2**      | OC20-SurfAds    | 0.5234   | [download](https://zenodo.org/records/23294701/files/eqv2_seed0.pt?download=1)                                                                                  |
| **EquiformerV2-SAFE** | OC20-SurfAds    | 0.4464   | [download](https://zenodo.org/records/23294701/files/eqv2_safe_seed0.pt?download=1)                                                                                  |
| **EquiformerV2**      | OC20-SurfAds-AF | 0.3939   | [download](https://zenodo.org/records/23294701/files/eqv2-af.pt?download=1)       |                                                                  |
| **EquiformerV2-SAFE** | OC20-SurfAds-AF | 0.3278   | [download](https://zenodo.org/records/23294701/files/eqv2-safe-af.pt?download=1)  |                                                              |

Due to the large number of trained models and checkpoints generated in this study, only the main models are listed above. Additional pretrained models and checkpoints are available from https://doi.org/10.5281/zenodo.23294701.

## Acknowledgements

This repository builds on the [Open Catalyst Project / OCP](https://github.com/Open-Catalyst-Project/ocp) and its broader ecosystem. We acknowledge the OCP contributors and the authors of the model architectures and libraries used in the codebase.

For the original OCP dataset and tasks, see:

> Chanussot, L. *et al.* **Open Catalyst 2020 (OC20) Dataset and Community Challenges.** *ACS Catalysis* (2021). [DOI: 10.1021/acscatal.0c04525](https://doi.org/10.1021/acscatal.0c04525)

If you use a particular model implementation, dataset, or upstream component, please also cite its original publication. 

## License

This repository includes an [MIT License](LICENSE.md). Please retain applicable attribution and license notices for code originating from the Open Catalyst Project and other dependencies.

---

<div align="center">

**SAFE** · [Repository](https://github.com/liudean998/SAFE) · [Issues](https://github.com/liudean998/SAFE/issues)

</div>
