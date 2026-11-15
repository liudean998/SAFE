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

**Note:** If the installation is incomplete or dependency-related errors occur, you can try running the training command. Install any missing dependencies indicated by the error messages in the activated environment, and then rerun the command.

## Training and Inference

SAFE supports both **training from scratch** and **inference using pretrained models**.

The datasets used in this study are described in the [Datasets](#datasets) section, and pretrained model checkpoints are provided in the [Models](#models) section.

### Train from Scratch

To train a model from scratch, specify the corresponding YAML configuration file:

```bash
python main.py --mode train --config-yml "$CONFIG" --seed $SEED
```

Here, `$CONFIG` is a placeholder for the path to a YAML configuration file, which can be specified either as an **absolute path** or as a **relative path from the project root directory**. The `--seed` argument specifies the random seed used for reproducibility (default: `0`).

By default, trained model checkpoints are saved in the `checkpoints/` directory, training logs are stored in the `logs/` directory, and prediction results are saved in the `results/` directory.

**Example:**

```bash
python main.py --mode train --config-yml configs/is2re-safe-paper/SAAs/equiformer_v2.yml --seed 0
```

### Multi-GPU Training

For faster training, multiple GPUs can be used with PyTorch Distributed Data Parallel (DDP). For example, to train a model using two GPUs on a single node, run:

```bash
torchrun --standalone --nproc_per_node=$NUM_GPUS main.py --num-gpus $NUM_GPUS --mode train --config-yml "$CONFIG" --seed $SEED
```

**For Example:**

```bash
torchrun --standalone --nproc_per_node=2 main.py --num-gpus 2 --mode train \
--config-yml configs/is2re-safe-paper/SAAs/equiformer_v2.yml --seed 0
```

The arguments are explained below:

- `torchrun`: Launches distributed training processes using PyTorch.
- `--standalone`: Enables standalone distributed training on a single node.
- `--nproc_per_node`: Launches two training processes on the node, typically one per GPU.
- `--num-gpus`: Specifies the total number of GPUs used by the training script.
- `--mode train`: Sets the execution mode to training.
- `--config-yml "$CONFIG"`: Specifies the YAML configuration file containing the model architecture, dataset paths, and training hyperparameters.
- `--seed $SEED`: Sets the random seed.

The number of GPUs can be adjusted according to the available hardware. For single-node training, `--nproc_per_node` and `--num-gpus` should be set consistently with the number of GPUs used.

### Inference with a Pretrained Model

To perform inference using a pretrained checkpoint:

```bash
python main.py --mode predict --config-yml "$CONFIG" --checkpoint "$CHECKPOINT"
```

The input dataset and prediction output path are determined by the selected configuration.

**For Example:**

```bash
python main.py --mode predict --config-yml configs/is2re-safe-paper/SAAs/equiformer_v2.yml \
--checkpoint example/eqv2-saas0-seed0.pt
```

### Evaluation

After training or performing prediction, the generated results are saved in the `results/` directory.

To evaluate the results, locate the corresponding `.npz` file in the `results/` directory and the dataset file used for the experiment. Then, open `data_analyse/energy.ipynb` and specify the absolute paths to these files in the `result_npz_path` and `mdb_path` variables, respectively.

**For example:**

```
result_npz_path = "/home/code/safe/example/2026-10-14-23-10-56/is2rfse_predictions_0.npz"
mdb_path = "/home/code/safe-gemnet-oc/dataset/test"
```

**Note:**

- Both `result_npz_path` and `mdb_path` must be specified using absolute paths. The example paths should be replaced with the corresponding absolute paths on your local machine.
- `result_npz_path` should point to the specific `.npz` prediction file, whereas `mdb_path` should point to the dataset directory. The notebook automatically reads the `data.mdb` file inside the specified directory, so there is no need to specify the `data.mdb` filename explicitly.
- For evaluation using `energy.ipynb`, the dataset file needs to be renamed from `.lmdb` to `.mdb`. Only the file extension needs to be changed.
- This renaming is only required for the evaluation notebook. During model training, the dataset should retain its original `.lmdb` filename, as the training pipeline uses PyTorch to load the LMDB dataset.

After setting the paths, run the cells in `energy.ipynb` to evaluate the model predictions and calculate the corresponding energy errors.

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

## Reproducing Experiments with SAFE

Four model variants are used in our experiments: GemNet-OC, GemNet-OC-SAFE, EquiformerV2, and EquiformerV2-SAFE. We evaluate these models on three datasets: OC20-SurfAds, OC20-SurfAds-AF, and SAAs-SurfAds. For brevity, the `SurfAds` suffix is omitted from the dataset names in the paper. Nevertheless, all datasets used in our experiments consist of surface–adsorbate systems.

### Reproduction Example: GemNet-OC on OC20-SurfAds

We use the training of GemNet-OC on OC20-SurfAds as an example to illustrate the reproduction procedure.

**Step 1: Download the dataset.**

Download the OC20-SurfAds dataset following the instructions in the **Datasets** section.

**Step 2: Configure the dataset paths.**

Open `configs/is2re-common/base/base-energy.yml` and replace the three `-src` paths with the absolute paths to the training, validation, and test datasets, respectively.

**Step 3: Run the training command.**

Execute the following command from the repository root directory:

```
python main.py --mode train --config-yml "configs/is2re-safe-paper/OC20-OC20AF/gemnet-oc.yml"
```

**Step 4: Calculate MAE and EWT metrics.**

After training is completed, rename the dataset files to use the `.mdb` extension. Then, open `data_analyse/energy.ipynb` and specify the absolute paths to the dataset directory and the generated `.npz` result file, respectively. Run the notebook to obtain the MAE and EWT evaluation results.

### Experimental Configurations and Environments

The datasets, configuration files, GPU specifications, and training commands used in all SAFE experiments are summarized in the following table. The number of GPUs indicates the number of GPUs required for each experiment.

| Model                 | Dataset         | No. of GPUs | GPU                     | Training Command                                             |
| --------------------- | --------------- | ----------- | ----------------------- | ------------------------------------------------------------ |
| **GemNet-OC**         | OC20-SurfAds    | 1           | NVIDIA A800 80GB GPU    | python main.py --mode train --config-yml configs/is2re-safe-paper/OC20-OC20AF/gemnet-oc.yml |
| **GemNet-OC-SAFE**    | OC20-SurfAds    | 1           | NVIDIA A800 80GB GPU    | python main.py --mode train --config-yml configs/is2re-safe-paper/OC20-OC20AF/gemnet-oc-safe.yml |
| **GemNet-OC**         | OC20-SurfAds-AF | 1           | NVIDIA A800 80GB GPU    | python main.py --mode train --config-yml configs/is2re-safe-paper/OC20-OC20AF/gemnet-oc.yml |
| **GemNet-OC-SAFE**    | OC20-SurfAds-AF | 1           | NVIDIA A800 80GB GPU    | python main.py --mode train --config-yml configs/is2re-safe-paper/OC20-OC20AF/gemnet-oc-safe.yml |
| **EquiformerV2**      | OC20-SurfAds    | 2           | NVIDIA A800 80GB GPU    | torchrun --standalone --nproc_per_node=2 main.py --num-gpus 2 --mode train --config-yml configs/is2re-safe-paper/OC20-OC20AF/equiformer_v2.yml |
| **EquiformerV2-SAFE** | OC20-SurfAds    | 2           | NVIDIA A800 80GB GPU    | torchrun --standalone --nproc_per_node=2 main.py --num-gpus 2 --mode train --config-yml configs/is2re-safe-paper/OC20-OC20AF/equiformer_v2-safe.yml |
| **EquiformerV2**      | OC20-SurfAds-AF | 2           | NVIDIA A800 80GB GPU    | torchrun --standalone --nproc_per_node=2 main.py --num-gpus 2 --mode train --config-yml configs/is2re-safe-paper/OC20-OC20AF/equiformer_v2.yml |
| **EquiformerV2-SAFE** | OC20-SurfAds-AF | 2           | NVIDIA A800 80GB GPU    | torchrun --standalone --nproc_per_node=2 main.py --num-gpus 2 --mode train --config-yml configs/is2re-safe-paper/OC20-OC20AF/equiformer_v2-safe.yml |
| **GemNet-OC**         | SAAs-SurfAds    | 1           | NVIDIA GeForce RTX 4090 | python main.py --mode train --config-yml configs/is2re-safe-paper/SAAs/gemnet-oc.yml |
| **GemNet-OC-SAFE**    | SAAs-SurfAds    | 1           | NVIDIA GeForce RTX 4090 | python main.py --mode train --config-yml configs/is2re-safe-paper/SAAs/gemnet-oc-safe.yml |
| **EquiformerV2**      | SAAs-SurfAds    | 1           | NVIDIA GeForce RTX 4090 | python main.py --mode train --config-yml configs/is2re-safe-paper/SAAs/equiformer_v2.yml |
| **EquiformerV2-SAFE** | SAAs-SurfAds    | 1           | NVIDIA GeForce RTX 4090 | python main.py --mode train --config-yml configs/is2re-safe-paper/SAAs/equiformer_v2-safe.yml |
| **GemNet-OC ER**      | OC20-SurfAds    | 1           | NVIDIA A800 80GB GPU    | python main.py --mode train --config-yml configs/is2re-safe-paper/OC20-OC20AF/gemnet-oc-er.yml |
| **GemNet-OC ED**      | OC20-SurfAds    | 1           | NVIDIA A800 80GB GPU    | python main.py --mode train --config-yml configs/is2re-safe-paper/OC20-OC20AF/gemnet-oc-ed.yml |
| **EquiformerV2 ER**   | OC20-SurfAds    | 2           | NVIDIA A800 80GB GPU    | torchrun --standalone --nproc_per_node=2 main.py --num-gpus 2 --mode train --config-yml configs/is2re-safe-paper/OC20-OC20AF/equiformer_v2-er.yml |
| **EquiformerV2 EV**   | OC20-SurfAds    | 2           | NVIDIA A800 80GB GPU    | torchrun --standalone --nproc_per_node=2 main.py --num-gpus 2 --mode train --config-yml configs/is2re-safe-paper/OC20-OC20AF/equiformer_v2-ev.yml |

### Notes on Reproducing the Experiments

**1. Dataset Preparation and Configuration**

Before running an experiment, users should download the corresponding dataset following the instructions in the **Datasets** section. The absolute or relative paths to the training, validation, and test datasets must then be specified in the corresponding base configuration files.

The relevant base configuration files can be identified by examining the configuration YAML file specified in the training command and checking its included configuration files. Users should update the dataset paths in these base configurations accordingly.

**2. Random Seed Configuration**

For experiments requiring a specific random seed, users can append the `--seed` argument to the corresponding training command. For example, to use random seed 1, append `--seed 1` to the command:

```
python main.py --mode train --config-yml configs/is2re-safe-paper/OC20-OC20AF/gemnet-oc.yml --seed 1
```

The same procedure applies to the other models and datasets.

**3. SAAs Dataset Splits**

The SAAs-SurfAds dataset contains 10 different data splits. For conciseness, the corresponding configurations and training commands for all 10 splits are not listed individually in the table. The provided commands illustrate the general procedure for running experiments on this dataset.

**4. Ablation Experiments**

The last four rows of the table correspond to the ablation experiments reported in **Table 4** of the paper. In terms of model naming, GemNet-OC-ERD and EquiformerV2-ERV in Table 4 correspond to GemNet-OC-SAFE and EquiformerV2-SAFE, respectively, in the README configuration table.

These experiments are conducted using the corresponding ER/ED configurations for GemNet-OC and ER/EV configurations for EquiformerV2, as specified in the training commands above.

## SAFE Implementation Details

The SAFE implementation is developed based on the Open Catalyst Project (OCP) codebase. We implement SAFE by modifying the model forward-pass outputs and the corresponding trainer evaluation procedures for GemNet-OC and EquiformerV2, respectively. In addition, the specific loss formulations required by SAFE are implemented in the OCP evaluator module.

The model registration names, source files, and corresponding trainer implementations are summarized below.

| Model Variant     | Registered Model Name | Model Implementation                                   | Trainer Implementation                                    |
| ----------------- | --------------------- | ------------------------------------------------------ | --------------------------------------------------------- |
| GemNet-OC-SAFE    | `gemnet_is2rse`       | `ocpmodels/models/gemnet_oc/gemnet_oc_is2rse.py`       | `ocpmodels/trainers/is2rse_trainer.py`                    |
| EquiformerV2-SAFE | `equiformer_v2_ERD`   | `ocpmodels/models/equiformer_v2/equiformer_v2_oc20.py` | `ocpmodels/models/equiformer_v2/trainers/safe_trainer.py` |

The main modifications to the OCP codebase are described as follows:

**1. Model Implementations**

The SAFE-specific model implementations are provided in the corresponding GemNet-OC and EquiformerV2 model files. The primary modifications are made to the output sections of their `forward()` methods to support the outputs required by the SAFE framework.

- **GemNet-OC-SAFE:** Implemented in `ocpmodels/models/gemnet_oc/gemnet_oc_is2rse.py`, registered under the model name `gemnet_is2rse`.
- **EquiformerV2-SAFE:** Implemented in `ocpmodels/models/equiformer_v2/equiformer_v2_oc20.py`, registered under the model name `equiformer_v2_ERD`.

**2. Trainer Implementations**

The corresponding trainer implementations handle the SAFE-specific evaluation procedures during model training and evaluation.

- **GemNet-OC-SAFE:** Uses `ocpmodels/trainers/is2rse_trainer.py`.
- **EquiformerV2-SAFE:** Uses `ocpmodels/models/equiformer_v2/trainers/safe_trainer.py`.

**3. Loss Implementations**

The loss formulations required by SAFE are implemented in `ocpmodels/modules/evaluator.py`, where the corresponding loss functions are added to support the training and evaluation of the SAFE models.

These modifications establish the connection between the SAFE-specific model architectures, training procedures, and evaluation metrics within the OCP framework.

For users who wish to understand or further develop the SAFE implementation, the above files provide the main entry points for examining the model modifications, trainer logic, and loss implementations.

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
