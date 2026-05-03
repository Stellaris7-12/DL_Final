# Reproduction Guide

This document explains how to reproduce the implementation, environment, training, evaluation, and reported results of this project.

The target is that another reader can follow this file alone and obtain the same type of artifacts used in the final submission.

## 1. Project Goal

This project implements autoregressive speech prediction on discrete codec representations with:

- `FACodec`
- `EnCodec`

and evaluates the predicted speech with:

- `STOI`
- `PESQ`
- `DNSMOS`

The dataset is `WSJCAM0`.

## 2. Recommended Runtime Environment

Recommended execution platform:

- OS: `Ubuntu 22.04`
- Python: `3.9`
- PyTorch: `2.8.0`
- TorchVision: `0.23.0`
- TorchAudio: `2.8.0`
- CUDA wheel source: `cu128`
- Environment manager: `conda`

The project was designed for GPU execution on a Linux server with a self-created conda environment.

## 3. Dependency Installation

### 3.1 Create the conda environment

```bash
conda create -y -n finalproject26-py39 python=3.9
```

### 3.2 Optional mirror settings

If direct access to Hugging Face is unstable:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com
```

If a pip mirror is also needed:

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.3 Install PyTorch first

```bash
conda run -n finalproject26-py39 python -m pip install --upgrade pip setuptools wheel
conda run -n finalproject26-py39 pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

### 3.4 Install project dependencies

Run from the project root:

```bash
conda run -n finalproject26-py39 pip install -r requirements.txt
```

Main dependencies include:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `seaborn`
- `soundfile`
- `librosa`
- `pyyaml`
- `tabulate`
- `torchmetrics[audio]`
- `pesq`
- `pystoi`
- `huggingface_hub`
- `einops`
- `encodec`
- `naturalspeech3_facodec` from GitHub

### 3.5 Register the Jupyter kernel

```bash
conda run -n finalproject26-py39 python -m ipykernel install --user --name finalproject26-py39 --display-name "Python (finalproject26-py39)"
```

## 4. Dataset Preparation

The code expects the dataset in the following structure:

```text
Final_Project/
|- wsj0/
|  |- si_tr_s/
|  |- si_dt_05/
|  `- si_et_05/
|- src/
|- configs/
`- notebooks/
```

Dataset assumptions:

- the audio files are already converted to `wav`
- the code recursively scans `.wav` files
- no manual manifest creation is required

The implementation automatically builds manifests and stores them under:

```text
artifacts/manifests/
```

## 5. Notebook Entry Point

Open:

```text
notebooks/run_project.ipynb
```

and switch the kernel to:

```text
Python (finalproject26-py39)
```

The notebook performs these stages:

1. sets Hugging Face mirror environment variables
2. prints Python / Torch / CUDA information
3. scans the dataset and reports split sizes
4. trains `FACodec`
5. evaluates `FACodec`
6. trains `EnCodec`
7. evaluates `EnCodec`
8. exports the cross-codec comparison table
9. exports report-ready figures and tables

## 6. How to Reproduce the Reported Results

### 6.1 Important note

The checked-in report and local artifacts correspond to the accepted submission setting:

- `10 epochs` for `FACodec`
- `10 epochs` for `EnCodec`

The repository defaults may be larger for general experimentation, so the exact reported result should be reproduced through notebook overrides.

### 6.2 Notebook override for the reported results

Inside `notebooks/run_project.ipynb`, set `COMMON_OVERRIDES` to include:

```python
COMMON_OVERRIDES = {
    'dataset': {
        'root': DATASET_ROOT,
    },
    'project': {
        'artifacts_dir': ARTIFACTS_DIR,
    },
    'codecs': {
        'hub_endpoint': HF_ENDPOINT,
    },
    'training': {
        'max_epochs': 10,
        'num_workers': 0,
    },
}
```

This is the recommended way to reproduce the submitted numbers without permanently changing the repository defaults.

## 7. Training Procedure

For each codec, the training pipeline is:

1. load a pretrained frozen codec
2. encode the waveform into embeddings and discrete tokens
3. sample a context window from the encoded sequence
4. predict the next-frame embedding with the transformer
5. quantize the predicted embedding
6. optimize the combined pre-quant and post-quant L1 losses
7. save the best checkpoint according to validation loss

Implementation notes relevant to reproducibility:

- each new codec run clears old files for that codec:
  - `artifacts/<codec>/checkpoints/`
  - `artifacts/<codec>/evaluation/`
  - `artifacts/<codec>/history.json`
- warnings are suppressed in the notebook to reduce clutter
- training prints a persistent summary every `5` epochs
- `num_workers=0` is used by default to avoid Jupyter multiprocessing shutdown issues

## 8. Inference and Evaluation Procedure

Evaluation is performed on:

- `wsj0/si_et_05`

For each codec:

1. load the best checkpoint
2. encode each test utterance
3. perform autoregressive rollout frame by frame
4. convert predicted embeddings to tokens
5. decode predicted tokens back to waveform
6. align predicted and reference future speech
7. compute STOI, PESQ, and DNSMOS

Special handling for EnCodec:

- EnCodec internally runs at its codec sample rate
- decoded EnCodec predictions are resampled to `16 kHz`
- the reference waveform is aligned to the same evaluation rate before scoring

## 9. Output Files

### 9.1 Per-codec outputs

For each codec, the following files are produced:

```text
artifacts/<codec>/
|- history.json
|- checkpoints/
|  `- best.pt
`- evaluation/
   |- test_manifest.csv
   |- per_utterance_metrics.csv
   |- summary_metrics.json
   `- samples/
```

### 9.2 Combined outputs

The notebook also generates:

```text
artifacts/codec_comparison.csv
artifacts/report/codec_summary.csv
artifacts/report/codec_summary.md
artifacts/report/codec_summary.tex
artifacts/report/codec_metric_comparison.png
artifacts/report/facodec_training_curves.png
artifacts/report/encodec_training_curves.png
```

These files are used directly by the LaTeX report.

## 10. Report Compilation

The LaTeX report is stored in:

```text
report/
```

Compile it with:

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Before submission, replace the placeholder fields in `report/main.tex`:

- `Your Name`
- `Student ID`
- `Your Affiliation`

## 11. Files Needed for a Reproducible Submission

The most important files for code-and-report submission are:

- `src/`
- `configs/`
- `notebooks/run_project.ipynb`
- `requirements.txt`
- `README.md`
- `REPRODUCTION.md`
- `INSTALL_SERVER.md`
- `report/main.tex`
- `report/references.bib`
- `report/figures/`
- compiled report PDF

Normally, large raw datasets should not be uploaded with the code submission unless explicitly requested by the course platform.

## 12. Current Reported Quantitative Results

The accepted result table used in the report is:

| codec | num_files | stoi_mean | stoi_std | pesq_mean | pesq_std | dnsmos_mean | dnsmos_std |
|:--|--:|--:|--:|--:|--:|--:|--:|
| encodec | 651 | 0.335698 | 0.124314 | 1.196353 | 0.366403 | 1.106710 | 0.015287 |
| facodec | 651 | 0.362790 | 0.127164 | 1.080040 | 0.103604 | 1.350038 | 0.265406 |

The same values are also stored in:

- `artifacts/codec_comparison.csv`
- `artifacts/report/codec_summary.csv`

## 13. Minimal Reproduction Checklist

1. Create the `finalproject26-py39` conda environment
2. Install PyTorch 2.8.0 and all project dependencies
3. Prepare the `wsj0/` dataset directory
4. Open `notebooks/run_project.ipynb`
5. Set `max_epochs=10` in `COMMON_OVERRIDES`
6. Run all notebook cells
7. Confirm that:
   - `artifacts/codec_comparison.csv` exists
   - `artifacts/report/codec_metric_comparison.png` exists
   - `report/main.tex` compiles successfully
