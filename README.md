# WSJCAM0 Autoregressive Speech Prediction

This repository implements the CUHK-Shenzhen deep learning final project on autoregressive speech prediction over discrete codec representations.

The project compares:

- `FACodec`
- `EnCodec`

on the `WSJCAM0` dataset using:

- `STOI`
- `PESQ`
- `DNSMOS`

## Layout

- `notebooks/run_project.ipynb`: one-click entry notebook
- `src/`: reusable project code
- `configs/`: experiment defaults
- `artifacts/`: generated manifests, checkpoints, metrics, and samples

## Dataset

The notebook assumes the unpacked dataset is available at:

```text
./wsj0/
  si_tr_s/
  si_dt_05/
  si_et_05/
```

The local workspace already matches this structure.

## Runtime model

The notebook now assumes dependencies are already installed in the target environment.

Install the environment from the terminal first, then open the notebook.

Detailed terminal steps are documented in:

- `INSTALL_SERVER.md`

If the server cannot directly access `huggingface.co`, set a mirror endpoint before running codec downloads:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com
```

or set `codecs.hub_endpoint` in `configs/default.yaml`.

Because the notebook no longer installs dependencies, the workflow is simpler:

1. prepare the environment in the terminal
2. open the notebook
3. switch to `Python (finalproject26-py39)`
4. run the remaining cells with `Run All`

## Reproducibility

The intended production environment is AutoDL with a self-managed Python 3.9 environment inside the base image.

The implementation keeps the main logic in Python modules so the same pipeline can be run from the notebook or from scripts.
