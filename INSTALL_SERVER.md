# Server Installation Guide

This project no longer installs dependencies from inside the notebook.

Use the terminal on the server instead, then open the notebook with the prepared kernel.

## 1. Enter the project directory

```bash
cd ~/Final_Project
```

## 2. Create the conda environment

```bash
conda create -y -n finalproject26-py39 python=3.9
```

## 3. Activate the mirror settings

For Hugging Face model downloads:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com
```

If your server also needs a pip mirror, replace the example URL with the mirror you actually use:

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

## 4. Install PyTorch first

```bash
conda run -n finalproject26-py39 python -m pip install --upgrade pip setuptools wheel
conda run -n finalproject26-py39 pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

## 5. Install project dependencies

```bash
conda run -n finalproject26-py39 pip install -r requirements.txt
```

## 6. Register the notebook kernel

```bash
conda run -n finalproject26-py39 python -m ipykernel install --user --name finalproject26-py39 --display-name "Python (finalproject26-py39)"
```

## 7. Launch Jupyter

After the environment is ready, open `notebooks/run_project.ipynb` and switch the kernel to:

```text
Python (finalproject26-py39)
```

## Notes

- Default training is now `50 epochs` for `FACodec` and `50 epochs` for `EnCodec`.
- Default `num_workers` is `0` to avoid Jupyter DataLoader multiprocessing shutdown issues.
- The notebook config already sets the Hugging Face mirror endpoint for runtime downloads.
- Training logs now print a persistent summary every `5` epochs.
