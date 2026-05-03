# WSJCAM0 Autoregressive Speech Prediction

This repository contains the final project implementation for autoregressive speech prediction over discrete codec representations on `WSJCAM0`.

The project compares:

- `FACodec`
- `EnCodec`

with the required objective metrics:

- `STOI`
- `PESQ`
- `DNSMOS`

## Main Files

- `notebooks/run_project.ipynb`: end-to-end experiment notebook
- `src/`: reusable code for data, codecs, model, training, and evaluation
- `configs/`: default experiment settings
- `artifacts/`: generated manifests, checkpoints, metrics, figures, and audio samples
- `report/`: LaTeX report source

## Submission-Oriented Documents

- [REPRODUCTION.md](./REPRODUCTION.md): full environment, data, training, evaluation, and expected-output guide
- [SUBMISSION_CHECKLIST.md](./SUBMISSION_CHECKLIST.md): what should be submitted for the final project
- [INSTALL_SERVER.md](./INSTALL_SERVER.md): terminal-only environment setup for the server
- [report/README.md](./report/README.md): how to compile the LaTeX report

## Current Reported Result Basis

The checked-in report and downloaded artifacts correspond to the currently accepted experiment setting used for submission:

- `10 epochs` for `FACodec`
- `10 epochs` for `EnCodec`

The exact quantitative comparison is stored in:

- `artifacts/codec_comparison.csv`
- `artifacts/report/codec_summary.csv`
- `artifacts/report/codec_summary.tex`

## Dataset Layout

The code expects the unpacked dataset at:

```text
./wsj0/
  si_tr_s/
  si_dt_05/
  si_et_05/
```

## Quick Start

1. Prepare the environment by following `REPRODUCTION.md` or `INSTALL_SERVER.md`.
2. Open `notebooks/run_project.ipynb`.
3. Switch to the prepared kernel.
4. Run the notebook cells.

## Notes

- Each new codec run automatically clears old checkpoints, evaluation files, and history for that codec.
- The notebook exports report-ready plots and tables to `artifacts/report/`.
- The LaTeX report uses local figure copies in `report/figures/`.
