# Submission Checklist

This checklist summarizes what should be prepared for the final project submission.

## 1. Main Deliverables

- Final English report in PDF format
- Source code for the project
- Markdown reproduction guide

## 2. Recommended Files to Submit

### Report

- `report/main.tex`
- `report/references.bib`
- `report/figures/`
- compiled PDF report

### Code

- `src/`
- `configs/`
- `notebooks/run_project.ipynb`
- `requirements.txt`

### Documentation

- `README.md`
- `REPRODUCTION.md`
- `INSTALL_SERVER.md`
- `report/README.md`

## 3. Usually Do Not Submit

Unless explicitly required by the course platform, do not include:

- the full `wsj0/` dataset
- large downloaded checkpoints if size is restricted
- temporary notebook checkpoints
- large intermediate cache files

## 4. What the Markdown Guide Should Cover

The assignment requires a Markdown document explaining reproduction. The most important one in this repository is:

- `REPRODUCTION.md`

It already covers:

- environment setup
- library versions
- data layout
- notebook execution
- training and evaluation steps
- output locations
- how to reproduce the reported results

## 5. Final Pre-Submission Checks

- The report is fully in English
- Author name, student ID, and affiliation are filled in
- Report sections match the assignment requirement
- The reproduction guide is up to date
- Paths in the report figures are correct
- Quantitative results in the report match `artifacts/codec_comparison.csv`
- The notebook can be understood by another reader
