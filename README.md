# Holistic APT attribution through data-level abstraction: code and data

Code and data of the final experiments of the PhD thesis *Holistic approach to Advanced Persistent Threat (APT)
attribution through behavioural and evidentiary artefacts using machine learning techniques* (University of
Greenwich): the Data-Level Abstraction Attribution Framework (Chapter 4), the Hybrid Attribution Framework (Chapter 5)
and the false flag analysis (Chapter 6), on the 163 groups of MITRE ATT&CK release 16.

## What is here

| Folder | Content |
|---|---|
| `FinalToolsPhD_6feat/` | the six-feature configuration: TTPs, software, CAPEC, CWE, CVE and AlienVault OTX pulses |
| `FinalToolsPhD_3feat/` | the three-feature configuration: TTPs, software and CVE |
| `ai_written/` | the parts written by an AI model, kept apart: the dynamic visualisation tool with its attack generator, the feature-combination sweep and the two checks of Appendix F (see `ai_written/README.md`) |

The two tool folders have the same layout:

| Path | Content |
|---|---|
| `BaseData/` | the dataset: `EncodedData.csv` (730,910 records, 163 groups) and `comparison_Encoded_df.csv`, which maps the encoded values back to the identifiers |
| `1.XGboost/`, `2.GNN/`, `3.SiameseNetwork/`, `4.Attention-Based_MIL/`, `5.RF_FeatureEngineering/` | the five models: `train.py`, `query.py`, `eval.py` and the false-flag script of each |
| `Jaccards_Correlation/`, `featureMatching/`, `TVD_Total_Variation_Distance/` | the three similarity measures |
| `6.hybrid/` | the weighted reciprocal rank fusion and the search for its weights |
| `CTI_eval/` | the 29 CTI reports (`CTI_Eval_empty_template.json`) and their evaluation |
| `common/protocol.py` | the evaluation protocol shared by every approach: profiles, halved profiles, synthetic false-flag profiles, ranking |
| `run_all.sh` | the complete evaluation |
| `results_archive/` | the result tables of the run reported in the thesis (`summary.txt`, `excel_blocks.csv`) |

## Running the evaluation

Python 3 with numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, joblib, torch, torch_geometric and xgboost.

    cd FinalToolsPhD_6feat
    bash run_all.sh --smoke      # quick end-to-end check
    bash run_all.sh              # the complete evaluation

`run_all.sh` trains XGBoost again (fixed seed), loads the four other models from their saved files, runs the
attribution, false-flag and CTI evaluations of the eight approaches, fuses the stored rankings into the three hybrid
configurations and gathers the tables in `results_archive/<date_time>/`. On a MacBook Pro M1 Max with 64 GB of RAM the
six-feature evaluation takes about 1 hour 30 minutes and the three-feature one about 40 minutes.

A model is trained with the `train.py` of its folder (`cd 2.GNN && python train.py`). The GNN, MIL and Siamese models
were trained without a fixed seed, so a new training gives a different model and different results (thesis, Appendix F
and Section 7.3); the numbers of the thesis are those of the saved models.

## What is not in the repository

- The saved MIL and Random Forest models (about 1 GB and 98 MB with six features, 536 MB and 94 MB with three). They
  are listed in `.gitignore`; with Git LFS they can be added by removing those lines.
- The figures and the large rank and score matrices of the false-flag evaluation (`results/ff_*.npy`); `run_all.sh`
  writes them again.
- `Cyber_-Necronomicon/`, the MongoDB dumps from which the dataset was built (Chapter 3). Only the visualisation tool
  reads them; it expects that folder next to the two tool folders.

## Data sources

MITRE ATT&CK release 16 (October 2024), CAPEC 3.9, CWE 4.17, CVE records up to 3 April 2025 and AlienVault OTX pulses
up to 23 April 2025. The experiments use identifiers only (technique, software, CAPEC, CWE, CVE and pulse
identifiers).

## How the code was written

For the final implementation (thesis, Section 5.5.2 and Appendix F), the evaluation code was restructured with the
assistance of an AI model (Claude, Anthropic), so that the complete evaluation runs in hours instead of weeks. What
that model wrote entirely is in `ai_written/`.
