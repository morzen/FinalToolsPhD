# Feature-combination sweep and TTP-only comparison

Written by an AI model (Claude, Anthropic); it runs the training and query programs of the tool folders. In the
thesis: Section 5.6.4 (the 63 combinations of feature types) and the TTP-only tables of Section 4.6.

A combination is a set of the six feature types, named with one letter per type: T techniques, S software, A CAPEC,
W CWE, V CVE, P pulses. `T` is the TTP-only configuration, `TSV` the three-feature one and `TSAWVP` the six-feature
one; there are 63 combinations. The tool for a combination is the six-feature tool with its code unchanged, run on
data in which every feature type outside the combination is set to -1 (the value for "missing"); the models are
trained again on these data with their own `train.py`.

| File | Role |
|---|---|
| `run_neural_sweep.sh` | trains and evaluates the eight approaches for the chosen combinations (`CONFIGS=all` for the 63, `REPEATS=n` for repeated trainings) |
| `run_rf_sweep_mac.sh` | the Random Forest for the 63 combinations |
| `run_ttp_evidence.sh` | the six- and three-feature tools as they are, every query restricted to its TTPs |
| `make_outputs.sh` | rebuilds `tables/`, `figures/` and `RESULTS.md` from the saved rank matrices (no model is run) |
| `code/` | the programs behind these scripts |
| `results_mac/`, `results_ttp_evidence/`, `results_check/` | the saved rank matrices and checks of the runs |
| `tables/`, `figures/`, `RESULTS.md` | the outputs |

The comments inside the scripts were written while the work was in progress and address the author. Where they speak
of 181 groups or 30 CTI reports, they describe the dataset before the 18 AlienVault OTX "groups" were removed; the
programs read the number of groups and of reports from the data, and `RESULTS.md`, the tables and the figures are
those of the final dataset (163 groups, 29 reports).
