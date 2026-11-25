# The two checks of Appendix F

Both scripts were written by an AI model (Claude, Anthropic).

## First check: the final code against the original scoring code

    bash ai_written/checks/run_first_check.sh        (about 40 minutes for the two tools)

`first_check/verify_against_original.py` scores the same evidence (12 group profiles and 12 false-flag pairs, each in
both scenarios: 48 queries per approach) with the scorer of the final code and with the original query code of the
approach, and compares the scores and the ranks of all groups.

`first_check/original_code/` holds that original code. It is not AI-written: these are the author's scripts as they
were on 22 October 2026, before the evaluation code was restructured, in which only the feature-weight values were
replaced by the rank-sum weights of the thesis, so that both sides use the same weights.

`verify_report_6feat_2026-11-23.txt` and `verify_report_3feat_2026-11-23.txt` are the reports of the run of
23 November 2026 on the final models (the folder of the original code is shown under its place in this repository).

The check needs the saved models and the Python environment of the tool folders.

## Second check: the tables recomputed from the saved rankings

    bash ai_written/checks/run_second_check.sh       (about 1 minute)

`second_check/recheck_numbers_163.py` uses numpy and pandas only and none of the code of the tools. It reads the saved
rank matrices of the eight approaches and recomputes the Top-1, 5, 10 and 20 of the attribution tests, the four
outcomes of the false-flag tests, the three hybrid configurations (from the ranks of the eight approaches, compared
rank by rank with the saved hybrid ranks), the CTI scores and the numbers of the archived summary. Any difference is
printed on a line that starts with PROBLEM.

`second_check_output_6feat.txt` and `second_check_output_3feat.txt` are the outputs of the run of 25 November 2026 on
the results of the thesis. The check needs the rank matrices that `run_all.sh` writes; they are too large for the
repository, so `run_all.sh` has to be run first.

## The check of the visualisation tool

`bash validate_explorer.sh` in the folder of the tool (thesis, Section 5.8).
