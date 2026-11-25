# Original scoring code

The author's scripts as they were on 22 October 2026, before the evaluation code was restructured (the first
implementation of Section 5.5.2 of the thesis). One change was made for the first check of Appendix F: the
feature-type weights written in the scripts were replaced by the rank-sum weights of the thesis (6/6, 5/6, 4/6, 3/6,
2/6 and 1/6; 60 to 10 for the edges of the GNN; the starting values of MIL), so that the comparison is made with the
same weights. Only the Python files are here; they read the data and the saved models of the tool folders.
