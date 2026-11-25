# Parts written by an AI model

Everything in this folder was written by an AI model (Claude, Anthropic) at the author's request, and is kept apart
from the two tool folders for that reason. The one exception is `checks/first_check/original_code/`, which holds the
author's own scoring code (see `checks/README.md`).

| Folder | What it is | In the thesis |
|---|---|---|
| `dynamic_visualisation_attribution_tool_and_threat_modelling/` | a local web application that shows how each approach and the hybrid rank the 163 groups for a piece of evidence, and the attack generator | Section 5.8 and Appendix G |
| `feature_combination_sweep/` | the program that runs the 63 combinations of the six feature types and the TTP-only comparison, and draws their tables and figures; it uses the training and query programs of the tool folders | Section 5.6.4 and the TTP-only tables of Section 4.6 |
| `checks/` | the two checks of the results | Appendix F |

The folders have to stay where they are: the tool and the sweep look for `FinalToolsPhD_6feat/` and
`FinalToolsPhD_3feat/` two folders above them.
