# Dynamic visualisation attribution tool and threat modelling

Written entirely by an AI model (Claude, Anthropic); thesis Section 5.8 and Appendix G.

A local web app over the attribution tools of the thesis. It has two interactive 3-D views:

- **3D attribution**: type evidence (TTPs, software, CAPEC, CWE, CVE, pulses) and watch, live, how each of the
  eight approaches of the Data-Level Abstraction Attribution Framework ranks the 163 groups, and how the Hybrid
  Attribution Framework fuses them.
- **Attack generator** (threat modelling): choose a group (APT profile) and it builds one possible attack from the
  group's own records, step by step down the levels of abstraction, with the mitigations and detections. You can
  choose the tactics, and items at any level (or a CTI report) that the attack must contain.

Everything runs on your computer. The page needs no internet and no JavaScript library.

## What it needs

The thesis tools and the MongoDB dumps, read only. By default the app looks for them two folders up:

```
<repository root>/
├── FinalToolsPhD_6feat/      the tools, with their Python (venv)
├── FinalToolsPhD_3feat/      (optional) the three-feature tools
├── Cyber_-Necronomicon/      the MongoDB dumps (ATT&CK, CAPEC, CWE, schema)
└── ai_written/
    └── dynamic_visualisation_attribution_tool_and_threat_modelling/   this project
```

If the project is somewhere else, give the folder that holds them: `PHD_DIR=/path/to/that/folder bash run_explorer.sh`.

## Start it

```
cd dynamic_visualisation_attribution_tool_and_threat_modelling
bash run_explorer.sh          # six-feature tools; the page opens at http://127.0.0.1:8765
bash run_explorer.sh 3f       # three-feature tools
```

Ctrl+C in the Terminal stops it. The first start reads the display names from the dumps and keeps them in
`names.json` next to the scripts (about half a minute, once). The attack generator reads the records in the
background (about 20 s).

In the 3-D views you can drag to turn, scroll to zoom, double-click to reset, hover for details, and click a group
or a step.

## 3D attribution

The levels of abstraction are stacked as discs, TTPs at the top and pulses at the bottom. The evidence items sit on
their level, and the 163 groups sit on a ring around them, closer to the centre when they are ranked higher. The
scene updates each time you add or remove an item. "Clear all" goes back to the empty scene.

You can type IDs or names, paste a report (the IDs are picked out) or load a thesis query: one of the 29 CTI
reports, a group's full or half profile, or a synthetic false flag. The group fields search by name, ATT&CK ID or
alias, for example `apt29`, `G0016` or `cozy bear`.

One button per approach shows its reasoning:

| Approach | What the 3-D view shows |
|---|---|
| Jaccard, FM, TVD | threads from each evidence item to the top groups that hold it, and the selected group's per-level column (similarity x level weight) |
| Random Forest | the 100 trees, green where the evidence falls in the same leaf as the selected group's profile, and the tests on the path |
| XGBoost | threads sized by each item's contribution to the group's raw score (dashed red when it lowers it) and the probabilities |
| GNN | the evidence as a new graph node, hop 1 to its items, hop 2 to the groups, and the embedding space |
| Siamese | the embedding space and the cosine similarity of the query to each group |
| MIL | the items sized by their attention weight, and the per-level similarity x learned weight |
| Hybrid (Equal, Self-eval, CTI-based) | what each approach adds to the WRRF score, w / (60 + rank), and where the selected group stands in each approach |

The right panel gives the ten best groups and why the selected one is there. It also shows "recomputed from the
model: identical to the tool's scores" when the panel's own computation matches the tool.

## Attack generator

For the chosen group, platform and objective (espionage and impact, espionage, or destruction), it builds one
possible attack:

1. One technique of the group for each ATT&CK tactic it covers, in kill-chain order (release 16). The technique is
   drawn at random among the group's techniques for that tactic, each equally likely (the dataset records which
   techniques a group uses, not how often).
2. For each step:
   - how ATT&CK says the group, or its software, did it;
   - the software;
   - one TTP -> CAPEC -> CWE -> CVE -> Pulse path of the group's records, preferring CVEs with a public exploit;
   - the ATT&CK mitigations and detections.
3. A defence plan: the mitigations, and the detections grouped by ATT&CK data source, that cover the most steps.

You can also shape the attack:

- **Tactics**: tick the tactics that get a step drawn from the group's profile. By default, all the group's tactics
  that the objective keeps are ticked.
- **Include in the attack**: techniques, software, CAPEC, CWE, CVEs or pulses the attack must contain. Type them,
  take them from one of the 29 CTI reports of the thesis (the report's group becomes the attacker), or paste any
  report (its IDs are picked out).
  - An included technique is a step of its own, under its tactic.
  - Included software is attached to a step whose technique it implements (ATT&CK procedure examples or the
    dataset records), or gets a step of its own.
  - An included CAPEC, CWE, CVE or pulse becomes the attack path of a step whose technique leads to it in the
    group's records, or in other groups' records when the group's records have none.

Steps of included items are marked with a star, and are in the attack whether their tactic is ticked or not. The
ticked tactics they do not cover are drawn from the profile as in step 1. Items that cannot be placed are listed
under the steps, with the reason. With the default tactics and nothing included, the attack is the one described
above.

The same choices and seed always give the same attack. The seed is shown and can be typed. Play runs the attack
step by step: space plays or pauses and the arrow keys step through it.

Nothing is invented: every name, path and text comes from the dataset records and the dumps.

## 2-D panels

The earlier page with 2-D panels of the same computations is at http://127.0.0.1:8765/index.html (link "2-D
panels" at the top).

## Faithful to the tools

- The tools are imported read only, with their own Python. Nothing is written into the tool folders: no bytecode
  and no results.
- Each ranking comes from the approach's own scorer object, the one its `hybrid_query` uses, and is ranked with
  the tools' `protocol.ranked_results`. The hybrid is `protocol.wrrf_ranked` with `protocol.HYBRID_WEIGHTS`.
- `bash validate_explorer.sh` takes 10-15 minutes and writes nothing. It compares the app's rankings with each
  approach's own `hybrid_query`. It also compares them with the ranks the tools saved in `CTI_eval/`, for:
  - the 29 CTI reports,
  - the five combined-evidence sets,
  - 20 halved profiles.

  It ends with ALL IDENTICAL when they match.

## Files

| File | Role |
|---|---|
| `run_explorer.sh`, `server.py` | start the local server (Python standard library) |
| `engine.py` | loads the eight approaches; computes the rankings, what each approach does, and the hybrid |
| `attack.py`, `tm_data.py` | the attack generator and the data it reads (records, ATT&CK, CAPEC, CWE) |
| `names.py`, `minibson.py` | display names read from the MongoDB dumps |
| `static/studio.html`, `studio.css`, `studio.js` | the page |
| `static/engine3d.js` | the small 3-D engine (Canvas 2-D) |
| `static/attr3d.js`, `static/attack3d.js` | the 3D attribution and attack generator views |
| `static/common.js`, `static/picker.js` | shared helpers and the type-to-search group fields |
| `static/index.html`, `static/app.js`, `static/style.css` | the 2-D panels |
| `validate_explorer.sh`, `validate_explorer.py` | the check that the app shows what the tools compute |
