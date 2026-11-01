# Configuration groups

Each market directory contains four entry configurations:

| File | Run |
|---|---|
| base.yaml | Original non-starred Static ML and default Portfolio ML; reports contain two methods |
| iteration_1.yaml | Original benchmark-relative Portfolio ML; 1%, 2%, 4% TE targets |
| iteration_2.yaml | Original 250-stock breadth/capacity configuration, inheriting iteration 1 |
| iteration_3.yaml | Original 3/5/10-year and no-decay half-life choices, inheriting iteration 2 |

keats_base.yaml is the shared parameter schema, not a standalone market run. Group files set separate outputs/<group>/<market> paths. The original 50 features and group-specific parameters are retained. Shared fields used only by iteration reference calculations do not add strategies to the base results.

Legacy aliases active_pfml, universe_250 and halflife resolve to iteration_1, iteration_2 and iteration_3 through script_groups.py. No separate linear base profile is restored. See ../scripts/README.md for execution and dependency guidance.
