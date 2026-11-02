# Current characteristic set and archive removal

At the user's request, the previous characteristic archives were removed on 2 November 2026:

- `data/archives/20261102_121716_previous_50`
- `outputs/archive/20261102_121716_previous_50`
- `../Characteristics_tryout/data/selection_archive/20261102_122021_previous_cross_market`

This removed roughly 7.4 GB of obsolete data, fitted models and reports. No Git backup of these archives was created. Existing ML implementation code is retained; current configurations and prepared inputs use only the new US high-cap-selected 50. No fresh ML model was fitted during this cleanup.

The broader source cache remains for provenance. Current selection admits only 153 names verified against both published JKP reference files; 152 qualify and 50 are selected. Original `keats_pipeline` remains a separate replication pipeline.

The current chart is `outputs/us_high_cap_selection/charts/selection_decision_path_jkp_only.pdf`. It shows every selected name, score, group and A/T/S reason. Large full-period input and processed caches remain local under the existing ignore policy.
