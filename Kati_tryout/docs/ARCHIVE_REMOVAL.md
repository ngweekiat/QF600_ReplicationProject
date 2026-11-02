# Current characteristic set and archive removal

At the user's request, the previous characteristic archives were removed on 2 November 2026:

- `data/archives/20261102_121716_previous_50`
- `outputs/archive/20261102_121716_previous_50`
- `../Characteristics_tryout/data/selection_archive/20261102_122021_previous_cross_market`

This removed roughly 7.4 GB of obsolete data, fitted models and reports. No Git backup of these archives was created. Existing ML implementation code is retained; current configurations and prepared inputs use only the new US high-cap-selected 50. No fresh ML model was fitted during this cleanup.

The source candidate cache remains available to reproduce why 50 were selected from 404. It is not an alternative active predictor list. Original `keats_pipeline` remains a separate replication pipeline.

The published selection chart is `outputs/us_high_cap_selection/charts/selection_decision_path.pdf`. It shows every selected name, score, group and A/T/S reason. Current report outputs and the compact US selection sample are included in Git; large Kati full-period model-input and processed caches remain local under the existing ignore policy.
