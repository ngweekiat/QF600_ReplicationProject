# 200-characteristic research dataset

Stored in Git LFS. Retrieve with `git lfs pull --include="Characteristics_tryout/data/**"` from the repository root.

- `panels/`: 64 US/EU yearly raw panels, formation years 1994-2025, with 200 characteristics. US high-cap and mid-cap share the US panel; membership is in metadata.
- `metadata_US.parquet`, `metadata_EU.parquet`: stock-month memberships and original return targets.
- `pilot/`: broader candidate screening inputs, including supplemental US price observations.
- `lag_history/`: supplementary lag observations; unavailable rows remain explicitly flagged.
- `reference/`: JKP characteristic definitions and themes.
- CSV/JSON files record selection, source schema, acquisition status and download checksums.

Raw missing characteristic values are preserved. Keys are id and eom. The source forward-return audit column is not a predictor. Formation dates span December 1994 through November 2025; target returns span 1995-2025. There are 2,886,029 stock-month rows across the two regional panels.

`PUBLICATION_MANIFEST.json` records source paths, sizes and SHA256 hashes for the copied files. Every copied file was compared with its Downloads original. Paths are provenance, not runtime dependencies.

Excluded: duplicate derived `selected50_ranked` and preliminary `selected50_ranked_initial_proxy_selection` exports (regenerable), and generated analysis reports. The recommended 50 are research suggestions; Keats original 50 and ML settings remain unchanged. No new model training or feature analysis was performed for publication.
