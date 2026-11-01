# Relocation validation

Validated on 1 November 2026 using Python 3.13:

- All 18 characteristic/relocation tests passed, including the existing cache/schema checks against the downloaded study.
- The statistical, selection, analysis and independent-verification modules retain their archived source bytes.
- CLI help and the read-only `paths` command work from the repository root; tests also invoke the entry point from a different working directory.
- Default generated data and reports resolve inside Characteristics_tryout, outside keats_pipeline.
- Cache/report overrides are applied before calculation modules load. The read-only path preview creates no data directory and opens no database connection.
- A mocked first-acquisition test verifies that missing source column metadata is fetched once and cached.

No WRDS connection, new download, full analysis run or model refit was performed for this code upload. Existing Downloads data and Keats model settings were not changed. Local test scratch/cache files and logs are ignored by Git.
