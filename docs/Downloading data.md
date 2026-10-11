# Downloading data for the Python replication

The missing `src/jkmp/data/` package has been reconstructed from the existing
configuration, tests, documented Python design and reference R code. It is not a
recovered copy of the original missing Python implementation.

Run these commands in the project root in VS Code's PowerShell terminal.

## Install into the interpreter you use

```powershell
$pythonExe = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
& $pythonExe -m pip install -e ".[dev]"
```

## Download the data

```powershell
# Optional: use your WRDS username instead of entering it at the login prompt.
$env:WRDS_USERNAME = "YOUR_WRDS_USERNAME"

# Check access and columns in the configured WRDS tables.
& $pythonExe scripts/00_download.py explore

# Download the WRDS data, French factors, and JKP metadata.
& $pythonExe scripts/00_download.py pull

# Prepare the panel for the remaining numbered scripts.
& $pythonExe scripts/01_prepare.py
```

The WRDS client uses the normal WRDS username/password login. If necessary, it
prompts for the password without displaying it. It can reuse PostgreSQL's saved
password file (on Windows: `%APPDATA%\postgresql\pgpass.conf`). An API key is
not used by the installed WRDS PostgreSQL client. Do not put a password in code or YAML.

Automatic local login can also use `configs/wrds_login.local.json`, containing
`username` and `password` string fields. This file is Git-ignored and stores the
password as plain text locally. The downloader reads it without logging the values.
An environment username override only uses its password when the usernames match.

You need WRDS access to `contrib.global_factor` and `crsp.dsf_v2`. Access to the
WRDS website alone does not guarantee subscriptions to both products. The
`explore` command reports whether the configured tables and columns are accessible.

Public files can be downloaded without a WRDS account:

```powershell
& $pythonExe scripts/00_download.py pull --public-only
```

That command downloads monthly/daily French factors, `Factor Details.xlsx`,
`Cluster Labels.csv`, and their normalised Parquet files. It does not download
the licensed stock data, so it is insufficient for `01_prepare.py` on its own.

## Outputs and resume behaviour

Full downloads go to `data/raw/`: `jkp_usa.parquet`, `crsp_daily.parquet`,
`top500_by_month.parquet`, `jkp_usa_top500.csv`, public factors/metadata and
`row_counts_by_month.csv`. The top-N number follows the configuration.

Only the configured columns and stock filters are queried. Stock data is queried
year by year and streamed into temporary Parquet files. A file is published only
after its download finishes. Rerunning `pull` reuses completed datasets with
matching settings; an interrupted individual dataset starts again. A settings
manifest prevents reuse of WRDS files from a different configuration.

To refresh the existing data explicitly:

```powershell
& $pythonExe scripts/00_download.py pull --force
```

## Verification

```powershell
& $pythonExe -m pytest -q
```

Offline tests cover the parsers, timing, screens, buffered stock universe, complete
synthetic data flow and interrupted-download recovery. Tests against actual CRSP
returns are skipped until the licensed raw data is present. Passing offline tests
does not verify your account's live schema, entitlements or the numerical results
of the missing original Python package.

The root `.gitignore` uses `/data/` so downloaded data stays excluded while
`src/jkmp/data/` source files can be tracked.
