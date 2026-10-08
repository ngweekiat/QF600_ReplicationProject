"""Stage 0 runner: locate and pull the raw data into data/raw/.

    python scripts/00_download.py explore     # list candidate WRDS tables and check them
    python scripts/00_download.py pull        # JKP, top-N list, CRSP daily, Ken French, JKP reference files
    python scripts/00_download.py pull --force

Credentials: set WRDS_USERNAME and WRDS_API_KEY in the environment (see jkmp.data.wrds_pull.connect).
"""

from __future__ import annotations

import argparse
import logging

from jkmp.config import load_config
from jkmp.data import wrds_pull


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["explore", "pull"])
    parser.add_argument("--config", default=None, help="YAML config (default: configs/base.yaml)")
    parser.add_argument("--override", action="append", default=[], help="dotted override, e.g. dates.test_end=2024-12-31")
    parser.add_argument("--force", action="store_true", help="pull: re-download files that already exist")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.override)

    with wrds_pull.connect() as db:
        if args.step == "explore":
            print(wrds_pull.format_report(wrds_pull.explore(db, cfg)))
        else:
            for name, path in wrds_pull.pull_all(db, cfg, force=args.force).items():
                logging.info("%-15s %s", name, path)


if __name__ == "__main__":
    main()
