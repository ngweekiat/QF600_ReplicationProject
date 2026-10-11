"""Download the configured data with VS Code's Run Python File button."""

import logging

from jkmp.config import load_config
from jkmp.data.wrds_pull import connect, pull_all


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config()
    logging.info("Connecting to WRDS. Authentication may take a moment.")
    with connect() as db:
        logging.info("Connected to WRDS. Starting downloads.")
        files = pull_all(db, cfg)
    for name, path in files.items():
        logging.info("%-15s %s", name, path)
    logging.info("Downloads complete. Run scripts/01_prepare.py next.")


if __name__ == "__main__":
    main()
