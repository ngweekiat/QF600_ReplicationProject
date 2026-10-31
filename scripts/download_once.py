"""Attended download with one login attempt and credentials held only in memory."""

import argparse
import getpass
import logging
import sys

import wrds

from jkmp.config import load_config
from jkmp.data.wrds_pull import pull_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrds-username", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config()
    username = args.wrds_username or input("WRDS username: ").strip()
    if not username:
        parser.error("WRDS username must not be empty")
    password = getpass.getpass("WRDS password (typing is hidden): ")
    if not password:
        parser.error("WRDS password must not be empty")

    print("One connection attempt. Open Duo Mobile and approve the WRDS push.", flush=True)
    # Bypass both the local login file and WRDS's automatic prompt/retry flow.
    db = wrds.Connection(autoconnect=False, wrds_username=username, wrds_password=password)
    try:
        db._Connection__make_sa_engine_conn(raise_err=True)
    except Exception:
        logging.error("Connection failed. No retry was made. Resolve the login/Duo issue before running again.")
        raise
    try:
        for name, path in pull_all(db, cfg).items():
            logging.info("%s: %s", name, path)
    finally:
        db.close()
    print("Downloads complete. Run scripts/01_prepare.py next.")


if __name__ == "__main__":
    try:
        main()
    except EOFError:
        print("No keyboard input available. Run this script as a file in the PowerShell terminal; do not pipe Python code into it.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Stopped. No automatic retry will be made.", file=sys.stderr)
        sys.exit(130)
