"""Stage 1 runner: data/raw -> data/processed/.

Writes panel.parquet, wealth.parquet, universe_membership.parquet, screen_log.csv and
universe_stats.csv, and prints the universe size and turnover per year.

    python scripts/01_prepare.py
"""

from __future__ import annotations

import argparse
import logging

from jkmp.config import load_config
from jkmp.data import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/base.yaml)")
    parser.add_argument("--override", action="append", default=[], help="dotted override, e.g. investor.wealth=1.0e+9")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.override)

    data = prepare.build_panel(cfg)
    out = cfg.paths.processed
    out.mkdir(parents=True, exist_ok=True)
    data.panel.to_parquet(out / "panel.parquet", index=False)
    data.wealth.to_parquet(out / "wealth.parquet", index=False)
    data.membership.to_parquet(out / "universe_membership.parquet", index=False)
    data.screen_log.to_csv(out / "screen_log.csv", index=False)
    data.universe_stats.to_csv(out / "universe_stats.csv", index=False)

    p = data.panel
    print("\nUniverse size and turnover per year (December reconstitution -> next calendar year):")
    print(data.universe_stats.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    logging.info("Wrote %s: panel %d rows (%d valid, %d in universe)", out, len(p),
                 int(p["valid_data"].sum()), int(p["in_universe"].sum()))


if __name__ == "__main__":
    main()
