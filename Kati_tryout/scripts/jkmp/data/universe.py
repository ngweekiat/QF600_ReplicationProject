"""Investable universe: the largest stocks with a rank buffer, reconstituted once a year.

At each reconstitution month-end (December of year y), among stocks with ``valid_data``:

1. rank by market cap ``me`` (1 = largest; ties broken by ``id``);
2. keep current members still ranked <= ``exit_rank`` (100);
3. add every non-member ranked <= ``entry_rank`` (70);
4. fill with the next-largest non-members until there are ``target_n`` (80).

Steps 2–3 can give more than ``target_n`` members (no trimming), so the size is reported each year.

Members hold for the 12 formation months after reconstitution (Jan–Dec of y+1). A member leaves
early only when it has no valid row — it delists or fails a screen. Its last month in the universe is
its last valid row, whose ``ret_ld1`` is its final return (JKP returns include delisting returns), and
it does not re-enter before the next reconstitution, so it cannot flicker in and out.

Replaces R's size screen + 12/12 addition/deletion rule (docs/deviations.md D3).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HOLD_MONTHS = 12


def reconstitute(panel: pd.DataFrame, target_n: int, entry_rank: int, exit_rank: int,
                 month: int) -> pd.DataFrame:
    """Membership decisions: one row per (``recon_eom``, ``id``) with ``me_rank`` and ``status``.

    ``status`` is ``kept`` (incumbent within the exit rank), ``added`` (newcomer within the entry
    rank) or ``filled`` (next-largest newcomer, to reach ``target_n``).
    """
    eligible = panel.loc[panel["valid_data"] & (panel["eom"].dt.month == month), ["eom", "id", "me"]]
    members: set[int] = set()
    decisions = []
    for recon_eom, cs in eligible.groupby("eom", sort=True):
        cs = cs.sort_values(["me", "id"], ascending=[False, True]).reset_index(drop=True)
        cs["me_rank"] = np.arange(1, len(cs) + 1)
        incumbent = cs["id"].isin(members)
        keep = incumbent & (cs["me_rank"] <= exit_rank)
        add = ~incumbent & (cs["me_rank"] <= entry_rank)
        fill = pd.Series(False, index=cs.index)
        short = target_n - int((keep | add).sum())
        if short > 0:
            fill[cs.index[~incumbent & ~add][:short]] = True  # cs is in rank order
        chosen = cs[keep | add | fill].assign(
            status=np.select([keep, add, fill], ["kept", "added", "filled"], default="")[keep | add | fill],
            recon_eom=recon_eom,
        )
        decisions.append(chosen[["recon_eom", "id", "me_rank", "status"]])
        members = set(chosen["id"])
    if not decisions:
        raise ValueError(f"no valid stock-months in reconstitution month {month}")
    return pd.concat(decisions, ignore_index=True)


def _add_months(eom: pd.Series, k: pd.Series | int) -> pd.Series:
    return (eom.dt.to_period("M") + k).dt.end_time.dt.normalize()


def flag_universe(panel: pd.DataFrame, membership: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mark ``in_universe`` on the panel and record how each membership ended.

    Returns the panel with ``recon_eom`` (the reconstitution behind the membership; NaT outside it) and
    ``in_universe``, and the membership table with ``last_eom`` and ``exit``: ``held`` (all months to
    the next reconstitution or the end of the data), ``left_sample`` (no later valid rows at all —
    delisted, merged or dropped by JKP) or ``screened_out`` (later rows exist but failed a screen).
    """
    last_month = panel["eom"].max()
    grid = membership[["recon_eom", "id"]].merge(pd.DataFrame({"k": range(1, HOLD_MONTHS + 1)}), how="cross")
    grid["eom"] = _add_months(grid["recon_eom"], grid["k"])
    grid = grid[grid["eom"] <= last_month]

    valid = panel.loc[panel["valid_data"], ["id", "eom"]].assign(ok=1)
    grid = grid.merge(valid, on=["id", "eom"], how="left").fillna({"ok": 0})
    grid = grid.sort_values(["recon_eom", "id", "k"])
    grid["active"] = grid.groupby(["recon_eom", "id"])["ok"].cummin().astype(bool)  # out once a month fails

    active = grid[grid["active"]]
    out = panel.merge(active[["id", "eom", "recon_eom"]].assign(in_universe=True), on=["id", "eom"], how="left")
    out["in_universe"] = out["in_universe"].notna()  # True where matched to an active membership

    ends = active.groupby(["recon_eom", "id"]).agg(last_eom=("eom", "max"), n_months=("k", "size")).reset_index()
    possible = grid.groupby(["recon_eom", "id"]).size().rename("n_possible").reset_index()
    m = membership.merge(possible, on=["recon_eom", "id"], how="left").merge(ends, on=["recon_eom", "id"], how="left")
    m["n_months"] = m["n_months"].fillna(0).astype(int)
    last_valid = valid.groupby("id")["eom"].max()
    held = m["n_months"] >= m["n_possible"].fillna(0)
    gone = m["last_eom"].fillna(m["recon_eom"]) >= m["id"].map(last_valid)
    m["exit"] = np.select([held, gone], ["held", "left_sample"], default="screened_out")
    return out, m.drop(columns="n_possible")


def universe_stats(membership: pd.DataFrame) -> pd.DataFrame:
    """Per membership year: size, kept / added / filled, exits at reconstitution, turnover, early exits.

    ``turnover`` = share of names that are new at reconstitution (added + filled) / size.
    """
    rows, prev_size = [], 0
    for recon_eom, g in membership.groupby("recon_eom", sort=True):
        counts = g["status"].value_counts()
        kept, entrants = int(counts.get("kept", 0)), int(counts.get("added", 0) + counts.get("filled", 0))
        exits = g["exit"].value_counts()
        rows.append({
            "year": recon_eom.year + 1,
            "size": len(g),
            "kept": kept,
            "added": int(counts.get("added", 0)),
            "filled": int(counts.get("filled", 0)),
            "dropped_at_recon": prev_size - kept,
            "turnover": entrants / len(g),
            "early_exit_left_sample": int(exits.get("left_sample", 0)),
            "early_exit_screened_out": int(exits.get("screened_out", 0)),
            "avg_monthly_size": g["n_months"].sum() / max(g["n_months"].max(), 1),
        })
        prev_size = len(g)
    return pd.DataFrame(rows)


def build_universe(panel: pd.DataFrame, target_n: int, entry_rank: int, exit_rank: int,
                   month: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reconstitute, flag the panel, and summarise: returns (panel, membership, stats)."""
    membership = reconstitute(panel, target_n, entry_rank, exit_rank, month)
    panel, membership = flag_universe(panel, membership)
    stats = universe_stats(membership)
    log.info("Universe: %d reconstitutions, size %d-%d, mean turnover %.1f%%", len(stats),
             stats["size"].min(), stats["size"].max(), 100 * stats["turnover"].iloc[1:].mean())
    return panel, membership, stats
