"""Buffered top-N universe, reconstituted annually using only current information.

This is the Python design documented in deviations.md (D3), rather than the
original R code's 12-month addition/deletion rule. Stocks that fail eligibility
during a membership year cannot re-enter before the next reconstitution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _choose(rows: pd.DataFrame, previous: set[int], target_n: int,
            entry_rank: int, exit_rank: int) -> pd.DataFrame:
    candidates = rows.loc[rows["valid_data"] & rows["me"].notna(), ["id", "me"]].copy()
    candidates = candidates.sort_values(["me", "id"], ascending=[False, True])
    candidates["me_rank"] = np.arange(1, len(candidates) + 1)
    kept = candidates["id"].isin(previous) & (candidates["me_rank"] <= exit_rank)
    added = ~candidates["id"].isin(previous) & (candidates["me_rank"] <= entry_rank)
    selected = candidates.loc[kept | added].copy()
    selected["status"] = np.where(selected["id"].isin(previous), "kept", "added")
    slots = max(target_n - len(selected), 0)
    if slots:
        fill = candidates.loc[~candidates["id"].isin(selected["id"])].head(slots).copy()
        fill["status"] = "filled"
        selected = pd.concat([selected, fill], ignore_index=True)
    # Never trim retained members just to enforce exactly target_n.
    return selected.sort_values("me_rank").reset_index(drop=True)


def _validate(panel: pd.DataFrame, target_n: int, entry_rank: int, exit_rank: int, month: int) -> None:
    if not 0 < entry_rank <= target_n <= exit_rank or not 1 <= month <= 12:
        raise ValueError("invalid universe ranks or reconstitution month")
    if panel.duplicated(["id", "eom"]).any():
        raise ValueError("universe input has duplicate stock-months")


def reconstitute(panel: pd.DataFrame, target_n: int, entry_rank: int, exit_rank: int,
                 month: int = 12) -> pd.DataFrame:
    """Decisions at each reconstitution date, in market-cap order."""
    _validate(panel, target_n, entry_rank, exit_rank, month)
    previous, rows = set(), []
    for eom, cross_section in panel.loc[panel["eom"].dt.month == month].groupby("eom", sort=True):
        selected = _choose(cross_section, previous, target_n, entry_rank, exit_rank)
        rows.append(selected.assign(recon_eom=eom))
        previous = set(selected["id"])
    if not rows:
        return pd.DataFrame(columns=["id", "me", "me_rank", "status", "recon_eom"])
    return pd.concat(rows, ignore_index=True)


def build_universe(panel: pd.DataFrame, target_n: int, entry_rank: int, exit_rank: int,
                   month: int = 12) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return panel flags, member histories, and annual membership statistics.

    The month after reconstitution starts the membership year. No future values
    determine selection; later rows only determine when a member leaves.
    """
    _validate(panel, target_n, entry_rank, exit_rank, month)
    panel = panel.copy().reset_index(drop=True)
    panel["in_universe"] = False
    panel["recon_eom"] = pd.NaT
    membership, statistics = [], []
    previous_active, previous_selected = set(), set()
    if panel.empty:
        return panel, pd.DataFrame(), pd.DataFrame()
    end = panel["eom"].max()
    positions = {(row.eom, row.id): i for i, row in enumerate(panel[["eom", "id"]].itertuples(index=False))}
    eligible = panel["valid_data"].fillna(False).to_numpy(dtype=bool)
    for eom, cross_section in panel.loc[panel["eom"].dt.month == month].groupby("eom", sort=True):
        if eom >= end:
            continue
        selected = _choose(cross_section, previous_active, target_n, entry_rank, exit_rank)
        year_end = min(eom + pd.offsets.MonthEnd(12), end)
        months = pd.date_range(eom + pd.offsets.MonthEnd(1), year_end, freq="ME")
        survivors, exits = set(), 0
        for member in selected.itertuples(index=False):
            last, exit_reason = pd.NaT, "held"
            for date in months:
                pos = positions.get((date, member.id))
                if pos is None:
                    exit_reason = "left_sample"
                    break
                if not eligible[pos]:
                    exit_reason = "screened_out"
                    break
                panel.at[pos, "in_universe"] = True
                panel.at[pos, "recon_eom"] = eom
                last = date
            if exit_reason == "held":
                survivors.add(member.id)
            else:
                exits += 1
            membership.append({
                "recon_eom": eom, "id": member.id, "me": member.me, "me_rank": member.me_rank,
                "status": member.status, "last_eom": last, "exit": exit_reason,
            })
        current = set(selected["id"])
        dropped = len(previous_selected - current)
        counts = selected["status"].value_counts()
        newcomers = int(counts.get("added", 0) + counts.get("filled", 0))
        # Mean of the entry and exit fractions; initial establishment is not turnover.
        turnover = ((newcomers / max(len(current), 1) + dropped / max(len(previous_selected), 1)) / 2
                    if previous_selected else 0.0)
        statistics.append({
            "year": (eom + pd.offsets.MonthEnd(1)).year, "recon_eom": eom, "size": len(selected),
            "kept": int(counts.get("kept", 0)), "added": int(counts.get("added", 0)),
            "filled": int(counts.get("filled", 0)), "dropped": dropped,
            "turnover": turnover, "early_exits": exits,
        })
        previous_active, previous_selected = survivors, current
    member_columns = ["recon_eom", "id", "me", "me_rank", "status", "last_eom", "exit"]
    stat_columns = ["year", "recon_eom", "size", "kept", "added", "filled", "dropped", "turnover", "early_exits"]
    return panel, pd.DataFrame(membership, columns=member_columns), pd.DataFrame(statistics, columns=stat_columns)
