"""Offline tests for the buffered top-N universe (jkmp.data.universe), on small synthetic panels.

Scaled-down rule: target 4, entry rank 3, exit rank 5 (the real config is 80 / 70 / 100).
"""

import pandas as pd

from jkmp.data.universe import build_universe, reconstitute

TARGET, ENTRY, EXIT = 4, 3, 5
DEC1, DEC2 = pd.Timestamp("2010-12-31"), pd.Timestamp("2011-12-31")


def _december(eom, me_by_id):
    return pd.DataFrame({"eom": eom, "id": list(me_by_id), "me": list(me_by_id.values()), "valid_data": True})


def _decisions(panel):
    m = reconstitute(panel, TARGET, ENTRY, EXIT, month=12)
    return {eom: dict(zip(g["id"], g["status"])) for eom, g in m.groupby("recon_eom")}


def test_first_reconstitution_takes_the_top_n():
    d = _decisions(_december(DEC1, {1: 90, 2: 80, 3: 70, 4: 60, 5: 50, 6: 40}))
    assert d[DEC1] == {1: "added", 2: "added", 3: "added", 4: "filled"}


def test_buffer_keeps_incumbents_to_exit_rank_and_adds_newcomers_in_entry_rank():
    year1 = _december(DEC1, {1: 90, 2: 80, 3: 70, 4: 60, 5: 50, 6: 40, 7: 30})
    # Year 2 ranks: 5 (newcomer) 1st, 1 2nd, 6 (newcomer) 3rd, 2 4th, 3 5th, 7 6th, 4 7th.
    year2 = _december(DEC2, {5: 95, 1: 90, 6: 85, 2: 80, 3: 75, 7: 70, 4: 65})
    d = _decisions(pd.concat([year1, year2]))
    # Incumbents 1, 2, 3 are within rank 5 -> kept; 4 at rank 7 -> dropped; newcomers 5, 6 in top 3 -> added.
    # 5 members > target 4: no trimming. Newcomer 7 at rank 6 is not needed for filling.
    assert d[DEC2] == {5: "added", 1: "kept", 6: "added", 2: "kept", 3: "kept"}


def test_fill_uses_next_largest_newcomers():
    year1 = _december(DEC1, {1: 90, 2: 80, 3: 70, 4: 60, 5: 50, 6: 40})
    year2 = _december(DEC2, {5: 90, 6: 80, 1: 70, 7: 60, 8: 50, 2: 40, 3: 30, 4: 20})
    d = _decisions(pd.concat([year1, year2]))
    # Newcomers 5, 6 at ranks 1-2 added; incumbent 1 (rank 3) kept; 2, 3, 4 below rank 5 dropped;
    # one slot left -> filled by the next-largest newcomer, 7 (rank 4).
    assert d[DEC2] == {5: "added", 6: "added", 1: "kept", 7: "filled"}


def test_membership_holds_jan_to_dec_and_delisted_stock_exits_at_its_last_row():
    months = pd.date_range("2010-12-31", periods=13, freq="ME")  # Dec 2010 .. Dec 2011
    rows = []
    for eom in months:
        for sid, me in {1: 90, 2: 80, 3: 70, 4: 60, 5: 50}.items():
            delisted = sid == 2 and eom > pd.Timestamp("2011-05-31")  # stock 2 last trades May 2011
            gap = sid == 3 and eom == pd.Timestamp("2011-03-31")      # stock 3 fails a screen in March
            if not delisted:
                rows.append({"eom": eom, "id": sid, "me": me, "valid_data": not gap})
    panel, membership, stats = build_universe(pd.DataFrame(rows), TARGET, ENTRY, EXIT, month=12)
    inu = panel[panel["in_universe"]]
    by_id = inu.groupby("id")["eom"].agg(["min", "max", "size"])
    assert by_id.loc[1].tolist() == [pd.Timestamp("2011-01-31"), pd.Timestamp("2011-12-31"), 12]
    assert by_id.loc[2].tolist() == [pd.Timestamp("2011-01-31"), pd.Timestamp("2011-05-31"), 5]  # exits at last row
    assert by_id.loc[3].tolist() == [pd.Timestamp("2011-01-31"), pd.Timestamp("2011-02-28"), 2]  # no re-entry after gap
    assert 5 not in by_id.index  # never selected
    assert not panel.loc[panel["eom"] == DEC1, "in_universe"].any()  # Dec 2010 formation precedes membership
    exits = membership[membership["recon_eom"] == DEC1].set_index("id")["exit"].to_dict()
    assert exits == {1: "held", 2: "left_sample", 3: "screened_out", 4: "held"}
    assert stats.loc[stats["year"] == 2011, "size"].item() == 4
