"""Executed candidate books for active Portfolio-ML (research log trials 6–7; the team's Kati_tryout iterations
4.2 and 4.3, adapted to this replication).

The original December selection scores each (g, p, λ) by its model-implied utility r̃ᵀβ − ½βᵀΣ̃β. Here every
candidate is also *executed*: its own book is traded month by month exactly as ``pfml_weights`` and
``accounting`` would trade it (value-weight start, drift, aim recursion with the trading speed m, exit costs), so its
realised active return net of cost and its realised active risk can score it instead. All candidates share m and
the universe, so they are run together as the columns of one matrix.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from jkmp.evaluation.bootstrap import executed_score, score_standard_error
from jkmp.portfolio.portfolio_ml import SearchResult, model_columns, return_year
from jkmp.portfolio.recursion import MonthInputs
from jkmp.portfolio.trading_speed import trading_speed

MONTH = pd.offsets.MonthEnd(1)

AimsFn = Callable[[pd.Timestamp, np.ndarray], np.ndarray]


def candidate_table(results: Mapping[str, SearchResult], lambdas: Sequence[float]) -> pd.DataFrame:
    """Every (tag, g, p, λ) of the search, in the order of the original validation rows (its tie order)."""
    rows = []
    for tag, res in results.items():
        first = min(res.coefs)
        for p in res.coefs[first]:
            for l_no, lam in enumerate(lambdas):
                rows.append({"tag": tag, "g": res.g, "p": int(p), "l_no": l_no, "l": float(lam)})
    return pd.DataFrame(rows).rename_axis("cand").reset_index()


def candidate_aims(results: Mapping[str, SearchResult], table: pd.DataFrame) -> AimsFn:
    """Aim weights of every candidate at a formation month, for the universe ``ids``: S_t[:, cols(p)] β_y(p, λ),
    with β of the month's return year (as ``aim_portfolios``). Returns an (n_ids × n_candidates) matrix."""
    groups = [(tag, int(p), g["l_no"].to_numpy()) for (tag, p), g in table.groupby(["tag", "p"], sort=False)]

    def aims(eom: pd.Timestamp, ids: np.ndarray) -> np.ndarray:
        y = return_year(eom)
        blocks = []
        for tag, p, l_nos in groups:
            res = results[tag]
            s_ids, signal = res.signals[eom]
            pos = pd.Index(s_ids).get_indexer(ids)
            if (pos < 0).any():
                raise ValueError(f"universe stocks without a signal at {eom.date()}")
            with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked below
                blocks.append(signal[np.ix_(pos, model_columns(res, p))] @ res.coefs[y][p][:, l_nos])
        a = np.hstack(blocks)
        if not np.isfinite(a).all():
            raise FloatingPointError(f"non-finite candidate aims at {eom.date()}")
        return a

    return aims


def executed_books(inputs: Mapping[pd.Timestamp, MonthInputs], aims: AimsFn, panel: pd.DataFrame,
                   wealth: pd.DataFrame, rf: pd.Series, *, gamma: float, mu: float, iterations: int,
                   charge_exit_tc: bool) -> dict[str, np.ndarray | pd.DatetimeIndex]:
    """Benchmark-relative books for every column of ``aims`` over the (consecutive) months of ``inputs``: the
    ``pfml_weights`` recursion w = π^VW + m(π_start − π^VW) + (I − m)A and ``portfolio_returns`` accounting, vectorised
    over candidates. Returns ``eom_ret`` and (months × candidates) arrays ``r``, ``tc``, ``turnover``."""
    dates = sorted(inputs)
    look = {e: g.set_index("id") for e, g in panel.loc[panel["eom"].isin(dates), ["id", "eom", "ret_ld1", "tr_ld1",
                                                                                  "lambda"]].groupby("eom")}
    mu_ld1 = wealth.set_index("eom")["mu_ld1"]
    r_out, tc_out, to_out = [], [], []
    prev_ids, prev = None, None
    for eom in dates:
        x = inputs[eom]
        ids, pt = x.ids, look[eom]
        a = aims(eom, ids)
        me = x.panel_t.set_index("id")["me"].reindex(ids).to_numpy()
        vw = me / me.sum()
        exit_tc = exit_to = 0.0
        if prev_ids is None:
            held = np.repeat(vw[:, None], a.shape[1], axis=1)
        else:
            pos = pd.Index(prev_ids).get_indexer(ids)
            held = np.zeros((len(ids), a.shape[1]))
            held[pos >= 0] = prev[pos[pos >= 0]]
            leaving = ~pd.Index(prev_ids).isin(ids) & pd.Index(prev_ids).isin(x.panel_t["id"])
            if charge_exit_tc and leaving.any():
                lam_exit = pt["lambda"].reindex(prev_ids[leaving]).to_numpy()
                if np.isnan(lam_exit).any():
                    raise ValueError(f"exiting stocks without lambda at {eom.date()}")
                d = prev[leaving]
                exit_tc = 0.5 * x.wealth * (lam_exit[:, None] * d**2).sum(axis=0)
                exit_to = np.abs(d).sum(axis=0)
        m = trading_speed(x.sigma, x.lam, x.wealth, gamma, float(rf[eom + MONTH]), mu, iterations)
        with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked below
            w = vw[:, None] + m @ (held - vw[:, None]) + a - m @ a
        if not np.isfinite(w).all():
            raise FloatingPointError(f"non-finite candidate weights at {eom.date()}")
        ret = pt["ret_ld1"].reindex(ids).to_numpy()
        if np.isnan(ret).any():
            raise ValueError(f"universe stocks without ret_ld1 at {eom.date()}")
        dw = w - held
        with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked below
            r_t = ret @ w
        if not np.isfinite(r_t).all():
            raise FloatingPointError(f"non-finite candidate returns at {eom.date()}")
        r_out.append(r_t)
        tc_out.append(0.5 * x.wealth * (x.lam[:, None] * dw**2).sum(axis=0) + exit_tc)
        to_out.append(np.abs(dw).sum(axis=0) + exit_to)
        tr = pt["tr_ld1"].reindex(ids).to_numpy()
        held_any = (w != 0.0).any(axis=1)
        if np.isnan(tr[held_any]).any():
            raise ValueError(f"held positions lack tr_ld1 at {eom.date()}")
        prev = w * (1.0 + np.nan_to_num(tr))[:, None] / (1.0 + float(mu_ld1[eom]))
        prev_ids = ids
    return {"eom_ret": pd.DatetimeIndex(dates) + MONTH, "r": np.vstack(r_out), "tc": np.vstack(tc_out),
            "turnover": np.vstack(to_out)}


def active_series(books: Mapping, bench: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Net and gross active returns of every candidate vs the benchmark (``eom_ret, r, tc``), months × candidates."""
    b = bench.set_index("eom_ret").reindex(books["eom_ret"])
    if b[["r", "tc"]].isna().any().any():
        raise ValueError("benchmark missing for some candidate months")
    rb, tcb = b["r"].to_numpy()[:, None], b["tc"].to_numpy()[:, None]
    return (books["r"] - books["tc"]) - (rb - tcb), books["r"] - rb


def executed_scores(books: Mapping, net: np.ndarray, gross: np.ndarray, decisions: Sequence[pd.Timestamp],
                    gamma: float) -> np.ndarray:
    """Executed score of every candidate at each December return month in ``decisions``, over all candidate months
    up to it: (decisions × candidates)."""
    eom_ret = books["eom_ret"]
    return np.vstack([executed_score(net[eom_ret <= d], gross[eom_ret <= d], gamma, axis=0) for d in decisions])


def choose(table: pd.DataFrame, picks: Sequence[int], decisions: Sequence[pd.Timestamp], score: np.ndarray) -> pd.DataFrame:
    """The ``chosen`` table (year, eom_ret, tag, g, p, l_no, l, h, score) for picked candidate indices."""
    rows = []
    for d, k, s in zip(decisions, picks, score):
        c = table.iloc[k]
        rows.append({"year": d.year + 1, "eom_ret": d, "tag": c["tag"], "g": c["g"], "p": int(c["p"]),
                     "l_no": int(c["l_no"]), "l": c["l"], "h": np.inf, "cand": int(c["cand"]), "score": float(s)})
    return pd.DataFrame(rows)


def one_se_pick(table: pd.DataFrame, scores: np.ndarray, net: np.ndarray, gross: np.ndarray, gamma: float,
                reps: int, block: int, seed: int) -> tuple[int, dict]:
    """Within one bootstrap standard error of the best executed score, the simplest candidate: fewest RFF features p,
    then the largest ridge penalty λ, then the smaller RFF variance g. ``net``/``gross``: the months up to the
    December, months × candidates."""
    best = int(np.argmax(scores))
    se = score_standard_error(net[:, best], gross[:, best], gamma, reps, block, seed)
    eligible = np.flatnonzero(scores >= scores[best] - se)
    order = table.iloc[eligible].sort_values(["p", "l", "g"], ascending=[True, False, True], kind="stable")
    return int(order.index[0]), {"best": best, "se": se, "n_eligible": len(eligible)}
