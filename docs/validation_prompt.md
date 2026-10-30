# Claude Code Prompt — Validate the Implementation Against the Research Framework

Paste everything inside the code block into a **new** Claude Code session started from `QF600_Replication/`.
It is a read-only audit: the only file it writes is `docs/validation_report.md`.

```text
ROLE
You are an independent model validator (think SR 11-7 "effective challenge"). Your job is to check whether
this repository implements the requirements in docs/research_framework.md, and to verify the numbers the
team reports. You did not build this code; assume nothing is correct until you have seen the evidence.

READ FIRST
- docs/research_framework.md   — the requirements (written 2026-10-29, BEFORE the rebaseline below)
- CLAUDE.md, configs/base.yaml, configs/experiments/*.yaml, src/jkmp/config.py
- docs/research_log.md, docs/deviations.md, docs/claude_code_prompts.md (the phase plan actually followed)
- git log --stat, git status, git tag

HARD RULES
1. Read-only. Do not edit, delete, move or reformat any file except creating docs/validation_report.md.
   Do not commit. Do not change configs.
2. Do NOT unlock the holdout and do not re-run any pipeline script (00–15). Holdout outputs that already
   exist (outputs/holdout*, outputs/holdout_final) may be READ; never regenerate them.
3. You may run: `.venv/bin/pytest`, `git` read commands, and short read-only Python that loads existing
   files in outputs/ and data/processed/ to recompute statistics. Keep each script under ~2 minutes.
4. Every verdict cites evidence: a file path + line number (code/config/docs) or a file path + the
   recomputed number (outputs). No evidence → verdict is "Not verified", never "Met".
5. If the code and the research log disagree, the code and the outputs win; report the discrepancy.

SUPERSESSIONS (decisions taken AFTER the framework was written — judge these requirements in their
superseded form, and say so in the verdict)
S1. Timeline. Framework: weights from 2015-12, decide on 2016–2019, report 2020–2025.
    Now: the paper's timeline — data from 1952-12, validation 1971–1980, test 1981-01..2020-12 (480 months,
    used for keep/kill decisions), holdout 2021-01..2025-12 locked until the end and run once.
    → Baseline fix 1 is replaced by this; protocol rule 3 becomes "decide on 1981–2020, holdout untouched
      until the final run"; go/no-go criteria "measured on 2020–2025" become "measured on the 2021–2025
      holdout" (and also report the 1981–2020 value).
S2. Benchmark for net IR. Changed during Trial 3 from the traded VW universe to a "paper index" (benchmark
    trading cost = 0). Check that this change is documented, and that the Trial 3 decision is the same
    under both definitions.
S3. Housekeeping items "report has no executive summary / 23 pages" are out of scope for code validation:
    record them as "Pending — report not yet rewritten" unless docs show otherwise.

VERDICT SCALE
Met · Met (superseded form) · Partial · Not met · Deviates — documented (in deviations.md or the
research log) · Deviates — undocumented · Not verified.

REQUIREMENTS TO CHECK (keep these IDs in the report)

A. Baseline fixes
A1  (S1) Timeline in configs/base.yaml matches S1; holdout keys exist; data pull and wealth path cover
    1952–2025 with $10B anchored at 2020-12-31 (rolled back AND forward). Check the start-weight
    sensitivity the framework worried about: from outputs/table2_test.csv inputs (the pf files), recompute
    Table 2 Utility for 1981–2020 and for 1982–2020 (drop the first 12 months); does the ranking of
    Portfolio-ML, Static-ML*, Static-ML change?
A2  Grids: portfolio_ml.p_grid contains 8, 16, 32; static_ml.u_grid contains 0 and 0.1; grid-edge shares
    are reported (outputs/tables/grid_edges.csv). Flag any hyperparameter with > 25% of picks on an edge.
A3  Feature importance averaged over 20 seeds with standard errors; Figure 5 shows error bars.
A4  `import wrds` only inside pull functions (grep the whole src/ tree for top-level imports); tests run
    without WRDS; CLAUDE.md "Design" and "Status" sections describe the CURRENT timeline (flag any
    1993 / 2015 / 2020–2025 statements left over); no WRDS-licensed CSV is tracked by git
    (`git ls-files | grep -i csv`) — a source reference is given instead.

B. Research protocol
B1  Every trial in docs/research_log.md has a pre-registration (hypothesis, change, metric, keep threshold,
    kill criterion, date) placed before its result. Note that ordering in time cannot be proven from the
    log alone; check output file modification times against the log's statements where possible and say
    what can and cannot be verified.
B2  One change at a time: diff each experiment YAML against base.yaml and against the previous kept
    version. List every key that changes. Flag any change not named in the trial's pre-registration
    (e.g. the gamma_te re-calibration in Trial 3 — is it pre-registered?).
B3  (S1) Decisions use 1981–2020 only. Inspect jkmp.evaluation.holdout (the guard): is it called on every
    evaluation path (metrics/table2, grids, feature importance, regimes, active, breadth, half-life,
    monitor)? List any path that can emit a return month > 2020-12 while holdout_locked is true.
    Run its tests.
B4  Keep rule: the posterior probability P = Φ(mean(d) / (sd(d)/√T)) ≥ 80%. Find the function that
    computes it (e.g. metrics.prob_outperformance) and check the formula, the ddof and T. Recompute P for
    Trials 2, 3 and 4 from the saved monthly series and compare with the research log.
B5  Trial count is kept and complete (count every evaluated variant, including calibration runs if they
    looked at returns — they should not have). Deflated Sharpe ratio for the final model: does an output
    exist? (scripts/15_holdout.py writes deflated_sharpe.csv.) If not, "Not met".
B6  Robustness battery for each KEPT change: 20 RFF seeds, sub-periods, γ ∈ {5, 10, 20}, wealth
    $1B–$250B. Report which of the four were actually run for Trials 2 and 3.

C. Iteration 1 — benchmark-relative Portfolio-ML (Trial 2)
C1  `portfolio_ml.add_constant` flag in scale_signals; default reproduces the baseline.
C2  `investor.gamma_te` (and the calibration grid / targets) in config.py and base.yaml; calibration used
    validation months only (1972–1980) and ex-ante TE only — verify in scripts/12_active_pfml.py that no
    return is read during calibration.
C3  Weight recursion on active weights a_t = m·a_inh + (I − m)·A_t, then π = π^VW + a. Check that the
    traded Δπ equals Δa and that the benchmark's own turnover/cost is reported.
C4  Metrics implemented: active return, ex-ante and realised TE, gross and net IR, β to benchmark, theme
    exposures (Xᵀa), share of negative total weights. Recompute net IR, realised TE and β for the 1%, 2%,
    4% budgets from outputs/active/*/pf.parquet and compare with outputs/active/summary.csv and the log.
C5  Long-only limitation stated somewhere (log, deviations, or docs).
C6  Kill criterion evaluated as pre-registered; decision recorded.

D. Iteration 2 — breadth vs capacity (Trial 3)
D1  configs/experiments/universe_250.yaml: target_n 250, entry/exit ranks scaled; benchmark = that
    universe's own VW portfolio; covariance factors still from the top 500.
D2  Run-time measured at N = 250 before the full run; N = 500 decision documented.
D3  Rank IC by size bucket (top 80 vs 81–250) by decade — recompute from data/processed predictions and
    returns and compare with the log.
D4  Capacity curve over the wealth grid; capacity = AUM where net IR < 0.3 (check the interpolation).
D5  Cost sensitivity is evaluation-only (no retraining): VIX-scaled Λ (1990–2020) and stock-level
    Λ·σ_i,t/σ̄_i,t with σ̄ from data ≤ t ONLY — inspect the code line by line for look-ahead.
    Capacity reported under all three cost models.
D6  (S2) Benchmark-definition change documented; recompute the Trial 3 keep probability under both
    definitions.

E. Iteration 3 — time-decayed estimation (Trial 4)
E1  Discounted running sums in coefs_from_sums and december_utility; δ = 0.5^(1/h); normalisation uses
    the discounted month counts.
E2  `portfolio_ml.half_life_years` = {3, 5, 10, ∞}; h chosen each December jointly with (g, p, λ) by the
    same cumulative validation score.
E3  A test shows h = ∞ reproduces the previous β exactly (find it; run it).
E4  Chosen h by year saved and reported; kill criterion applied as pre-registered (35/40 years h = ∞,
    P = 2.6% per the log — recompute).

F. Closure — go/no-go and monitoring
F1  Go/no-go thresholds were recorded in the research log BEFORE the baseline ran (check placement).
F2  Evaluate every go/no-go criterion for the final kept version (Trial 3), on the holdout (S1) and on
    1981–2020, from saved outputs only: net IR ≥ 0.3; P(beat index) ≥ 80%; deflated Sharpe probability
    > 0.95; realised TE within ±20% of the 2% budget; capacity ≥ $10B under every cost model.
    Give PASS / FAIL / NOT COMPUTED per criterion with the number.
F3  scripts/11_monitor.py (+ library code): implements all six checks with the framework's flag rules —
    rolling 36-month IR < 0 or drawdown > 2 × TE budget; β / theme exposure outside a stated limit (are
    the limits defined anywhere?); rolling 12-month rank IC < 0 for 6 consecutive months; realised > predicted
    trading cost for 3 consecutive months; half-life jump > 1 grid step in a year; bias statistic outside
    1 ± √(2/T). Check each rule in code against the framework wording. Inspect
    outputs/holdout_final/monitor/monitor.csv: are flags firing for sensible reasons (e.g. flag_tc in
    1981 — is "predicted cost" defined correctly)? Confirm a dashboard file exists.

G. Project-guideline deliverables (QF600)
G1  Execution scripts exist for data, model development, validation/backtesting and monitoring.
G2  Datasets: CSV or link + source reference; nothing WRDS-licensed redistributed.
G3  Report and slides: record presence and whether the report reflects the CURRENT design
    (report/QF600_Replication_Report.pdf vs the 1981–2020 timeline). Do not edit them.

H. Integrity and reproducibility
H1  `.venv/bin/pytest` — report pass/fail counts.
H2  git: list uncommitted / untracked files that results depend on (scripts, configs, src). Check tags
    (baseline-v1 and per-trial tags) point at code that can reproduce each result.
H3  Recompute Table 2 (Market, Factor-ML, Static-ML*, Portfolio-ML, Portfolio-ML linear) for 1981–2020
    from the saved monthly pf files and compare with outputs/tables/table2_vs_paper.csv (tolerance 1e-6).

OUTPUT — write docs/validation_report.md with exactly these sections:
1. Verdict (3–5 sentences): is the framework implemented, and can the reported results be trusted?
2. Traceability matrix: | ID | Requirement (short) | Verdict | Evidence (path:line or recomputed number) |
   — one row per ID above, in order.
3. Recomputed numbers vs reported: | Quantity | Reported (where) | Recomputed | Match? |
4. Go/no-go scorecard (F2).
5. Gaps, ranked by severity: Critical (result could be wrong) · Major (requirement unmet) · Minor
   (documentation). For each: what, evidence, the smallest fix.
6. What could not be verified, and why.

STOP after writing the report. Print the Verdict section and the Critical/Major gaps in chat, nothing else.
```

---

### Notes for you
- Save the framework doc as `docs/research_framework.md` first; the prompt reads it from there. (Done in your repo alongside this file.)
- Commit your iteration work **before** running the audit, so H2 tests the committed state, not your working tree.
- Expect at least these findings, from my earlier review: deflated Sharpe ratio not yet computed (B5), CLAUDE.md out of date (A4), realised TE above the ±20% band (F2), report not rewritten (G3), robustness battery partly run (B6).
