"""Verify local inputs, exact feature substitution and orchestration without fits.

The checks hash the immutable source snapshot, inspect every prepared panel's
keys/timing, and independently recompute ranks on three full cross-sections.
The report distinguishes a verified build from completed model experiments.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from workflow import ROOT, UNIVERSES, config_path, originals_unchanged
from jkmp.config import load_config
from jkmp.features import load_theme_directions, theme_scores
from prepare_inputs import bootstrap, digest
from run_all import plan
from script_groups import normalise_group, script_path


def verify():
    checks = []

    def check(name, passed, detail=""):
        checks.append(dict(check=name, status="PASS" if bool(passed) else "FAIL", detail=detail))

    manifest = bootstrap()  # Cached path verifies local hashes, without siblings.
    names = json.loads((ROOT / "data/characteristics/recommended_50.json").read_text())
    check("exactly_50_unique_inputs", len(names) == len(set(names)) == 50)
    forbidden = {"id", "eom", "ret_exc", "ret_exc_lead1m", "ret_ld1", "tr_ld1", "lambda", "g"}
    check("no_targets_or_identifiers_added_as_predictors", not set(names) & forbidden)
    check("original_keats_source_unchanged", originals_unchanged())
    if manifest.get('selection_scope')=='us_large':
        selection=json.loads((ROOT/'outputs/us_high_cap_selection/selection_manifest.json').read_text())
        check('association_selection_is_us_high_cap_2005_2014_only',selection['scope']=='us_large only' and
              selection['selection_return_start']=='2005-01-31' and selection['selection_return_end']=='2014-12-31' and not selection['other_markets_used'])
        check('no_ml_regression_or_test_outcomes_in_selection',not selection['models_fitted'] and
              not selection['regression_r2_used'] and not selection['test_outcomes_used'])
        check('active_names_match_us_association_report',json.loads((ROOT/'outputs/us_high_cap_selection/recommended_50.json').read_text())==names)
        clusters=pd.read_parquet(ROOT/'data/reference/cluster_labels.parquet')
        details=pd.read_parquet(ROOT/'data/reference/factor_details.parquet')
        check('all_50_in_both_published_jkp_references',set(names).issubset(set(clusters.characteristic)&set(details.abr_jkp.dropna())))
        check('jkp_only_policy_enforced',selection.get('jkp_only') and selection['published_group_count']==50 and selection['proxy_group_count']==0)
        baseline=json.loads((ROOT/'data/characteristics/implementation_baseline.json').read_text())
        core=list((ROOT/'scripts/jkmp/models').glob('*.py'))+list((ROOT/'scripts/jkmp/portfolio').glob('*.py'))
        old_core=[p for p in core if p.relative_to(ROOT).as_posix() in baseline['source_hashes']]
        # The return-model validation population is an authorised adaptation.
        # All other original coefficient/portfolio mathematics stays pinned.
        preserved_core=[p for p in old_core if p.relative_to(ROOT).as_posix()!='scripts/jkmp/models/return_model.py']
        check('other_ml_and_weight_learning_source_preserved',all(digest(p)==baseline['source_hashes'][p.relative_to(ROOT).as_posix()] for p in preserved_core))
        check('only_authorised_static_star_library_added',set(p.name for p in core if p not in old_core)=={'static_ml_star.py'})
        from jkmp.config import _read_yaml
        same=True
        for config in (ROOT/'configs').rglob('*.yaml'):
            if config.stem.startswith('iteration_'):continue
            previous=baseline['non_feature_configs'][config.relative_to(ROOT).as_posix()].copy()
            current=_read_yaml(config,seen=())
            previous.pop('features',None);current.pop('features',None)
            expected_scope='all' if config.name=='kati_base.yaml' else 'universe'
            check(f'{config.relative_to(ROOT)}:authorised_rff_validation_scope',
                  current['return_model'].get('validation_scope')==expected_scope)
            current['return_model'].pop('validation_scope',None)
            expected_static=dict(k_grid=[1.0,1/3,0.2,0.1,0.05],u_grid=[0.0,0.1,0.25,0.5,1.0],g_grid=[0.0,1.0,2.0,3.0])
            if config.parent.name=='us_large' and current.get('static_ml')==expected_static:
                # Authorised base-only grid expansion; iteration loading pins
                # its unused Static ML settings to the original shared defaults.
                current['static_ml']=previous['static_ml']
            # The base model mandate and plot labels changed; economic/model settings did not.
            for section,key in [('reporting','main_types'),('regimes','plot_types')]:
                previous.get(section,{}).pop(key,None);current.get(section,{}).pop(key,None)
            same=same and previous==json.loads(json.dumps(current,default=str))
        check('non_feature_economic_model_settings_preserved',same)
        check('old_characteristic_archives_removed',not any((ROOT/'data/archives').glob('*')) and not any((ROOT/'outputs/archive').glob('*')))
    check("snapshot_all_three_markets_and_training_panels", len(manifest["regional_audits"]) == 2 and
          all(x["exact_broad_keys_preserved"] and x["absent_source_rows"] == 0 for x in manifest["regional_audits"]))
    spec = pd.read_csv(ROOT / "data/characteristics/theme_directions.csv")
    selected_metadata=pd.read_csv(ROOT/'data/characteristics/selected_feature_metadata.csv')
    check("frozen_group_metadata_matches_selected_50", len(spec) == 50 and
          int(spec.published_jkp_153.sum()) == int(selected_metadata.published_jkp_153.sum()) and spec.model_theme.nunique() == 13)
    check("theme_sign_policy", spec.loc[spec.published_jkp_153, "model_theme_direction"].equals(
          spec.loc[spec.published_jkp_153, "direction"]) and
          np.array_equal(spec.loc[~spec.published_jkp_153, "model_theme_direction"], spec.loc[~spec.published_jkp_153, "training_direction"]))

    configs = [(u, "base") for u in UNIVERSES] + [("us_large", f"iteration_{i}") for i in range(1,6)]
    for market, group in configs:
        cfg = load_config(config_path(market, group))
        check(f"{market}/{group}:rff_tuning_scope", cfg.return_model.validation_scope == 'universe')
        check(f"{market}/{group}:ridge_penalties", len(cfg.ridge_grid.values)==101 and cfg.ridge_grid.values[0]==0.0)
        check(f"{market}/{group}:frozen_50_only", len(cfg.features.names) == 50 and set(cfg.features.names) == set(names))
        check(f"{market}/{group}:local_paths", all(p.resolve().is_relative_to(ROOT) for p in
              (cfg.paths.raw, cfg.paths.processed, cfg.paths.outputs)))
        directions = load_theme_directions(cfg.paths.raw, cfg.features.themes)
        check(f"{market}/{group}:all_covariance_themes_supported", len(directions) == 50 and directions.theme.nunique() == 13)

    dates = [pd.Timestamp(d) for d in ("2004-12-31", "2019-12-31", "2025-11-30")]
    for market in UNIVERSES:
        cfg = load_config(config_path(market))
        path = cfg.paths.processed / "panel.parquet"
        check(f"{market}:prepared_panel_exists", path.exists())
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["id", "eom", "in_universe", "ret_ld1", "ret_ld1_original", "ret_ld1_imputed_zero"])
        check(f"{market}:unique_formation_keys", not frame.duplicated(["id", "eom"]).any())
        check(f"{market}:declared_horizon", frame.eom.max() == pd.Timestamp("2025-11-30") and
              frame.eom.dt.is_month_end.all() and (frame.eom + pd.offsets.MonthEnd()).le(pd.Timestamp(cfg.dates.test_end)).all())
        assumed = frame.ret_ld1_imputed_zero
        check(f"{market}:explicit_zero_return_policy", frame.loc[assumed, "ret_ld1_original"].isna().all() and
              frame.loc[assumed, "ret_ld1"].eq(0).all() and frame.loc[assumed, "in_universe"].all(), f"{int(assumed.sum())} labelled assumptions")
        source = ROOT / f"data/source/screened_{'EU' if market=='eu_large' else 'US'}.parquet"
        source_keys = pd.read_parquet(source, columns=["id", "eom"])
        a = pd.MultiIndex.from_frame(frame[["id", "eom"]])
        b = pd.MultiIndex.from_frame(source_keys)
        check(f"{market}:broad_training_population_preserved", len(a) == len(b) and a.difference(b).empty and b.difference(a).empty)
        sample = pd.read_parquet(source, columns=["id", "eom", *names], filters=[("eom", "in", dates)])
        expected = sample[names].groupby(sample.eom).rank(method="average", pct=True).fillna(.5)
        expected.index = pd.MultiIndex.from_frame(sample[["id", "eom"]])
        actual = pd.read_parquet(path, columns=["id", "eom", *names], filters=[("eom", "in", dates)]).set_index(["id", "eom"])
        actual = actual.reindex(expected.index)[names]
        check(f"{market}:independent_three_month_rank_recalculation", np.allclose(expected, actual, atol=1e-12, rtol=0),
              f"{len(actual)} rows; all 50 ranks, missing raw observations become 0.5")
        ds = load_theme_directions(cfg.paths.raw, cfg.features.themes)
        themes = theme_scores(actual, ds)
        saved = pd.read_parquet(path, columns=["id", "eom", *themes.columns], filters=[("eom", "in", dates)]).set_index(["id", "eom"]).reindex(expected.index)
        check(f"{market}:independent_theme_recalculation", np.allclose(themes, saved[themes.columns], rtol=0, atol=1e-12))
        actual_liquidity=pd.read_parquet(path,columns=['id','eom','dolvol'],filters=[('eom','in',dates)]).set_index(['id','eom']).reindex(expected.index).dolvol
        raw_liquidity=pd.read_parquet(source,columns=['id','eom','dolvol_126d'],filters=[('eom','in',dates)]).set_index(['id','eom']).reindex(expected.index).dolvol_126d
        check(f"{market}:cost_liquidity_stays_raw_even_when_selected_as_signal",np.allclose(actual_liquidity,raw_liquidity,equal_nan=True))

    jobs = plan(UNIVERSES, "full", "all")
    available = set()
    for job in jobs:
        check(f"plan:{job.name}:dependencies_precede_job", set(job.dependencies).issubset(available))
        available.add(job.name)
        if job.stage in {"03", "04", "05", "06", "07", "10", "11", "12", "13", "14", "monitor","experiment"}:
            check(f"plan:{job.name}:script_available", script_path(job.stage, normalise_group(job.variant)).is_file())
    check("iterations_only_us_high_cap", all(job.universe == "us_large" for job in jobs if job.variant != "base"))
    check('five_sequential_experiments',len([j for j in jobs if j.stage=='experiment'])==5)
    check("static_star_only_us_high_cap_base",[j.universe for j in jobs if j.stage=='05']==['us_large'])
    cfg=load_config(config_path('us_large'))
    check('expanded_static_star_grid_has_100_candidates',len(cfg.static_ml.k_grid)*len(cfg.static_ml.u_grid)*len(cfg.static_ml.g_grid)==100)
    check("frontier_toggle_independent", any(j.stage == "figure1" for j in jobs) and
          not any(j.stage == "figure1" for j in plan(UNIVERSES, "full", "all", skip_figure1=True)))

    # Check the actual lag-key contract before covariance/ML fits are started.
    # The earliest covariance window is unavailable without fitting; checking
    # from 2014 onward covers all headline and validation portfolio decisions.
    for market, group in [(u, "base") for u in UNIVERSES] + [("us_large", "iteration_2")]:
        cfg = load_config(config_path(market, group))
        path = cfg.paths.processed / "panel.parquet"
        if not path.exists():
            continue
        keys = pd.read_parquet(path, columns=["id", "eom", "in_universe"])
        held = keys.loc[keys.in_universe & keys.eom.ge("2013-12-31"), ["id", "eom"]]
        needed = pd.concat([held.assign(eom=held.eom-pd.offsets.MonthEnd(i)) for i in range(cfg.universe.lookback_months+2)]).drop_duplicates()
        available = pd.MultiIndex.from_frame(keys[["id", "eom"]])
        region = "EU" if market == "eu_large" else "US"
        lag = pd.read_parquet(ROOT / f"data/inputs/lag_history_{region}.parquet", columns=["id", "eom"])
        available = available.union(pd.MultiIndex.from_frame(lag))
        absent = pd.MultiIndex.from_frame(needed).difference(available)
        check(f"{market}/{group}:actual_12_month_signal_history_keys", len(absent) == 0, f"{len(needed)} required distinct keys; {len(absent)} unavailable after exclusions")

    for path in list((ROOT / "scripts").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    check("all_python_sources_parse", True)
    notebook = ROOT / "Kati_tryout.ipynb"
    if notebook.exists():
        book = json.loads(notebook.read_text(encoding="utf-8"))
        cells = book["cells"]
        check("every_code_cell_has_prior_markdown", all(i > 0 and cells[i-1]["cell_type"] == "markdown" for i, cell in enumerate(cells) if cell["cell_type"] == "code"))
        embedded = book["metadata"].get("kati_source_manifest", {})
        # The builder stores every source path/hash as provenance metadata.
        if isinstance(embedded, dict) and embedded:
            check("notebook_embedded_source_matches_files", all(hashlib.sha256(((ROOT / p).read_text(encoding="utf-8").rstrip()+"\n").encode("utf-8")).hexdigest() == h for p, h in embedded.items()))
            actual_paths = {p.relative_to(ROOT).as_posix() for p in (ROOT / "scripts").rglob("*.py")}
            actual_paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").rglob("*.py"))
            check("notebook_contains_all_python_source", actual_paths.issubset(embedded))
            for cell in cells:
                source = cell["source"]
                if cell["cell_type"] == "code" and source.startswith("%%writefile "):
                    header, body = source.split("\n", 1)
                    relative = header.removeprefix("%%writefile ").strip()
                    check(f"notebook_cell:{relative}", (ROOT / relative).read_text(encoding="utf-8").rstrip() == body.rstrip())
    out = ROOT / "outputs/verification"
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status="FAIL" if any(c["status"] == "FAIL" for c in checks) else "PASS", checks=len(checks),
                  passed=sum(c["status"] == "PASS" for c in checks), models_refitted=False, model_backtest_complete=False,
                  data_stock_months=sum(x["stock_months"] for x in manifest["regional_audits"]),
                  details=checks)
    (out / "build_verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    pd.DataFrame(checks).to_csv(out / "build_checks.csv", index=False)
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
    return 0 if report["status"] == "PASS" else 2


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
