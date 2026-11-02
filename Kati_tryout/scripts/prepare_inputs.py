"""Create an independent local input snapshot for the frozen research 50.

This is a one-time, offline adapter, not a characteristic selector or ML fit.
The first call reads the two sibling projects; subsequent calls verify and reuse
the local snapshot. Pipeline calculations need only Kati_tryout/data afterward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
MARKETS = {"US": ("us_large", "us_mid"), "EU": ("eu_large",)}
KEYS = ["id", "eom"]
BASE_COLUMNS = ["id", "eom", "sic", "size_grp", "me", "ret_exc_lead1m", "hist_months", "dolvol_126d"]
MEMBER_COLUMNS = ["id", "eom", "form_year", "mcap_rank", "status"]
REFERENCES = ("ff_factors_monthly.parquet", "ff_factors_daily.parquet", "factor_details.parquet", "cluster_labels.parquet")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(frame):
    frame = frame.copy()
    frame["id"] = pd.to_numeric(frame.id, errors="raise").astype("int64")
    frame["eom"] = pd.to_datetime(frame.eom)
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Missing or duplicate stock-month keys")
    if not frame.eom.dt.is_month_end.all():
        raise ValueError("Expected month-end formation dates")
    return frame


def same_keys(left, right):
    a = pd.MultiIndex.from_frame(left[KEYS])
    b = pd.MultiIndex.from_frame(right[KEYS])
    return len(a) == len(b) and a.is_unique and b.is_unique and a.difference(b).empty and b.difference(a).empty


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def compare_source_vintages(original, pipeline=ROOT):
    """Compare shared raw features, without replacing either source vintage.

    This is an offline bootstrap audit. It is not needed by subsequent model
    runs, which use the independent local snapshot only.
    """
    original = Path(original)
    pipeline = Path(pipeline)
    data = pipeline / "data"
    names = json.loads((data / "characteristics/recommended_50.json").read_text())
    records = []
    for region in MARKETS:
        source_path = original / f"screened_{region}.parquet"
        shared = [name for name in names if name in pq.read_schema(source_path).names]
        new = normalize(pd.read_parquet(data / f"characteristics/raw50_{region}.parquet", columns=[*KEYS, *shared]))
        old = normalize(pd.read_parquet(source_path, columns=[*KEYS, *shared]))
        old = old.loc[old.eom.between(new.eom.min(), new.eom.max())].copy()
        if not same_keys(old, new):
            raise AssertionError("Source-vintage comparison requires exact matching stock-month keys")
        joined = old.merge(new, on=KEYS, validate="one_to_one", suffixes=("_old", "_new"))
        for name in shared:
            a = joined[name + "_old"].to_numpy(dtype=float, na_value=np.nan)
            b = joined[name + "_new"].to_numpy(dtype=float, na_value=np.nan)
            observed_a, observed_b = np.isfinite(a), np.isfinite(b)
            joint = observed_a & observed_b
            difference = np.abs(a[joint] - b[joint])
            records.append({"region": region, "characteristic": name, "stock_months": len(joined),
                "joint_nonmissing": int(joint.sum()), "both_missing": int((~observed_a & ~observed_b).sum()),
                "old_missing_new_observed": int((~observed_a & observed_b).sum()),
                "old_observed_new_missing": int((observed_a & ~observed_b).sum()),
                "different_at_absolute_tolerance_1e_10": int((difference > 1e-10).sum()),
                "maximum_absolute_difference": float(difference.max()) if len(difference) else None,
                "mean_absolute_difference": float(difference.mean()) if len(difference) else None,
                "comparison": "original Keats raw screened feature vs frozen Characteristics_tryout raw value; absolute tolerance1e-10, no rescaling"})
    report = pd.DataFrame(records)
    path = data / "characteristics/source_vintage_comparison.csv"
    report.to_csv(path, index=False)
    summary = {"status": "PASS", "comparison_completed": True, "feature_region_pairs": len(report),
        "overlap_features": report.characteristic.unique().tolist(),
        "pairs_with_observed_value_revisions": int(report.different_at_absolute_tolerance_1e_10.gt(0).sum()),
        "pairs_with_missingness_revisions": int((report.old_missing_new_observed + report.old_observed_new_missing).gt(0).sum()),
        "observed_values_different": int(report.different_at_absolute_tolerance_1e_10.sum()),
        "files_or_models_modified": "Only this comparison CSV and audit JSON; neither source values nor feature selection/model caches changed",
        "interpretation": "Any revisions mean the experiment combines the frozen new feature set with its supplied source vintage, rather than isolating feature names alone."}
    _write_json(pipeline / "outputs/validation/source_vintage_comparison.json", summary)
    return path, summary


def bootstrap(repo_root=None, pipeline_root=None):
    """Materialize local inputs, or verify every previously snapshotted file.

    Existing snapshots are immutable: a changed/damaged file raises instead of
    silently replacing inputs underneath saved results. No network is used.
    """
    pipeline = Path(pipeline_root).resolve() if pipeline_root else ROOT
    repo = Path(repo_root).resolve() if repo_root else pipeline.parent
    data = pipeline / "data"
    manifest_path = data / "SNAPSHOT_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["files"]:
            target = (pipeline / item["path"]).resolve()
            if not target.is_relative_to(data.resolve()):
                raise ValueError(f"Snapshot manifest path escapes data: {target}")
            if not target.is_file() or digest(target) != item["sha256"]:
                raise ValueError(f"Snapshot changed or incomplete: {target}. Existing results were not modified.")
        if json.loads((data / "characteristics/recommended_50.json").read_text()) != manifest["features"]:
            raise ValueError("Frozen feature list differs from snapshot")
        print(f"Verified independent local snapshot: {len(manifest['files'])} files; 50 characteristics", flush=True)
        return manifest

    if (pipeline/'settings.json').exists() and json.loads((pipeline/'settings.json').read_text()).get('selection_scope')=='us_large':
        raise FileNotFoundError('Restore the current US high-cap input snapshot; do not bootstrap the archived three-market feature selection. See scripts/update_us_selection.py.')

    keats = repo / "keats_pipeline"
    tryout = repo / "Characteristics_tryout/data"
    settings = json.loads((keats / "settings.json").read_text(encoding="utf-8"))
    original = (keats / settings["source"]).resolve()
    reference = (keats / settings["reference_data"]).resolve()
    selected_path = tryout / "recommended_50.json"
    names = json.loads(selected_path.read_text(encoding="utf-8"))
    if len(names) != 50 or len(set(names)) != 50 or set(names) & {"source_ret_exc_lead1m", "ret_exc", "ret_exc_lead1m", "id", "eom"}:
        raise ValueError("Expected exactly 50 unique predictor characteristics")
    selected_metadata_path = repo / "Characteristics_tryout/outputs/selection_explanation/source_tables/recommended_50.csv"
    if not selected_metadata_path.exists():
        selected_metadata_path = repo / "Characteristics_tryout/outputs/recommended_50.csv"
    selected = pd.read_csv(selected_metadata_path)
    if selected.characteristic.tolist() != names or not selected.training_direction.isin([-1, 1]).all():
        raise ValueError("Frozen selection evidence does not match the 50 names/directions")

    files = []
    sources = []
    audits = []

    def source_record(path):
        item = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)}
        sources.append(item)
        return item

    def register(path, role, source=None):
        item = {"path": str(path.relative_to(pipeline)).replace("\\", "/"), "bytes": path.stat().st_size,
                "sha256": digest(path), "role": role}
        if source is not None:
            item["copied_from"] = str(source.resolve())
        files.append(item)

    def copy(path, destination, role):
        source_record(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and digest(destination) != digest(path):
            raise ValueError(f"Refusing to overwrite a different input: {destination}")
        if not destination.exists():
            shutil.copy2(path, destination)
        if digest(path) != digest(destination):
            raise AssertionError(f"Copy hash mismatch: {path}")
        register(destination, role, path)

    def parquet(frame, destination, role):
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".building")
        frame.to_parquet(temporary, index=False)
        if destination.exists():
            if digest(temporary) != digest(destination):
                raise ValueError(f"Refusing to overwrite a different generated input: {destination}")
            temporary.unlink()
        else:
            temporary.replace(destination)
        check = pd.read_parquet(destination, columns=KEYS)
        if not same_keys(frame, check):
            raise AssertionError(f"Written keys changed: {destination}")
        register(destination, role)

    copy(selected_path, data / "characteristics/recommended_50.json", "Frozen, ordered predictor list")
    copy(selected_metadata_path, data / "characteristics/selected_feature_metadata.csv", "Frozen group, sign and selection evidence; proxy labels explicitly retained")
    directions = selected[["characteristic", "theme", "theme_assignment_method", "published_jkp_153", "direction", "training_direction"]].copy()
    directions["model_theme"] = directions.theme.str.strip().str.lower().str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False)
    directions["model_theme_direction"] = np.where(directions.published_jkp_153, directions.direction, directions.training_direction)
    directions["model_direction_source"] = np.where(directions.published_jkp_153, "published JKP direction", "2005-2014 learned direction for provisional proxy group")
    if not directions.model_theme_direction.isin([-1, 1]).all():
        raise ValueError("Every theme exposure requires an explicit valid direction")
    directions_path = data / "characteristics/theme_directions.csv"
    directions.to_csv(directions_path, index=False)
    register(directions_path, "Theme covariance direction policy: published28 signs, training22 proxy signs; raw ML ranks remain unflipped")
    for filename in REFERENCES:
        copy(reference / filename, data / "reference" / filename, "Unchanged reference input")
    # Infrastructure copies deliberately exclude old normalized/raw feature lag
    # files and the old aggregate manifest; new lag files are built below.
    for path in sorted((keats / "data/inputs").iterdir()):
        if path.is_file() and path.suffix in {".parquet", ".json", ".csv"} and not (
                path.name.startswith(("lag_history", "unavailable_lag_history")) or path.name == "manifest.json"):
            copy(path, data / "inputs" / path.name, "Unchanged returns, daily covariance, regime or infrastructure audit input")

    for region, universes in MARKETS.items():
        print(f"Snapshot {region}: joining frozen raw50 to original infrastructure metadata", flush=True)
        metadata_path = tryout / f"metadata_{region}.parquet"
        source_record(metadata_path)
        meta = normalize(pd.read_parquet(metadata_path))
        parts = []
        for path in sorted((tryout / "panels").glob(f"{region}_*.parquet")):
            source_record(path)
            frame = normalize(pd.read_parquet(path, columns=[*KEYS, *names, "source_row_found"]))
            frame[names] = frame[names].apply(pd.to_numeric, errors="raise").astype(float)
            if np.isinf(frame[names].to_numpy()).any():
                raise ValueError(f"Infinite source characteristic: {path}")
            parts.append(frame)
        if not parts:
            raise ValueError(f"No characteristic partitions for {region}")
        raw = normalize(pd.concat(parts, ignore_index=True)).sort_values(["eom", "id"], ignore_index=True)
        if not same_keys(meta, raw):
            raise AssertionError(f"Downloaded {region} keys differ from the frozen broad training/held-stock keys")
        # Source flags refer to missing entire observations; feature-level NaNs
        # are retained independently and become 0.5 only after monthly ranking.
        parquet(raw, data / f"characteristics/raw50_{region}.parquet", "Raw selected characteristics; original NaNs and source-row flags preserved")

        source_path = original / f"screened_{region}.parquet"
        source_record(source_path)
        broad = normalize(pd.read_parquet(source_path, columns=BASE_COLUMNS))
        key_index = pd.MultiIndex.from_frame(raw[KEYS])
        in_scope = pd.MultiIndex.from_frame(broad[KEYS]).isin(key_index)
        excluded_beyond_horizon = broad.loc[~in_scope, KEYS].copy()
        broad = broad.loc[in_scope].copy()
        if not same_keys(broad, raw):
            raise AssertionError(f"Original metadata does not exactly cover frozen {region} input keys")
        if len(excluded_beyond_horizon) and not excluded_beyond_horizon.eom.gt(raw.eom.max()).all():
            raise AssertionError("A pre-horizon original source key was unexpectedly omitted")
        broad = broad.merge(raw, on=KEYS, how="left", validate="one_to_one").sort_values(["eom", "id"], ignore_index=True)
        parquet(broad, data / f"source/screened_{region}.parquet", "Original screened population and infrastructure joined to raw selected50; no source forward-return audit field admitted")
        ranks = raw[names].groupby(raw.eom).rank(method="average", pct=True).fillna(0.5)
        rank_frame = pd.concat([raw[KEYS], ranks.add_suffix("_rank")], axis=1)
        a = ranks.to_numpy()
        if not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
            raise AssertionError("Ranked characteristics outside [0,1]")
        coverage = raw[names].notna().mean().rename("observed_fraction").rename_axis("characteristic").reset_index()
        coverage["region"] = region
        coverage_path = data / f"characteristics/coverage_{region}.csv"
        coverage.to_csv(coverage_path, index=False)
        register(coverage_path, "Raw feature availability; includes warmup and test dates without inspecting return outcomes")

        membership_audit = {}
        for universe in universes:
            member_path = original / f"panel_{universe}.parquet"
            source_record(member_path)
            members_all = normalize(pd.read_parquet(member_path, columns=MEMBER_COLUMNS))
            members = members_all.loc[members_all.eom.between(raw.eom.min(), raw.eom.max())].copy()
            prepared = members.merge(rank_frame, on=KEYS, how="left", validate="one_to_one", indicator=True)
            if not prepared._merge.eq("both").all() or not same_keys(members, prepared):
                raise AssertionError(f"Missing characteristic data for original {universe} membership")
            prepared = prepared.drop(columns="_merge")
            parquet(prepared, data / f"source/panel_{universe}.parquet", "Original declared-period membership metadata with newly computed selected50 ranks")
            membership_audit[universe] = {"members_original_in_declared_period": len(members), "members_snapshotted": len(prepared),
                "exact_membership_keys_preserved": True, "original_member_rows_beyond_input_horizon": len(members_all) - len(members)}

        lag_path = tryout / f"lag_history/{region}.parquet"
        source_record(lag_path)
        lag = normalize(pd.read_parquet(lag_path, columns=[*KEYS, *names, "source_row_found"]))
        old_lag_path = keats / f"data/inputs/lag_history_{region}.parquet"
        source_record(old_lag_path)
        old_lag = normalize(pd.read_parquet(old_lag_path, columns=[*KEYS, "g", "growth_imputed_neutral"]))
        old_unavailable_path = keats / f"data/inputs/unavailable_lag_history_{region}.csv"
        source_record(old_unavailable_path)
        old_unavailable = normalize(pd.read_csv(old_unavailable_path, parse_dates=["eom"]))
        requested = pd.concat([old_lag[KEYS], old_unavailable[KEYS]], ignore_index=True).drop_duplicates(KEYS)
        if not same_keys(lag, requested):
            raise AssertionError(f"Supplemental {region} requested lag keys differ from Keats")
        actual = lag.loc[lag.source_row_found].copy()
        usable = actual.merge(old_lag, on=KEYS, how="inner", validate="one_to_one")
        # A usable feature source must have an actual saved infrastructure growth
        # observation. Absence remains a key exclusion, never a synthetic row.
        unavailable_keys = pd.MultiIndex.from_frame(requested[KEYS]).difference(pd.MultiIndex.from_frame(usable[KEYS]))
        unavailable = unavailable_keys.to_frame(index=False).sort_values(KEYS, ignore_index=True)
        normalized_lag = usable[[*KEYS, "g", "growth_imputed_neutral"]].copy()
        for name in names:
            normalized_lag[name] = 0.5
        grouped_reference = {date: group for date, group in raw.groupby("eom", sort=False)}
        for date, group in usable.groupby("eom", sort=False):
            if date not in grouped_reference:
                raise ValueError(f"No actual regional rank reference for supplemental month {date}")
            reference_group = grouped_reference[date]
            for name in names:
                baseline = np.sort(reference_group[name].dropna().to_numpy(dtype=float))
                values = pd.to_numeric(group[name], errors="raise").to_numpy(dtype=float, na_value=np.nan)
                valid = np.isfinite(values)
                normalized = np.full(len(group), 0.5)
                if len(baseline):
                    lo = np.searchsorted(baseline, values[valid], side="left")
                    hi = np.searchsorted(baseline, values[valid], side="right")
                    normalized[valid] = (lo + hi + 2.0) / (2.0 * (len(baseline) + 1))
                normalized_lag.loc[group.index, name] = normalized
        normalized_lag["ret_ld1"] = np.nan
        normalized_lag["lambda"] = np.nan
        normalized_lag["lag_history_only"] = True
        if not np.isfinite(normalized_lag[names].to_numpy()).all():
            raise AssertionError("Non-finite supplemental feature rank")
        parquet(normalized_lag, data / f"inputs/lag_history_{region}.parquet", "Selected50 normalized pre-entry history; saved growth infrastructure, no training targets")
        parquet(lag, data / f"characteristics/lag_raw50_{region}.parquet", "Raw selected50 supplemental history, including explicit unavailable source rows")
        unavailable_path = data / f"inputs/unavailable_lag_history_{region}.csv"
        unavailable.to_csv(unavailable_path, index=False)
        register(unavailable_path, "Actual missing supplemental keys; apply explicit membership-exclusion policy")
        lag_info_path = data / f"inputs/lag_history_{region}.json"
        _write_json(lag_info_path, {"source": "../characteristics/lag_raw50_" + region + ".parquet",
            "feature_list": "../characteristics/recommended_50.json", "requested_keys": len(requested), "usable_keys": len(usable),
            "unavailable_keys": len(unavailable), "rank_reference": "New raw selected50 regional month plus inserted actual observation; average ties",
            "missing_feature_or_reference_rank": 0.5, "growth": "Unchanged original normalized Keats lag g by exact key",
            "purpose": "Input history only: ret_ld1 and lambda deliberately missing, never additional training targets"})
        register(lag_info_path, "Supplemental history provenance")
        audits.append({"region": region, "stock_months": len(raw), "formation_start": str(raw.eom.min().date()),
            "formation_end": str(raw.eom.max().date()), "exact_broad_keys_preserved": True, "absent_source_rows": int((~raw.source_row_found).sum()),
            "minimum_full_history_feature_coverage": float(coverage.observed_fraction.min()), "memberships": membership_audit,
            "original_broad_rows_beyond_declared_return_horizon": len(excluded_beyond_horizon),
            "lag_requested": len(requested), "lag_usable": len(usable), "lag_unavailable": len(unavailable),
            "zero_filled_raw_features": False, "source_forward_return_audit_used_as_predictor": False})
        print(f"{region}: {len(raw):,} broad rows; {len(usable)} usable lag rows; {len(unavailable)} unavailable keys", flush=True)

    vintage_path, vintage_summary = compare_source_vintages(original, pipeline)
    register(vintage_path, "Shared-feature source-vintage audit; values are never substituted")
    manifest = {"status": "PASS", "pipeline": "Kati_tryout", "features": names, "feature_count": len(names),
        "network_used": False, "models_refitted": False, "runtime_requires_sibling_projects": False,
        "source_vintage_comparison": vintage_summary,
        "theme_direction_policy": {"published_direction_count": int(directions.published_jkp_153.sum()),
            "training_direction_proxy_count": int((~directions.published_jkp_153).sum()),
            "published_vs_training_sign_disagreements": directions.loc[directions.published_jkp_153 & directions.direction.ne(directions.training_direction), "characteristic"].tolist(),
            "applies_to": "theme covariance exposures only; the 50 raw ML ranks are unflipped"},
        "source_hashes": sources, "files": files, "regional_audits": audits,
        "assumptions": ["Original Keats memberships, returns, liquidity, daily covariance observations and regimes are preserved.",
            "Formation December2025 is excluded because its next return is January2026, outside the declared 1995–2025 return horizon.",
            "Raw feature NaNs remain NaN; monthly regional average percentile ranks impute missing observations to 0.5.",
            "The selected50 were chosen with pre2020 evidence; published and proxy labels/signs remain explicitly distinguished.",
            "Theme exposures use published JKP directions for28 published members and 2005–2014 learned directions for22 proxies; raw ML feature ranks are not sign-flipped.",
            "No original feature rank is reused; dolvol_126d remains an infrastructure cost input, not an extra ML predictor.",
            "Old outputs, fitted coefficients, covariance estimates and model caches are not copied."]}
    _write_json(manifest_path, manifest)
    _write_json(pipeline / "outputs/validation/data_snapshot_audit.json", {k: v for k, v in manifest.items() if k not in {"files", "source_hashes"}})
    print(f"Independent input snapshot complete: {manifest_path}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None, help="Existing repository to read during first snapshot only")
    args = parser.parse_args(argv)
    bootstrap(repo_root=args.repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
