"""Discover and align existing outputs, preserving an explicit source inventory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def discover(root: Path, universes=None):
    """Only primary saved portfolios; grid, permutation and capacity runs stay separate."""
    from jkmp.config import load_config
    inventories, frames, missing = [], [], []
    for universe in universes or ['us_large', 'us_mid', 'eu_large']:
        for variant in ['base', 'universe_250', 'halflife']:
            cfg_path = root / 'keats_pipeline/configs' / universe / f'{variant}.yaml'
            cfg = load_config(cfg_path)
            output, processed = cfg.paths.outputs, cfg.paths.processed
            dataset = f'{universe}/{output.name}'
            files = []
            if variant != 'halflife':
                for sub in ['benchmarks/pf_test.parquet', 'static_ml/pf.parquet',
                            'portfolio_ml/rff/pf.parquet', 'portfolio_ml/raw/pf.parquet']:
                    source = output / sub
                    weights = source.with_name('weights_test.parquet' if sub.startswith('benchmarks') else 'weights.parquet')
                    if source.exists():
                        files.append((source, weights, None))
                    else:
                        missing.append(dict(dataset=dataset, artifact=sub, status='SKIPPED', reason='No saved output; no model was fitted.'))
            active_sources=sorted((output / 'active').glob('gte*/pf.parquet'))
            if not active_sources:
                missing.append(dict(dataset=dataset,artifact='active/gte*/pf.parquet',status='SKIPPED',
                    reason='No saved active/half-life strategy returns; no fitting or calibration performed.'))
            for source in active_sources:
                try:
                    gamma_te = float(source.parent.name.removeprefix('gte'))
                except ValueError:
                    continue
                label = f'Portfolio-ML (active; gamma_TE={gamma_te:g})'
                calibration = output / 'active/calibrated.csv'
                if calibration.exists():
                    cal = pd.read_csv(calibration)
                    match = cal.loc[np.isclose(cal.gamma_te, gamma_te)]
                    if len(match) == 1:
                        label = f'Portfolio-ML (active; TE {100 * match.iloc[0].te_target:g}%)'
                files.append((source, source.with_name('weights.parquet'), label))
            for source, weights, label in files:
                frame = pd.read_parquet(source)
                required = {'type', 'eom', 'eom_ret', 'r', 'tc', 'turnover', 'inv', 'shorting'}
                if not required.issubset(frame):
                    raise ValueError(f'Unexpected saved portfolio schema: {source}')
                digest = sha256(source)
                for source_type, g in frame.groupby('type', sort=False):
                    strategy = label or source_type
                    g = g.copy().assign(dataset=dataset, strategy=strategy, universe=universe, variant=variant)
                    g['eom'] = pd.to_datetime(g.eom)
                    g['eom_ret'] = pd.to_datetime(g.eom_ret)
                    g = g.loc[g.eom_ret.between(pd.Timestamp(cfg.dates.test_start), pd.Timestamp(cfg.dates.test_end))]
                    if not len(g):
                        continue
                    if g.duplicated(['eom_ret']).any():
                        raise ValueError(f'Duplicate return months: {source}, {source_type}')
                    if not np.isfinite(g[['r', 'tc', 'turnover', 'inv', 'shorting']]).all().all():
                        raise ValueError(f'Nonfinite saved performance: {source}, {source_type}')
                    if not (g.eom + pd.offsets.MonthEnd(1)).eq(g.eom_ret).all():
                        raise ValueError(f'Formation/return-month mismatch: {source}')
                    frames.append(g)
                    effective_cfg=cfg_path
                    overrides=[]
                    if label is not None:
                        active_gamma=float(source.parent.name.removeprefix('gte'))
                        if variant=='base':
                            effective_cfg=cfg_path.with_name('active_pfml.yaml')
                        overrides.append(f'investor.gamma_te={active_gamma:g}')
                    elif variant=='universe_250' and source_type=='Portfolio-ML':
                        # run_all.plan explicitly selects the absolute-return
                        # baseline despite the active u250 parent configuration.
                        overrides=['portfolio_ml.benchmark_relative=false','portfolio_ml.add_constant=true']
                    inventories.append(dict(dataset=dataset, universe=universe, variant=variant,
                        strategy=strategy, source_type=source_type, pf_path=str(source),
                        weights_path=str(weights) if weights.exists() else '', processed_path=str(processed),
                        config_path=str(effective_cfg), effective_overrides=json.dumps(overrides),
                        source_sha256=digest, wealth_scale=1.0,
                        n_saved=len(g), sample_start=str(g.eom_ret.min().date()), sample_end=str(g.eom_ret.max().date()),
                        evaluation_gamma=float(cfg.investor.gamma), wealth_anchor=float(cfg.investor.wealth),
                        wealth_anchor_date=str(cfg.investor.wealth_end_date), feature_count=len(cfg.features.names)))
    if not frames:
        raise ValueError('No saved primary Keats portfolios found.')
    return pd.DataFrame(inventories), pd.concat(frames, ignore_index=True), pd.DataFrame(missing)


def align(monthly: pd.DataFrame):
    """Common months per dataset, and then across datasets; never fill absent returns."""
    if monthly.empty:
        raise ValueError('No performance observations to align.')
    monthly=monthly.copy()
    monthly['eom_ret']=pd.to_datetime(monthly.eom_ret)
    if monthly[['dataset','strategy','eom_ret']].isna().any().any():
        raise ValueError('Missing dataset, strategy or return month.')
    if monthly.duplicated(['dataset','strategy','eom_ret']).any():
        raise ValueError('Duplicate dataset/strategy/return-month observations.')
    if not monthly.eom_ret.dt.is_month_end.all():
        raise ValueError('Saved monthly returns must be dated at month-end.')
    aligned, logs = [], []
    for dataset, group in monthly.groupby('dataset', sort=False):
        date_sets = [set(g.eom_ret) for _, g in group.groupby('strategy')]
        common = set.intersection(*date_sets)
        if len(common) < 12:
            raise ValueError(f'Fewer than 12 common observations: {dataset}')
        ordered=pd.DatetimeIndex(sorted(common)).to_period('M').asi8
        if not np.all(np.diff(ordered)==1):
            raise ValueError(f'Calendar gaps in common monthly sample: {dataset}; no missing return is fabricated.')
        for strategy, g in group.groupby('strategy'):
            drop = g.loc[~g.eom_ret.isin(common)]
            logs.append(dict(dataset=dataset, strategy=strategy, original_months=len(g),
                common_months=len(common), excluded_months=len(drop),
                excluded_dates=';'.join(drop.eom_ret.dt.strftime('%Y-%m-%d'))))
        aligned.append(group.loc[group.eom_ret.isin(common)])
    within = pd.concat(aligned, ignore_index=True).sort_values(['dataset','strategy','eom_ret'])
    common_all = set.intersection(*(set(g.eom_ret) for _, g in within.groupby('dataset')))
    across = within.loc[within.eom_ret.isin(common_all)].copy()
    return within, across, pd.DataFrame(logs)


def snapshot(root: Path):
    """Byte hashes of existing model code/config/output files; excludes run logs/caches."""
    paths = []
    comparison_root=(root/'keats_pipeline/outputs/ML comparison and reasoning').resolve()
    for sub, patterns in [('keats_pipeline', ['*.py']),
                          ('keats_pipeline/configs', ['*.yaml']), ('keats_pipeline/outputs', ['*.parquet','*.csv'])]:
        for pattern in patterns:
            paths.extend(p for p in (root / sub).rglob(pattern)
                if 'run_logs' not in p.parts and not p.resolve().is_relative_to(comparison_root))
    for name in ['README.md','settings.json','original_manifest.json']:
        path=root/'keats_pipeline'/name
        if path.is_file():paths.append(path)
    return pd.DataFrame([dict(path=str(p.relative_to(root)), bytes=p.stat().st_size,
        modified_ns=p.stat().st_mtime_ns, sha256=sha256(p)) for p in sorted(set(paths))])
