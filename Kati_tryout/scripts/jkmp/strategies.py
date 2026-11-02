"""The public strategy set for the base results; iterations have their own mandates."""
STRATEGIES = ("Static-ML*", "Portfolio-ML")


def base_strategies(universe):
    return STRATEGIES if universe == "us_large" else ("Portfolio-ML",)


def base_result_paths(outputs):
    """Active base files only; never discover historical fixed Static-ML files."""
    return ([outputs/"static_ml_star/pf.parquet"] if outputs.name=="us_large" else []) + [outputs/"portfolio_ml/rff/pf.parquet"]


def base_sources_marker(outputs,folder):
    """Refresh derived tables/charts whenever their saved model returns change."""
    import hashlib
    signature=hashlib.sha256()
    for path in base_result_paths(outputs):
        signature.update(str(path.relative_to(outputs)).encode())
        if not path.exists():return outputs/folder/"sources_pending.json"
        with path.open("rb") as stream:
            signature.update(hashlib.file_digest(stream,"sha256").digest())
    return outputs/folder/f"sources_{signature.hexdigest()[:16]}.json"


def portfolio_tuning_marker(cfg):
    """Recompute base portfolio coefficients when penalties, inputs or settings change."""
    import hashlib
    import json
    from pathlib import Path
    plain=cfg.as_dict()
    payload={key:plain[key] for key in ("seed","features","dates","investor","portfolio","portfolio_ml","ridge_grid")}
    payload["prediction_years"]=cfg.return_model.pred_years
    payload["market"]=cfg.paths.outputs.name
    inputs=[cfg.paths.processed/name for name in ("panel.parquet","wealth.parquet","covariance/cov_factor.parquet","covariance/cov_loadings.parquet")]
    for g in cfg.portfolio_ml.rff_variance_grid:
        folder=cfg.paths.processed/"pfml"/f"rff_g{g:.4f}"
        inputs += sorted(folder.glob("*.npz")) + [folder/"meta.json",folder/"rff_w.npy"]
    payload["inputs"]={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) if p.exists() else None for p in inputs}
    scripts=Path(__file__).resolve().parents[1]
    sources=[scripts/"base/07_portfolio_ml.py",scripts/"jkmp/portfolio/portfolio_ml.py",scripts/"jkmp/models/ridge.py"]
    payload["sources"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    token=hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()[:16]
    return cfg.paths.outputs/"portfolio_ml/rff"/f"tuning_{token}.json"


def require_strategies(frame, complete=False, universe="us_large"):
    allowed=base_strategies(universe)
    names=set(frame["type"])
    if names-set(allowed) or (complete and names!=set(allowed)):
        raise ValueError(f"Expected only {allowed}; found {sorted(names)}")
    if frame.duplicated(["type","eom_ret"]).any():
        raise ValueError("Duplicate strategy/return-month rows")
    return frame
