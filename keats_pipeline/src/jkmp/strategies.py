"""The public strategy set for the base results; iterations have their own mandates."""
STRATEGIES = ("Static-ML", "Portfolio-ML")


def require_strategies(frame, complete=False):
    names=set(frame["type"])
    if names-set(STRATEGIES) or (complete and names!=set(STRATEGIES)):
        raise ValueError(f"Expected only {STRATEGIES}; found {sorted(names)}")
    if frame.duplicated(["type","eom_ret"]).any():
        raise ValueError("Duplicate strategy/return-month rows")
    return frame
