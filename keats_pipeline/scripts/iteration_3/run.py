"""iteration_3: original active Portfolio-ML engine; use calibrate/run/capacity/report as applicable."""
from pathlib import Path
import runpy,sys
ROOT=Path(__file__).resolve().parents[2]
if "--config" not in sys.argv:sys.argv += ["--config",str(ROOT/"configs/us_large/iteration_3.yaml")]
runpy.run_path(str(ROOT/"scripts/shared/active_engine.py"),run_name="__main__")
