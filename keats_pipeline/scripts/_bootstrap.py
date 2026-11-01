"""Resolve only the isolated package; never silently import the original jkmp."""
from pathlib import Path
import sys
for _stream in (sys.stdout,sys.stderr):
    if hasattr(_stream,"reconfigure"):_stream.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from jkmp.imputation_audit import enable_reporting, reporting_enabled
if not reporting_enabled():
    enable_reporting(Path(sys.argv[0]).name)


def configure(script_file):
    """Safe defaults for running a copied script directly in VS Code."""
    if "--config" not in sys.argv and Path(script_file).name not in {
        "00_download.py", "01_prepare.py", "02_covariance.py", "download_once.py",
        "15_holdout.py", "compare_breadth.py", "compare_halflife.py",
    }:
        folder=Path(script_file).parent.name
        group=folder if folder in ("base","iteration_1","iteration_2","iteration_3") else "base"
        if Path(script_file).name in ("04_benchmarks.py","active_engine.py"):group="iteration_1"
        cfg = ROOT / "configs" / "us_large" / f"{group}.yaml"
        sys.argv += ["--config", str(cfg)]
