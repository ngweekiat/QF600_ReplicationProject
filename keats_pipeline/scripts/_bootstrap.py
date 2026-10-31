"""Resolve only the isolated package; never silently import the original jkmp."""
from pathlib import Path
import sys

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
        "11_monitor.py", "13_breadth.py", "14_halflife.py", "15_holdout.py",
    }:
        cfg = ROOT / "configs" / "us_large" / (
            "active_pfml.yaml" if Path(script_file).name == "12_active_pfml.py" else "base.yaml")
        sys.argv += ["--config", str(cfg)]
