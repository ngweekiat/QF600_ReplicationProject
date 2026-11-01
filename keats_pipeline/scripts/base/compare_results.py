"""Refresh base-only tables and charts from saved Static ML / Portfolio ML outputs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"outputs/base"))
from analysis.ml_comparison.cli import main
if __name__=="__main__":raise SystemExit(main())
