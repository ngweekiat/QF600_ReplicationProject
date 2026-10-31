"""Bootstrap the relocated evaluation package for standalone pytest runs."""
from pathlib import Path
import sys

COMPARISON_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMPARISON_ROOT))
from analysis.ml_comparison.paths import REPOSITORY_ROOT
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / 'keats_pipeline/src'))
