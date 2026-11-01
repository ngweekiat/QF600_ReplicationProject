"""Keats adapter. Use run.py to select universe and variant."""
from pathlib import Path as _ScriptPath
import sys as _script_sys
_script_sys.path.insert(0,str(_ScriptPath(__file__).resolve().parents[1]))
from _bootstrap import ROOT
from workflow import covariance

if __name__ == "__main__":
    covariance("us_large")
