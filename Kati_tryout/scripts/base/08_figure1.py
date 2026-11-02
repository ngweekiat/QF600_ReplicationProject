"""Retained-model Figure 1; see Kati_tryout/scripts/figure1.py for the implementation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from figure1 import main
if __name__=="__main__":raise SystemExit(main())
