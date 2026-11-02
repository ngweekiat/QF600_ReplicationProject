"""Gamma 5 on the last accepted US high-cap setup; paired test-P comparison."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from iteration_sequence import main
if __name__=='__main__':raise SystemExit(main(['--upto','2',*sys.argv[1:]]))
