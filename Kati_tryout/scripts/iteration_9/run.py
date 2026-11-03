"""Run benchmark-aware recursive cost learning on the last accepted setup."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from iteration_sequence import main
if __name__=='__main__':raise SystemExit(main(['--upto','9','--start-from','9',*sys.argv[1:]]))
