"""Market plus pretest-calibrated 1% TE tilt, inheriting only accepted changes."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from iteration_sequence import main
if __name__=='__main__':raise SystemExit(main(['--upto','4',*sys.argv[1:]]))
