"""Market plus pretest-calibrated 1% TE tilt directly from Base; no gamma-5 parent."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from iteration_sequence import main
if __name__=='__main__':
    from base_anchor_revision_reports import pretest
    pretest()
    raise SystemExit(main(['--upto','4','--only','4',*sys.argv[1:]]))
