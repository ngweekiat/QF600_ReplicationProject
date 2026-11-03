"""Run one-SE executed active-utility selection, retaining stage 4.2."""
from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from active_validation import main
if __name__=='__main__':
    status=main(['--stage','4.3',*sys.argv[1:]])
    if status==0:
        from base_anchor_revision_reports import results
        results()
    raise SystemExit(status)
