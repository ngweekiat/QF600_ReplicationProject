"""Run executed active-utility selection after the saved iteration 4."""
from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from active_validation import main
if __name__=='__main__':
    status=main(['--stage','4.2',*sys.argv[1:]])
    root=Path(__file__).resolve().parents[2]
    manifests=[root/f'outputs/iteration_{s}/us_large/fit_manifest.json' for s in ('4.2','4.3')]
    if status==0 and all(p.is_file() for p in manifests) and len({json.loads(p.read_text())['signature'] for p in manifests})==1:
        from active_validation_reports import build_reports
        status=build_reports()
    raise SystemExit(status)
