from pathlib import Path
import os
import sys
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def pytest_configure(config):
    if config.option.basetemp is not None:
        return
    scratch=(ROOT/'outputs/test_tmp'/uuid4().hex).resolve()
    if not scratch.is_relative_to(ROOT) or scratch.exists():
        raise RuntimeError('Unexpected test scratch path')
    scratch.parent.mkdir(parents=True,exist_ok=True)
    config.option.basetemp=('\\\\?\\'+str(scratch)) if os.name=='nt' else str(scratch)
