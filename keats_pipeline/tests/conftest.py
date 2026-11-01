from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

sys.path.insert(0,str(ROOT/"outputs/base"))


def pytest_configure(config):
    """Keep test scratch files in this workspace, including on restricted Windows."""
    if config.option.basetemp is not None:
        return
    import os
    from uuid import uuid4
    scratch=(ROOT/'outputs/organisation_audit/test_tmp'/uuid4().hex).resolve()
    if not scratch.is_relative_to(ROOT) or scratch.exists():
        raise RuntimeError('Unexpected test scratch path')
    scratch.parent.mkdir(parents=True,exist_ok=True)
    config.option.basetemp=('\\\\?\\'+str(scratch)) if os.name=='nt' else str(scratch)
