"""Keep this experiment's writes outside the active Keats pipeline."""
from pathlib import Path
import os

TRYOUT_ROOT=Path(__file__).resolve().parents[1]
PROJECT=TRYOUT_ROOT.parent
KEATS=PROJECT/'keats_pipeline'
DATA=Path(os.environ.get('CHARACTERISTICS_DATA_DIR',TRYOUT_ROOT/'data')).expanduser().resolve()
OUTPUT=Path(os.environ.get('CHARACTERISTICS_OUTPUT_DIR',TRYOUT_ROOT/'outputs')).expanduser().resolve()


def relative_link(parent,target):
    """Portable report links, including when an existing cache is selected."""
    try:
        return os.path.relpath(target,parent).replace('\\','/')
    except ValueError:
        return Path(target).resolve().as_posix()
