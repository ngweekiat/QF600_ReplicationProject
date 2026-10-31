"""Locate the repository independently of the comparison folder's nesting."""
from pathlib import Path

COMPARISON_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / 'keats_pipeline/settings.json').is_file()
)
