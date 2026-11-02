"""No separate unseen holdout follows the saved Keats test period."""
from pathlib import Path as _ScriptPath
import sys as _script_sys
_script_sys.path.insert(0,str(_ScriptPath(__file__).resolve().parents[1]))
from _bootstrap import ROOT

def main():
    print("No separate Keats holdout: 2021-2025 is already in the declared 2020-2025 test period.")

if __name__ == "__main__":
    main()
