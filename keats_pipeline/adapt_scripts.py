"""One-time adaptations to the private script snapshot; no original is modified."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def adapt():
    scripts = ROOT / "scripts"
    for name in ("00_download.py", "download_once.py"):
        (scripts / name).write_text('''"""Offline audit of existing Keats inputs. No downloads."""
from _bootstrap import ROOT
from workflow import audit

if __name__ == "__main__":
    audit()
''')
    for name, action in (("01_prepare.py", "prepare"), ("02_covariance.py", "covariance")):
        (scripts / name).write_text(f'''"""Keats adapter. Use run.py to select universe and variant."""
from _bootstrap import ROOT
from workflow import {action}

if __name__ == "__main__":
    {action}("us_large")
''')
    (scripts / "15_holdout.py").write_text('''"""No separate unseen holdout follows the saved Keats test period."""
from _bootstrap import ROOT

def main():
    print("No separate Keats holdout: 2021-2025 is already in the declared 2020-2025 test period.")

if __name__ == "__main__":
    main()
''')
    for name in ("13_breadth.py", "14_halflife.py"):
        path = scripts / name
        text = path.read_text(encoding="utf-8")
        for old, new in (("1981–2020", "2020–2025"), ("1990–2020", "2020–2025 (VIX subset)"),
                         ("1971–2020", "2015–2025"), ("end-2020", "end-2019"),
                         ("top-80", "80-stock"), ("top-250", "250-stock")):
            text = text.replace(old, new)
        text = text.replace('"configs/experiments/active_pfml.yaml"', '"configs/us_large/active_pfml.yaml"')
        text = text.replace('"configs/experiments/universe_250.yaml"', '"configs/us_large/universe_250.yaml"')
        text = text.replace('"configs/experiments/halflife.yaml"', '"configs/us_large/halflife.yaml"')
        if name == "14_halflife.py":
            text = text.replace('dates, g = var.dates, var.investor.gamma_te', '''dates = var.dates
    cal = pd.read_csv(ref.paths.outputs / "active/calibrated.csv")
    g = float(cal.loc[(cal.te_target - 0.02).abs().lt(1e-12), "gamma_te"].iloc[0])''')
        path.write_text(text, encoding="utf-8")
    path = scripts / "11_tables.py"
    text = path.read_text(encoding="utf-8").replace('PAPER = PROJECT_ROOT / "docs"', 'PAPER = PROJECT_ROOT.parent / "docs"')
    text = text.replace('    paper = pd.read_csv', '    print("Paper results use a different sample/universe: descriptive reference only, not a replication match.")\n    paper = pd.read_csv', 1)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    adapt()
