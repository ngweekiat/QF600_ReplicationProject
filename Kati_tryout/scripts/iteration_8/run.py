"""Run the one-SE selection experiment using the saved accepted predecessor."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from iteration_sequence import main
if __name__=='__main__':raise SystemExit(main(['--upto','8','--start-from','8',*sys.argv[1:]]))
