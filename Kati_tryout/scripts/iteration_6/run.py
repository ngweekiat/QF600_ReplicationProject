"""More selective trading on the saved accepted setup; retain actual cost accounting."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from iteration_sequence import main
if __name__=='__main__':raise SystemExit(main(['--upto','6','--start-from','6',*sys.argv[1:]]))
