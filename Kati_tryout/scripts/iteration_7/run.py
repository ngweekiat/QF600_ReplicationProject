"""Test stock-specific risk responsiveness on the saved last accepted portfolio."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from iteration_sequence import main
if __name__=='__main__':raise SystemExit(main(['--upto','7','--start-from','7',*sys.argv[1:]]))
