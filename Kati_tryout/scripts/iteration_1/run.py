"""Annual refitting: retained as already part of Base. No extra fitted experiment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from iteration_sequence import main
if __name__=='__main__':raise SystemExit(main(['--upto','1',*sys.argv[1:]]))
