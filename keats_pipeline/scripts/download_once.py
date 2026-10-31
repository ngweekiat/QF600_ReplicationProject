"""Offline audit of existing Keats inputs. No downloads."""
from _bootstrap import ROOT
from workflow import audit

if __name__ == "__main__":
    audit()
