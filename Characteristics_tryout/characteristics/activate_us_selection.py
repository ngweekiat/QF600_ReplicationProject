"""Promote the US high-cap association recommendation; preserve prior metadata."""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import shutil

from .paths import DATA, OUTPUT


def activate():
    source=OUTPUT/'us_high_cap_selection'
    names=json.loads((source/'recommended_50.json').read_text())
    previous=DATA/'recommended_50.json'
    if previous.exists() and json.loads(previous.read_text())!=names:
        archive=DATA/'selection_archive'/dt.datetime.now().strftime('%Y%m%d_%H%M%S_previous_cross_market')
        archive.mkdir(parents=True)
        for name in ('recommended_50.json','design.json'):
            if (DATA/name).exists():shutil.copy2(DATA/name,archive/name)
    shutil.copy2(source/'recommended_50.json',previous)
    target=DATA/'us_high_cap_selection'
    target.mkdir(exist_ok=True)
    for name in ('recommended_50.json','recommended_50.csv','candidate_200.csv','selection_manifest.json'):
        shutil.copy2(source/name,target/name)
    (DATA/'current_selection.json').write_text(json.dumps(dict(scope='us_large',period='2005–2014',
        method='correlation/covariance association analysis; A/T/S diversity rules',
        selected_50='us_high_cap_selection/recommended_50.json',candidate_200='us_high_cap_selection/candidate_200.csv',
        report=str(source),models_fitted=False,historical_download_200_file='download_200.json'),indent=2)+'\n',encoding='utf-8')
    design=json.loads((DATA/'design.json').read_text())
    design.update(scope='US high-cap held-stock observations only for characteristic selection',
        final_selection='2005–2014 US high-cap association ranking; 13 A, 13 T and 24 S; no ML/regression fits',
        current_selection='us_high_cap_selection/recommended_50.json',
        selection_return_start='2005-01-31',selection_return_end='2014-12-31',
        validation_return_start=None,validation_return_end=None,held_out_from_selection_start='2015-01-31',
        preselection_score='0.55 absolute US high-cap mean monthly rank IC percentile + 0.20 US coverage percentile + 0.15 yearly association-sign stability percentile + 0.10 rank persistence percentile',
        preselection_pilot='Only US high-cap held-stock observations in the 2005–2014 measurement window; no ML fit',
        validation_outcomes_used_for_current_selection=False,models_fitted_for_selection=False,
        historical_download_note='download_200.json describes the earlier full-period data acquisition cache, not the current US-only 200-candidate cap')
    (DATA/'design.json').write_text(json.dumps(design,indent=2)+'\n',encoding='utf-8')
    print('Current Characteristics_tryout recommendation is the US high-cap-only association 50.')


if __name__=='__main__':activate()
