"""Remove the obsolete feature archives explicitly requested by the user.

Keeps only hashes as evidence that the current ML implementation was preserved.
Candidate data used to reproduce the current selection are retained.
"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    from jkmp.config import _read_yaml
    evidence = ROOT / 'data/characteristics/implementation_baseline.json'
    if not evidence.exists():
        files = list((ROOT/'scripts/jkmp/models').glob('*.py')) + list((ROOT/'scripts/jkmp/portfolio').glob('*.py'))
        configs = {}
        for path in (ROOT/'configs').rglob('*.yaml'):
            value = _read_yaml(path, seen=())
            value.pop('features', None)
            configs[path.relative_to(ROOT).as_posix()] = value
        record = dict(source_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                      non_feature_configs=configs, old_feature_archives_removed=True)
        evidence.write_text(json.dumps(record, indent=2, default=str)+'\n', encoding='utf-8')
    targets = [ROOT/'data/archives/20261102_121716_previous_50',
               ROOT/'outputs/archive/20261102_121716_previous_50',
               ROOT.parent/'Characteristics_tryout/data/selection_archive/20261102_122021_previous_cross_market']
    removed = []
    for target in targets:
        resolved = target.resolve()
        if not resolved.is_relative_to(ROOT.parent.resolve()) or resolved == ROOT.parent.resolve():
            raise ValueError(f'Unsafe archive target: {resolved}')
        if resolved.exists():
            shutil.rmtree('\\\\?\\' + str(resolved))
            removed.append(str(resolved.relative_to(ROOT.parent)))
    for path, key in [(ROOT/'settings.json','old_feature_archive'), (ROOT/'data/SNAPSHOT_MANIFEST.json','previous_feature_archive')]:
        value = json.loads(path.read_text())
        value.pop(key, None)
        value['old_feature_archives_removed'] = True
        path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    report = dict(status='PASS', old_feature_archives_removed=True, removed=removed,
                  active_selection='US high-cap 2005–2014 association 50', models_fitted=False,
                  recovery='Removed from local disk; no repository backup of these archives was created.')
    (ROOT/'outputs/validation/archive_removal.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
