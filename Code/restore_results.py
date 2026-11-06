from pathlib import Path
import gzip
import hashlib
import json
import shutil
import zipfile

folder = Path(__file__).resolve().parents[1] / 'Result'
archives = folder / 'Archives'
expected = json.loads((archives / 'manifest.json').read_text())
for archive in sorted(archives.glob('analysis-*.zip')):
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            destination = (folder / entry.filename).resolve()
            if not destination.is_relative_to(folder.resolve()):
                raise ValueError('Unsafe archive path')
        source.extractall(folder)
    print(archive.name, 'extracted')
for archive in sorted((folder / 'NoAuction_Zscore').glob('influence_h*.csv.gz')):
    target = archive.with_suffix('')
    with gzip.open(archive, 'rb') as source, target.open('wb') as destination:
        shutil.copyfileobj(source, destination)
for name, record in expected.items():
    target = folder / name
    if target.stat().st_size != record['bytes'] or hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Checksum mismatch: ' + name)
print(len(expected), 'original result files restored and verified')
