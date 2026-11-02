"""Package only real trained outputs; never fabricate a placeholder model."""
import hashlib
import json
from pathlib import Path
import zipfile

root=Path('artifacts')
required=['model.joblib','model.joblib.sha256','splits.csv','evaluation.json','validation.json','selection.json','class_counts.json']
for name in required:
    if not (root/name).is_file(): raise SystemExit(f'Missing {name}: finish training first')
files=[root/name for name in required]+[Path('requirements-lock.txt')]
checks={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(root/'bundle-checksums.json').write_text(json.dumps(checks,indent=2))
with zipfile.ZipFile(root/'ser-model-bundle.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in files+[root/'bundle-checksums.json']: z.write(p,p.name)
p=root/'ser-model-bundle.zip'
(root/'ser-model-bundle.zip.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
print(p)
