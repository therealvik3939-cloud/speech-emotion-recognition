"""Download exactly the requested Zenodo file; verify Zenodo's published MD5."""
from pathlib import Path
import hashlib
import json
import zipfile
import requests

DEST = Path("data")
DEST.mkdir(exist_ok=True)
response = requests.get("https://zenodo.org/api/records/1188976", timeout=60)
response.raise_for_status()
record = response.json()
entry = next(f for f in record["files"] if f["key"] == "Audio_Speech_Actors_01-24.zip")
(DEST / "provenance.json").write_text(json.dumps({"record": record["id"], "metadata": record["metadata"], "file": entry}, indent=2))
archive = DEST / entry["key"]
def valid():
    return archive.exists() and "md5:" + hashlib.md5(archive.read_bytes()).hexdigest() == entry["checksum"]
if not valid():
    temporary = archive.with_suffix(".part")
    with requests.get(entry["links"]["self"], stream=True, timeout=120) as r:
        r.raise_for_status()
        with temporary.open("wb") as f:
            for chunk in r.iter_content(1024*1024):
                f.write(chunk)
    temporary.replace(archive)
if not valid():
    raise ValueError("Archive checksum mismatch")
with zipfile.ZipFile(archive) as z:
    for member in z.infolist():
        if not (DEST.resolve() / member.filename).resolve().is_relative_to(DEST.resolve()):
            raise ValueError("Unsafe archive path")
    z.extractall(DEST)
print("Verified and extracted", archive)
