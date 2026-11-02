# Explainable Speech Emotion Recognition with Noise Robustness Analysis

CPU classical ML, speaker-disjoint evaluation, acoustic feature ablation, controlled white noise, forest SHAP, and an offline Streamlit demo.

**Current status:** source implementation and synthetic software tests are available. RAVDESS download is blocked by the managed network policy. Real training, measured emotion results, and a downloadable trained bundle remain unavailable. Do not treat test fixtures as trained models.

## Install (Python 3.12)
Linux:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```
Windows PowerShell:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
```
If PowerShell activation is restricted, invoke `.venv\Scripts\python.exe` directly. Run all commands from the repository root. The complete lock captures the tested Linux/Python 3.12 environment; Windows installation is documented but not yet tested. Native binary wheels are used; audio decoding uses SoundFile, not FFmpeg. `requirements.txt` lists the direct compatible versions.

## Reproduce real experiments
```bash
python scripts/download_data.py
python -m ser.train --data data --output artifacts
python scripts/render_results.py
python scripts/package_bundle.py
python -m pytest -q
```
Only `Audio_Speech_Actors_01-24.zip` is downloaded. Internet access to `zenodo.org`, PyPI and wheel hosts is needed for preparation. The data downloader stores live provenance and verifies Zenodo's MD5; training stores per-recording SHA-256 hashes. Read [the report](docs/REPORT.md) for license restrictions and the exact protocol. Do not redistribute the raw corpus in Git or the model bundle.

Feature extraction is the slowest CPU step; rerunning reuses completed feature checkpoints. Training searches restart deterministically. Do not compare incomplete/stale output directories; use a new `--output` when changing code or dependencies. Default output is `artifacts`; rendering and packaging currently use that default. Raw data, model files, caches, and archives are Git-ignored. Small evaluation tables/plots under `results` can be committed after checking they contain no audio.

## Offline demo
After installing dependencies and obtaining a trusted trained bundle, extract its contents into `artifacts/` so `artifacts/model.joblib` and `artifacts/model.joblib.sha256` exist. Then:
```bash
python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
```
All inference, feature extraction, SHAP, and noise generation run locally; no model/API download occurs. `SER_MODEL` can select another trusted checksum-protected bundle. Never load untrusted joblib/pickle files: a matching checksum checks integrity, not publisher authenticity. The app gracefully displays an unavailable-model message before training.

The prediction model is always the combined-feature Random Forest; its Tree SHAP explains that same model. The report records the separately validation-selected benchmark winner. Scores are uncalibrated model probabilities, not certainty about a person's feelings.

## Persistent model release procedure (not yet completed)
No trained model exists yet and no working download link is claimed. After successful real training:
1. Run packaging above. It refuses to package missing model/evaluation files and writes a ZIP plus SHA-256 and per-file checksums.
2. With authorized repository release access, publish `artifacts/ser-model-bundle.zip` and its `.sha256` as GitHub Release assets. Include dataset attribution/license metadata and the exact source commit in release notes.
3. Download both assets from the resulting public/authorized release URL into a fresh directory. Check the archive SHA-256, extract, check `bundle-checksums.json`, then run a real WAV inference and the app smoke test against the downloaded bundle.
4. Record the actual release URL and retrieval result in `results/STATUS.md`. Until then, bundle delivery remains blocked/unverified.

Linux checksum: `sha256sum -c ser-model-bundle.zip.sha256`.
Windows checksum: `Get-FileHash ser-model-bundle.zip -Algorithm SHA256`, comparing to the published checksum. A Linux wheelhouse is not portable to Windows; offline installation requires wheels prepared on the target OS (`python -m pip download -r requirements-lock.txt -d wheelhouse`), then `python -m pip install --no-index --find-links wheelhouse -r requirements-lock.txt`.

## Deliverables
- [Report and architecture diagram](docs/REPORT.md)
- [Eight-minute presentation and four-minute demo script](docs/PRESENTATION.md)
- [Viva questions with answers](docs/VIVA.md)
- [Measured execution status](results/STATUS.md)

No team identities or experimental results are fabricated. The contribution table is deliberately left for the team.
