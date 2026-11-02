# Explainable Speech Emotion Recognition with Noise Robustness Analysis

## Status and motivation
Speech interfaces may benefit from recognizing expressed emotion. Acoustic patterns depend on speaker identity and recording conditions; random recording splits can exaggerate performance. This project compares classical CPU models with actor isolation, feature ablation, and controlled noise. This implementation is complete enough to exercise locally, but **RAVDESS experiments have not run**: the managed proxy denied Zenodo (HTTP CONNECT 403). No accuracy or F1 is claimed. No trained bundle has been published.

## Dataset analysis and provenance
Source: Livingstone, S. R. and Russo, F. A. (2018), *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*, Zenodo record https://zenodo.org/records/1188976, DOI https://doi.org/10.5281/zenodo.1188976. Download only `Audio_Speech_Actors_01-24.zip`. The downloader captures the live record metadata and file checksum before extracting; the live metadata has not yet been verified in this environment.

RAVDESS is distributed under CC BY-NC-SA 4.0 (attribution, noncommercial, share alike); verify the live record terms before redistribution or deployment. Cite the source and retain the license and provenance with derivative data. Do not assume model distribution removes dataset-license obligations.

Expected speech subset: 24 professional actors, 12 female and 12 male, North American English, two lexical statements, normal and strong intensity except neutral. Filename fields are modality, channel, emotion, intensity, statement, repetition, actor. Expected counts (not observed locally yet):

| Emotion | Total | Train (16 actors) | Validation (4) | Test (4) |
|---|---:|---:|---:|---:|
| Neutral | 96 | 64 | 16 | 16 |
| Calm | 192 | 128 | 32 | 32 |
| Happy | 192 | 128 | 32 | 32 |
| Sad | 192 | 128 | 32 | 32 |
| Angry | 192 | 128 | 32 | 32 |
| Fearful | 192 | 128 | 32 | 32 |
| Disgust | 192 | 128 | 32 | 32 |
| Surprised | 192 | 128 | 32 | 32 |
| Total | 1440 | 960 | 240 | 240 |

The trainer validates unique recording IDs and content hashes. It writes observed class counts and a per-recording SHA-256 split manifest. Dataset files and feature caches are ignored by Git.

## Architecture
```mermaid
flowchart LR
  Z[Zenodo speech ZIP] --> V[Checksum and metadata]
  V --> W[WAV validation / mono / 16 kHz]
  W --> F[35 acoustic features]
  F --> S[Fixed actor partitions]
  S --> C[Training-only grouped CV / fold-local scaling]
  C --> M[Dummy / Logistic / SVM / Forest]
  M --> A[Validation model selection]
  A --> T[Final clean test and paired noisy test]
  T --> R[Metrics / manifests / bundle]
  R --> U[Offline Streamlit]
  U --> H[Forest prediction and forest SHAP]
```

## Implementation and experimental protocol
Waveforms are decoded using SoundFile, channels averaged, and resampled to 16 kHz. No silence trimming or amplitude normalization is applied. Inputs must be finite, non-silent, and 0.25–30 seconds. Each file yields 13 MFCC means and standard deviations (26), pYIN voiced pitch mean/std and voiced-frame fraction (3), RMS mean/std (2), spectral centroid mean/std (2), and zero-crossing-rate mean/std (2). Pitch is limited to 65–500 Hz; unvoiced recordings use zero pitch statistics with voiced fraction zero. Frames use 512 samples and 160-sample hops. Pitch estimation may be unreliable outside this range or under noise.

Actors 01–16 are training, 17–20 validation, 21–24 test. These are fixed before fitting and include both actor-sex categories. Four-fold StratifiedGroupKFold uses actor IDs exclusively inside training; StandardScaler is inside every candidate pipeline. GridSearchCV optimizes macro-F1: logistic C={0.1,1,10}; SVM C={1,10}, kernel={linear,rbf}; forest max_depth={None,12}, min_samples_leaf={1,3}, 200 trees; dummy predicts the training prior. Both MFCC-only and all-feature representations are tested. Parallelism is bounded to two workers; random seed 42. Validation chooses the benchmark winner; test outcomes never select hyperparameters or the winner. Models are not refitted on validation/test data. The demo deliberately uses the combined forest regardless of the benchmark winner, to support Tree SHAP of exactly its prediction model. Forest scores are uncalibrated vote probabilities, not validated confidence estimates. SVM benchmark uses decision labels, without recording-level probability-calibration leakage.

No training augmentation is used. Noise is generated only for already held-out recordings. Additive Gaussian white noise is scaled to the exact waveform power for 20, 10, and 0 dB SNR, with seed 42 plus stable test-row index. Signals are not clipped or normalized. Clean and noisy predictions share identical test recordings and fixed fitted models; there is no noise-specific retuning. This measures synthetic additive noise, not real acoustic robustness. Feature checkpoints are content-addressed and written atomically. Interrupted feature extraction resumes; model searches restart deterministically, preserving completed candidate models and validation outputs on disk. A full rerun after source or dependency changes should use a fresh artifacts directory to avoid stale feature caches.

## Results
Not measured. The pipeline writes macro-F1, balanced accuracy, per-class precision/recall/F1/support, confusion matrices with explicit class order, all candidate validation scores, all clean/noisy test scores, and manifests. `scripts/render_results.py` generates shareable CSVs and labeled confusion plots after successful training. Populate this section from those outputs only. Synthetic inference tests are software checks and are not speech-emotion performance evidence.

## Limitations and responsible interpretation
Only 24 actors and four final test speakers make estimates sensitive to speaker choice. Two statements, acted performances, studio microphones, limited accent and language diversity, and unequal neutral counts limit generalization. Expressed emotion is not mental state. There is no naturalistic validation, fairness audit, clinical validation, confidence calibration, or repeated-split confidence interval. White noise does not represent reverberation, music, babble, codecs, or microphone distortion. SHAP describes the forest, not the SVM/logistic winner or causal vocal mechanisms; correlated MFCCs complicate attribution. Do not use this demo for diagnosis, surveillance, hiring, or consequential assessments.

## Conclusion
A leakage-conscious classical workflow and explainable offline interface have been implemented. Empirical conclusions await authorized dataset access, full experiments, and validation of a persistently downloadable trained bundle.

## Team contributions (complete with actual work)
| Team member | Contribution | Evidence / commit | Review |
|---|---|---|---|
| To fill | To fill | To fill | To fill |
| To fill | To fill | To fill | To fill |
| To fill | To fill | To fill | To fill |
