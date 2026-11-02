# Eight-minute presentation outline
- 0:00–0:45 Motivation: expressed emotion in speech; distinguish from internal state.
- 0:45–1:40 Dataset: RAVDESS provenance, license, expected counts, acted studio limitations.
- 1:40–2:40 Architecture: WAV validation, resampling, 35 acoustic statistics, offline artifacts.
- 2:40–3:50 Evaluation design: actor isolation, fold-local preprocessing, grouped CV, bounded search, dummy baseline.
- 3:50–4:50 Experiments: MFCC ablation, clean/noisy paired test at 20/10/0 dB. Show measured tables only when generated; currently explicitly unrun.
- 4:50–5:45 Explainability: forest Tree SHAP, class-specific attribution, correlated-feature limitations; explain why demo forest can differ from benchmark winner.
- 5:45–7:10 Demo walkthrough: scores, explanation, noise comparison, graceful invalid-input behavior. If bundle missing, show honest unavailable state, not fabricated predictions.
- 7:10–8:00 Limitations, conclusion, remaining access/publication work, actual contribution table.

# Four-minute demo script
- 0:00–0:30 Launch locally using the README command. Explain that inference requires no network once dependencies and a trusted bundle are installed.
- 0:30–1:10 Upload a short WAV the presenter has rights to use; play it. Read predicted expressed emotion and explain that scores are uncalibrated forest probabilities.
- 1:10–2:00 Inspect class-specific SHAP features; explain positive/negative relative contributions and why these are associations, not causes.
- 2:00–3:00 Compare the same waveform at 20, 10, 0 dB white noise. Listen and inspect score changes; do not assume monotonic degradation on an individual recording.
- 3:00–3:30 Upload silence or an invalid WAV to demonstrate validation.
- 3:30–4:00 Show evaluation manifests and actual metrics if available. State held-out speaker counts and licensing limitations.

If data access is still blocked, show the unavailable-model screen and tests; explicitly state that no trained model or measured results exist yet.
