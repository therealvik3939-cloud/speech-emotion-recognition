# Viva questions and answers
1. **Why speaker-disjoint splits?** Recording-level splits can let models memorize speaker identity. Actor groups prevent the same speaker entering training and evaluation.
2. **Why macro-F1?** It weights each emotion equally; neutral has fewer examples. Balanced accuracy averages class recalls. Neither alone describes every error.
3. **Where is scaling fitted?** Inside the pipeline on each CV training fold, then on all training actors for the selected hyperparameters. Never on validation/test actors.
4. **Why have validation and CV?** Training CV tunes hyperparameters; validation selects the benchmark model/feature family; test estimates final held-out performance.
5. **Can noisy copies leak?** They would if split independently. Here no training augmentation is used; noisy variants are generated only from already isolated test actors.
6. **What does SNR mean?** 10 log10(signal mean-square / noise mean-square). Noise is scaled exactly per recording; clipping would change that relationship.
7. **Why classical ML?** Small data and CPU constraints favor low-cost, interpretable baselines. This is not evidence they outperform deep models.
8. **What does SHAP explain?** The combined forest's class output relative to its background expectation, not a different benchmark winner or causal physiology.
9. **Are model scores calibrated?** No. Forest probabilities are tree vote averages; they must not be represented as calibrated certainty.
10. **Why keep a dummy baseline?** It shows whether learned models improve over a training-prior strategy under identical evaluation.
11. **Does white-noise robustness imply real-world robustness?** No: real noise is often nonstationary and includes reverberation and channel shift.
12. **What results were obtained?** RAVDESS experiments remain unrun because dataset access was denied. Report only saved measured results after execution.
13. **Can it identify true feelings?** No. Labels represent acted expressed emotion; internal state cannot be inferred reliably from this dataset.
14. **What remains reproducible after interruption?** Content-hash feature checkpoints resume; deterministic model searches can restart. Manifests record exact files and actor assignments.
