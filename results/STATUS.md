# Execution evidence

- Python 3.12.14, Linux CPU; direct dependency versions and full resolved lock saved.
- Initial local pytest run: 6 passed in 23.09 seconds, 14 third-party matplotlib/pyparsing deprecation warnings.
- RAVDESS download attempted twice: proxy HTTP CONNECT 403 for zenodo.org. No archive was downloaded, no real features/training/evaluation completed.
- Expected dataset counts in the report are protocol expectations, not measured observations.
- Trained bundle: does not exist; packaging/release/retrieval unverified.
- GitHub CLI authentication: failed. Native Git proxy publishing is tested separately.
- Windows installation: unverified.

## Follow-up validation
- Expanded local pytest run: **10 passed in 4.26 seconds**, 14 third-party deprecation warnings. Full output: pytest.txt.
- Native HTTPS Git proxy push succeeded; branch project/explainable-ser exists remotely.
- PR creation attempted with gh: API GraphQL request returned Forbidden. There is no callable GitHub app connector in this session. This API failure does not establish that native Git authentication is missing.
- Runtime network still reports no custom allowed hosts; draft now requests zenodo.org and api.github.com. Apply through environment settings before retrying.

- GitHub branch source ZIP downloaded successfully (26,653 bytes), archive integrity checked, source and measured pytest output confirmed inside. Permanent download: https://github.com/therealvik3939-cloud/speech-emotion-recognition/archive/refs/heads/project/explainable-ser.zip
- Dependency integrity: pip check reports no broken requirements.
