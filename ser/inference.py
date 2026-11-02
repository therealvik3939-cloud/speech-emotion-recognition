import hashlib
from pathlib import Path
import joblib
import numpy as np
from ser.audio import features, NAMES

def load_bundle(path):
    path = Path(path)
    checksum = path.with_suffix(path.suffix + ".sha256")
    if not checksum.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum.read_text().split()[0]:
        raise ValueError("Missing or incorrect model checksum.")
    # Joblib is executable: load only bundles from a trusted publisher.
    bundle = joblib.load(path)
    if bundle["feature_names"] != NAMES:
        raise ValueError("Incompatible feature schema.")
    return bundle

def predict(bundle, y):
    x = features(y).reshape(1, -1)
    model = bundle["model"]
    scores = model.predict_proba(x)[0]
    return str(model.classes_[np.argmax(scores)]), dict(zip(model.classes_, map(float, scores))), x

def explain(bundle, x):
    import shap
    model = bundle["explanation_model"]
    transformed = model["scale"].transform(x)
    values = shap.TreeExplainer(model["clf"])(transformed)
    return values.values[0], model.classes_
