import io
import hashlib
import joblib
import numpy as np
import pytest
import soundfile as sf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from ser.audio import *
from ser.train import SPLITS, split_for, metrics
from ser.inference import load_bundle,predict,explain

def tone(): return (.1*np.sin(2*np.pi*220*np.arange(SR)/SR)).astype(np.float32)
def wav(y):
    b=io.BytesIO(); sf.write(b,y,SR,format="WAV"); return b.getvalue()
def test_features():
    a=features(tone()); assert a.shape==(35,); assert np.isfinite(a).all()
    np.testing.assert_array_equal(a,features(tone()))
    assert 180<a[26]<260

def test_invalid():
    for data in (b"bad",wav(np.zeros(SR)),b"x"*(MAX_BYTES+1),wav(np.ones(100))):
        with pytest.raises(ValueError): load_audio(data)

def test_split_isolation():
    assert sorted(sum(SPLITS.values(),[]))==list(range(1,25))
    for a in range(1,25):
        # Any variant of a recording retains the actor's assignment.
        variants = [(f"03-01-01-01-01-01-{a:02d}", noise) for noise in [None,0,10]]
        assignments = {split_for(int(record.split("-")[-1])) for record, noise in variants}
        assert assignments == {split_for(a)}
    assert not set(SPLITS["train"]) & set(SPLITS["test"])

def test_noise():
    y=tone(); n=add_noise(y,10)
    assert np.isclose(10*np.log10(np.mean(y*y)/np.mean((n-y)**2)),10)
    np.testing.assert_array_equal(n,add_noise(y,10))
    assert not np.array_equal(n,add_noise(y,10,43))
    with pytest.raises(ValueError): add_noise(np.zeros(SR),10)

def test_inference(tmp_path):
    x=features(tone()); rng=np.random.default_rng(42)
    model=make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=5,random_state=42))
    model.fit(np.stack([x+rng.normal(size=35) for _ in range(12)]),["calm","happy"]*6)
    # Named steps follow production bundle schema.
    from sklearn.pipeline import Pipeline
    model=Pipeline([("scale",model[0]),("clf",model[1])])
    p=tmp_path/"model.joblib"; joblib.dump({"model":model,"explanation_model":model,"feature_names":NAMES},p)
    p.with_suffix(".joblib.sha256").write_text(hashlib.sha256(p.read_bytes()).hexdigest())
    b=load_bundle(p); label,scores,z=predict(b,tone())
    assert label in scores and sum(scores.values())==pytest.approx(1)
    vals,classes=explain(b,z); assert vals.shape==(35,2)
    p.write_bytes(b"corrupt")
    with pytest.raises(ValueError): load_bundle(p)

def test_app_smoke(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("SER_MODEL","/nonexistent/model.joblib")
    app=AppTest.from_file("app.py").run()
    assert not app.exception and len(app.warning)==1
