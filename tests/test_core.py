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

def test_grouped_cv_and_fold_scaling():
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    groups=np.repeat(np.arange(1,17),8)
    labels=np.tile(np.arange(8),16)
    X=np.column_stack([groups*100.,labels])
    cv=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=42)
    for train,held in cv.split(X,labels,groups):
        assert not set(groups[train]) & set(groups[held])
        assert set(groups[train]) <= set(SPLITS['train'])
        pipe=Pipeline([('scale',StandardScaler()),('clf',__import__('sklearn.dummy',fromlist=['DummyClassifier']).DummyClassifier())])
        pipe.fit(X[train],labels[train])
        np.testing.assert_allclose(pipe['scale'].mean_,X[train].mean(axis=0))
        before=pipe['scale'].mean_.copy(); pipe.predict(X[held]+10000)
        np.testing.assert_array_equal(before,pipe['scale'].mean_)

def test_metrics_include_all_classes():
    m=metrics(LABELS,LABELS)
    assert m['macro_f1']==1 and m['balanced_accuracy']==1
    assert np.array_equal(m['confusion_matrix'],np.eye(8))
    assert m['class_order']==LABELS

def test_resample_and_stereo():
    t=np.arange(32000)/32000
    y=np.column_stack([.1*np.sin(2*np.pi*220*t)]*2)
    b=io.BytesIO(); sf.write(b,y,32000,format='WAV')
    mono=load_audio(b.getvalue())
    assert mono.ndim==1 and len(mono)==SR
    assert np.sqrt(np.mean(mono**2))==pytest.approx(.1/np.sqrt(2),rel=.01)

def test_app_with_synthetic_bundle(tmp_path,monkeypatch):
    # Tests readiness with a toy model; never distributed as a trained SER model.
    from sklearn.pipeline import Pipeline
    from streamlit.testing.v1 import AppTest
    model=Pipeline([('scale',StandardScaler()),('clf',RandomForestClassifier(n_estimators=2,random_state=42))])
    model.fit(np.random.default_rng(42).normal(size=(16,35)),LABELS*2)
    p=tmp_path/'model.joblib'; joblib.dump({'model':model,'explanation_model':model,'feature_names':NAMES},p)
    p.with_suffix('.joblib.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest())
    monkeypatch.setenv('SER_MODEL',str(p))
    app=AppTest.from_file('app.py').run()
    assert not app.exception and not app.warning
