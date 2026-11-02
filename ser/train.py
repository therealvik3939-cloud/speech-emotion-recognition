"""CPU bounded experiments. Resume feature checkpoints; never augment before splitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from ser.audio import load_audio, features, add_noise, LABELS, NAMES

SPLITS = {"train": list(range(1,17)), "validation": [17,18,19,20], "test": [21,22,23,24]}
def split_for(actor):
    return next(k for k,v in SPLITS.items() if actor in v)

def metrics(y, pred):
    return {"macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0), "balanced_accuracy": balanced_accuracy_score(y,pred), "classes": classification_report(y,pred,labels=LABELS,output_dict=True,zero_division=0), "confusion_matrix": confusion_matrix(y,pred,labels=LABELS).tolist(), "class_order": LABELS}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data"); p.add_argument("--output",default="artifacts"); args=p.parse_args()
    out=Path(args.output); out.mkdir(exist_ok=True,parents=True); cache=out/"cache"; cache.mkdir(exist_ok=True)
    rows=[]; xs=[]
    for path in sorted(Path(args.data).rglob("*.wav")):
        parts=path.stem.split("-")
        if len(parts)!=7 or parts[:2]!=["03","01"]: continue
        actor=int(parts[-1]); emotion=LABELS[int(parts[2])-1]
        digest=hashlib.sha256(path.read_bytes()).hexdigest(); saved=cache/(digest+".npy")
        if not saved.exists():
            value=features(load_audio(path)); temp=saved.with_suffix(".tmp")
            with temp.open("wb") as f: np.save(f,value)
            os.replace(temp,saved)
        xs.append(np.load(saved)); rows.append({"path":str(path),"sha256":digest,"actor":actor,"emotion":emotion,"split":split_for(actor),"recording_id":path.stem})
    frame=pd.DataFrame(rows)
    if len(frame)!=1440 or frame.recording_id.nunique()!=1440 or frame.sha256.nunique()!=1440:
        raise ValueError("Expected exactly 1440 unique RAVDESS speech recordings")
    frame.to_csv(out/"splits.csv",index=False)
    (out/"class_counts.json").write_text(frame.groupby(["split","emotion"]).size().to_json())
    X=np.stack(xs); y=frame.emotion.to_numpy(); train=frame.split.eq("train"); val=frame.split.eq("validation"); test=frame.split.eq("test")
    models={"dummy":(DummyClassifier(strategy="prior"),{}),"logistic":(LogisticRegression(max_iter=2000,random_state=42),{"clf__C":[.1,1,10]}),"svm":(SVC(probability=False,random_state=42),{"clf__C":[1,10],"clf__kernel":["linear","rbf"]}),"forest":(RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=2),{"clf__max_depth":[None,12],"clf__min_samples_leaf":[1,3]})}
    results={}; fitted={}
    cv=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=42)
    for feature_set,n in [("mfcc",26),("combined",len(NAMES))]:
        for name,(clf,grid) in models.items():
            key=feature_set+"_"+name
            pipe=Pipeline([("scale",StandardScaler()),("clf",clf)])
            search=GridSearchCV(pipe,grid,cv=cv,scoring="f1_macro",n_jobs=2,error_score="raise")
            search.fit(X[train,:n],y[train],groups=frame.actor[train])
            model=search.best_estimator_; fitted[key]=model
            results[key]={"cv_macro_f1":search.best_score_,"parameters":search.best_params_,"validation":metrics(y[val],model.predict(X[val,:n]))}
            joblib.dump(model,out/(key+".joblib"))
            (out/"validation.json").write_text(json.dumps(results,indent=2))
    # Select by validation, before opening test outcomes. No validation/test fitting.
    selected=max(results,key=lambda k:results[k]["validation"]["macro_f1"])
    (out/"selection.json").write_text(json.dumps({"selected":selected,"seed":42,"splits":SPLITS,"noise":"additive white Gaussian; no clipping; seed=42+test row index","snrs_db":[20,10,0]}))
    test_results={k:metrics(y[test],m.predict(X[test,:26 if k.startswith("mfcc") else len(NAMES)])) for k,m in fitted.items()}
    noisy={}
    for snr in [20,10,0]:
        nx=np.stack([features(add_noise(load_audio(row.path),snr,42+i)) for i,row in enumerate(frame[test].itertuples())])
        noisy[str(snr)]={k:metrics(y[test],m.predict(nx[:,:26 if k.startswith("mfcc") else len(NAMES)])) for k,m in fitted.items()}
    (out/"evaluation.json").write_text(json.dumps({"clean":test_results,"noise":noisy},indent=2))
    # Demo intentionally uses the combined forest: its probabilities and SHAP describe the same model.
    bundle={"model":fitted["combined_forest"],"explanation_model":fitted["combined_forest"],"feature_names":NAMES,"metadata":{"prediction_model":"combined_forest","benchmark_selected":selected,"seed":42}}
    target=out/"model.joblib"; joblib.dump(bundle,target)
    target.with_suffix(".joblib.sha256").write_text(hashlib.sha256(target.read_bytes()).hexdigest()+"  model.joblib\n")
    print(json.dumps({"selected":selected,"test":test_results[selected]},indent=2))
if __name__=="__main__": main()
