import os
from pathlib import Path
import io
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
from ser.audio import load_audio, add_noise, NAMES, SR
from ser.inference import load_bundle, predict, explain

st.set_page_config(page_title="Voice · Emotion Lab",page_icon="🎙️",layout="wide")
st.title("🎙️ Voice · Emotion Lab")
st.caption("Explainable speech emotion recognition · local, offline inference")
st.info("Estimates acted, expressed emotion—not a person's internal state. Not for diagnosis or high-stakes decisions.")
path=Path(os.environ.get("SER_MODEL", "artifacts/model.joblib"))
if not path.exists():
    st.warning("Trained model unavailable. Follow README training or verified model-bundle installation instructions.")
else:
    try:
        bundle=load_bundle(path)
        st.caption("Prediction and SHAP model: combined-feature Random Forest. Scores are uncalibrated class vote probabilities.")
        upload=st.file_uploader("Upload WAV · 0.25–30 seconds · up to 20 MiB",type=["wav"])
        if upload:
            y=load_audio(upload.getvalue()); st.audio(upload.getvalue())
            emotion,scores,x=predict(bundle,y); st.metric("Predicted expressed emotion",emotion)
            st.bar_chart(pd.Series(scores,name="Model score"))
            values,classes=explain(bundle,x); idx=list(classes).index(emotion)
            contributions=values[:,idx] if values.ndim==2 else values
            st.subheader("Why this forest prediction?")
            st.caption("Tree SHAP contributions for the predicted class relative to the forest background expectation; associations, not causes.")
            st.bar_chart(pd.Series(contributions,index=NAMES).sort_values(key=abs).tail(12))
            snr=st.select_slider("White-noise SNR (dB)",options=[0,10,20],value=10)
            noisy=add_noise(y,snr); label,ns,_=predict(bundle,noisy)
            st.subheader("Clean / noisy comparison")
            st.write(f"Clean: {emotion} · Noisy: {label}")
            st.bar_chart(pd.DataFrame({"Clean":scores,"Noisy":ns}))
            buffer=io.BytesIO(); sf.write(buffer,noisy,SR,format="WAV",subtype="FLOAT"); st.audio(buffer.getvalue())
            st.caption("Playback may clip at high noise levels; inference uses the unclipped waveform.")
    except (ValueError,RuntimeError,OSError) as e:
        st.error(str(e))
