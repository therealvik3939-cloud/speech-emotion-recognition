import io
import numpy as np
import librosa
import soundfile as sf

SR = 16000
MAX_BYTES = 20 * 1024 * 1024
LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
NAMES = [f"mfcc_{i+1}_{s}" for s in ("mean", "std") for i in range(13)] + ["pitch_mean", "pitch_std", "voiced_fraction", "rms_mean", "rms_std", "centroid_mean", "centroid_std", "zcr_mean", "zcr_std"]

def load_audio(source):
    if isinstance(source, bytes):
        if len(source) > MAX_BYTES:
            raise ValueError("WAV exceeds 20 MiB.")
        source = io.BytesIO(source)
    try:
        with sf.SoundFile(source) as f:
            if f.format != "WAV" or not 4000 <= f.samplerate <= 192000:
                raise ValueError("Upload a WAV with sample rate 4–192 kHz.")
            if not .25 <= len(f) / f.samplerate <= 30:
                raise ValueError("Audio must last 0.25–30 seconds.")
            rate = f.samplerate
            y = f.read(dtype="float32", always_2d=True).mean(axis=1)
    except (RuntimeError, sf.LibsndfileError) as e:
        raise ValueError("Invalid or unsupported WAV.") from e
    if not np.isfinite(y).all() or np.sqrt(np.mean(y*y)) < 1e-5:
        raise ValueError("Audio is silent or contains invalid samples.")
    return librosa.resample(y, orig_sr=rate, target_sr=SR) if rate != SR else y

def features(y):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or len(y) < SR//4 or not np.isfinite(y).all() or np.sqrt(np.mean(y*y)) < 1e-5:
        raise ValueError("Expected finite, non-silent mono audio of at least 0.25 seconds.")
    # No peak normalization: retain energy information; all experiments use this same path.
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, n_fft=512, hop_length=160)
    pitch, voiced, _ = librosa.pyin(y, fmin=65, fmax=500, sr=SR, frame_length=512, hop_length=160)
    p = pitch[np.isfinite(pitch)]
    out = list(mfcc.mean(axis=1)) + list(mfcc.std(axis=1))
    out += [float(p.mean()) if len(p) else 0., float(p.std()) if len(p) else 0., float(voiced.mean())]
    for arr in (librosa.feature.rms(y=y, frame_length=512, hop_length=160), librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=512, hop_length=160), librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=160)):
        out += [float(arr.mean()), float(arr.std())]
    return np.asarray(out, dtype=np.float64)

def add_noise(y, snr_db, seed=42):
    y = np.asarray(y, dtype=np.float64)
    power = np.mean(y*y)
    if not np.isfinite(snr_db) or power <= 0 or not np.isfinite(y).all():
        raise ValueError("Noise requires finite SNR and nonzero finite audio.")
    noise = np.random.default_rng(seed).normal(size=y.shape)
    noise *= np.sqrt(power / (10**(snr_db/10) * np.mean(noise*noise)))
    return y + noise  # deliberately no clipping or renormalization
