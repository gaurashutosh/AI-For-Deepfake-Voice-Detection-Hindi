import streamlit as st
import librosa
import numpy as np
import tempfile
import subprocess
import os
from joblib import load

# ---- CONFIG ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/xgb_mfcc_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.joblib")
SR = 16000

st.set_page_config(page_title="Deepfake Voice Detector", page_icon="🎙️")

# ---- check model/scaler existence ----
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    st.error(
        "Model or scaler not found. Make sure files exist in the models/ folder:\n"
        "- xgb_mfcc_model.joblib\n- scaler.joblib"
    )
    st.stop()

# load model & scaler
model = load(MODEL_PATH)
scaler = load(SCALER_PATH)

st.title("🎙️ Deepfake Voice Detector")
st.write(
    "Upload an audio file (wav/mp3/m4a/flac). The app will convert it to 16kHz mono WAV, extract MFCC features, and show a fake-probability score."
)

uploaded_file = st.file_uploader(
    "Choose an audio file", type=["wav", "mp3", "m4a", "flac"], accept_multiple_files=False
)

def convert_to_wav_bytes(uploaded_file_bytes, target_sr=SR):
    """Save uploaded bytes to a temp file and convert using ffmpeg to 16kHz mono WAV. Return path to temp wav."""
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file_bytes.name)[1]).name
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    # write bytes
    with open(tmp_in, "wb") as f:
        f.write(uploaded_file_bytes.getbuffer())
    # ffmpeg conversion
    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_in,
        "-ar", str(target_sr),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        tmp_out
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        try:
            os.remove(tmp_in)
        except:
            pass
        raise RuntimeError(f"ffmpeg conversion failed: {e}")
    try:
        os.remove(tmp_in)
    except:
        pass
    return tmp_out

def extract_mfcc_features(path, sr=16000, n_mfcc=20):
    y, _ = librosa.load(path, sr=sr, mono=True)   # ✅ removed backend
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)))
    else:
        y = y[:sr]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return feat

if uploaded_file is not None:
    # show audio player
    st.audio(uploaded_file)

    with st.spinner("⏳ Analyzing your audio... please wait..."):
        # convert -> extract -> predict
        try:
            wav_path = convert_to_wav_bytes(uploaded_file, target_sr=SR)
        except Exception as e:
            st.error(f"Audio conversion failed: {e}\nMake sure ffmpeg is installed and in PATH.")
            st.stop()

        try:
            feat = extract_mfcc_features(wav_path)
            X = feat.reshape(1, -1)
            Xs = scaler.transform(X)
            prob_fake = model.predict_proba(Xs)[0, 1]
        except Exception as e:
            st.error(f"Feature extraction or prediction failed: {e}")
            try:
                os.remove(wav_path)
            except:
                pass
            st.stop()

        # cleanup converted file
        try:
            os.remove(wav_path)
        except:
            pass

    # 🎉 After spinner finishes, show result
    st.success("✅ Analysis complete!")

    # display result
    st.markdown(f"**Fake probability:** **{prob_fake:.3f}**")
    if prob_fake >= 0.5:
        st.error("⚠️ Model says: Likely DEEPFAKE")
    else:
        st.success("✅ Model says: Likely REAL")

    # optional note
    # st.info("Note: threshold 0.5 used. You can tune this based on precision/recall needs.")
