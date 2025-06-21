import streamlit as st
from audio_recorder_streamlit import audio_recorder
import numpy as np
import soundfile as sf
import tempfile
import pickle
import os
from utils.model_utils import extract_features_from_audio_file, predict

st.set_page_config(page_title="ABC", layout="centered")
st.title("ABC (Audio Binary Classification)")
st.caption('Press the Button -> Say "_puppy_" or "_barbie_" -> Get classified!')

@st.cache_resource
def load_model():
    with open("model/model_cnn_mlp_manual.pkl", "rb") as f:
        return pickle.load(f)
model = load_model()

# audio record
audio_bytes = audio_recorder(
    text="",
)

if audio_bytes:
    # simpan audio record
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_bytes)
        temp_path = tmpfile.name

    # playback audio
    st.audio(temp_path, format="audio/wav")

    mfcc, extras = extract_features_from_audio_file(temp_path)

    # debug
    st.write(f"MFCC shape: {mfcc.shape}, Extras: {extras}")

    # prediksi
    label, conf = predict(mfcc, extras, model)
    st.success(f"🤖 Predicted: **{label}** ({conf} confidence)")

    os.remove(temp_path)
