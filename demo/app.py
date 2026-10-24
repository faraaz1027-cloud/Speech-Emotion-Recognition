"""Bangla SER coursework portfolio: audio feature explorer and study results.

No fitted model or training data were supplied with the original project files.
The app MUST NOT infer emotions until an independently trained artifact exists.
"""
from __future__ import annotations

import io
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from features import load_audio, extract_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ser_svm_pipeline.joblib"

st.set_page_config(page_title="Bangla Speech Emotion Research", page_icon="🎙️", layout="wide")
st.title("🎙️ Bangla Speech Emotion Recognition")
st.caption("CSE424 collaborative coursework · Acoustic-feature explorer & reproducibility project")

st.warning(
    "**Research demonstration, not a validated emotion detector.** The original "
    "notebook and report contain historical classification experiments, but the "
    "supplied ZIP contains **no fitted model or training audio**. Uploaded speech "
    "is processed here to visualize acoustic features; it is not classified."
)

st.subheader("Explore your own audio")
st.write(
    "Upload a short **WAV** speech recording to view its waveform, MFCCs, "
    "spectral statistics, and the 58-dimensional feature vector used in the coursework."
)
upload = st.file_uploader("WAV recording, maximum 15 MB and 30 seconds", type=["wav"])
if upload is None:
    st.info("Upload a WAV recording to explore audio features. No example speech or private dataset is bundled.")
else:
    raw = upload.getvalue()
    st.audio(raw, format="audio/wav")
    try:
        with st.spinner("Computing 58 acoustic features..."):
            audio, sr = load_audio(raw)
            features = extract_features(audio, sr)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        st.error("The recording could not be analyzed. Try a short uncompressed WAV file.")
    else:
        a, b, c = st.columns(3)
        a.metric("Duration after trimming", f"{len(audio)/sr:.2f} s")
        b.metric("Sample rate", f"{sr:,} Hz")
        c.metric("Extracted features", str(len(features)))

        decimation = max(1, len(audio) // 1600)
        t = np.arange(0, len(audio), decimation) / sr
        st.subheader("Normalized audio waveform")
        st.line_chart(pd.DataFrame({"Amplitude": audio[::decimation]}, index=t), x_label="Time (seconds)")

        st.subheader("Mel-frequency cepstral coefficients")
        fig, ax = plt.subplots(figsize=(10, 3))
        img = librosa.display.specshow(mfcc, sr=sr, hop_length=512, x_axis="time", ax=ax)
        ax.set_ylabel("MFCC index")
        fig.colorbar(img, ax=ax, format="%+2.0f")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Acoustic summary")
        a, b, c = st.columns(3)
        a.metric("Spectral centroid", f"{features['spectral_centroid_mean']:.0f} Hz")
        b.metric("Zero-crossing rate", f"{features['zcr_mean']:.3f}")
        c.metric("RMS energy", f"{features['rms_mean']:.3f}")

        with st.expander("Full 58-feature vector"):
            df = pd.DataFrame(
                {"Feature": list(features.keys()), "Value": list(features.values())}
            )
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.download_button(
                "Download feature vector as CSV",
                df.to_csv(index=False),
                file_name="bangla_ser_audio_features.csv",
                mime="text/csv",
            )
        st.caption(
            "These are signal descriptors, **not** emotion predictions. "
            "No uploaded audio is saved to a GitHub repository or application database."
        )

st.divider()
st.subheader("Recorded coursework results")
x, y, z = st.columns(3)
x.metric("SVM internal-test accuracy", "76.5%")
y.metric("SVM external-test accuracy", "33.6%")
z.metric("SVM external macro F1", "0.334")
st.caption(
    "The notebook reports five classes: Angry, Happy, Natural (neutral), Sad, "
    "and Surprised. External validation uses a balanced 500-clip set."
)
st.error(
    "**Important validation limitation:** The original notebook fits its "
    "StandardScaler on all extracted features *before* train/test splitting "
    "and cross-validation. Therefore the internal result is potentially optimistic. "
    "External performance also drops substantially. These values are historical "
    "coursework outputs, not independently reproduced here."
)
st.info(
    "To create a legitimate live classifier, authorized training audio is needed. "
    "The repository contains a separate training workflow that fits preprocessing "
    "only on the training split. A trained model will only be presented here "
    "after it has been produced, tested, and reviewed."
)
st.markdown(
    "**Project sources:** [GitHub repository]"
    "(https://github.com/faraaz1027-cloud/Speech-Emotion-Recognition) · "
    "[Original team notebook and report]"
    "(https://github.com/waysal1/Speech-Emotion-Recognition) · "
    "[BanglaMood dataset and license]"
    "(https://data.mendeley.com/datasets/9cc8k5v7w4) · "
    "[KBES dataset and license]"
    "(https://data.mendeley.com/datasets/vsn37ps3rx)"
)
st.caption(
    "Collaborative CSE424 project; the original contributors retain appropriate "
    "credit. Do not treat this educational research UI as a clinical, "
    "behavioral, or employment-related emotion assessment."
)
