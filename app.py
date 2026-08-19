"""Streamlit interface for the English Accent Classifier."""

from __future__ import annotations

import io
from pathlib import Path

import joblib
import streamlit as st

from audio_features import extract_mfcc


MODEL_FILENAMES = ("accent_model.pkl", "model.pkl")
DISPLAY_NAMES = {
    "indian_accent": "Indian Accent",
    "British_accent": "British Accent",
    "American_accent": "American Accent",
}
FLAGS = {"indian_accent": "🇮🇳", "British_accent": "🇬🇧", "American_accent": "🇺🇸"}


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the complete scaler-and-classifier pipeline once per app session."""
    model_path = next(
        (Path(__file__).with_name(filename) for filename in MODEL_FILENAMES if Path(__file__).with_name(filename).is_file()),
        None,
    )
    if model_path is None:
        raise FileNotFoundError("The trained model is not available yet.")
    return joblib.load(model_path)


def probability_chart(probabilities: dict[str, float]) -> None:
    """Draw a dependency-free horizontal probability chart."""
    rows = []
    for label, probability in probabilities.items():
        display_name = DISPLAY_NAMES.get(label, label.replace("_", " ").title())
        width = max(probability * 100, 0.5)
        rows.append(
            f"<div class='probability-row'><span>{display_name}</span>"
            f"<div class='track'><div class='fill' style='width:{width:.2f}%'></div></div>"
            f"<strong>{probability:.1%}</strong></div>"
        )
    st.markdown("<div class='probability-chart'>" + "".join(rows) + "</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="English Accent Classifier", page_icon="🎙️", layout="centered")
    st.markdown(
        """<style>
        .probability-row { display:flex; align-items:center; gap:10px; margin:12px 0; }
        .probability-row span { width:135px; font-size:0.95rem; }
        .probability-row strong { width:52px; text-align:right; }
        .track { flex:1; height:14px; border-radius:99px; background:#e6e9ef; overflow:hidden; }
        .fill { height:100%; background:#4f7cff; border-radius:99px; }
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("English Accent Classifier")
    st.caption("Speech-based accent classification using MFCC features and SVM")

    uploaded_file = st.file_uploader("Upload a WAV speech recording", type=["wav"])
    if uploaded_file is not None:
        audio_bytes = uploaded_file.getvalue()
        if not audio_bytes:
            st.error("The uploaded file is empty. Please choose a valid WAV recording.")
        else:
            st.audio(audio_bytes, format="audio/wav")

    if st.button("Predict Accent", type="primary", disabled=uploaded_file is None):
        try:
            model = load_model()
            features = extract_mfcc(io.BytesIO(uploaded_file.getvalue())).reshape(1, -1)
            probabilities_array = model.predict_proba(features)[0]
            classes = model.classes_
            probabilities = dict(zip(classes, probabilities_array))
            predicted_label = classes[probabilities_array.argmax()]
            display_name = DISPLAY_NAMES.get(predicted_label, predicted_label.replace("_", " ").title())
            flag = FLAGS.get(predicted_label, "🎙️")

            st.subheader("Predicted Accent")
            st.success(f"## {flag} {display_name}")
            st.subheader("Class probabilities")
            probability_chart(probabilities)
        except FileNotFoundError:
            st.error("Model file missing. Place `accent_model.pkl` beside `app.py`.")
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error("We could not process this recording. Please try another valid WAV file.")

    with st.expander("How it works"):
        st.markdown("""Audio  
↓  
MFCC Feature Extraction  
↓  
26-dimensional Feature Vector  
↓  
SVM Classifier  
↓  
Accent Prediction

This is a demonstration/prototype trained on a small dataset. It may not generalize to every speaker, recording condition, or English accent.""")


if __name__ == "__main__":
    main()
