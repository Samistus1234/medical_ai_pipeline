from streamlit_webrtc import WebRtcMode
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import queue
from summarize_notes import summarize_notes
from streamlit_shap import st_shap
import pandas as pd
import shap
from lab_predictor import train_lab_model, predict_lab_result, explain_prediction
import streamlit as st
from analyze_xray import analyze_xray
from PIL import Image
import tempfile

st.set_page_config(page_title="Medical AI Assistant", layout="wide")
st.title("🩺 Medical AI Assistant")

# --- Section 1: Chest X-ray Interpretation ---
st.header("1. Chest X-ray Interpretation")
uploaded_image = st.file_uploader("Upload Chest X-ray Image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_image.read())
        temp_path = tmp_file.name

    result = analyze_xray(temp_path)

    st.subheader("Prediction Scores")
    st.write({k: float(f"{v:.2f}") for k, v in result.items() if v > 0.1})
# --- Section 2: Lab Interpretation ---
st.header("2. Lab Result Interpretation")

with st.form("lab_form"):
    st.write("Enter basic lab results:")
    wbc = st.number_input("WBC (White Blood Cell count)", min_value=0.0, step=0.1)
    hb = st.number_input("Hemoglobin (Hb)", min_value=0.0, step=0.1)
    na = st.number_input("Sodium (Na)", min_value=0.0, step=0.1)
    submitted = st.form_submit_button("Predict Diagnosis")

    if submitted:
        from lab_predictor import train_lab_model, predict_lab_result, explain_prediction
        model, background = train_lab_model()
        input_data = {"WBC": wbc, "Hb": hb, "Na": na}
        result = predict_lab_result(model, input_data)

        st.success(f"Predicted Diagnosis: **{result}**")

               # SHAP Explanation (corrected version)
        st.subheader("🧠 Why did the model make this prediction?")

        explainer, shap_values = explain_prediction(model, background, input_data)

        st_shap(
            shap.force_plot(
                base_value=explainer.expected_value[0],
                shap_values=shap_values[0][0],
                features=pd.DataFrame([input_data]),
                matplotlib=False
            ),
            height=150
        )
# --- Section 3: Clinical Note Summarization ---
st.header("3. Clinical Note Summarization")

clinical_note = st.text_area("Paste clinical notes here (e.g., HPI, physical exam, etc.):")

if st.button("Summarize Note"):
    if clinical_note.strip():
        with st.spinner("Generating summary..."):
            summary = summarize_notes(clinical_note)
        st.success("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter a clinical note to summarize.")
st.markdown("### 🎙️ Voice Dictation")

st.info("Click below to start recording your clinical note.")

# Stream audio input into a queue
audio_queue = queue.Queue()

class AudioProcessor(AudioProcessorBase):
    def recv(self, frame):
        audio_queue.put(frame.to_ndarray().flatten())
        return frame

ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={
        "video": False,
        "audio": True
    },
    rtc_configuration={  # optional, helps some devices
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    async_processing=True,
    audio_receiver_size=1024,
)

