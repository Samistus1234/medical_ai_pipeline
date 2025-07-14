import streamlit as st
from analyze_xray import analyze_xray
from summarize_notes import summarize_notes
import pandas as pd

st.set_page_config(page_title="Medical AI Assistant", layout="wide")

st.title("🩺 Medical AI Assistant")

# --- Section 1: Chest X-ray ---
st.header("1. Chest X-ray Interpretation")
uploaded_image = st.file_uploader("Upload Chest X-ray Image", type=["png", "jpg", "jpeg"])
if uploaded_image:
    with open("temp_xray.png", "wb") as f:
        f.write(uploaded_image.read())
    result = analyze_xray("temp_xray.png")
    st.subheader("Prediction Scores")
    st.write(result)

# --- Section 2: Lab Data ---
st.header("2. Lab Result Analysis (Coming Soon)")
st.info("Upload CSV feature for lab analysis will be activated in the next version.")

# --- Section 3: Clinical Notes Summarization ---
st.header("3. Clinical Notes Summarization")
clinical_note = st.text_area("Paste clinical notes here:")
if st.button("Summarize"):
    summary = summarize_notes(clinical_note)
    st.subheader("Summary")
    st.write(summary)
