import streamlit as st
import torch
import os
import time
import base64
import tempfile
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="VoiceSHIELD – Malicious Audio Detector",
    page_icon="🛡️",
    layout="wide"
)
st.sidebar.image("emvo.png")

st.title("🛡️ VoiceSHIELD")
st.caption("AI-powered detection of malicious and unsafe audio content")

# ------------------------------------------------------------
# Model Loader (cached)
# ------------------------------------------------------------
@st.cache_resource
def load_voiceshield():
    MODEL_ID = "Emvo-ai/voiceSHIELD-small"

    with st.spinner("Loading VoiceShield model..."):
        model_path = snapshot_download(repo_id=MODEL_ID)
        import sys
        sys.path.insert(0, model_path)

        config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )

        from modeling_voiceshield import VoiceShieldForAudioClassification
        from pipeline_voiceshield import VoiceShieldPipeline

        model = VoiceShieldForAudioClassification(config)
        weights_file = os.path.join(model_path, "model.safetensors")
        state_dict = load_file(weights_file)
        model.load_state_dict(state_dict, strict=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_id = 0 if torch.cuda.is_available() else -1

        model = model.to(device)
        model.eval()

        pipe = VoiceShieldPipeline(model=model, device=device_id)
        return pipe, device

pipe, device = load_voiceshield()

# ------------------------------------------------------------
# Sample Audio Files
# ------------------------------------------------------------
AUDIO_FILES = {
    "Jailbreak – Normal Conversation": "audio_0033.wav",
    "Jailbreak – Voice": "audio_0022.wav",
    "Prompt Injection – Voicemail": "audio_0116.wav",
    "Prompt Injection – Voice": "test2.wav",
    "Prompt Injection – Audio": "test3.wav",
    "Custom Voice Sample": "my_voice_sample.wav"
}

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)

    st.markdown("### About VoiceSHIELD")
    st.markdown(
        """
        Detects malicious or unsafe audio content such as:
        - Scam calls
        - Prompt injection attempts
        - Fraudulent voice messages

        **Output classes**
        - 🟢 Safe
        - 🔴 Malicious
        """
    )

    st.markdown("---")
    st.caption(f"Running on: **{device}**")

# ------------------------------------------------------------
# Result Display Function
# ------------------------------------------------------------
def show_results(result, latency):
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Latency", f"{latency:.2f} s")

    with col2:
        label = result["label"].lower()
        if label == "malicious":
            st.error("Prediction: MALICIOUS 🔴")
        else:
            st.success("Prediction: SAFE 🟢")

    confidence = result["confidence"]
    st.progress(confidence, text=f"Confidence: {confidence:.2%}")

    transcript = result.get("transcript", "")
    if transcript:
        st.markdown("**Transcript**")
        st.write(transcript)

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2 = st.tabs(["Sample Audio", "Upload & Test"])

# ============================================================
# Tab 1: Sample Audio
# ============================================================
with tab1:
    st.subheader("Test Pre-recorded Audio")

    selected_name = st.selectbox(
        "Select audio sample",
        list(AUDIO_FILES.keys())
    )
    selected_path = AUDIO_FILES[selected_name]

    if os.path.exists(selected_path):
        st.audio(selected_path, format="audio/wav")

        if st.button("Run Detection", type="primary"):
            with st.spinner("Analyzing audio..."):
                start = time.perf_counter()
                result = pipe(selected_path)
                latency = time.perf_counter() - start

            show_results(result, latency)
    else:
        st.warning("Audio file not found.")

# ============================================================
# Tab 2: Live Recording
# ============================================================
# ============================================================
# Tab 2: Upload or Record Audio
# ============================================================
with tab2:
    st.subheader("Upload or Record Audio")

    uploaded_file = st.file_uploader(
        "Upload a WAV audio file",
        type=["wav"]
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("Analyze Uploaded Audio"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner("Analyzing..."):
                start = time.perf_counter()
                result = pipe(tmp_path)
                latency = time.perf_counter() - start

            os.unlink(tmp_path)
            show_results(result, latency)

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("© 2026 EMVO AI – VoiceShield Security Model")
