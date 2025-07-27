import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import numpy as np

# ----------------------------
# Page Config & Styles
# ----------------------------
st.set_page_config(page_title="Banana Disease Detection", page_icon="🍌", layout="centered")

st.markdown(
    """
    <style>
    h1 {
        text-align: center;
        background: linear-gradient(90deg, #FFB300, #FF6F00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
    }
    div.stButton > button {
        background: #FFA000;
        color: #000000;
        border-radius: 10px;
        padding: 0.6em 1em;
        font-size: 16px;
        font-weight: bold;
        border: none;
        margin-bottom: 10px;
    }
    div.stButton > button:hover {
        background: #FFC107;
        color: #000000;
    }
    [data-testid="stSidebar"] {
        background: #FFF9E5;
        color: #212121;
    }
    .custom-tip {
        background-color: #FFF176;
        color: #212121;
        padding: 10px;
        border-radius: 8px;
        border-left: 6px solid #FBC02D;
        font-weight: bold;
    }
    .result-card {
        background: #FFF8E1;
        padding: 12px;
        border-radius: 10px;
        margin: 8px 0;
        font-size: 16px;
        border-left: 5px solid #FFB300;
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Sidebar Instructions
# ----------------------------
with st.sidebar:
    st.markdown(
        """
        <h2 style='color:#E65100;'>🍌 Banana Disease Detection</h2>
        <h3 style="color:#FF6F00; font-weight:bold;">How it works:</h3>
        <ol style="color:#3E2723;">
            <li>Click <b>Access Camera</b> to enable camera preview and capture</li>
            <li>OR Upload a clear banana leaf image</li>
            <li>Click <b>Detect Disease</b></li>
            <li>View annotated result</li>
        </ol>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="custom-tip">
        💡 Use <b>clear, well-lit banana leaf images</b> for the most accurate results.
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# Load YOLO Model
# ----------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Make sure best.pt is in same folder
    return model

model = load_model()

# ----------------------------
# Title
# ----------------------------
st.markdown("<h1>🍌 Banana Disease Detection </h1>", unsafe_allow_html=True)
st.write("Choose an option below to provide a banana leaf image for detection:")

# ----------------------------
# Initialize camera access flag
# ----------------------------
if "camera_access" not in st.session_state:
    st.session_state.camera_access = False

# ----------------------------
# Layout: Two columns for Camera Access button & File uploader
# ----------------------------
col1, col2 = st.columns(2)

selected_image = None

with col1:
    if not st.session_state.camera_access:
        if st.button("📸 Access Camera"):
            st.session_state.camera_access = True
    else:
        # Show cancel button to close camera preview
        if st.button("❌ Cancel Camera Access"):
            st.session_state.camera_access = False

        cam_img = st.camera_input("Capture Banana Leaf")
        if cam_img:
            selected_image = Image.open(cam_img)

with col2:
    uploaded_img = st.file_uploader("📂 Upload a Banana Leaf Image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        selected_image = Image.open(uploaded_img)

# ----------------------------
# Detection Logic
# ----------------------------
if selected_image:
    st.image(selected_image, caption="📥 Selected Banana Leaf", use_column_width=True)

    if st.button("🚀 Detect Disease"):
        with st.spinner("🔄 Analyzing leaf for diseases..."):

            img_np = np.array(selected_image)

            results = model.predict(img_np, conf=0.1, iou=0.3, save=True)

        if results and len(results[0].boxes) > 0:
            img_path = Path(results[0].path)
            save_dir = Path(results[0].save_dir)
            result_img_path = save_dir / img_path.name

            st.subheader("✅ Detection Result")
            st.image(str(result_img_path), caption="Annotated Detection", use_column_width=True)

            # Prediction Details removed as requested

            with open(result_img_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Annotated Result",
                    data=f,
                    file_name="banana_disease_result.jpg",
                    mime="image/jpeg"
                )
        else:
            st.error("❌ No detections! Either no disease or model didn't load correctly.")
