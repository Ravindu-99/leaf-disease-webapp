import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import numpy as np

# ✅ Page Config
st.set_page_config(
    page_title="Banana Disease Detection",
    page_icon="🍌",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ✅ Dark Theme with Matching Buttons
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Title gradient */
    h1 {
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFB300, #FF6F00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #1f1f1f;
        color: #e0e0e0;
        padding: 1rem 1.5rem;
    }

    /* Sidebar boxes */
    .sidebar-box {
        border: 2px solid #ffb300;
        border-radius: 12px;
        padding: 15px 20px;
        margin-bottom: 1rem;
        background-color: #2a2a2a;
        color: #ffcc80;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    .custom-tip {
        background-color: #333300;
        color: #ffff99;
        padding: 15px;
        border-radius: 12px;
        border-left: 8px solid #ffb300;
        font-weight: 600;
        margin-top: 1rem;
        font-size: 0.9rem;
    }

    /* Captions for images */
    .image-caption {
        font-size: 1rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1rem;
        color: #ffb300;
    }

    /* Buttons: Access, Upload, Detect, Reset, Download */
    div.stButton > button {
        background: #ffb300;
        color: #121212;
        font-weight: 700;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-size: 0.95rem;
        border: none;
        margin: 8px 0;
        box-shadow: 0 4px 8px rgba(255, 179, 0, 0.3);
        transition: background 0.3s ease;
    }
    div.stButton > button:hover {
        background: #ffa000;
        color: #121212;
        box-shadow: 0 6px 12px rgba(255, 160, 0, 0.4);
    }

    /* Download Button */
    div.stDownloadButton > button {
        background: #ffb300;
        color: #121212;
        font-weight: 700;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-size: 0.95rem;
        margin-top: 15px;
        transition: background 0.3s ease;
        box-shadow: 0 4px 8px rgba(255, 179, 0, 0.3);
    }
    div.stDownloadButton > button:hover {
        background: #ffa000;
        color: #121212;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Sidebar (No images)
with st.sidebar:
    st.markdown(
        """
        <h1>🍌 Banana Disease Detection</h2>
        <div class="sidebar-box">
            <h2>How it works:</h3>
            <ol>
                <li>Click <b>Access Camera</b> to capture a banana leaf photo</li>
                <li>OR upload a clear banana leaf image</li>
                <li>Click <b>Detect Disease</b> to get results</li>
                <li>View annotated image and download results</li>
            </ol>
        </div>
        <div class="custom-tip">
            💡 Use <b>clear, well-lit banana leaf images</b> for the most accurate results.
        </div>
        """,
        unsafe_allow_html=True
    )

# ✅ Load YOLO Model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Replace with your trained model

model = load_model()

st.markdown("<h1>🍌 Banana Disease Detection</h1>", unsafe_allow_html=True)
st.write("Choose an option below to provide a banana leaf image for detection:")

# ✅ Session States
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0
if "camera_access" not in st.session_state:
    st.session_state.camera_access = False
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

# ✅ Camera + Upload Buttons
col1, col2 = st.columns(2)

with col1:
    if not st.session_state.camera_access:
        if st.button("📸 Access Camera"):
            st.session_state.camera_access = True
    else:
        if st.button("❌ Cancel Camera"):
            st.session_state.camera_access = False
            st.session_state.selected_image = None
        cam_img = st.camera_input(
            "Capture Banana Leaf",
            key=f"camera_input_{st.session_state.reset_counter}"
        )
        if cam_img:
            st.session_state.selected_image = Image.open(cam_img)

with col2:
    uploaded_img = st.file_uploader(
        "📂 Upload a Banana Leaf Image",
        type=["jpg", "jpeg", "png"],
        key=f"file_uploader_{st.session_state.reset_counter}"
    )
    if uploaded_img:
        st.session_state.selected_image = Image.open(uploaded_img)

# ✅ Show Selected Image
if st.session_state.selected_image:
    # Caption for selected image
    st.markdown('<div class="image-caption">📥 Selected Banana Leaf</div>', unsafe_allow_html=True)
    st.image(st.session_state.selected_image, use_container_width=True)

    # ✅ Detect Button
    if st.button("🚀 Detect Disease"):
        with st.spinner("🔄 Analyzing leaf for diseases..."):
            img_np = np.array(st.session_state.selected_image)
            results = model.predict(img_np, conf=0.1, iou=0.3, save=True)

        if results and len(results[0].boxes) > 0:
            img_path = Path(results[0].path)
            save_dir = Path(results[0].save_dir)
            result_img_path = save_dir / img_path.name

            # Caption for detection result
            st.markdown('<div class="image-caption">📥 Annotated Detection</div>', unsafe_allow_html=True)
            st.image(str(result_img_path), use_container_width=True)

            # ✅ Download Button
            with open(result_img_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Annotated Result",
                    data=f,
                    file_name="banana_disease_result.jpg",
                    mime="image/jpeg"
                )
        else:
            st.error("❌ No detections found! Try another image.")

# ✅ Reset Button (same theme)
if st.button("🔄 Reset", key="reset_button"):
    st.session_state.camera_access = False
    st.session_state.selected_image = None
    st.session_state.reset_counter += 1
    st.success("✅ Reset done — inputs cleared!")
