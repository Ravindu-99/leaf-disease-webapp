import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

# ----------------------------
# 1️⃣ Page Config & Custom CSS
# ----------------------------
st.set_page_config(page_title="🌿 Leaf Disease Detection", page_icon="🍃", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Center main titles */
    .css-10trblm { text-align: center; font-size: 28px; font-weight: bold; }

    /* Modern buttons */
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1em;
        font-size: 16px;
        font-weight: bold;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: #f4f9f4;
    }

    /* Detection result cards */
    .result-card {
        background: #f1fff1;
        padding: 10px;
        border-radius: 10px;
        margin: 8px 0;
        font-size: 16px;
        border-left: 5px solid #4CAF50;
        color: #000000; /* Force black text */
    }

    /* General heading color */
    h3 {
        color: #1B5E20 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# 2️⃣ Sidebar with colored text and styled tip box
# ----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/7666/7666443.png", width=100)

    st.markdown(
        """
        <h2 style='color:#1B5E20; text-align:center;'>🍃 Leaf Detector</h2>
        <h3 style="color:#2E7D32; font-weight:bold;">How it works:</h3>
        <ol style="color:#4A8C2F; padding-left: 20px;">
            <li>Upload a clear leaf image</li>
            <li>Click <b>Detect</b></li>
            <li>View annotated result + disease prediction</li>
        </ol>
        <p style="color:#1B5E20; font-weight:bold;">Model: YOLOv8 Custom</p>

        <style>
        /* Change the color and background of the info box */
        div[role="alert"] {
            background-color: #DFF0D8 !important;
            color: #3C763D !important;
            border-left: 6px solid #4CAF50 !important;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.info("💡 Tip: Use well-lit, clear images for best results")

# ----------------------------
# 3️⃣ Load Model (cached)
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ----------------------------
# 4️⃣ Main Title with blue color
# ----------------------------
st.markdown("<h2 style='text-align:center; color:#1565C0;'>🌱 Leaf Disease Detection Web App</h2>", unsafe_allow_html=True)
st.write("Upload a leaf image and let the AI model detect diseases with confidence levels.")

# ----------------------------
# 5️⃣ Upload Section with white tips list on dark background
# ----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📷 Upload a Leaf Image", type=["jpg", "jpeg", "png"])

with col2:
    st.markdown(
        """
        <h3 style="color:#E65100;">📝 Tips for Better Detection</h3>
        <ul style="color:#FFFFFF; background-color:#4A8C2F; padding:10px; border-radius:8px;">
            <li>✅ Use <b>clear, high-resolution</b> images</li>
            <li>✅ Avoid <b>blurry photos</b></li>
            <li>✅ Ensure <b>good lighting</b></li>
            <li>✅ Only <b>one leaf per image</b> for best accuracy</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# 6️⃣ Detection Section
# ----------------------------
if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="📥 Uploaded Image", use_column_width=True)

    # Detection button
    detect_btn = st.button("🚀 Detect Disease")

    if detect_btn:
        st.write("🔄 Running model...")
        results = model.predict(image, save=True)  # YOLO prediction & save annotated image

        if results:
            # Get result image path
            img_path = Path(results[0].path)         # Original filename
            save_dir = Path(results[0].save_dir)     # Output dir
            result_img_path = save_dir / img_path.name  # Final saved result path

            # Show detection result image
            st.markdown(
                "<h3 style='color:#1B5E20; background:#E8F5E9; padding:6px; border-radius:6px;'>✅ Detection Result</h3>",
                unsafe_allow_html=True
            )
            st.image(str(result_img_path), caption="Annotated Detection", use_column_width=True)

            # Show Prediction Details with clear styling
            st.markdown(
                "<h3 style='color:#1B5E20; background:#E8F5E9; padding:6px; border-radius:6px;'>📊 Prediction Details</h3>",
                unsafe_allow_html=True
            )
            
            for box in results[0].boxes:
                cls_id = int(box.cls.item())        # class ID
                conf = float(box.conf.item())       # confidence
                label = results[0].names[cls_id]    # class name

                st.markdown(
                    f"""
                    <div class="result-card">
                        <b>🌿 Class:</b> {label}<br>
                        <b>🎯 Confidence:</b> {conf:.2%}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Add a download button for annotated image
            with open(result_img_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Annotated Result",
                    data=f,
                    file_name="detection_result.jpg",
                    mime="image/jpeg"
                )
        else:
            st.error("❌ No results detected!")
