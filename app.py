import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tempfile
import os
import gdown
import matplotlib.pyplot as plt

# ------------------------------
# Download model from Google Drive if missing
# ------------------------------
MODEL_PATH = "brain_tumor_model.h5"  # local path
FILE_ID = "1ywdCfQsYOTmfEeywv8qE-9dSIiJvj2bc"  # your Drive file ID
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(URL, MODEL_PATH, quiet=False, fuzzy=True)

# Load the model
model = load_model(MODEL_PATH)

# ------------------------------
# Class names and colors
# ------------------------------
class_names = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
class_colors = {
    "glioma_tumor": "#FF9999",
    "meningioma_tumor": "#99FF99",
    "no_tumor": "#99CCFF",
    "pituitary_tumor": "#FFCC99"
}

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload MRI images and see predictions with confidence charts.")

# Sidebar legend
st.sidebar.header("Tumor Types Legend")
for tumor, color in class_colors.items():
    st.sidebar.markdown(
        f"""
        <div style='
            background-color:{color};
            padding:10px;
            border-radius:5px;
            text-align:center;
            margin-bottom:8px;
        '>{tumor}</div>
        """,
        unsafe_allow_html=True
    )

# File uploader
uploaded_files = st.file_uploader(
    "Choose MRI images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True
)

# ------------------------------
# Process uploaded images
# ------------------------------
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown("---")
        st.subheader(f"File: {uploaded_file.name}")

        # Save temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Load and preprocess
        img = image.load_img(tfile.name, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediction
        predictions = model.predict(img_array)
        pred_idx = np.argmax(predictions[0])
        pred_label = class_names[pred_idx]
        pred_conf = np.max(predictions[0]) * 100

        # Display image
        st.image(img, caption="Uploaded MRI", use_container_width=True)

        # Colored prediction box
        st.markdown(
            f"""
            <div style="
                background-color: {class_colors[pred_label]};
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                font-size: 20px;
                font-weight: bold;
            ">
                Prediction: {pred_label} ({pred_conf:.2f}%)
            </div>
            """,
            unsafe_allow_html=True
        )

        # Prediction probability bar chart
        st.subheader("Prediction Confidence for All Tumor Types")
        fig, ax = plt.subplots()
        ax.bar(class_names, predictions[0], color=[class_colors[name] for name in class_names])
        ax.set_ylim([0, 1])
        ax.set_ylabel("Probability")
        ax.set_title("Model Prediction Probabilities")
        for i, v in enumerate(predictions[0]):
            ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontweight='bold')
        st.pyplot(fig)
