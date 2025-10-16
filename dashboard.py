import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# 🌸 Setup Page
# ==========================
st.set_page_config(
    page_title="Image Detector Bulan",
    page_icon="🌸",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            background-color: #FBE8E7;
        }

        .stApp {
            background-color: #FBE8E7;
        }

        .title {
            text-align: center;
            color: #D77FA1;
            font-weight: 700;
            font-size: 2.2em;
        }

        .sub {
            text-align: center;
            color: #555;
            font-size: 1em;
        }

        .stButton button {
            background-color: #F2B5D4;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            font-weight: 600;
            transition: 0.3s;
        }

        .stButton button:hover {
            background-color: #E59BC0;
        }

        .css-1cpxqw2, .stTextInput, .stFileUploader label {
            color: #444;
        }

        .uploadedImage {
            border-radius: 20px;
            box-shadow: 0px 0px 12px rgba(0,0,0,0.1);
        }

    </style>
""", unsafe_allow_html=True)

# ==========================
# 🌷 Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Meilani Bulandari Hsb_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Meilani Bulandari Hsb_Laporan 2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# 💫 UI Header
# ==========================
st.markdown("<h1 class='title'>AI Image Detection & Classification</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Deteksi objek dan klasifikasikan gambar menjadi Animal, Fashion, Food, atau Nature 🪷</p>", unsafe_allow_html=True)

# ==========================
# 📂 File Upload
# ==========================
uploaded_file = st.file_uploader("Upload gambar kamu di sini!", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_container_width=True, output_format="auto", clamp=True)

    mode = st.radio(
        "Pilih mode analisis:",
        ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"],
        horizontal=True
    )

    if st.button("✨ Jalankan Deteksi"):
        if mode == "Deteksi Objek (YOLO)":
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

        elif mode == "Klasifikasi Gambar":
            img_resized = img.resize((96, 96))  # sesuaikan ke input model
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            classes = ["Animal", "Fashion", "Food", "Nature"]
            predicted_class = classes[class_index]

            st.success(f"🌼 Gambar ini termasuk dalam kategori: **{predicted_class}**")
            st.write("📊 Probabilitas:", round(np.max(prediction)*100, 2), "%")

# ==========================
# ✨ Footer
# ==========================
st.markdown("""
<hr style='border: 1px solid #F2B5D4;'>
<p style='text-align: center; color: #888; font-size: 0.9em;'>
Didesain dengan 💕 oleh <b>Meilani Bulandari</b> — Powered by YOLO & TensorFlow
</p>
""", unsafe_allow_html=True)
