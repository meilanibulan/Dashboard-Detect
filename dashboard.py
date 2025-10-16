import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# ==========================
# Custom CSS untuk tampilan mirip UI cantik pastel
# ==========================
st.markdown("""
    <style>
    body {
        background-color: #fdfdfd;
    }
    .main {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 30px;
    }
    .stApp {
        background-color: #FCEBEA;
    }
    .title {
        color: #070F4E;
        text-align: center;
        font-weight: 700;
        font-size: 32px;
    }
    .subtitle {
        color: #555;
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .upload-box {
        background-color: #FFE7E6;
        padding: 20px;
        border-radius: 20px;
        text-align: center;
    }
    .result-card {
        background-color: white;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Meilani Bulandari Hsb_Laporan 4.pt")  # YOLO model
    classifier = tf.keras.models.load_model("model/Meilani Bulandari Hsb_Laporan 2.h5")  # CNN classifier
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.markdown("<h1 class='title'>AI Image Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Deteksi dan klasifikasikan gambar ke dalam kategori Animal, Fashion, Food, atau Nature 🌸</p>", unsafe_allow_html=True)

menu = st.sidebar.radio("📂 Pilih Mode", ["🕵️‍♀️ Deteksi Objek (YOLO)", "🧩 Klasifikasi Gambar"])
uploaded_file = st.file_uploader("Unggah gambar kamu di sini 💖", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(image_bytes))
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if menu == "🕵️‍♀️ Deteksi Objek (YOLO)":
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("🔍 Hasil Deteksi Objek")
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        elif menu == "🧩 Klasifikasi Gambar":
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("🎯 Hasil Klasifikasi")

            # Preprocessing
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Prediksi
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            class_labels = ["Animal", "Fashion", "Food", "Nature"]
            result = class_labels[class_index]

            st.success(f"Gambar ini termasuk kategori: **{result}** 🌼")
            st.metric(label="Probabilitas", value=f"{np.max(prediction)*100:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("📸 Unggah gambar terlebih dahulu untuk melihat hasil deteksi atau klasifikasi ya, Sayang 💕")
