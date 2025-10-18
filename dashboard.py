import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
import numpy as np
from PIL import Image
import io
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Image Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# THEME / STYLES (gradien pastel + kartu membulat)
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
* { font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
body, .stApp {
  background: radial-gradient(1200px 600px at 0% 0%, #ffdfe9 0%, #ffeccf 30%, #e5e3ff 65%, #f1f5ff 100%);
}
.sidebar .sidebar-content { background: transparent !important; }
.block-card {
  background:#fff9; backdrop-filter: blur(4px);
  border: 1px solid #ffffff55; border-radius: 22px; padding: 22px;
  box-shadow: 0 10px 24px #0000001a;
}
.badge {
  display:inline-block; padding:6px 14px; border-radius: 999px;
  background:#e9e2ff; color:#4d3aa6; font-weight:600; font-size: 12px;
}
.btn-main {
  border-radius: 999px; padding: 10px 18px; font-weight:600; border: none;
  background: linear-gradient(135deg, #ff9cc7, #b494ff); color: white;
}
.card-cta { border-radius:18px; padding:16px; background:#ffffffaa; border:1px solid #fff; }
.category-card { text-align:center; border-radius:22px; padding:14px; background:#fff; border:1px solid #00000010; }
.category-card:hover { box-shadow:0 12px 20px #00000012; transform: translateY(-2px); transition: all .2s; }
.upload-box { border:3px dashed #e2d9ff; border-radius:22px; padding: 6px 10px; }
.result-panel { background:#eafff1aa; border:1px solid #b6ffd3; border-radius:16px; padding:14px; min-height:220px; }
.result-panel-2 { background:#f2f8ffaa; border:1px solid #cfe2ff; border-radius:16px; padding:14px; min-height:110px; }
.caption-soft { color:#8b8fa7; font-size:13px }
.sidebar-thanks { color:#7e859a; font-size:12px; margin-top:12px }
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "stats" not in st.session_state:
    st.session_state.stats = {c: 0 for c in ["Animal", "Fashion", "Food", "Nature"]}
if "last_results" not in st.session_state:
    st.session_state.last_results = []   # list of tuples: (label, prob)
if "selected_theme" not in st.session_state:
    st.session_state.selected_theme = None

CLASS_NAMES = ["Animal", "Fashion", "Food", "Nature"]  # urutkan sesuai model klasifikasi kamu

# =========================
# LOAD MODELS
# =========================
@st.cache_resource(show_spinner=True)
def load_models():
    # ganti path sesuai file kamu
    yolo = YOLO("model/Meilani Bulandari Hsb_Laporan 4.pt")
    clf = tf.keras.models.load_model("model/Meilani Bulandari Hsb_Laporan 2.h5")
    return yolo, clf

try:
    yolo_model, classifier = load_models()
    MODELS_READY = True
except Exception as e:
    MODELS_READY = False
    load_err = str(e)

# =========================
# SIDEBAR (Menu kiri sesuai screenshot)
# =========================
st.sidebar.markdown("### Features That\nCan Be Used")
menu = st.sidebar.radio(
    "",
    ["Home", "Image Detection", "Image Classification", "Statistics", "Dataset", "About"],
    index=0,
)
st.sidebar.markdown("<div class='sidebar-thanks'>Thank you for using this website</div>", unsafe_allow_html=True)

# =========================
# REUSABLE: uploader
# =========================
def image_uploader(label="Select an image (jpg/png)"):
    st.markdown(f"<span class='badge'>{label}</span>", unsafe_allow_html=True)
    file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    return file

def read_image(file):
    pil = Image.open(file).convert("RGB")
    return pil

# =========================
# HOME
# =========================
if menu == "Home":
    st.markdown("## WELCOME TO MY IMAGE DETECTION")
    st.markdown("Welcome to Bulandari’s image detection website! Choose the features that best suit your needs.")
    with st.container():
        st.markdown("<div class='block-card'>", unsafe_allow_html=True)
        st.markdown("You can use this website to detect images by theme:")
        # Kartu kategori 2x2
        c1, c2 = st.columns(2)
        with c1:
            colA, colB = st.columns(2, gap="large")
            with colA:
                if st.button("🐾 Animal", use_container_width=True):
                    st.session_state.selected_theme = "Animal"
                    st.switch_page("app.py")  # tetap di halaman, hanya set state
                st.markdown("<div class='category-card'><img src='https://placehold.co/240x140?text=Animal' width='100%'></div>", unsafe_allow_html=True)
            with colB:
                if st.button("👗 Fashion", use_container_width=True):
                    st.session_state.selected_theme = "Fashion"
                st.markdown("<div class='category-card'><img src='https://placehold.co/240x140?text=Fashion' width='100%'></div>", unsafe_allow_html=True)
        with c2:
            colC, colD = st.columns(2, gap="large")
            with colC:
                if st.button("🍰 Food", use_container_width=True):
                    st.session_state.selected_theme = "Food"
                st.markdown("<div class='category-card'><img src='https://placehold.co/240x140?text=Food' width='100%'></div>", unsafe_allow_html=True)
            with colD:
                if st.button("🌲 Nature", use_container_width=True):
                    st.session_state.selected_theme = "Nature"
                st.markdown("<div class='category-card'><img src='https://placehold.co/240x140?text=Nature' width='100%'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# IMAGE DETECTION (YOLO)
# =========================
elif menu == "Image Detection":
    st.markdown("## UPLOAD IMAGE")
    st.caption("Insert the image according to what you want (jpg/png)")
    st.write("")
    with st.container():
        st.markdown("<div class='block-card'>", unsafe_allow_html=True)
        file = image_uploader("Select an image (jpg/png)")
        run = st.button("Run Detection", help="Jalankan deteksi objek pada gambar yang diunggah.", use_container_width=False)
        st.write("")
        c1, c2 = st.columns([1.2, 1], gap="large")

        if not MODELS_READY:
            st.error(f"Model belum siap: {load_err}")

        if file is not None:
            img = read_image(file)
            c1.image(img, caption="Input", use_container_width=True)
        else:
            c1.info("Silakan unggah gambar terlebih dahulu.")

        with c2:
            st.markdown("<div class='result-panel'><b>Result</b><br>", unsafe_allow_html=True)
            detected_list = []
            if MODELS_READY and run and file is not None:
                # inference
                results = yolo_model(img)
                # render
                plot = results[0].plot()  # numpy array (BGR)
                st.image(plot, caption="Hasil Deteksi", use_container_width=True)
                # list objek
                names = results[0].names if hasattr(results[0], "names") else yolo_model.names
                cls_ids = results[0].boxes.cls.cpu().numpy().astype(int).tolist() if len(results[0].boxes) else []
                for cid in cls_ids:
                    detected_list.append(names.get(cid, f"id:{cid}"))
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown("**Detected objects:**")
        if detected_list:
            st.write(", ".join(detected_list))
        else:
            st.write("—")

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# IMAGE CLASSIFICATION (TensorFlow)
# =========================
elif menu == "Image Classification":
    st.markdown("## UPLOAD IMAGE")
    st.caption("Insert the image according to what you want (jpg/png)")
    st.write("")
    with st.container():
        st.markdown("<div class='block-card'>", unsafe_allow_html=True)
        file = image_uploader("Select an image (jpg/png)")
        run = st.button("Run Classification", help="Klasifikasikan tema gambar: Animal/Fashion/Food/Nature")
        st.write("")
        c1, c2 = st.columns([1.2, 1], gap="large")

        label_out, prob_out = None, None

        if not MODELS_READY:
            st.error(f"Model belum siap: {load_err}")

        if file is not None:
            img = read_image(file)
            c1.image(img, caption="Input", use_container_width=True)
        else:
            c1.info("Silakan unggah gambar terlebih dahulu.")

        if MODELS_READY and run and file is not None:
            # preprocessing — sesuaikan ke ukuran modelmu (ganti 224 bila perlu)
            img_resized = img.resize((224, 224))
            arr = kimage.img_to_array(img_resized)
            arr = np.expand_dims(arr, axis=0) / 255.0
            pred = classifier.predict(arr, verbose=0)
            if pred.ndim == 2 and pred.shape[1] == len(CLASS_NAMES):
                idx = int(np.argmax(pred[0]))
                label_out = CLASS_NAMES[idx]
                prob_out = float(np.max(pred[0]))
            else:
                # fallback bila model output 1 angka (binary) atau bentuk lain
                idx = int(np.argmax(pred))
                label_out = CLASS_NAMES[idx % len(CLASS_NAMES)]
                prob_out = float(np.max(pred))

            # simpan ke session (Dataset & Statistics)
            st.session_state.last_results.append((label_out, prob_out))
            st.session_state.stats[label_out] = st.session_state.stats.get(label_out, 0) + 1

        with c2:
            st.markdown("<div class='result-panel-2'><b>Result</b><br>", unsafe_allow_html=True)
            if label_out is not None:
                st.metric(label="Predicted class", value=label_out)
                st.write(f"Confidence: **{prob_out*100:.2f}%**")
            else:
                st.write("—")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# STATISTICS (bar chart + daftar)
# =========================
elif menu == "Statistics":
    st.markdown("## SESSION STATISTICS")
    st.caption("Displays the total number of processes you have completed.")
    st.write("")
    with st.container():
        st.markdown("<div class='block-card'>", unsafe_allow_html=True)

        df = pd.DataFrame({
            "Category": list(st.session_state.stats.keys()),
            "Count": list(st.session_state.stats.values())
        })
        chart = st.bar_chart(df.set_index("Category"))

        # mirror list on right like screenshot
        st.write("")
        st.write("### Summary")
        for cat, cnt in st.session_state.stats.items():
            st.write(f"- **{cat}** — {cnt}")

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# DATASET (list hasil terakhir sesi)
# =========================
elif menu == "Dataset":
    st.markdown("## DATASET")
    st.caption("Upload sample images to build a dataset or review the files you've processed in this session.")
    st.write("")
    with st.container():
        st.markdown("<div class='block-card'>", unsafe_allow_html=True)
        st.write("**Last results (session):**")
        if st.session_state.last_results:
            for i, (lab, p) in enumerate(st.session_state.last_results[-20:], start=1):
                st.write(f"{i}. {lab} — {p*100:.2f}%")
        else:
            st.write("—")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# ABOUT
# =========================
elif menu == "About":
    st.markdown("## ABOUT")
    with st.container():
        st.markdown("<div class='block-card'>", unsafe_allow_html=True)
        st.write(
            "Image Detection by **Meilani Bulandari Hasibuan** — "
            "A clean UI for performing image detection and classification. "
            "Uses **ultralytics YOLO** for detection and **TensorFlow** for classification. "
            "Ready to deploy to **Streamlit Cloud**."
        )
        st.markdown("</div>", unsafe_allow_html=True)
