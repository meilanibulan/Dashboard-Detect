import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
import numpy as np
from PIL import Image
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
# GLOBAL STYLES (Tema pastel + Sidebar seperti mockup kanan)
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
* { font-family:'Poppins', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }

/* App background */
.stApp{
  background: radial-gradient(1200px 600px at 0% 0%, #FFF5D9 0%, #FFECE8 32%, #ECEBFF 70%);
}

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #FFF7DC 0%, #FFF2E9 50%, #FFEAF4 100%) !important;
  border-right: 1px solid #EAE0FF;
}
section[data-testid="stSidebar"] .sidebar-content{ padding: 18px 16px 24px; }
section[data-testid="stSidebar"] h3{ margin:6px 0 0 0; color:#A36464; font-weight:700; letter-spacing:.2px; }
section[data-testid="stSidebar"] .cap{ margin:2px 0 14px 0; color:#777E90; font-size:13px; }

/* Radio → pill + spacing */
section[data-testid="stSidebar"] [data-testid="stRadio"]{ margin-top:10px; }
section[data-testid="stSidebar"] [role="radiogroup"] > div{ margin:6px 0; }
section[data-testid="stSidebar"] [role="radio"]{
  border:1px solid #EFE7FF; background:#FFFFFFE6; color:#40465A;
  padding:8px 14px; border-radius:999px; box-shadow:0 4px 10px #00000008;
}
section[data-testid="stSidebar"] [role="radio"] p{ font-size:14px; font-weight:600; margin:0; }
section[data-testid="stSidebar"] [role="radio"][aria-checked="true"]{
  background: linear-gradient(90deg, #FFB9D3 0%, #F6C7FF 100%);
  color:#FFFFFF !important; border-color:transparent;
}
section[data-testid="stSidebar"] [role="radio"][aria-checked="true"] p{ color:#FFFFFF !important; }

.sidebar-thanks{ color:#7E859A; font-size:12px; margin-top:10px }

/* ---- Containers & spacing ---- */
.block-card{
  background:#FFFFFFE6; border:1px solid #ffffff66; border-radius:22px; padding:22px;
  box-shadow:0 10px 24px #00000012; margin-top:8px;
}
.badge{ display:inline-block; padding:6px 14px; border-radius:999px; background:#e9e2ff; color:#4d3aa6; font-weight:600; font-size:12px; }
.mt-8{ margin-top:8px; } .mt-16{ margin-top:16px; } .mt-24{ margin-top:24px; }

/* ---- File uploader: scoped supaya tidak bikin “garis putih” di halaman lain ---- */
.uploader-scope [data-testid="stFileUploaderDropzone"]{
  background:#FFFFFFE6 !important; border:0 !important; box-shadow:none !important;
  border-radius:20px !important; padding:14px 18px !important;
}

/* Home category card: teks saja */
.cat-card{
  display:flex; align-items:center; justify-content:center;
  height:110px; border-radius:22px; background:#FFFFFF; border:1px solid #00000010;
  font-weight:700; font-size:22px; color:#9AA0AF;
}
.cat-card:hover{ box-shadow:0 12px 20px #00000012; transform: translateY(-2px); transition:all .2s; }

/* Result panels */
.result-panel{ background:#f2f8ffaa; border:1px solid #cfe2ff; border-radius:16px; padding:14px; min-height:120px; }
.result-panel-green{ background:#eafff1aa; border:1px solid #b6ffd3; border-radius:16px; padding:14px; min-height:220px; }
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "stats" not in st.session_state:
    st.session_state.stats = {c: 0 for c in ["Animal", "Fashion", "Food", "Nature", "total"]}
if "last_results" not in st.session_state:
    st.session_state.last_results = []   # list[(label, prob)]

CLASS_NAMES = ["Animal", "Fashion", "Food", "Nature"]  # urutan output layer model .h5

# =========================
# LOAD MODELS
# =========================
@st.cache_resource(show_spinner=True)
def load_models():
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
# SIDEBAR
# =========================
st.sidebar.markdown("### Features That")
st.sidebar.markdown("<div class='cap'>Can Be Used</div>", unsafe_allow_html=True)

labels = [
    "🏠 Home", "🖼️ Image Detection", "🧪 Image Classification",
    "📊 Statistics", "🗂️ Dataset", "ℹ️ About"
]
label2key = {
    "🏠 Home":"Home",
    "🖼️ Image Detection":"Image Detection",
    "🧪 Image Classification":"Image Classification",
    "📊 Statistics":"Statistics",
    "🗂️ Dataset":"Dataset",
    "ℹ️ About":"About",
}
choice = st.sidebar.radio("", labels, index=0, key="nav_radio")
menu = label2key[choice]
st.sidebar.markdown("<div class='sidebar-thanks'>Thank you for using this website</div>", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
def read_image(file): return Image.open(file).convert("RGB")

def add_stats(label, prob):
    st.session_state.last_results.append((label, prob))
    st.session_state.stats[label] = st.session_state.stats.get(label, 0) + 1
    st.session_state.stats["total"] += 1

# =========================
# HOME
# =========================
if menu == "Home":
    st.markdown("## WELCOME TO MY IMAGE DETECTION")
    st.markdown("Welcome to Bulandari’s image detection website! Choose the features that best suit your needs.")
    st.markdown("<div class='mt-16'></div>", unsafe_allow_html=True)

    st.markdown("You can use this website to detect images by theme:")
    st.markdown("<div class='mt-8'></div>", unsafe_allow_html=True)

    # Kartu kategori teks saja (tanpa gambar)
    col1, col2, col3, col4 = st.columns(4, gap="large")
    with col1:
        if st.button("🐾 Animal", use_container_width=True): pass
        st.markdown("<div class='cat-card'>Animal</div>", unsafe_allow_html=True)
    with col2:
        if st.button("👗 Fashion", use_container_width=True): pass
        st.markdown("<div class='cat-card'>Fashion</div>", unsafe_allow_html=True)
    with col3:
        if st.button("🍰 Food", use_container_width=True): pass
        st.markdown("<div class='cat-card'>Food</div>", unsafe_allow_html=True)
    with col4:
        if st.button("🌲 Nature", use_container_width=True): pass
        st.markdown("<div class='cat-card'>Nature</div>", unsafe_allow_html=True)

# =========================
# IMAGE DETECTION (YOLO)
# =========================
elif menu == "Image Detection":
    st.markdown("## UPLOAD IMAGE")
    st.caption("Insert the image according to what you want (jpg/png)")
    st.markdown("<div class='mt-8'></div>", unsafe_allow_html=True)

    # uploader hanya di halaman ini → tidak ada “garis putih” di tempat lain
    st.markdown("<div class='uploader-scope'>", unsafe_allow_html=True)
    file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='mt-16'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1], gap="large")

    if not MODELS_READY:
        st.error(f"Model belum siap: {load_err}")

    if file is not None:
        img = read_image(file)
        c1.image(img, caption="Input", use_container_width=True)
    else:
        c1.info("Silakan unggah gambar terlebih dahulu.")

    with c2:
        st.markdown("<div class='result-panel-green'><b>Result</b><br>", unsafe_allow_html=True)
        detected_list = []
        if MODELS_READY and file is not None:
            results = yolo_model(img)
            plot = results[0].plot()  # ndarray
            st.image(plot, caption="Hasil Deteksi", use_container_width=True)
            names = results[0].names if hasattr(results[0], "names") else yolo_model.names
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int).tolist() if len(results[0].boxes) else []
            for cid in cls_ids:
                detected_list.append(names.get(cid, f"id:{cid}"))
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("**Detected objects:**", ", ".join(detected_list) if detected_list else "—")

# =========================
# IMAGE CLASSIFICATION (TensorFlow)
# =========================
elif menu == "Image Classification":
    st.markdown("## UPLOAD IMAGE")
    st.caption("Insert the image according to what you want (jpg/png)")
    st.markdown("<div class='mt-8'></div>", unsafe_allow_html=True)

    st.markdown("<div class='uploader-scope'>", unsafe_allow_html=True)
    file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='mt-16'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1], gap="large")
    label_out, prob_out = None, None

    if not MODELS_READY:
        st.error(f"Model belum siap: {load_err}")

    if file is not None:
        img = read_image(file)
        c1.image(img, caption="Input", use_container_width=True)
        if MODELS_READY:
            # sesuaikan ukuran jika modelmu dilatih selain 224
            arr = kimage.img_to_array(img.resize((224, 224)))
            arr = np.expand_dims(arr, axis=0) / 255.0
            pred = classifier.predict(arr, verbose=0)
            if pred.ndim == 2 and pred.shape[1] == len(CLASS_NAMES):
                idx = int(np.argmax(pred[0]))
                label_out = CLASS_NAMES[idx]
                prob_out = float(np.max(pred[0]))
            else:
                idx = int(np.argmax(pred))
                label_out = CLASS_NAMES[idx % len(CLASS_NAMES)]
                prob_out = float(np.max(pred))
            add_stats(label_out, prob_out)
    else:
        c1.info("Silakan unggah gambar terlebih dahulu.")

    with c2:
        st.markdown("<div class='result-panel'><b>Result</b><br>", unsafe_allow_html=True)
        if label_out is not None:
            st.metric(label="Predicted class", value=label_out)
            st.write(f"Confidence: **{prob_out*100:.2f}%**")
        else:
            st.write("—")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# STATISTICS
# =========================
elif menu == "Statistics":
    st.markdown("## SESSION STATISTICS")
    st.caption("Displays the total number of processes you have completed.")
    st.markdown("<div class='mt-16'></div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='block-card'>", unsafe_allow_html=True)
        df = pd.DataFrame({
            "Category": list(st.session_state.stats.keys()),
            "Count": list(st.session_state.stats.values())
        })
        st.bar_chart(df.set_index("Category"))
        st.markdown("<div class='mt-16'></div>", unsafe_allow_html=True)
        st.write("### Summary")
        for k, v in st.session_state.stats.items():
            st.write(f"- **{k}** — {v}")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# DATASET
# =========================
elif menu == "Dataset":
    st.markdown("## DATASET")
    st.caption("Upload sample images to build a dataset or review the files you've processed in this session.")
    st.markdown("<div class='mt-16'></div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='block-card'>", unsafe_allow_html=True)
        st.write("**Last results (session):**")
        if st.session_state.last_results:
            for i, (lab, p) in enumerate(st.session_state.last_results[-50:], start=1):
                st.write(f"{i}. {lab} — {p*100:.2f}%")
        else:
            st.write("—")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# ABOUT
# =========================
elif menu == "About":
    st.markdown("## 🌸 ABOUT")
    st.markdown("<div class='mt-8'></div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='block-card'>", unsafe_allow_html=True)
        st.write(
            "Image Detection by **Meilani Bulandari Hasibuan** — "
            "A clean UI for performing image detection and classification. "
            "Uses **ultralytics YOLO** for detection and **TensorFlow** for classification. "
            "Ready to deploy to **Streamlit Cloud**."
        )
        st.markdown("</div>", unsafe_allow_html=True)
