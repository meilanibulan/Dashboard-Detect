import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
import numpy as np
from PIL import Image
import pandas as pd

# ========= Page config =========
st.set_page_config(
    page_title="Image Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========= Global styles (Pastel palette) =========
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
* { font-family:'Poppins', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }

/* App background: smooth vertical gradient from the palette */
.stApp{
  background: linear-gradient(
    180deg,
    #FF99C8 0%,
    #E8C0FC 22%,
    #A8DEFA 45%,
    #D0F4E0 70%,
    #FCF5BF 100%
  );
}

/* Sidebar gradient + pill nav */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #FCF5BF 0%, #D0F4E0 35%, #A8DEFA 65%, #E8C0FC 100%) !important;
  border-right: 1px solid #ffffff55;
}
section[data-testid="stSidebar"] .sidebar-content{ padding:18px 16px 24px; }
section[data-testid="stSidebar"] h3{ margin:6px 0 0 0; color:#5e5a6b; font-weight:800; letter-spacing:.2px; }
section[data-testid="stSidebar"] .cap{ margin:2px 0 14px 0; color:#6f758a; font-size:13px; }

section[data-testid="stSidebar"] [data-testid="stRadio"]{ margin-top:10px; }
section[data-testid="stSidebar"] [role="radiogroup"] > div{ margin:6px 0; }
section[data-testid="stSidebar"] [role="radio"]{
  border:1px solid #ffffff80; background:#ffffffcc; color:#40465A;
  padding:8px 14px; border-radius:999px; box-shadow:0 4px 10px #0000000d;
}
section[data-testid="stSidebar"] [role="radio"] p{ font-size:14px; font-weight:600; margin:0; }
section[data-testid="stSidebar"] [role="radio"][aria-checked="true"]{
  background: linear-gradient(90deg, #FF99C8 0%, #E8C0FC 100%);
  color:#FFFFFF !important; border-color:transparent;
}
section[data-testid="stSidebar"] [role="radio"][aria-checked="true"] p{ color:#FFFFFF !important; }
.sidebar-thanks{ color:#6f758a; font-size:12px; margin-top:10px }

/* Cards & spacing */
.block-card{
  background:#ffffffd9; border:1px solid #ffffff80; border-radius:22px; padding:22px;
  box-shadow:0 10px 24px #00000012; margin-top:8px;
}
.badge{ display:inline-block; padding:6px 14px; border-radius:999px; background:#E8C0FC; color:#40304e; font-weight:700; font-size:12px; }
.mt-8{ margin-top:8px; } .mt-16{ margin-top:16px; } .mt-24{ margin-top:24px; }

/* Hide ALL uploaders globally… */
[data-testid="stFileUploaderDropzone"]{ display:none !important; }
/* …except the ones we scope explicitly */
.uploader-scope [data-testid="stFileUploaderDropzone"]{
  display:block !important;
  background:#ffffff; border:0 !important; box-shadow:none !important;
  border-radius:20px !important; padding:14px 18px !important;
}

/* Home category: text-only tiles */
.cat-card{
  display:flex; align-items:center; justify-content:center;
  height:110px; border-radius:22px; background:#ffffff; border:1px solid #00000010;
  font-weight:700; font-size:22px; color:#8a8fa3;
}
.cat-card:hover{ box-shadow:0 12px 20px #00000012; transform: translateY(-2px); transition:all .2s; }

/* Result panels */
.result-panel{ background:#D0F4E0cc; border:1px solid #c2e9d0; border-radius:16px; padding:14px; min-height:120px; }
.result-panel-green{ background:#A8DEFA33; border:1px solid #A8DEFA; border-radius:16px; padding:14px; min-height:220px; }

/* Headings */
h2, h3 { letter-spacing:.1px; color:#2e3140; }
</style>
""", unsafe_allow_html=True)

# ========= Session state =========
if "stats" not in st.session_state:
    st.session_state.stats = {k: 0 for k in ["Animal", "Fashion", "Food", "Nature", "total"]}
if "last_results" not in st.session_state:
    st.session_state.last_results = []   # list[(label, prob)]

CLASS_NAMES = ["Animal", "Fashion", "Food", "Nature"]  # urutkan sesuai model

# ========= Models =========
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

# ========= Sidebar =========
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

# ========= Helpers =========
def read_image(file): return Image.open(file).convert("RGB")

def add_stats(label, prob):
    st.session_state.last_results.append((label, prob))
    st.session_state.stats[label] = st.session_state.stats.get(label, 0) + 1
    st.session_state.stats["total"] += 1

def prepare_for_model(pil_img, model):
    """
    Resize & format image to match model.input_shape.
    Supports C=1 (grayscale) or C=3 (RGB). Returns (batch_array, (H,W,C)).
    """
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    _, H, W, C = input_shape
    H = H or 224; W = W or 224; C = C or 3

    img = pil_img.convert("RGB").resize((W, H))
    arr = kimage.img_to_array(img)  # (H,W,3)

    if C == 1:
        arr = np.dot(arr[...,:3], [0.2989, 0.5870, 0.1140]).astype("float32")  # (H,W)
        arr = np.expand_dims(arr, -1)  # (H,W,1)
    elif C == 3:
        pass
    else:
        arr = np.resize(arr, (H, W, C))

    arr = arr / 255.0
    arr = np.expand_dims(arr, 0)  # (1,H,W,C)
    return arr, (H, W, C)

# ========= Pages =========
if menu == "Home":
    st.markdown("## WELCOME TO MY IMAGE DETECTION")
    st.markdown("Welcome to Bulandari’s image detection website! Choose the features that best suit your needs.")
    st.markdown("<div class='mt-16'></div>", unsafe_allow_html=True)
    st.markdown("You can use this website to detect images by theme:")
    st.markdown("<div class='mt-8'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        st.button("🐾 Animal", use_container_width=True)
        st.markdown("<div class='cat-card'>Animal</div>", unsafe_allow_html=True)
    with c2:
        st.button("👗 Fashion", use_container_width=True)
        st.markdown("<div class='cat-card'>Fashion</div>", unsafe_allow_html=True)
    with c3:
        st.button("🍰 Food", use_container_width=True)
        st.markdown("<div class='cat-card'>Food</div>", unsafe_allow_html=True)
    with c4:
        st.button("🌲 Nature", use_container_width=True)
        st.markdown("<div class='cat-card'>Nature</div>", unsafe_allow_html=True)

elif menu == "Image Detection":
    st.markdown("## UPLOAD IMAGE")
    st.caption("Insert the image according to what you want (jpg/png)")
    st.markdown("<div class='mt-8'></div>", unsafe_allow_html=True)

    st.markdown("<div class='uploader-scope'>", unsafe_allow_html=True)
    file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='mt-16'></div>", unsafe_allow_html=True)

    colL, colR = st.columns([1.2, 1], gap="large")

    if not MODELS_READY:
        st.error(f"Model belum siap: {load_err}")

    if file is not None:
        img = read_image(file)
        colL.image(img, caption="Input", use_container_width=True)
    else:
        colL.info("Silakan unggah gambar terlebih dahulu.")

    with colR:
        st.markdown("<div class='result-panel-green'><b>Result</b><br>", unsafe_allow_html=True)
        detected = []
        if MODELS_READY and file is not None:
            results = yolo_model(img)
            plot = results[0].plot()  # ndarray (BGR)
            st.image(plot, caption="Hasil Deteksi", use_container_width=True)
            names = results[0].names if hasattr(results[0], "names") else yolo_model.names
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int).tolist() if len(results[0].boxes) else []
            detected = [names.get(cid, f"id:{cid}") for cid in cls_ids]
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("**Detected objects:**", ", ".join(detected) if detected else "—")

elif menu == "Image Classification":
    st.markdown("## UPLOAD IMAGE")
    st.caption("Insert the image according to what you want (jpg/png)")
    st.markdown("<div class='mt-8'></div>", unsafe_allow_html=True)

    st.markdown("<div class='uploader-scope'>", unsafe_allow_html=True)
    file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='mt-16'></div>", unsafe_allow_html=True)

    colL, colR = st.columns([1.2, 1], gap="large")
    label_out, prob_out = None, None

    if not MODELS_READY:
        st.error(f"Model belum siap: {load_err}")

    if file is not None:
        img = read_image(file)
        colL.image(img, caption="Input", use_container_width=True)
        if MODELS_READY:
            try:
                arr, target = prepare_for_model(img, classifier)
                pred = classifier.predict(arr, verbose=0)

                if pred.ndim == 2 and pred.shape[1] == len(CLASS_NAMES):
                    idx = int(np.argmax(pred[0]))
                    label_out = CLASS_NAMES[idx]
                    prob_out = float(np.max(pred[0]))
                elif pred.ndim == 2 and pred.shape[1] == 1:
                    prob_out = float(pred[0][0])
                    label_out = CLASS_NAMES[1] if prob_out >= 0.5 else CLASS_NAMES[0]
                else:
                    idx = int(np.argmax(pred))
                    label_out = CLASS_NAMES[idx % len(CLASS_NAMES)]
                    prob_out = float(np.max(pred))

                add_stats(label_out, prob_out)
            except Exception as e:
                st.error("Klasifikasi gagal. Kemungkinan besar ukuran/kanal input tidak sesuai.")
                st.code(
                    f"Expected input (H,W,C): {classifier.input_shape} • "
                    f"Used: {target if 'target' in locals() else 'unknown'}\nDetail: {e}"
                )
    else:
        colL.info("Silakan unggah gambar terlebih dahulu.")

    with colR:
        st.markdown("<div class='result-panel'><b>Result</b><br>", unsafe_allow_html=True)
        if label_out is not None:
            st.metric(label="Predicted class", value=label_out)
            st.write(f"Confidence: **{prob_out*100:.2f}%**")
        else:
            st.write("—")
        st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Statistics":
    st.markdown("## SESSION STATISTICS")
    st.caption("Displays the total number of processes you have completed.")
    st.markdown("<div class='mt-16'></div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='block-card'>", unsafe_allow_html=True)
        st.write("### Summary")
        for k in ["Animal", "Fashion", "Food", "Nature"]:
            st.write(f"- **{k}** — {st.session_state.stats.get(k, 0)}")
        st.write(f"- **total** — {st.session_state.stats.get('total', 0)}")
        st.markdown("</div>", unsafe_allow_html=True)

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
