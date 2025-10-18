import streamlit as st
import numpy as np
from PIL import Image

# ====== CONFIG ======
st.set_page_config(page_title="Vision Studio", page_icon="🌸", layout="wide")

PALETTE = {
    "pink":  "#FF99C8",
    "lilac": "#E8C0FC",
    "sky":   "#A8DEFA",
    "mint":  "#D0F4E0",
    "butter":"#FCF5BF",
    "ink":   "#2e3140",
    "muted": "#6f758a",
}

# ====== THEME (clean, baru) ======
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
* {{ font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
.stApp {{
  background: linear-gradient(180deg, {PALETTE['pink']} 0%, {PALETTE['lilac']} 22%,
                                     {PALETTE['sky']} 45%, {PALETTE['mint']} 70%, {PALETTE['butter']} 100%);
}}
/* Sidebar */
section[data-testid="stSidebar"]{{
  background: linear-gradient(180deg, {PALETTE['butter']} 0%, {PALETTE['mint']} 35%, {PALETTE['sky']} 65%, {PALETTE['lilac']} 100%) !important;
  border-right: 1px solid #ffffff66;
}}
section[data-testid="stSidebar"] .sidebar-content{{ padding:18px 16px 24px; }}
section[data-testid="stSidebar"] [role="radio"]{{
  border:1px solid #ffffff88; background:#ffffffcc; border-radius:999px;
  padding:8px 14px; margin:6px 0; box-shadow:0 4px 10px #0000000d;
}}
section[data-testid="stSidebar"] [role="radio"][aria-checked="true"]{{
  background:linear-gradient(90deg, {PALETTE['pink']} 0%, {PALETTE['lilac']} 100%); color:#fff;
  border-color:transparent;
}}
.sidebar-note{{ color:{PALETTE['muted']}; font-size:12px; margin-top:10px }}

/* Cards / spacing */
.card{{ background:#fff; border:1px solid #ffffff90; border-radius:22px; padding:18px;
       box-shadow:0 12px 30px rgba(0,0,0,.08); }}
.row{{ display:grid; gap:16px; }}
.row.cols-2{{ grid-template-columns: 1fr 1fr; }}
.row.cols-3{{ grid-template-columns: 1fr 1fr 1fr; }}
.mt-8{{ margin-top:8px; }} .mt-16{{ margin-top:16px; }} .mt-24{{ margin-top:24px; }}

/* Uploader: hanya tampil jika dibungkus scope */
[data-testid="stFileUploaderDropzone"]{{ display:none !important; }}
.uploader [data-testid="stFileUploaderDropzone"]{{ display:block !important; background:#fff;
  border:0; border-radius:18px; padding:14px 18px; box-shadow:none; }}
h1,h2,h3{{ color:{PALETTE['ink']}; letter-spacing:.1px; }}
.small{{ color:{PALETTE['muted']}; font-size:12px; }}
</style>
""", unsafe_allow_html=True)

# ====== NAV ======
st.sidebar.markdown("### Vision Studio")
menu = st.sidebar.radio(
    "", ["🏠 Home", "🖼️ Detect", "🧪 Classify", "ℹ️ About"], index=0
)
st.sidebar.markdown("<div class='sidebar-note'>Welcome 🌸</div>", unsafe_allow_html=True)

# ====== STATE ======
if "history" not in st.session_state:
    st.session_state.history = []  # (type, result)

# ====== PLACEHOLDERS for models (optional to load later) ======
def load_detector():
    """Return YOLO model or None if not available."""
    try:
        from ultralytics import YOLO
        return YOLO("model/detector.pt")
    except Exception:
        return None

def load_classifier():
    """Return tf model or None."""
    try:
        import tensorflow as tf
        return tf.keras.models.load_model("model/classifier.h5")
    except Exception:
        return None

DETECTOR = None
CLASSIFIER = None

# ====== UTIL ======
def prepare_for_model(pil_img, model):
    """
    Resize & format image to match Keras model.input_shape.
    Supports C=1/3. Returns batch array and (H,W,C).
    """
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as kimage
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    _, H, W, C = input_shape
    H = H or 224; W = W or 224; C = C or 3
    img = pil_img.convert("RGB").resize((W, H))
    arr = kimage.img_to_array(img)
    if C == 1:
        arr = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140]).astype("float32")
        arr = np.expand_dims(arr, -1)
    elif C != 3:
        arr = np.resize(arr, (H, W, C))
    arr = arr / 255.0
    return np.expand_dims(arr, 0), (H, W, C)

# ====== PAGES ======
if menu == "🏠 Home":
    st.markdown("## Welcome ✨")
    st.markdown("Template baru yang ringan & modular. Mulai dari sini.")
    st.markdown("<div class='row cols-3 mt-16'>", unsafe_allow_html=True)
    # Card 1
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Quick Start**")
        st.markdown("<div class='small'>Upload gambar lalu pilih halaman Detect atau Classify.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    # Card 2
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Palette**")
        st.code("#E8C0FC  #A8DEFA  #D0F4E0  #FCF5BF  #FF99C8")
        st.markdown("</div>", unsafe_allow_html=True)
    # Card 3
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Session Summary**")
        if st.session_state.history:
            for i, (typ, res) in enumerate(st.session_state.history[-6:][::-1], 1):
                st.write(f"{i}. {typ}: {res}")
        else:
            st.write("— belum ada aktivitas —")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "🖼️ Detect":
    st.markdown("## Object Detection")
    st.markdown("<div class='small'>Model: YOLO (opsional—boleh kosong dulu)</div>", unsafe_allow_html=True)
    st.markdown("<div class='uploader mt-16'>", unsafe_allow_html=True)
    file = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"])
    st.markdown("</div>", unsafe_allow_html=True)

    colL, colR = st.columns([1.2, 1])
    if file:
        img = Image.open(file).convert("RGB")
        colL.image(img, caption="Input", use_container_width=True)

        if DETECTOR is None:
            DETECTOR = load_detector()
        if DETECTOR is None:
            colR.warning("Model YOLO belum tersedia. Letakkan file di `model/detector.pt`.")
        else:
            results = DETECTOR(img)
            out = results[0].plot()  # ndarray (BGR)
            colR.image(out, caption="Result", use_container_width=True)
            names = results[0].names if hasattr(results[0], "names") else DETECTOR.names
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int).tolist() if len(results[0].boxes) else []
            labels = [names.get(i, f"id:{i}") for i in cls_ids]
            st.session_state.history.append(("Detect", ", ".join(labels) if labels else "No objects"))
            st.write("**Detected:**", ", ".join(labels) if labels else "—")
    else:
        st.info("Silakan unggah gambar.")

elif menu == "🧪 Classify":
    st.markdown("## Image Classification")
    st.markdown("<div class='small'>Model: Keras/TensorFlow (opsional—boleh kosong dulu)</div>", unsafe_allow_html=True)
    st.markdown("<div class='uploader mt-16'>", unsafe_allow_html=True)
    file = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"], key="clf")
    st.markdown("</div>", unsafe_allow_html=True)

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Input", use_container_width=True)

        if CLASSIFIER is None:
            CLASSIFIER = load_classifier()
        if CLASSIFIER is None:
            st.warning("Model klasifikasi belum tersedia. Letakkan file di `model/classifier.h5`.")
        else:
            try:
                arr, target = prepare_for_model(img, CLASSIFIER)
                pred = CLASSIFIER.predict(arr, verbose=0)
                # Interpretasi fleksibel (multiclass / binary)
                if pred.ndim == 2 and pred.shape[1] > 1:
                    idx = int(np.argmax(pred[0]))
                    conf = float(np.max(pred[0]))
                    result = f"class_{idx} ({conf*100:.2f}%)"
                else:
                    conf = float(pred.squeeze())
                    result = f"positive {conf*100:.2f}%" if conf >= 0.5 else f"negative {(1-conf)*100:.2f}%"
                st.success(f"Prediction: **{result}**")
                st.session_state.history.append(("Classify", result))
            except Exception as e:
                st.error("Gagal memproses: cek ukuran/kanal input model.")
                st.code(f"Expected: {CLASSIFIER.input_shape} • Error: {e}")
    else:
        st.info("Silakan unggah gambar.")

elif menu == "ℹ️ About":
    st.markdown("## 🌸 About")
    st.markdown(
        "Template baru — minimal, modular, dan siap diberi fitur. "
        "Semua warna mengikuti palet pastel pilihanmu."
    )
