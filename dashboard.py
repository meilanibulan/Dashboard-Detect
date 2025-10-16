# dashboard.py
# Vision Studio by Meilani — Pastel dashboard (Image Detection & Classification)
import streamlit as st
from pathlib import Path
import io
import time
from PIL import Image, ImageOps
import numpy as np

# Optional heavy libs (import safely)
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    _ultralytics_err = str(e)

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as keras_image
except Exception as e:
    tf = None
    keras_image = None
    _tf_err = str(e)

# ---------------------------------------------------------------------
# Config - change these model paths if needed
MODEL_YOLO_PATH = "model/Meilani Bulandari Hsb_Laporan 4.pt"
MODEL_CLASS_PATH = "model/Meilani Bulandari Hsb_Laporan 2.h5"
CLASS_LABELS = ["Animal", "Fashion", "Food", "Nature"]  # change to your model's class order
# ---------------------------------------------------------------------

st.set_page_config(page_title="Vision Studio by Meilani", page_icon="🌸", layout="wide")

# ----------------------
# Styles (Poppins + pastel UI)
# ----------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    :root{
      --bg: #FBF7F6;            /* main page bg (light pastel) */
      --panel: #FFFFFF;         /* card background */
      --muted: #7c7c86;
      --sidebar-bg: #0F1720;    /* dark sidebar */
      --accent-pink: #FFBBCB;   /* pink card */
      --accent-yellow: #FFF2C7; /* yellow */
      --accent-green: #DFF7EE;  /* green */
      --accent-blue: #DCEFFD;   /* blue */
      --soft-border: #EFECEC;
      --card-shadow: 0 8px 24px rgba(16,24,40,0.06);
    }

    html, body, [data-testid="stAppViewContainer"] { background: var(--bg); font-family: 'Poppins', sans-serif; }
    .block-container{ padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1400px; }

    /* Sidebar styling */
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, var(--sidebar-bg), #0C1319);
        color: white;
        border-right: 0;
        padding-top: 18px;
    }
    .sidebar .stRadio > label { color: #fff; }

    /* Cards */
    .card {
        background: var(--panel);
        border-radius: 14px;
        padding: 18px;
        box-shadow: var(--card-shadow);
        border: 1px solid var(--soft-border);
    }
    .card.compact { padding: 12px; }
    .card-title { font-weight: 600; color: #222; margin-bottom:8px; }
    .muted { color: var(--muted); font-size: 0.95rem; }

    /* uploader */
    .upload-box {
        border: 1px dashed #F2DFE6;
        background: linear-gradient(180deg, #FFF7F8, #FFFFFF);
        padding: 14px;
        border-radius: 12px;
        text-align: center;
    }

    /* run button */
    .run-btn {
        background: linear-gradient(90deg, #FF9DB2, #FF7FA6);
        color: white; font-weight:700; padding:10px 16px; border-radius:12px; border:none;
    }
    .run-btn:hover { opacity: .97; cursor:pointer; }

    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------
# Model loaders (safe)
# ----------------------
@st.cache_resource(show_spinner=True)
def load_yolo(path: str = MODEL_YOLO_PATH):
    if YOLO is None:
        return None, f"ultralytics not installed: {_ultralytics_err if '_ultralytics_err' in globals() else ''}"
    if not Path(path).exists():
        return None, f"YOLO model file not found: {path}"
    try:
        model = YOLO(path)
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_resource(show_spinner=True)
def load_classifier(path: str = MODEL_CLASS_PATH):
    if tf is None:
        return None, f"tensorflow not installed: {_tf_err if '_tf_err' in globals() else ''}"
    if not Path(path).exists():
        return None, f"Classifier model file not found: {path}"
    try:
        clf = tf.keras.models.load_model(path)
        return clf, None
    except Exception as e:
        return None, str(e)

yolo_model, yolo_err = load_yolo()
classifier, clf_err = load_classifier()

# ----------------------
# Session stats
# ----------------------
if "stats" not in st.session_state:
    st.session_state.stats = {"total": 0, **{k: 0 for k in CLASS_LABELS}}

if "last_results" not in st.session_state:
    st.session_state.last_results = []  # store small history: tuples (label, prob)

# ----------------------
# Sidebar content
# ----------------------
with st.sidebar:
    st.markdown("<div style='padding:8px 12px'>"
                "<h2 style='margin:4px 0 2px 0; color: white;'>Vision Studio</h2>"
                "<div style='color:#ffb6c1; font-weight:600'>by Meilani</div>"
                "</div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", ("Dashboard", "Image Detection", "Image Classification", "Statistics", "Dataset", "About"))
    st.markdown("---")
    st.markdown("<div style='padding:8px 12px'><div style='color:#fff;font-weight:600;margin-bottom:6px'>Model Status</div>", unsafe_allow_html=True)
    model_ok = (yolo_model is not None) and (classifier is not None)
    st.markdown(f"<div class='card compact'><div style='font-weight:700'>{'OK' if model_ok else 'Missing'}</div><div class='muted'>YOLO & Classifier</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#ddd;font-size:13px'>Contact: meilani@example.com</div>", unsafe_allow_html=True)

# ----------------------
# Helper utilities
# ----------------------
def pil_from_array(arr):
    """Safe conversion numpy array (RGB/BGR) -> PIL"""
    try:
        if arr.dtype != np.uint8:
            arr = (arr * 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 3:
            # Try to create PIL directly (assume RGB)
            return Image.fromarray(arr)
        else:
            return Image.fromarray(arr)
    except Exception:
        # fallback: ensure uint8 and try again
        arr2 = (np.clip(arr, 0, 255)).astype(np.uint8)
        return Image.fromarray(arr2)

def run_yolo_on_pil(model, pil_img):
    """Run YOLO and return plotted PIL image and summary list"""
    results = model(pil_img)
    plotted = results[0].plot()  # many ultralytics return numpy RGB or BGR
    # try convert robustly
    try:
        plotted_pil = pil_from_array(plotted)
    except Exception:
        # if BGR, convert by reversing channel
        try:
            plotted_pil = Image.fromarray(plotted[:, :, ::-1])
        except Exception:
            # as last resort convert via numpy cast
            plotted_pil = pil_from_array(np.array(plotted))
    # get summary
    boxes = getattr(results[0], "boxes", None)
    names = getattr(results[0], "names", {})
    summary = []
    if boxes is not None and len(boxes) > 0:
        try:
            cls_list = boxes.cls.tolist()
            conf_list = boxes.conf.tolist()
            for cls_id, conf in zip(cls_list, conf_list):
                label = names[int(cls_id)] if int(cls_id) in names else str(int(cls_id))
                summary.append((label, float(conf)))
        except Exception:
            pass
    return plotted_pil, summary

def run_classifier_on_pil(clf, pil_img):
    """Run classifier and return (label, prob). Auto-detect input size when possible."""
    target_size = (224, 224)
    try:
        if clf is not None and hasattr(clf, "input_shape"):
            ish = clf.input_shape
            # ish often is (None, H, W, C) or (None, features)
            if isinstance(ish, tuple) and len(ish) >= 3:
                if ish[1] and ish[2]:
                    target_size = (int(ish[1]), int(ish[2]))
    except Exception:
        target_size = (224, 224)

    img_resized = pil_img.convert("RGB").resize(target_size)
    if keras_image is not None:
        arr = keras_image.img_to_array(img_resized)
    else:
        arr = np.array(img_resized).astype(np.float32)
    arr = np.expand_dims(arr, 0) / 255.0
    preds = clf.predict(arr)
    idx = int(np.argmax(preds))
    prob = float(np.max(preds))
    label = CLASS_LABELS[idx] if idx < len(CLASS_LABELS) else f"Class {idx}"
    return label, prob

# ----------------------
# PAGES
# ----------------------

# Dashboard page
if page == "Dashboard":
    # header / welcome
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("<div class='card'><div class='card-title'>Welcome to My Image Detection 💗</div>"
                    "<div class='muted'>Monitor detection runs, classification results and session statistics.</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card compact'><div class='card-title'>Processed</div>"
                    f"<div style='font-weight:700; font-size:20px'>{st.session_state.stats['total']}</div>"
                    "<div class='muted'>total images this session</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # cards row: summary pink, bar yellow, distribution green, activity blue
    a, b = st.columns([2, 1])
    with a:
        st.markdown("<div class='card'><div class='card-title'>Latest Result</div>", unsafe_allow_html=True)
        if st.session_state.last_results:
            last = st.session_state.last_results[-1]
            lbl, p = last
            color = {"Animal":"#CFF2D6","Fashion":"#EED6F6","Food":"#FFE6D1","Nature":"#D6F2E8"}.get(lbl, "#FFF")
            st.markdown(f"<div style='background:{color}; padding:14px; border-radius:12px'>"
                        f"<div style='font-weight:700; font-size:18px'>{lbl}</div>"
                        f"<div class='muted'>Confidence: {p*100:.2f}%</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='muted'>No results yet — run image detection or classification to populate this card.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown("<div class='card'><div class='card-title'>Session Categories</div>", unsafe_allow_html=True)
        # small summary values
        for k in CLASS_LABELS:
            st.markdown(f"<div style='display:flex;justify-content:space-between;margin-bottom:6px'><div class='muted'>{k}</div><div style='font-weight:700'>{st.session_state.stats.get(k,0)}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Image Detection page
elif page == "Image Detection":
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("<div class='card'><div class='card-title'>Upload Image</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Select an image (jpg/png)", type=["jpg", "jpeg", "png"], key="up_detect", label_visibility="visible")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='muted'>Choose an image and press Run to detect objects.</div>", unsafe_allow_html=True)
        run = st.button("Run Detection", key="run_detect", help="Run object detection")
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded:
            try:
                pil_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                st.image(pil_img, use_column_width=True, caption="Preview")
            except Exception as e:
                st.error(f"Failed to open image: {e}")
                pil_img = None
        else:
            pil_img = None

    with right:
        st.markdown("<div class='card'><div class='card-title'>Results</div>", unsafe_allow_html=True)
        if not pil_img:
            st.markdown("<div class='muted'>Upload an image on the left and press Run Detection.</div>", unsafe_allow_html=True)
        else:
            if yolo_model is None:
                st.warning(f"YOLO not available: {yolo_err}")
            else:
                if run:
                    with st.spinner("Running YOLO..."):
                        try:
                            plotted_pil, summary = run_yolo_on_pil(yolo_model, pil_img)
                            st.image(plotted_pil, use_column_width=True, caption="Detections")
                            if summary:
                                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                                st.markdown("**Detected objects:**")
                                for lab, conf in summary:
                                    st.write(f"- {lab} — {conf:.2f}")
                            else:
                                st.info("No objects detected.")
                            # update stats total (we do not map YOLO labels to our 4 classes automatically)
                            st.session_state.stats["total"] += 1
                        except Exception as e:
                            st.error(f"YOLO error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

# Image Classification page
elif page == "Image Classification":
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("<div class='card'><div class='card-title'>Upload Image</div>", unsafe_allow_html=True)
        uploaded2 = st.file_uploader("Select an image (jpg/png)", type=["jpg", "jpeg", "png"], key="up_class", label_visibility="visible")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='muted'>Choose an image and press Run to classify into Animal, Fashion, Food, or Nature.</div>", unsafe_allow_html=True)
        run2 = st.button("Run Classification", key="run_class")
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded2:
            try:
                pil_img2 = Image.open(io.BytesIO(uploaded2.read())).convert("RGB")
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                st.image(pil_img2, use_column_width=True, caption="Preview")
            except Exception as e:
                st.error(f"Failed to open image: {e}")
                pil_img2 = None
        else:
            pil_img2 = None

    with right:
        st.markdown("<div class='card'><div class='card-title'>Results</div>", unsafe_allow_html=True)
        if not pil_img2:
            st.markdown("<div class='muted'>Upload an image on the left and press Run Classification.</div>", unsafe_allow_html=True)
        else:
            if classifier is None:
                st.warning(f"Classifier not available: {clf_err}")
            else:
                if run2:
                    with st.spinner("Running classifier..."):
                        try:
                            label, prob = run_classifier_on_pil(classifier, pil_img2)
                            color_map = {"Animal":"#CFF2D6","Fashion":"#EED6F6","Food":"#FFE6D1","Nature":"#D6F2E8"}
                            bg = color_map.get(label, "#FFF")
                            st.markdown(f"<div style='background:{bg}; padding:14px; border-radius:12px;'>"
                                        f"<div style='font-weight:700; font-size:18px'>{label}</div>"
                                        f"<div class='muted'>Confidence: {prob*100:.2f}%</div></div>", unsafe_allow_html=True)
                            # update stats
                            st.session_state.stats["total"] += 1
                            if label in st.session_state.stats:
                                st.session_state.stats[label] += 1
                            st.session_state.last_results.append((label, prob))
                        except Exception as e:
                            st.error(f"Classifier error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

# Statistics page
elif page == "Statistics":
    st.markdown("<div class='card'><div class='card-title'>Session Statistics</div>", unsafe_allow_html=True)
    stats = st.session_state.stats
    st.write(f"- Total processed: **{stats['total']}**")
    col1, col2 = st.columns([2,1])
    with col1:
        # bar chart using st.bar_chart or altair
        try:
            import pandas as pd
            df = pd.DataFrame({
                "category": CLASS_LABELS,
                "count": [stats.get(k,0) for k in CLASS_LABELS]
            })
            st.bar_chart(df.set_index("category"))
        except Exception:
            st.write({k: stats.get(k,0) for k in CLASS_LABELS})
    with col2:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        for k in CLASS_LABELS:
            st.markdown(f"<div style='display:flex;justify-content:space-between;margin-bottom:6px'><div class='muted'>{k}</div><div style='font-weight:700'>{stats.get(k,0)}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Dataset page (simple)
elif page == "Dataset":
    st.markdown("<div class='card'><div class='card-title'>Dataset</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Upload sample images to build a dataset or review the files you've processed in this session.</div>", unsafe_allow_html=True)
    if st.session_state.last_results:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.write("Last results (session):")
        for i, (lbl, p) in enumerate(reversed(st.session_state.last_results[-10:])):
            st.write(f"{i+1}. {lbl} — {p*100:.2f}%")
    else:
        st.info("No dataset samples in session yet.")
    st.markdown("</div>", unsafe_allow_html=True)

# About page
else:
    st.markdown("<div class='card'><div class='card-title'>About Vision Studio</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Vision Studio by Meilani — a pastel, clean UI to run image detection and classification. "
                "Uses ultralytics YOLO for detection and TensorFlow for classification. "
                "Deploy-ready for Streamlit Cloud. Change model paths at top of script if needed.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# footer
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#777;font-size:12px'>Made with ♥ — Vision Studio</div>", unsafe_allow_html=True)
