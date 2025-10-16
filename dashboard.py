# app.py
# Pastel Vision Dashboard (white pastel sidebar + pink accents)
import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image
import io
import time

# try import heavy libs (fail gracefully on deploy/build if missing)
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

# ==========================
# Page config & CSS
# ==========================
st.set_page_config(page_title="Vision Dashboard (Pastel)", page_icon="🌸", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    :root{
      --bg: #FAF8F8;
      --panel: #FFFFFF;
      --muted: #8C8C98;
      --accent-pink: #F9C5D1;
      --accent-pink-2: #F3A6BF;
      --soft-border: #EFECEC;
      --card-shadow: 0 8px 24px rgba(16,24,40,0.06);
    }

    html, body, [data-testid="stAppViewContainer"] { background: var(--bg); font-family: 'Poppins', sans-serif; }
    .block-container{ padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1400px; }

    /* Sidebar */
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #FFFFFF, #FFF9FB);
        border-right: 1px solid var(--soft-border);
        box-shadow: none;
        padding-top: 1rem;
    }
    .sidebar .stButton>button { background: transparent; }

    /* Card */
    .card {
        background: var(--panel);
        border-radius: 16px;
        padding: 18px;
        box-shadow: var(--card-shadow);
        border: 1px solid var(--soft-border);
    }
    .card.compact { padding: 12px; }
    .card-title { font-weight: 600; color: #333333; margin-bottom:6px; }
    .muted { color: var(--muted); font-size: 0.95rem; }

    /* uploader */
    .upload-box {
        border: 1px dashed #F2DFE6;
        background: linear-gradient(180deg, #FFF7F8, #FFFFFF);
        padding: 14px;
        border-radius: 12px;
        text-align: center;
    }

    /* buttons */
    .run-btn {
        background: linear-gradient(90deg, var(--accent-pink), var(--accent-pink-2));
        color: white; font-weight:700; padding:8px 14px; border-radius:12px; border:none;
    }
    .run-btn:hover { opacity: .95; cursor:pointer; }

    /* small KPI */
    .kpi { background: linear-gradient(180deg,#fff,#fff); padding:12px; border-radius:12px; text-align:center; }
    .kpi .val { font-weight:700; font-size:1.2rem; color:#333; }

    /* hide default streamlit footer */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# Utility: load models safely
# ==========================
@st.cache_resource
def load_yolo_model(path: str = "model/Meilani Bulandari Hsb_Laporan 4.pt"):
    if YOLO is None:
        return None, "ultralytics not installed"
    if not Path(path).exists():
        return None, f"YOLO model file not found: {path}"
    try:
        model = YOLO(path)
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_classifier(path: str = "model/Meilani Bulandari Hsb_Laporan 2.h5"):
    if tf is None:
        return None, "tensorflow not installed"
    if not Path(path).exists():
        return None, f"Classifier model file not found: {path}"
    try:
        clf = tf.keras.models.load_model(path)
        return clf, None
    except Exception as e:
        return None, str(e)

# attempt load (paths can be changed if your file names differ)
yolo_model, yolo_err = load_yolo_model("model/Meilani Bulandari Hsb_Laporan 4.pt")
classifier, clf_err = load_classifier("model/Meilani Bulandari Hsb_Laporan 2.h5")

# ==========================
# Session stats (keeps counts during app session)
# ==========================
if "stats" not in st.session_state:
    st.session_state.stats = {"total": 0, "Animal": 0, "Fashion": 0, "Food": 0, "Nature": 0}

# ==========================
# Sidebar (white pastel with pink icons)
# ==========================
with st.sidebar:
    st.markdown("<div style='padding:10px 8px;'>"
                "<h2 style='margin:6px 0 0 0;'>🌸 Vision Studio</h2>"
                "<div style='color:#777;margin-top:4px;font-size:14px;'>Image Detection & Classification</div>"
                "</div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", ["Dashboard", "Deteksi & Klasifikasi", "Statistik", "About"], index=1, format_func=lambda x: x)
    st.markdown("---")
    st.markdown("<div style='display:flex;gap:8px;align-items:center'>"
                "<div style='width:36px;height:36px;border-radius:10px;background:linear-gradient(90deg,#F9C5D1,#F3A6BF);display:flex;align-items:center;justify-content:center;color:white;font-weight:700'>AI</div>"
                "<div><b>Meilani</b><div style='color:#888;font-size:12px'>Creator</div></div>"
                "</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px;color:#666'>Model status:</div>", unsafe_allow_html=True)
    model_status = "OK" if (yolo_model and classifier) else "Missing"
    st.markdown(f"<div class='kpi' style='margin-top:8px'><div class='val'>{model_status}</div><div style='color:var(--muted);font-size:12px'>YOLO & Classifier</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.caption("Theme: Pastel • Font: Poppins")

# ==========================
# Main content (tabs-like pages)
# ==========================
if page == "Dashboard":
    # greeting + small KPIs
    c1, c2, c3 = st.columns([3,1,1])
    with c1:
        st.markdown("<div class='card'><div class='card-title'>Welcome back 👋</div>"
                    "<h2 style='margin:6px 0 6px 0;'>Vision Dashboard</h2>"
                    "<div class='muted'>Monitor detection runs and classification results</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card compact'><div class='card-title'>Processed</div>"
                    f"<div style='padding-top:8px' class='muted'><div class='val' style='font-size:20px'>{st.session_state.stats['total']}</div></div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card compact'><div class='card-title'>Top Category</div>"
                    f"<div style='padding-top:8px' class='muted'><div class='val' style='font-size:20px'>{max(['Animal','Fashion','Food','Nature'], key=lambda k: st.session_state.stats.get(k,0))}</div></div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # small gallery / last results (if any)
    colA, colB = st.columns([2,1])
    with colA:
        st.markdown("<div class='card'><div class='card-title'>Quick Actions</div>"
                    "<div class='muted'>Upload an image to run detection or classification.</div>"
                    "<div style='height:10px'></div>"
                    "<a class='run-btn' href='#deteksisection'>Run detection / classify</a>"
                    "</div>", unsafe_allow_html=True)
    with colB:
        st.markdown("<div class='card'><div class='card-title'>Session Stats</div>"
                    f"<div style='margin-top:8px' class='muted'>Total processed: {st.session_state.stats['total']}</div>"
                    "<div style='height:6px'></div>"
                    "<div style='display:flex;gap:8px'>"
                    f"<div style='flex:1' class='kpi'><div style='font-size:16px'>{st.session_state.stats['Animal']}</div><div class='muted'>Animal</div></div>"
                    f"<div style='flex:1' class='kpi'><div style='font-size:16px'>{st.session_state.stats['Fashion']}</div><div class='muted'>Fashion</div></div>"
                    f"</div>"
                    "</div>", unsafe_allow_html=True)

elif page == "Deteksi & Klasifikasi":
    st.markdown("<a name='deteksisection'></a>", unsafe_allow_html=True)
    # Two tabs area (left: upload & preview, right: results)
    left, right = st.columns([1.05, 1])

    with left:
        st.markdown("<div class='card'><div class='card-title'>Unggah Gambar</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Pilih file (jpg/png)", type=["jpg","jpeg","png"], key="main_upload", label_visibility="visible")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='muted'>Pilih mode lalu tekan Run</div>", unsafe_allow_html=True)
        mode = st.radio("Mode", ["Deteksi (YOLO)", "Klasifikasi"], horizontal=True)
        run = st.button("Run", key="run_main", help="Jalankan mode yang dipilih", on_click=None)
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded:
            try:
                pil_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                st.image(pil_img, use_column_width=True, caption="Preview")
            except Exception as e:
                st.error(f"Gagal membuka gambar: {e}")
                pil_img = None
        else:
            pil_img = None

    with right:
        st.markdown("<div class='card'><div class='card-title'>Hasil</div>", unsafe_allow_html=True)

        if not pil_img:
            st.markdown("<div class='muted'>Unggah gambar di panel kiri lalu tekan Run untuk melihat hasil.</div>", unsafe_allow_html=True)
        else:
            if not (yolo_model or classifier):
                # show missing model messages
                if not yolo_model:
                    st.warning(f"YOLO tidak tersedia: {yolo_err if 'yolo_err' in globals() else 'ultralytics not installed'}")
                if not classifier:
                    st.warning(f"Classifier tidak tersedia: {clf_err if 'clf_err' in globals() else 'tensorflow not installed'}")
            else:
                if run:
                    if mode == "Deteksi (YOLO)":
                        with st.spinner("Menjalankan YOLO..."):
                            try:
                                results = yolo_model(pil_img)
                                # results[0].plot() may be BGR (opencv). Try convert robustly.
                                plotted = results[0].plot()
                                try:
                                    import cv2
                                    plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                                except Exception:
                                    pass
                                st.image(plotted, use_column_width=True, caption="Deteksi (Bounding boxes)")
                                # summary:
                                boxes = results[0].boxes
                                names = getattr(results[0], "names", {})
                                if boxes is not None and len(boxes) > 0:
                                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                                    st.markdown("**Ringkasan Deteksi:**")
                                    for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                                        label = names[int(cls_id)] if int(cls_id) in names else str(int(cls_id))
                                        st.write(f"- {label} — {conf:.2f}")
                                else:
                                    st.info("Tidak ada objek terdeteksi.")
                                # update stats: Naively map YOLO label names to categories if possible
                                st.session_state.stats["total"] += 1
                            except Exception as e:
                                st.error(f"Error saat YOLO: {e}")

                    elif mode == "Klasifikasi":
                        with st.spinner("Menjalankan klasifikasi..."):
                            try:
                                # determine classifier input size if available
                                target_size = (224, 224)
                                try:
                                    if classifier is not None and hasattr(classifier, "input_shape"):
                                        ish = classifier.input_shape  # e.g. (None, 224,224,3) or (None, 96,96,3)
                                        if isinstance(ish, tuple) and len(ish) >= 3:
                                            # last 3 usually (H,W,C)
                                            if ish[1] and ish[2]:
                                                target_size = (int(ish[1]), int(ish[2]))
                                except Exception:
                                    target_size = (224, 224)

                                img_resized = pil_img.resize(target_size)
                                arr = keras_image.img_to_array(img_resized) if keras_image is not None else np.array(img_resized)
                                arr = np.expand_dims(arr, 0) / 255.0

                                pred = classifier.predict(arr)
                                prob = float(np.max(pred))
                                idx = int(np.argmax(pred))
                                # label mapping: adjust to your own labels if different
                                labels = ["Animal", "Fashion", "Food", "Nature"]
                                pred_label = labels[idx] if idx < len(labels) else f"Class {idx}"

                                # result card pastel
                                color_map = {
                                    "Animal": "#CFF2D6",
                                    "Fashion": "#EED6F6",
                                    "Food": "#FFE6D1",
                                    "Nature": "#D6F2E8"
                                }
                                bg = color_map.get(pred_label, "#F5F5F5")
                                st.markdown(f"<div style='background:{bg}; padding:14px; border-radius:12px;'>"
                                            f"<div style='font-weight:700; font-size:18px'>{pred_label}</div>"
                                            f"<div class='muted'>Confidence: {prob*100:.2f}%</div>"
                                            f"</div>", unsafe_allow_html=True)

                                # update stats
                                st.session_state.stats["total"] += 1
                                if pred_label in st.session_state.stats:
                                    st.session_state.stats[pred_label] += 1
                            except Exception as e:
                                # common failure: mismatch model input shape or runtime error
                                st.error(f"Gagal menjalankan klasifikasi: {e}")
            # end run
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Statistik":
    st.markdown("<div class='card'><div class='card-title'>Statistik Sesi</div>", unsafe_allow_html=True)
    stats = st.session_state.stats
    st.write(f"- Total diproses: **{stats['total']}**")
    st.write(f"- Animal: **{stats['Animal']}**")
    st.write(f"- Fashion: **{stats['Fashion']}**")
    st.write(f"- Food: **{stats['Food']}**")
    st.write(f"- Nature: **{stats['Nature']}**")
    st.markdown("</div>", unsafe_allow_html=True)

else:  # About
    st.markdown("<div class='card'><div class='card-title'>Tentang Aplikasi</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Dibuat oleh Meilani Bulandari — Dashboard untuk deteksi dan klasifikasi gambar (Animal, Fashion, Food, Nature). "
                "Menggunakan YOLO (ultralytics) untuk object detection dan TensorFlow untuk classification. Pastel UI & Poppins font.</div>",
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer small
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#777;font-size:12px'>Made with ♥ — Vision Dashboard</div>", unsafe_allow_html=True)
