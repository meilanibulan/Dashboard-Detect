import os
import io
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import cv2

# =========================
# PAGE CONFIG & THEME
# =========================
st.set_page_config(
    page_title="Vision Dashboard",
    page_icon="🧠",
    layout="wide",
)

# ——— Global CSS: dark gradient, rounded cards, neumorphic feel
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

:root{
  --bg1:#0A0F13; --bg2:#0F1720;
  --panel:#0E1B1B; --panel2:#0F1E24;
  --accent:#22D3A3; --accent-2:#6EE7B7; --accent-3:#93FFE2;
  --text:#ECF5F3; --muted:#9BB5AE;
  --ring:#1F8A70;
}

* { font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
html, body, [data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 800px at 20% -10%, #14232C 0%, var(--bg2) 40%, var(--bg1) 100%) !important;
  color: var(--text);
}
h1, h2, h3, h4 { color: var(--text); margin: 0 0 .4rem 0; }
small, .muted { color: var(--muted); }

.block-container{ padding-top: 1.2rem; }

/* Sidebar */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0D151B 0%, #0B1318 100%) !important;
  border-right: 1px solid rgba(255,255,255,.04);
}
.sidebar-title{ font-weight: 700; font-size: 1.1rem; margin-bottom: .8rem; }

/* Search bar look */
.header-bar{
  display:flex; align-items:center; gap: 12px;
  background: rgba(255,255,255,.03);
  border: 1px solid rgba(255,255,255,.06);
  border-radius: 14px; padding: 10px 14px;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.02);
}
.header-chip{
  font-size:.85rem; padding:6px 10px; border-radius:999px;
  background: rgba(34,211,163,.12); color: var(--accent);
  border:1px solid rgba(34,211,163,.25);
}

/* Card */
.card{
  background: linear-gradient(180deg, var(--panel) 0%, var(--panel2) 100%);
  border: 1px solid rgba(255,255,255,.065);
  border-radius: 22px;
  padding: 18px 18px 16px 18px;
  box-shadow:
     0 12px 35px rgba(0,0,0,.35),
     inset 0 1px 0 rgba(255,255,255,.03);
}
.card h3{ font-size:1.05rem; font-weight:700; letter-spacing:.2px; }
.kpi{
  display:flex; align-items:baseline; gap:10px; margin-top:.5rem;
}
.kpi .big{ font-size:1.6rem; font-weight:800; letter-spacing:.4px;}
.kpi .pill{
  font-size:.8rem; padding:4px 8px; border-radius:8px;
  background: rgba(34,211,163,.12); color: var(--accent);
  border:1px solid rgba(34,211,163,.25);
}

/* Upload dropzone tweak */
[data-testid="stFileUploaderDropzone"]{
  border: 1px dashed rgba(255,255,255,.20) !important;
  background: rgba(255,255,255,.02);
  border-radius: 18px;
}

/* Buttons */
.stButton>button{
  background: linear-gradient(180deg, #11BFA0 0%, #0AAE8F 100%);
  border: 0; color: #00221B; font-weight: 800;
  border-radius: 12px; padding: 10px 14px;
}
.stButton>button:hover{ filter: brightness(1.06); }

/* Images */
img{ border-radius: 14px; }
hr{ border-color: rgba(255,255,255,.06); }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODELS (cached)
# =========================
@st.cache_resource(show_spinner=True)
def load_models():
    yolo, clf = None, None
    try:
        # Sesuaikan ke path model kamu
        yolo = YOLO("model/Meilani Bulandari Hsb_Laporan 4.pt")
    except Exception:
        pass
    try:
        clf = tf.keras.models.load_model("model/Meilani Bulandari Hsb_Laporan 2.h5")
    except Exception:
        pass
    return yolo, clf

yolo_model, classifier = load_models()

# =========================
# HEADER (seperti "My Dashboard")
# =========================
colA, colB = st.columns([3.2, 1])
with colA:
    st.markdown("<div class='header-bar'><span class='header-chip'>Vision App</span><div class='muted'>Image Classification & Object Detection</div></div>", unsafe_allow_html=True)
    st.markdown("<h1 style='margin:.8rem 0 0 0;'>My Dashboard</h1>", unsafe_allow_html=True)
with colB:
    st.markdown("<div class='card' style='text-align:right; height:100%; display:flex; flex-direction:column; justify-content:center;'><div class='muted'>Hi, Anisa!</div><div style='font-weight:700;'>🧠 Ready</div></div>", unsafe_allow_html=True)

st.write("")

# =========================
# SIDEBAR (ikon & mode)
# =========================
st.sidebar.markdown("<div class='sidebar-title'>⚙️ Controls</div>", unsafe_allow_html=True)
mode = st.sidebar.selectbox("Pilih Mode", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
st.sidebar.markdown("—")
st.sidebar.markdown("**Model Status**")
st.sidebar.write(f"YOLO: {'✅' if yolo_model else '⛔️'}  |  Classifier: {'✅' if classifier else '⛔️'}")

# =========================
# TOP ROW — Cards (Available / Income / Expense style)
# =========================
c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
with c1:
    st.markdown("<div class='card'><h3>Upload</h3><div class='muted'>Unggah gambar untuk dianalisis</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='card'><h3>YOLO</h3><div class='kpi'><div class='big'>"
                + ("Ready" if yolo_model else "N/A")
                + "</div><span class='pill'>object detection</span></div></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='card'><h3>Classifier</h3><div class='kpi'><div class='big'>"
                + ("Ready" if classifier else "N/A")
                + "</div><span class='pill'>image class</span></div></div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='card'><h3>Mode</h3><div class='kpi'><div class='big'>"
                + ("YOLO" if mode.startswith("Deteksi") else "Classifier")
                + "</div><span class='pill'>active</span></div></div>", unsafe_allow_html=True)

st.write("")

# =========================
# MAIN CONTENT
# =========================
left, right = st.columns([1.4, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Revenue Flow (Preview Area)")
    st.caption("Panel ini menampilkan gambar input serta pratinjau hasil sesuai mode pilihan.")
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
    original_img = None
    if uploaded_file is not None:
        original_img = Image.open(uploaded_file).convert("RGB")
        st.image(original_img, caption="Gambar yang diunggah", use_container_width=True)
    else:
        st.info("Belum ada gambar yang diunggah.")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### My Card (Results)")
    st.caption("Hasil deteksi/klasifikasi akan muncul di sini sebagai kartu ringkas seperti 'Transactions'.")

    if original_img is not None:
        if mode == "Deteksi Objek (YOLO)":
            if yolo_model is None:
                st.error("Model YOLO belum tersedia.")
            else:
                results = yolo_model(original_img, verbose=False)
                # results[0].plot() -> numpy array BGR, ubah ke RGB agar warna tepat
                plotted = results[0].plot()
                if plotted is not None:
                    # Convert BGR to RGB for correct Streamlit display
                    rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                    st.image(rgb, caption="Hasil Deteksi (YOLO)", use_container_width=True)

                # Ringkasan “Transactions”
                try:
                    boxes = results[0].boxes
                    n_det = int(boxes.shape[0]) if boxes is not None else 0
                except Exception:
                    n_det = 0

                st.markdown("---")
                st.markdown("**Deteksi Terdapat:**")
                st.write(f"- Total objek: **{n_det}**")
                if n_det > 0:
                    with st.expander("Detail skor & kelas"):
                        try:
                            cls_ids = boxes.cls.cpu().numpy().astype(int).tolist()
                            confs = boxes.conf.cpu().numpy().tolist()
                            for i, (cid, cf) in enumerate(zip(cls_ids, confs), 1):
                                st.write(f"{i}. class_id = **{cid}** | conf = {cf:.3f}")
                        except Exception:
                            st.write("Ringkasan tidak tersedia.")
        else:
            if classifier is None:
                st.error("Model klasifikasi belum tersedia.")
            else:
                # Preprocess
                img_resized = original_img.resize((224, 224))
                arr = keras_image.img_to_array(img_resized)
                arr = np.expand_dims(arr, axis=0) / 255.0

                pred = classifier.predict(arr, verbose=0)
                class_index = int(np.argmax(pred))
                prob = float(np.max(pred))

                st.markdown("<div class='kpi'><div class='big'>Prediction</div><span class='pill'>Classifier</span></div>", unsafe_allow_html=True)
                st.write(f"**Class index:** {class_index}")
                st.write(f"**Probability:** {prob:.4f}")
                st.markdown("---")
                st.caption("Catatan: mapping nama kelas mengikuti urutan label pada model kamu.")

    else:
        st.info("Unggah gambar pada panel kiri untuk melihat hasil di sini.")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# FOOTER MINI WIDGETS
# =========================
f1, f2, f3 = st.columns([1,1,1])
with f1:
    st.markdown("<div class='card'><h3>Add friends</h3><div class='muted'>Undang teman dan berkolaborasi menguji model.</div></div>", unsafe_allow_html=True)
with f2:
    st.markdown("<div class='card'><h3>Spending</h3><div class='muted'>Placeholder grafik kecil (opsional).</div></div>", unsafe_allow_html=True)
with f3:
    st.markdown("<div class='card'><h3>Tips</h3><div class='muted'>Gunakan gambar jelas & berukuran cukup agar hasil optimal.</div></div>", unsafe_allow_html=True)
