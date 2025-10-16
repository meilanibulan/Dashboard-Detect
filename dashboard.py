# app.py
# ---------------------------------------
# Dashboard Vision (Dark + Purple Accent)
# ---------------------------------------
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# PAGE & THEME
# ==========================
st.set_page_config(page_title="Vision Dashboard", page_icon="🧠", layout="wide")

# ---------- Custom CSS (dark + purple accent) ----------
st.markdown("""
<style>
:root{
  --bg:#0F0F14;
  --panel:#151520;
  --panel-2:#1B1B2A;
  --text:#E7E7F0;
  --muted:#9EA0B3;
  --accent:#7C3AED;   /* ungu */
  --accent-2:#9F67FF; /* ungu terang */
  --success:#8BE28B;
  --warning:#FFD166;
}
html, body, [data-testid="stAppViewContainer"]{ background: var(--bg); color: var(--text); }
.block-container{ padding-top: 1rem; padding-bottom: 2rem; max-width: 1300px; }
a{ color: var(--accent-2) !important; }
h1,h2,h3,h4{ color: var(--text); }

.card{
  background: linear-gradient(180deg, var(--panel), var(--panel-2));
  border: 1px solid #23233a;
  border-radius: 16px;
  padding: 18px 20px;
  box-shadow: 0 10px 24px rgba(0,0,0,.25);
}
.card.compact{ padding: 14px 16px; }
.card-title{
  font-weight: 700; font-size:1rem; margin-bottom:.4rem; color:#cfd1e6;
}
.caption{ color: var(--muted); font-size:.86rem; }
.kpi{
  display:flex; align-items:center; gap:.6rem; padding:.6rem .9rem;
  background:#10101a; border:1px solid #23233a; border-radius:12px;
}
.kpi .big{ font-weight:800; font-size:1.15rem; color:var(--text); }
.kpi .sub{ font-size:.8rem; color:var(--muted); }
.pill{
  background: rgba(124,58,237,.14);
  border: 1px solid rgba(124,58,237,.45);
  color: var(--accent-2);
  padding: .35rem .7rem; border-radius: 999px; font-size:.82rem; font-weight:600;
}
.action{
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  color:white; font-weight:700; padding:.55rem 1rem; border-radius:12px;
  display:inline-flex; align-items:center; gap:.5rem; text-decoration:none;
}
.upload-box{
  border:1px dashed #353555; background:#0f0f16; border-radius:16px; padding:16px;
}
.progress{ width:100%; height:10px; background:#24243a; border-radius:999px; overflow:hidden; }
.progress > span{ display:block; height:100%; width:0; background:linear-gradient(90deg,var(--accent),var(--accent-2)); }
hr{ border-color:#23233a }
[data-testid="stFileUploader"] section div{ color: var(--muted) !important; }
</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODELS
# ==========================
@st.cache_resource(show_spinner=True)
def load_models():
    yolo = YOLO("model/Meilani Bulandari Hsb_Laporan 4.pt")               # Deteksi objek
    clf  = tf.keras.models.load_model("model/Meilani Bulandari Hsb_Laporan 2.h5")  # Klasifikasi
    return yolo, clf

yolo_model, classifier = load_models()

# ==========================
# HEADER
# ==========================
c1, c2, c3 = st.columns([1.6, 1, 1])
with c1:
    st.markdown(
        "<div class='card'>"
        "<div class='card-title'>Dashboard</div>"
        "<h1 style='margin:0 0 .3rem 0;'>Welcome Back, Anisa! 👋</h1>"
        "<div class='caption'>Pantau progres deteksi & klasifikasi di sini</div>"
        "</div>", unsafe_allow_html=True
    )
with c2:
    st.markdown("""
    <div class='card compact'>
      <div class='card-title'>Model Status</div>
      <div class='kpi'><span class='big'>Ready ✅</span><span class='sub'>YOLO & Classifier</span></div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class='card compact' style='display:flex;justify-content:space-between;align-items:center;gap:1rem'>
      <span class='pill'>Vision Suite</span>
      <a class='action' href='#' onclick='return false;'>＋ Create Session</a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================
# TABS
# ==========================
tab1, tab2 = st.tabs(["🟣 Deteksi Objek (YOLO)", "🟣 Klasifikasi Gambar"])

def uploader_card(key_label:str):
    st.markdown("<div class='card'><div class='card-title'>Unggah Gambar</div>", unsafe_allow_html=True)
    file = st.file_uploader(" ", type=["jpg","jpeg","png"], key=key_label, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    return file

# -------- TAB 1: DETEKSI --------
with tab1:
    colL, colR = st.columns([1.04, 1])
    with colL:
        file_det = uploader_card("up_yolo")
        if file_det:
            img = Image.open(file_det).convert("RGB")
            st.markdown("<div class='card'><div class='card-title'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with colR:
        st.markdown("<div class='card'><div class='card-title'>Hasil Deteksi</div>", unsafe_allow_html=True)
        if not file_det:
            st.markdown("<div class='caption'>Unggah gambar di panel kiri untuk menjalankan deteksi.</div>", unsafe_allow_html=True)
        else:
            with st.spinner("Menjalankan YOLO..."):
                results = yolo_model(img)
                plotted = results[0].plot()           # ndarray BGR
                plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            st.image(plotted, use_container_width=True, caption="Deteksi (bounding boxes)")
            # Ringkasan deteksi
            names = results[0].names
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("**Ringkasan:**")
                for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                    st.write(f"- {names[int(cls_id)]} — conf: {conf:.2f}")
            else:
                st.info("Tidak ada objek terdeteksi.")
        st.markdown("</div>", unsafe_allow_html=True)

# -------- TAB 2: KLASIFIKASI --------
with tab2:
    colL2, colR2 = st.columns([1.04, 1])
    with colL2:
        file_cls = uploader_card("up_cls")
        if file_cls:
            img2 = Image.open(file_cls).convert("RGB")
            st.markdown("<div class='card'><div class='card-title'>Pratinjau</div>", unsafe_allow_html=True)
            st.image(img2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with colR2:
        st.markdown("<div class='card'><div class='card-title'>Hasil Klasifikasi</div>", unsafe_allow_html=True)
        if not file_cls:
            st.markdown("<div class='caption'>Unggah gambar di panel kiri untuk menjalankan klasifikasi.</div>", unsafe_allow_html=True)
        else:
            # Preprocess (ubah ukuran sesuai model Anda)
            img_resized = img2.resize((224, 224))
            arr = image.img_to_array(img_resized)
            arr = np.expand_dims(arr, axis=0) / 255.0
            with st.spinner("Mengklasifikasikan..."):
                pred = classifier.predict(arr)
            prob = float(np.max(pred))
            idx  = int(np.argmax(pred))
            st.markdown(f"**Label Prediksi:** `{idx}`")
            st.markdown(f"**Probabilitas:** `{prob:.4f}`")
            st.markdown("<div class='caption'>Catatan: ganti mapping label sesuai kelas model Anda.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# FOOTER MINI KPI (opsional)
# ==========================
st.markdown("<br>", unsafe_allow_html=True)
k1, k2, k3 = st.columns(3)
with k1:
    st.markdown("<div class='card compact'><div class='card-title'>Sessions</div><div class='kpi'><span class='big'>3</span><span class='sub'>today</span></div></div>", unsafe_allow_html=True)
with k2:
    st.markdown("<div class='card compact'><div class='card-title'>Avg. Inference</div><div class='kpi'><span class='big'>~120ms</span><span class='sub'>per image</span></div></div>", unsafe_allow_html=True)
with k3:
    st.markdown("<div class='card compact'><div class='card-title'>GPU Memory</div><div class='kpi'><span class='big'>OK</span><span class='sub'>util &lt; 60%</span></div></div>", unsafe_allow_html=True)
