import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
import numpy as np
from PIL import Image
import pandas as pd

# ===================== Page config =====================
st.set_page_config(
    page_title="Image Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== PALETTE =====================
PALETTE = {
    "pink":  "#FF99C8",
    "lilac": "#E8C0FC",
    "sky":   "#A8DEFA",
    "mint":  "#D0F4E0",
    "butter":"#FCF5BF",
    "ink":   "#2e3140",
    "muted": "#6f758a",
}

# ===================== Global styles =====================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
* {{ font-family:'Poppins', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}

/* App background */
.stApp{{
  background: linear-gradient(180deg, {PALETTE['pink']} 0%,
                                     {PALETTE['lilac']} 22%,
                                     {PALETTE['sky']} 45%,
                                     {PALETTE['mint']} 70%,
                                     {PALETTE['butter']} 100%);
}}

/* Sidebar gradient + pill nav */
section[data-testid="stSidebar"]{{
  background: linear-gradient(180deg, {PALETTE['butter']} 0%, {PALETTE['mint']} 35%, {PALETTE['sky']} 65%, {PALETTE['lilac']} 100%) !important;
  border-right: 1px solid #ffffff55;
}}
section[data-testid="stSidebar"] .sidebar-content{{ padding:18px 16px 24px; }}
section[data-testid="stSidebar"] h3{{ margin:6px 0 0 0; color:#5e5a6b; font-weight:800; letter-spacing:.2px; }}
section[data-testid="stSidebar"] .cap{{ margin:2px 0 14px 0; color:#6f758a; font-size:13px; }}

section[data-testid="stSidebar"] [data-testid="stRadio"]{{ margin-top:10px; }}
section[data-testid="stSidebar"] [role="radiogroup"] > div{{ margin:6px 0; }}
section[data-testid="stSidebar"] [role="radio"]{{
  border:1px solid #ffffff80; background:#ffffffcc; color:#40465A;
  padding:8px 14px; border-radius:999px; box-shadow:0 4px 10px #0000000d;
}}
section[data-testid="stSidebar"] [role="radio"] p{{ font-size:14px; font-weight:600; margin:0; }}
section[data-testid="stSidebar"] [role="radio"][aria-checked="true"]{{
  background: linear-gradient(90deg, {PALETTE['pink']} 0%, {PALETTE['lilac']} 100%);
  color:#FFFFFF !important; border-color:transparent;
}}
section[data-testid="stSidebar"] [role="radio"][aria-checked="true"] p{{ color:#FFFFFF !important; }}
.sidebar-thanks{{ color:#6f758a; font-size:12px; margin-top:10px }}

/* Cards & spacing */
.block-card{{
  background:#ffffffd9; border:1px solid #ffffff80; border-radius:22px; padding:22px;
  box-shadow:0 10px 24px #00000012; margin-top:8px;
}}
.badge{{ display:inline-block; padding:6px 14px; border-radius:999px; background:{PALETTE['lilac']}; color:#40304e; font-weight:700; font-size:12px; }}
.mt-8{{ margin-top:8px; }} .mt-16{{ margin-top:16px; }} .mt-24{{ margin-top:24px; }}

/* Hide ALL uploaders globally… */
[data-testid="stFileUploaderDropzone"]{{ display:none !important; }}
/* …except the ones we scope explicitly */
.uploader-scope [data-testid="stFileUploaderDropzone"]{{
  display:block !important; background:#ffffff; border:0 !important; box-shadow:none !important;
  border-radius:20px !important; padding:14px 18px !important;
}}

/* Home dashboard layout */
.dash-grid {{ display:grid; grid-template-columns: 1.35fr 1fr 1fr .95fr; gap:18px; }}
.dash-subgrid {{ display:grid; grid-template-columns: 1fr 1fr; gap:18px; }}
.card {{
  background:#fff; border:1px solid #ffffff90; border-radius:22px; padding:16px 18px;
  box-shadow: 0 12px 30px rgba(0,0,0,.08), inset 0 -2px 0 #00000008;
}}
.card.dark {{
  background: linear-gradient(180deg, #2c2f3b 0%, #242733 100%); color:#fff; border:1px solid #00000033;
}}
.title {{ font-weight:800; color:{PALETTE['ink']}; font-size:26px; margin:8px 0 6px 0; }}
.kpi {{ font-size:36px; font-weight:800; color:{PALETTE['ink']}; }}
.kpi-sub {{ font-size:13px; color:{PALETTE['muted']}; }}
.hr-soft {{ height:10px; border-radius:999px; background:linear-gradient(90deg,{PALETTE['pink']} 0%, {PALETTE['lilac']} 60%, {PALETTE['sky']} 100%); opacity:.25 }}
.badge-chip {{ display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border-radius:999px; background:#fff; border:1px solid #eee; color:{PALETTE['ink']}; font-weight:600; font-size:12px; box-shadow:0 4px 10px #0000000d; }}
.badge-chip .dot {{ width:8px; height:8px; border-radius:50%; background:{PALETTE['pink']}; }}
.badge-active {{ background:linear-gradient(90deg,{PALETTE['pink']} 0%, {PALETTE['lilac']} 100%); color:#fff; border-color:transparent; }}
.pill-progress {{ height:12px; background:#f2f3f7; border-radius:999px; overflow:hidden; }}
.pill-fill {{ height:100%; border-radius:999px; }}
.list {{ list-style:none; padding-left:0; margin:0; }}
.list li {{ display:flex; align-items:center; gap:10px; padding:8px 0; border-bottom:1px dashed #eee; }}
.list .dot {{ width:10px; height:10px; border-radius:50%; background:{PALETTE['sky']}; }}
.tag {{ padding:2px 8px; border-radius:999px; font-size:11px; font-weight:700; background:{PALETTE['mint']}; color:#2c574b; border:1px solid #cfe9de; }}
.small {{ font-size:12px; color:{PALETTE['muted']}; }}

/* Tiles teks (bila ingin tampilkan kategori di home) */
.cat-card{{ display:flex; align-items:center; justify-content:center; height:110px; border-radius:22px; background:#ffffff; border:1px solid #00000010; font-weight:700; font-size:22px; color:#8a8fa3; }}
.cat-card:hover{{ box-shadow:0 12px 20px #00000012; transform: translateY(-2px); transition:all .2s; }}

h2, h3 {{ letter-spacing:.1px; color:{PALETTE['ink']}; }}
</style>
""", unsafe_allow_html=True)

# ===================== Session =====================
if "stats" not in st.session_state:
    st.session_state.stats = {k: 0 for k in ["Animal", "Fashion", "Food", "Nature", "total"]}
if "last_results" not in st.session_state:
    st.session_state.last_results = []  # list[(label, prob)]

CLASS_NAMES = ["Animal", "Fashion", "Food", "Nature"]

# ===================== Models =====================
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

# ===================== Sidebar =====================
st.sidebar.markdown("### Features That")
st.sidebar.markdown("<div class='cap'>Can Be Used</div>", unsafe_allow_html=True)
labels = [" Home", "Image Detection", "Image Classification", "Statistics", "Dataset", "About"]
label2key = {"Home":"Home","Image Detection":"Image Detection","Image Classification":"Image Classification","Statistics":"Statistics","Dataset":"Dataset","About":"About"}
menu = label2key[st.sidebar.radio("", labels, index=0, key="nav_radio")]
st.sidebar.markdown("<div class='sidebar-thanks'>Thank you for using this website</div>", unsafe_allow_html=True)

# ===================== Helpers =====================
def read_image(file): return Image.open(file).convert("RGB")

def add_stats(label, prob):
    st.session_state.last_results.append((label, prob))
    st.session_state.stats[label] = st.session_state.stats.get(label, 0) + 1
    st.session_state.stats["total"] += 1

def prepare_for_model(pil_img, model):
    """Resize & format image to match model.input_shape (supports C=1/3)."""
    input_shape = model.input_shape
    if isinstance(input_shape, list): input_shape = input_shape[0]
    _, H, W, C = input_shape
    H = H or 224; W = W or 224; C = C or 3
    img = pil_img.convert("RGB").resize((W, H))
    arr = kimage.img_to_array(img)  # (H,W,3)
    if C == 1:
        arr = np.dot(arr[...,:3], [0.2989, 0.5870, 0.1140]).astype("float32")
        arr = np.expand_dims(arr, -1)
    elif C != 3:
        arr = np.resize(arr, (H, W, C))
    arr = arr / 255.0
    return np.expand_dims(arr, 0), (H, W, C)

# ===================== Home - Dashboard =====================
def render_dashboard():
    st.markdown("<div class='title'>Welcome in, Bulandari</div>", unsafe_allow_html=True)
    st.markdown("<div class='hr-soft'></div>", unsafe_allow_html=True)
    st.write("")
    col_chip1, col_chip2, col_chip3, col_chip4, _ = st.columns([1,1,1,1,4])
    with col_chip1: st.markdown("<span class='badge-chip badge-active'><span class='dot'></span> Dashboard</span>", unsafe_allow_html=True)
    with col_chip2: st.markdown("<span class='badge-chip'>People</span>", unsafe_allow_html=True)
    with col_chip3: st.markdown("<span class='badge-chip'>Devices</span>", unsafe_allow_html=True)
    with col_chip4: st.markdown("<span class='badge-chip'>Settings</span>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<div class='dash-grid'>", unsafe_allow_html=True)

    # Col 1
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        cA1, cA2 = st.columns([1.1, 1.9])
        with cA1:
            st.markdown("**Lora Peterson**  \nUX/UI Designer")
            st.markdown("<span class='tag'>Active</span>", unsafe_allow_html=True)
        with cA2:
            st.markdown("**Interviews**", unsafe_allow_html=True)
            st.markdown(f"<div class='pill-progress'><div class='pill-fill' style='width:15%;background:{PALETTE['pink']}'></div></div><div class='small'>15%</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown("**Hired**", unsafe_allow_html=True)
            st.markdown(f"<div class='pill-progress'><div class='pill-fill' style='width:13%;background:{PALETTE['lilac']}'></div></div><div class='small'>13%</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown("**Output**", unsafe_allow_html=True)
            st.markdown(f"<div class='pill-progress'><div class='pill-fill' style='width:10%;background:{PALETTE['sky']}'></div></div><div class='small'>10%</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='dash-subgrid'>", unsafe_allow_html=True)
        # Progress bars kecil
        with st.container():
            st.markdown("<div class='card'><b>Progress</b> <span class='small'>Work time this week</span>", unsafe_allow_html=True)
            pcols = st.columns(7); vals = [3.2, 4.1, 6.1, 2.5, 5.2, 3.6, 1.8]
            for i, pc in enumerate(pcols):
                h = int(vals[i] * 12)
                pc.markdown(f"<div style='display:flex;align-items:flex-end;height:120px;'><div style='width:24px;height:{h}px;background:{PALETTE['pink']};border-radius:8px;'></div></div><div class='small' style='text-align:center;'>{'SMTWTFS'[i]}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        # Calendar mini
        with st.container():
            st.markdown("<div class='card'><b>Schedule</b> <span class='small'>September 2024</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='tag' style='display:inline-block;background:{PALETTE['lilac']};color:#3b3050;border-color:#e9d1ff'>Weekly Team Sync — Tue 09:00</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='tag' style='display:inline-block;background:{PALETTE['sky']};color:#17465b;border-color:#bfe7fa'>Onboarding Session — Thu 14:00</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)  # /subgrid

    # Col 2 KPIs
    with st.container():
        st.markdown("<div class='card'><div class='kpi'>78</div><div class='kpi-sub'>Employees</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><div class='kpi'>56</div><div class='kpi-sub'>Hirings</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><div class='kpi'>203</div><div class='kpi-sub'>Projects</div></div>", unsafe_allow_html=True)

    # Col 3 Donut + Onboarding bars
    with st.container():
        st.markdown("<div class='card'><b>Time tracker</b>", unsafe_allow_html=True)
        progress_pct = 68
        st.markdown(f"""
        <div style='display:flex;gap:16px;align-items:center;'>
          <svg width="120" height="120" viewBox="0 0 36 36">
            <path d="M18 2 a 16 16 0 0 1 0 32 a 16 16 0 0 1 0 -32" fill="none" stroke="#eee" stroke-width="3"/>
            <path d="M18 2 a 16 16 0 0 1 0 32 a 16 16 0 0 1 0 -32"
                  fill="none" stroke="{PALETTE['pink']}" stroke-width="3"
                  stroke-dasharray="{progress_pct}, 100" />
            <text x="18" y="20" text-anchor="middle" font-size="6" fill="{PALETTE['ink']}" font-weight="800">{progress_pct}%</text>
          </svg>
          <div class='small'>Work time</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><b>Onboarding</b> <span class='small'>Completion</span>", unsafe_allow_html=True)
        for label, pct, col in [("Task",30,PALETTE["pink"]), ("Docs",25,PALETTE["lilac"]), ("Review",0,PALETTE["sky"])]:
            st.markdown(f"<div class='small' style='margin-top:8px'>{label}</div><div class='pill-progress'><div class='pill-fill' style='width:{pct}%;background:{col}'></div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Col 4 Task list
    with st.container():
        st.markdown("<div class='card dark'><b>Onboarding Task</b> <span class='small'>2/8</span>", unsafe_allow_html=True)
        st.markdown("<ul class='list'>"
                    "<li><span class='dot'></span>Interview</li>"
                    "<li><span class='dot'></span>Offer Letter</li>"
                    "<li><span class='dot'></span>Project Update</li>"
                    "<li><span class='dot'></span>Discuss QA Goods</li>"
                    "<li><span class='dot'></span>HR Policy Review</li>"
                    "</ul>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # /dash-grid

# ===================== Routing =====================
if menu == "Home":
    render_dashboard()

elif menu == "Image Detection":
    st.markdown("## UPLOAD IMAGE")
    st.caption("Insert the image according to what you want (jpg/png)")
    st.markdown("<div class='mt-8'></div>", unsafe_allow_html=True)
    st.markdown("<div class='uploader-scope'>", unsafe_allow_html=True)
    file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='mt-16'></div>", unsafe_allow_html=True)

    colL, colR = st.columns([1.2, 1], gap="large")
    if not MODELS_READY: st.error(f"Model belum siap: {load_err}")

    if file is not None:
        img = read_image(file)
        colL.image(img, caption="Input", use_container_width=True)
    else:
        colL.info("Silakan unggah gambar terlebih dahulu.")

    with colR:
        st.markdown("<div class='block-card'>", unsafe_allow_html=True)
        detected = []
        if MODELS_READY and file is not None:
            results = yolo_model(img)
            plot = results[0].plot()
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
    if not MODELS_READY: st.error(f"Model belum siap: {load_err}")

    if file is not None:
        img = read_image(file)
        colL.image(img, caption="Input", use_container_width=True)
        if MODELS_READY:
            try:
                arr, target = prepare_for_model(img, classifier)
                pred = classifier.predict(arr, verbose=0)
                if pred.ndim == 2 and pred.shape[1] == len(CLASS_NAMES):
                    idx = int(np.argmax(pred[0])); label_out = CLASS_NAMES[idx]; prob_out = float(np.max(pred[0]))
                elif pred.ndim == 2 and pred.shape[1] == 1:
                    prob_out = float(pred[0][0]); label_out = CLASS_NAMES[1] if prob_out >= 0.5 else CLASS_NAMES[0]
                else:
                    idx = int(np.argmax(pred)); label_out = CLASS_NAMES[idx % len(CLASS_NAMES)]; prob_out = float(np.max(pred))
                add_stats(label_out, prob_out)
            except Exception as e:
                st.error("Klasifikasi gagal. Kemungkinan besar ukuran/kanal input tidak sesuai.")
                st.code(f"Expected input (H,W,C): {classifier.input_shape} • Used: {target if 'target' in locals() else 'unknown'}\nDetail: {e}")
    else:
        colL.info("Silakan unggah gambar terlebih dahulu.")

    with colR:
        st.markdown("<div class='block-card'><b>Result</b><br>", unsafe_allow_html=True)
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
