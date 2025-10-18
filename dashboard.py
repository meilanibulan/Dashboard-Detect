PALETTE = {
    "pink": "#FF99C8",
    "lilac": "#E8C0FC",
    "sky": "#A8DEFA",
    "mint": "#D0F4E0",
    "butter": "#FCF5BF",
    "ink": "#2e3140",
    "muted": "#6f758a",
}

# ==== CSS spesifik untuk dashboard-style seperti mockup ====
st.markdown(f"""
<style>
/* bg lembut */
.stApp {{
  background: radial-gradient(1200px 600px at 0% 0%, {PALETTE['butter']} 0%,
                               {PALETTE['mint']} 35%, {PALETTE['sky']} 70%) !important;
}}
/* wrapper grid */
.dash-grid {{ display:grid; grid-template-columns: 1.35fr 1fr 1fr .95fr; gap:18px; }}
.dash-subgrid {{ display:grid; grid-template-columns: 1fr 1fr; gap:18px; }}
.card {{
  background:#fff; border:1px solid #ffffff90; border-radius:22px; padding:16px 18px;
  box-shadow: 0 12px 30px rgba(0,0,0,.08), inset 0 -2px 0 #00000008;
}}
.card.dark {{
  background: linear-gradient(180deg, #2c2f3b 0%, #242733 100%);
  color:#fff; border:1px solid #00000033;
}}
.title {{ font-weight:800; color:{PALETTE['ink']}; font-size:26px; margin:8px 0 6px 0; }}
.kpi {{ font-size:36px; font-weight:800; color:{PALETTE['ink']}; }}
.kpi-sub {{ font-size:13px; color:{PALETTE['muted']}; }}
.hr-soft {{ height:10px; border-radius:999px; background:linear-gradient(90deg,
            {PALETTE['pink']} 0%, {PALETTE['lilac']} 60%, {PALETTE['sky']} 100%); opacity:.25 }}
.badge-chip {{
  display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border-radius:999px;
  background:#fff; border:1px solid #eee; color:{PALETTE['ink']}; font-weight:600; font-size:12px;
  box-shadow:0 4px 10px #0000000d;
}}
.badge-chip .dot {{ width:8px; height:8px; border-radius:50%; background:{PALETTE['pink']}; }}
.badge-active {{
  background:linear-gradient(90deg,{PALETTE['pink']} 0%, {PALETTE['lilac']} 100%); color:#fff; border-color:transparent;
}}
.pill-progress {{ height:12px; background:#f2f3f7; border-radius:999px; overflow:hidden; }}
.pill-fill {{ height:100%; border-radius:999px; }}
.list {{ list-style:none; padding-left:0; margin:0; }}
.list li {{ display:flex; align-items:center; gap:10px; padding:8px 0; border-bottom:1px dashed #eee; }}
.list .dot {{ width:10px; height:10px; border-radius:50%; background:{PALETTE['sky']}; }}
.tag {{
  padding:2px 8px; border-radius:999px; font-size:11px; font-weight:700; background:{PALETTE['mint']}; color:#2c574b;
  border:1px solid #cfe9de;
}}
.small {{ font-size:12px; color:{PALETTE['muted']}; }}
</style>
""", unsafe_allow_html=True)

def render_dashboard():
    # header
    st.markdown("<div class='title'>Welcome in, Bulandari</div>", unsafe_allow_html=True)
    st.markdown("<div class='hr-soft'></div>", unsafe_allow_html=True)
    st.write("")

    # chips bar (seperti top filter di mockup)
    col_chip1, col_chip2, col_chip3, col_chip4, _ = st.columns([1,1,1,1,4])
    with col_chip1:
        st.markdown("<span class='badge-chip badge-active'><span class='dot'></span> Dashboard</span>", unsafe_allow_html=True)
    with col_chip2:
        st.markdown("<span class='badge-chip'>People</span>", unsafe_allow_html=True)
    with col_chip3:
        st.markdown("<span class='badge-chip'>Devices</span>", unsafe_allow_html=True)
    with col_chip4:
        st.markdown("<span class='badge-chip'>Settings</span>", unsafe_allow_html=True)

    st.write("")
    # grid utama
    st.markdown("<div class='dash-grid'>", unsafe_allow_html=True)

    # ==== Col 1 (besar kiri) ====
    with st.container():
        # Profile card + progress bars kecil
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        cA1, cA2 = st.columns([1.1, 1.9])
        with cA1:
            st.markdown("**Lora Peterson**  \nUX/UI Designer")
            st.markdown("<span class='tag'>Active</span>", unsafe_allow_html=True)
        with cA2:
            st.markdown("**Interviews**", unsafe_allow_html=True)
            st.markdown("""
                <div class='pill-progress'><div class='pill-fill' style="width:15%;background:{0}"></div></div>
                <div class='small'>15%</div>
            """.format(PALETTE["pink"]), unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown("**Hired**", unsafe_allow_html=True)
            st.markdown("""
                <div class='pill-progress'><div class='pill-fill' style="width:13%;background:{0}"></div></div>
                <div class='small'>13%</div>
            """.format(PALETTE["lilac"]), unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown("**Output**", unsafe_allow_html=True)
            st.markdown("""
                <div class='pill-progress'><div class='pill-fill' style="width:10%;background:{0}"></div></div>
                <div class='small'>10%</div>
            """.format(PALETTE["sky"]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        # subgrid: Progress chart + Calendar strip (dummy)
        st.markdown("<div class='dash-subgrid'>", unsafe_allow_html=True)

        # Progress (bar kecil sederhana)
        with st.container():
            st.markdown("<div class='card'><b>Progress</b> <span class='small'>Work time this week</span>", unsafe_allow_html=True)
            pcols = st.columns(7)
            vals = [3.2, 4.1, 6.1, 2.5, 5.2, 3.6, 1.8]
            for i, pc in enumerate(pcols):
                h = int(vals[i] * 12)  # tinggi pseudo
                pc.markdown(f"""
                <div style='display:flex;align-items:flex-end;height:120px;'>
                    <div style='width:24px;height:{h}px;background:{PALETTE['pink']};border-radius:8px;'></div>
                </div>
                <div class='small' style='text-align:center;'>{"SMTWTFS"[i]}</div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Calendar strip
        with st.container():
            st.markdown("<div class='card'><b>Schedule</b> <span class='small'>September 2024</span>", unsafe_allow_html=True)
            # bar agenda
            st.markdown(f"""
            <div class='tag' style='display:inline-block;background:{PALETTE['lilac']};color:#3b3050;border-color:#e9d1ff'>
                Weekly Team Sync — Tue 09:00
            </div>
            <div style='height:10px'></div>
            <div class='tag' style='display:inline-block;background:{PALETTE['sky']};color:#17465b;border-color:#bfe7fa'>
                Onboarding Session — Thu 14:00
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # /subgrid

    # ==== Col 2 (kpi cards) ====
    with st.container():
        st.markdown("<div class='card'><div class='kpi'>78</div><div class='kpi-sub'>Employees</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><div class='kpi'>56</div><div class='kpi-sub'>Hirings</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><div class='kpi'>203</div><div class='kpi-sub'>Projects</div></div>", unsafe_allow_html=True)

    # ==== Col 3 (time tracker donut + onboarding %) ====
    with st.container():
        # donut simple (tanpa matplotlib agar ringan)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<b>Time tracker</b>", unsafe_allow_html=True)
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

        # Onboarding %
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<b>Onboarding</b> <span class='small'>Completion</span>", unsafe_allow_html=True)
        for label, pct, col in [
            ("Task", 30, PALETTE["pink"]),
            ("Docs", 25, PALETTE["lilac"]),
            ("Review", 0,  PALETTE["sky"])
        ]:
            st.markdown(f"""
            <div class='small' style='margin-top:8px'>{label}</div>
            <div class='pill-progress'><div class='pill-fill' style="width:{pct}%; background:{col}"></div></div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ==== Col 4 (task list gelap) ====
    with st.container():
        st.markdown("<div class='card dark'>", unsafe_allow_html=True)
        st.markdown("<b>Onboarding Task</b> <span class='small'>2/8</span>", unsafe_allow_html=True)
        st.markdown("<ul class='list'>"
                    "<li><span class='dot'></span>Interview</li>"
                    "<li><span class='dot'></span>Offer Letter</li>"
                    "<li><span class='dot'></span>Project Update</li>"
                    "<li><span class='dot'></span>Discuss QA Goods</li>"
                    "<li><span class='dot'></span>HR Policy Review</li>"
                    "</ul>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # /dash-grid
