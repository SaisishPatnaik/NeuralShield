import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from html import escape
from components.live_monitor import render_live_monitor

# ============================================================
# CONFIGURATION
# ============================================================
import os

API_URL = os.getenv(
    "NEURALSHIELD_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

st.set_page_config(
    page_title="NEURALSHIELD | AI NETWORK DEFENSE",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL THEME
# ============================================================
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;600;700;800;900&family=Rajdhani:wght@500;600;700;800&display=swap');
:root {
    --cyan:#55eaff; --blue:#008cff; --dark:#050a12; --panel:#0b1c2b;
    --muted:#83a9bf; --red:#ff455d; --orange:#ffae42; --green:#47ffb0;
}
html, body, [data-testid="stAppViewContainer"] {
    background:radial-gradient(ellipse at 50% 0%,#102b40 0%,#050a12 58%);
    color:#e5faff; font-family:'Rajdhani',sans-serif; font-weight:600;
}
[data-testid="stHeader"] {background:rgba(5,10,18,.94);}
[data-testid="stAppViewContainer"] .main {
    background-image:linear-gradient(rgba(0,217,255,.022) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,217,255,.022) 1px,transparent 1px);
    background-size:38px 38px;
}
[data-testid="stMainBlockContainer"] {padding-top:2rem;padding-bottom:2rem;}
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#081522,#030810);
    border-right:1px solid #087e9d;box-shadow:5px 0 25px rgba(0,191,255,.08);
}
[data-testid="stSidebar"] * {font-family:'Rajdhani',sans-serif;font-weight:700;}
h1,h2,h3,h4 {font-family:'Orbitron',sans-serif!important;font-weight:800!important;letter-spacing:1.5px;color:#e4faff;}
p,label,[data-testid="stMetricLabel"] {font-weight:700!important;}
.main-title {
    font-family:'Orbitron',sans-serif;font-size:clamp(30px,4vw,48px);font-weight:900;
    letter-spacing:5px;color:var(--cyan);text-shadow:0 0 12px #00bfff,0 0 30px #007bff;
}
.main-title:hover {text-shadow:0 0 10px #00d9ff,0 0 25px #00bfff,0 0 50px #007bff;}
.subtitle {font-size:15px;font-weight:800;letter-spacing:3px;color:#83a9bf;text-transform:uppercase;}
.section-title {
    font-family:'Orbitron',sans-serif;font-size:15px;font-weight:800;color:#48dfff;
    letter-spacing:2.5px;border-bottom:1px solid #14516c;padding-bottom:11px;
    margin:28px 0 18px;transition:all .25s ease;
}
.section-title:hover {color:#a3f4ff;text-shadow:0 0 12px #00bfff;padding-left:7px;}
.scan-line {
    height:2px;width:100%;background:linear-gradient(90deg,transparent,#00d9ff,#fff,#00d9ff,transparent);
    background-size:200% 100%;box-shadow:0 0 12px #00d9ff;margin:17px 0 23px;
    animation:scan 3s linear infinite;
}
@keyframes scan {0%{background-position:100% 0;}100%{background-position:-100% 0;}}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color:#12617b!important;border-radius:7px!important;
    background:linear-gradient(135deg,#0b1c2b,#07111c)!important;transition:all .25s ease-in-out;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color:#55eaff!important;box-shadow:0 0 13px rgba(0,217,255,.28);transform:translateY(-3px);
}
.metric-label {color:#83a9bf;font-size:13px;font-weight:800;letter-spacing:2.5px;text-transform:uppercase;}
.metric-value {font-family:'Orbitron',sans-serif;font-weight:900;color:#55eaff;font-size:32px;margin-top:9px;text-shadow:0 0 12px rgba(0,191,255,.5);}
.metric-danger {color:#ff455d;}.metric-warning {color:#ffb347;}.metric-green {color:#47ffb0;}
.status-online {color:#47ffb0;font-family:'Orbitron',sans-serif;font-size:12px;font-weight:800;letter-spacing:1.5px;text-shadow:0 0 10px #47ffb0;}
.status-offline {color:#ff455d;font-family:'Orbitron',sans-serif;font-size:12px;font-weight:800;letter-spacing:1.5px;}
[data-testid="stSidebar"] [role="radiogroup"] label {
    border:1px solid transparent;border-radius:5px;padding:9px 10px;transition:all .22s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background:rgba(0,217,255,.10);border-color:#00bfff;box-shadow:0 0 12px rgba(0,217,255,.2);transform:translateX(4px);
}
div.stButton>button {
    background:linear-gradient(90deg,#063b52,#087d9b);border:1px solid #37dfff;color:white;
    border-radius:4px;font-family:'Orbitron',sans-serif;font-weight:800;letter-spacing:.8px;transition:all .22s ease!important;
}
div.stButton>button:hover {
    background:#0ba5cb;transform:translateY(-2px);box-shadow:0 0 10px #00d9ff,0 0 22px rgba(0,217,255,.4);
    color:white;border-color:#8af3ff;
}
[data-testid="stDataFrame"] {border:1px solid #12617b;border-radius:5px;transition:all .25s ease;}
[data-testid="stDataFrame"]:hover {border-color:#35dcff;box-shadow:0 0 18px rgba(0,217,255,.16);}
.threat-card,.model-card {
    background:linear-gradient(120deg,#0b1c2b,#07111c);border:1px solid #21445c;
    border-left:4px solid var(--cyan);padding:18px;margin-bottom:13px;border-radius:5px;
    color:#d9faff;font-family:'Rajdhani',sans-serif;font-size:16px;font-weight:700;line-height:1.65;transition:all .25s ease;
}
.threat-card:hover,.model-card:hover {transform:translateX(4px);border-color:var(--cyan);box-shadow:0 0 18px rgba(0,217,255,.18);}
.model-name {color:#55eaff;font-family:'Orbitron',sans-serif;font-weight:800;font-size:15px;}
.model-category {color:#83a9bf;font-size:13px;font-weight:800;letter-spacing:1.5px;margin-top:8px;}
.footer {text-align:center;color:#64869e;font-family:'Orbitron',sans-serif;font-size:10px;font-weight:700;letter-spacing:2px;padding:28px 10px;}
.main .block-container {animation:pageAppear .45s ease-out;}
@keyframes pageAppear {from{opacity:0;transform:translateY(7px);}to{opacity:1;transform:translateY(0);}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;transition-duration:.01ms!important;}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# API HELPERS
# ============================================================
def get_api_data(endpoint):
    try:
        response = requests.get(f"{API_URL}{endpoint}", timeout=8)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def resolve_incident(alert_id):
    try:
        response = requests.patch(f"{API_URL}/alerts/{alert_id}/resolve", timeout=8)
        response.raise_for_status()
        return True
    except requests.RequestException as error:
        st.error(f"Incident resolution failed: {error}")
        return False


def render_page_header(title, subtitle=None):
    st.markdown(f'<div class="main-title">{escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="subtitle">{escape(subtitle)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)


def render_section(title):
    st.markdown(f'<div class="section-title">{escape(title)}</div>', unsafe_allow_html=True)


def render_footer():
    st.markdown("""
        <div class="footer">
            NEURALSHIELD // AI NETWORK DEFENSE SYSTEM<br>
            ARTIFICIAL INTELLIGENCE • NETWORK SECURITY • THREAT INTELLIGENCE
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# LOAD BACKEND DATA
# ============================================================
health = get_api_data("/health")
alert_data = get_api_data("/alerts/")
detection_data = get_api_data("/alerts/detections")
backend_online = health is not None
alerts = alert_data.get("alerts", []) if alert_data else []
detections = detection_data.get("detections", []) if detection_data else []

# ============================================================
# SINGLE SIDEBAR AND NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("""
        <div style="text-align:center;padding:20px 0 16px;">
            <div style="font-size:48px;">🛡️</div>
            <div style="font-family:Orbitron;color:#55eaff;font-size:21px;font-weight:900;letter-spacing:3px;">
                NEURALSHIELD
            </div>
            <div style="font-size:10px;font-weight:800;letter-spacing:3px;color:#729cb5;">
                AI NETWORK DEFENSE
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### SYSTEM STATUS")
    if backend_online:
        st.markdown('<div class="status-online">● API ONLINE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-offline">● API OFFLINE</div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="color:#83a9bf;font-size:13px;font-weight:700;line-height:2.1;margin-top:15px;">
            AI CORE : INITIALIZED<br>
            DATABASE : ATLAS<br>
            DETECTION : HYBRID ML/DL<br>
            MODE : PASSIVE ANALYSIS
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["COMMAND CENTER", "NETWORK MONITOR", "INCIDENT CENTER", "MODEL LABORATORY"],
        key="neuralshield_main_navigation",
    )
    st.markdown("---")
    if st.button("⟳ REFRESH SYSTEM", use_container_width=True):
        st.rerun()
    st.caption("NEURALSHIELD BUILD 1.0.0")

# ============================================================
# COMMAND CENTER
# ============================================================
if page == "COMMAND CENTER":
    render_page_header("NEURALSHIELD", "AI-POWERED NETWORK DEFENSE // COMMAND CENTER")
    now = datetime.now().strftime("%d %b %Y | %H:%M:%S")
    st.caption(f"SYSTEM TIME: {now}   |   ENVIRONMENT: DEVELOPMENT   |   MONITORING: PASSIVE")

    total_detections = len(detections)
    critical = sum(a.get("risk_level") == "CRITICAL" for a in alerts)
    high = sum(a.get("risk_level") == "HIGH" for a in alerts)
    open_alerts = sum(a.get("status") == "OPEN" for a in alerts)

    render_section("01 // SYSTEM TELEMETRY")
    metric_columns = st.columns(4)
    metrics = [
        ("TOTAL DETECTIONS", total_detections, ""),
        ("CRITICAL THREATS", critical, "metric-danger"),
        ("HIGH RISK ALERTS", high, "metric-warning"),
        ("OPEN INCIDENTS", open_alerts, "metric-green"),
    ]
    for col, (label, value, color_class) in zip(metric_columns, metrics):
        with col:
            with st.container(border=True):
                st.markdown(f'<div class="metric-label">{label}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value {color_class}">{value:,}</div>', unsafe_allow_html=True)

    render_section("02 // THREAT INTELLIGENCE")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### THREAT DISTRIBUTION")
        if detections:
            detection_df = pd.DataFrame(detections)
            if "predicted_attack" in detection_df.columns:
                counts = detection_df["predicted_attack"].fillna("Unknown").value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=counts.index, values=counts.values, hole=.68,
                    textinfo="label+percent",
                    marker=dict(
                        colors=["#00d9ff", "#ff455d", "#ffae42", "#a16eff", "#47ffb0", "#438cff", "#ff7ac6", "#b4d455"],
                        line=dict(color="#050a12", width=3),
                    ),
                )])
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#d9faff", family="Rajdhani", size=13),
                    showlegend=True, legend=dict(orientation="h", y=-.15),
                    margin=dict(t=20, b=30, l=10, r=10), height=340,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Attack category data is not available in the detection records.")
        else:
            st.info("Awaiting detection telemetry.")

    with right:
        st.markdown("#### RISK CLASSIFICATION")
        risk_counts = pd.Series([a.get("risk_level", "UNKNOWN") for a in alerts], dtype="object").value_counts()
        if not risk_counts.empty:
            risk_colors = {"CRITICAL": "#ff304f", "HIGH": "#ff9f43", "MEDIUM": "#ffe66d", "LOW": "#47ffb0"}
            fig = go.Figure(go.Bar(
                x=risk_counts.index, y=risk_counts.values,
                marker_color=[risk_colors.get(level, "#00d9ff") for level in risk_counts.index],
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#d9faff", family="Rajdhani", size=13), height=340,
                margin=dict(t=20, b=30, l=10, r=10),
                xaxis=dict(showgrid=False, title="RISK LEVEL"),
                yaxis=dict(gridcolor="#173347", title="ALERT COUNT"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No alert distribution available yet.")

    render_section("03 // RECENT DETECTION LOG")
    if detections:
        recent = pd.DataFrame(detections)
        display_columns = ["predicted_attack", "malicious_probability", "risk_score", "risk_level", "final_status"]
        available_columns = [column for column in display_columns if column in recent.columns]
        if available_columns:
            st.dataframe(recent[available_columns].head(10), use_container_width=True, hide_index=True)
        else:
            st.info("Detection records do not contain the expected display fields.")
    else:
        st.info("No traffic records detected yet.")
    render_footer()

# ============================================================
# NETWORK MONITOR
# ============================================================
elif page == "NETWORK MONITOR":
    render_live_monitor()
    if detections:
        detection_df = pd.DataFrame(detections)
        st.divider()
        render_section("DETECTION TELEMETRY")
        st.metric("ANALYSED RECORDS", len(detection_df))
        if "risk_score" in detection_df.columns:
            fig = px.histogram(detection_df, x="risk_score", nbins=20, title="RISK SCORE DISTRIBUTION")
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font_color="#d9faff",
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(detection_df, use_container_width=True, hide_index=True)
    render_footer()

# ============================================================
# INCIDENT CENTER
# ============================================================
elif page == "INCIDENT CENTER":
    render_page_header("INCIDENT CENTER", "SECURITY EVENTS // ALERT TRIAGE AND RESOLUTION")
    render_section("01 // SECURITY INCIDENTS")
    if alerts:
        for alert in alerts:
            level = str(alert.get("risk_level", "UNKNOWN"))
            attack = escape(str(alert.get("predicted_attack", "Unknown")))
            risk_score = escape(str(alert.get("risk_score", "N/A")))
            status = escape(str(alert.get("status", "UNKNOWN")))
            alert_id = str(alert.get("_id", "N/A"))
            color = "#ff304f" if level == "CRITICAL" else "#ff9f43" if level == "HIGH" else "#00d9ff"
            safe_level = escape(level)
            card_html = f"""
                <div class="threat-card" style="border-left-color:{color}">
                    <div style="font-family:Orbitron;font-weight:900;color:{color};letter-spacing:2px;font-size:15px;">
                        {safe_level} ALERT
                    </div>
                    <div style="margin-top:10px;">ATTACK : {attack}</div>
                    <div>RISK SCORE : {risk_score}</div>
                    <div>STATUS : {status}</div>
                    <div style="color:#83a9bf;font-size:12px;">ID : {escape(alert_id)}</div>
                </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            if alert.get("status") == "OPEN":
                if st.button("RESOLVE INCIDENT", key=f"resolve_incident_{alert_id}"):
                    if resolve_incident(alert_id):
                        st.success("Incident resolved.")
                        st.rerun()
    else:
        st.success("NO ACTIVE SECURITY ALERTS")
    render_footer()

# ============================================================
# MODEL LABORATORY
# ============================================================
elif page == "MODEL LABORATORY":
    render_page_header("MODEL LABORATORY", "NEURAL DETECTION CORE // MODEL INVENTORY")
    render_section("01 // DETECTION MODELS")
    models = [
        ("XGBoost Binary Classifier", "SUPERVISED MACHINE LEARNING"),
        ("XGBoost Multiclass Classifier", "ATTACK CLASSIFICATION"),
        ("Deep Learning Autoencoder", "ANOMALY DETECTION"),
        ("Hybrid Decision Engine", "ENSEMBLE DETECTION"),
        ("Risk Scoring Engine", "THREAT PRIORITIZATION"),
    ]
    for name, category in models:
        model_html = f"""
            <div class="model-card">
                <div class="model-name">{escape(name)}</div>
                <div class="model-category">{escape(category)} // INTEGRATED</div>
            </div>
        """
        st.markdown(model_html, unsafe_allow_html=True)
    st.divider()
    st.caption(
        "Model performance should be interpreted using the latest held-out test evaluation. "
        "The model inventory describes configured project components, not a live runtime health check."
    )
    render_footer()
