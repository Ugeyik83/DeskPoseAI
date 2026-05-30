"""
PostureGuard — Masabaşı Duruş İzleme
streamlit-webrtc ile gerçek zamanlı video + MediaPipe overlay
"""

import os
import time
import math
import queue
import cv2
import streamlit as st
from pathlib import Path
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

sys.path.insert(0, str(Path(__file__).parent))
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

from core.pose_analyzer import PoseAnalyzer
from core.alert_manager import AlertManager, AlertConfig
from core.session_logger import SessionLogger


st.set_page_config(
    page_title="DeskPoseAI",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design System ────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg: #0d1117; --surface: #161b22; --surface2: #1c2130;
        --border: #30363d; --border-sub: #21262d;
        --good: #3fb950; --good-bg: rgba(63,185,80,0.10);
        --warn: #d29922; --warn-bg: rgba(210,153,34,0.10);
        --crit: #f85149; --crit-bg: rgba(248,81,73,0.12);
        --info: #58a6ff; --info-bg: rgba(88,166,255,0.10);
        --text: #e6edf3; --text-sec: #8b949e; --text-muted: #484f58;
        --r: 10px; --rs: 6px;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text);
    }

    /* ── Cards ─────────────────────────────────────────────────── */
    .pg-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: var(--r); padding: 14px 16px; margin-bottom: 8px;
        transition: border-color 0.15s;
    }
    .pg-card:hover { border-color: #444c56; }

    .pg-label {
        font-size: 10px; font-weight: 600; letter-spacing: 0.09em;
        color: var(--text-sec); text-transform: uppercase; margin-bottom: 5px;
    }
    .pg-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 30px; font-weight: 600; line-height: 1;
        display: flex; align-items: baseline; gap: 3px;
    }
    .pg-unit  { font-size: 13px; color: var(--text-sec); font-weight: 400; }
    .pg-desc  { font-size: 11px; color: var(--text-sec); margin-top: 5px; }
    .pg-disc  { font-size: 10px; color: var(--text-muted); font-style: italic; margin-top: 3px; }

    /* ── Progress bar ───────────────────────────────────────────── */
    .prog-track {
        height: 4px; background: var(--border-sub);
        border-radius: 2px; margin-top: 9px; overflow: hidden;
    }
    .prog-fill { height: 100%; border-radius: 2px; transition: width 0.35s ease; }

    /* ── Status bar ─────────────────────────────────────────────── */
    .status-bar {
        border-radius: var(--r); padding: 11px 16px;
        font-size: 13px; font-weight: 600; letter-spacing: 0.02em;
        margin-bottom: 10px; display: flex; align-items: center; gap: 10px;
    }
    .status-good     { background: var(--good-bg); border: 1px solid var(--good); color: var(--good); }
    .status-warning  { background: var(--warn-bg); border: 1px solid var(--warn); color: var(--warn); }
    .status-critical {
        background: var(--crit-bg); border: 1px solid var(--crit); color: var(--crit);
        animation: pulse-crit 1.8s ease-in-out infinite;
    }
    .status-idle        { background: rgba(72,79,88,0.15); border: 1px solid #484f58; color: var(--text-sec); }
    .status-calibrating { background: var(--info-bg); border: 1px solid var(--info); color: var(--info); }

    @keyframes pulse-crit {
        0%, 100% { box-shadow: 0 0 0 0 rgba(248,81,73,0); }
        50%       { box-shadow: 0 0 0 6px rgba(248,81,73,0); }
    }

    /* ── Feedback list ──────────────────────────────────────────── */
    .fb-list { margin-top: 6px; }
    .fb-item {
        display: flex; align-items: flex-start; gap: 8px;
        padding: 6px 0; font-size: 12px; color: var(--text-sec);
        border-bottom: 1px solid var(--border-sub); line-height: 1.4;
    }
    .fb-item:last-child { border-bottom: none; }
    .fb-dot {
        width: 6px; height: 6px; border-radius: 50%;
        margin-top: 4px; flex-shrink: 0;
    }

    /* ── Badges ─────────────────────────────────────────────────── */
    .badge {
        display: inline-flex; align-items: center;
        padding: 2px 7px; border-radius: 4px;
        font-size: 9px; font-weight: 700; letter-spacing: 0.1em;
        font-family: 'JetBrains Mono', monospace;
        vertical-align: middle; margin-left: 6px;
    }
    .badge-cal { background: var(--good-bg); border: 1px solid var(--good); color: var(--good); }
    .badge-est { background: rgba(72,79,88,0.2); border: 1px solid #484f58; color: var(--text-sec); }

    /* ── Alert log ──────────────────────────────────────────────── */
    .alert-item {
        display: flex; align-items: center; gap: 8px;
        padding: 5px 0; font-size: 11px;
        border-bottom: 1px solid var(--border-sub);
    }
    .alert-item:last-child { border-bottom: none; }
    .alert-time {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-muted); font-size: 10px; min-width: 52px;
    }

    /* ── Session stat chips ─────────────────────────────────────── */
    .stat-chips { display: flex; gap: 6px; margin-top: 10px; }
    .stat-chip  {
        flex: 1; text-align: center; padding: 8px 4px;
        background: var(--surface2); border-radius: var(--rs);
    }
    .stat-chip-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 17px; font-weight: 600; display: block; line-height: 1; margin-bottom: 2px;
    }
    .stat-chip-lbl { font-size: 9px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-sec); }

    /* ── Sidebar section titles ─────────────────────────────────── */
    .sb-title {
        font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--text-sec); margin: 4px 0 8px;
    }

    /* ── Threshold table ────────────────────────────────────────── */
    .thresh-table { font-size: 11px; width: 100%; border-collapse: collapse; }
    .thresh-table th, .thresh-table td { padding: 5px 6px; border: 1px solid #2a2d3a; text-align: center; }
    .thresh-table th { background: #1a1d27; color: #6b7280; font-weight: 600; }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)


# ─── Render helpers ──────────────────────────────────────────────────────────

def _score_ring(score, color):
    r, cx, cy = 50, 64, 64
    circ   = 2 * math.pi * r
    offset = circ * (1 - max(0.0, min(100.0, score)) / 100)
    label  = "İyi Duruş" if score >= 70 else "Geliştirilmeli" if score >= 40 else "Dikkat Gerekli"
    return f"""
    <div class="pg-card" style="text-align:center;padding:18px 16px 14px;">
        <div class="pg-label" style="margin-bottom:10px;">Oturum Skoru</div>
        <svg width="120" height="120" viewBox="0 0 128 128" style="display:block;margin:0 auto;">
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#21262d" stroke-width="10"/>
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="10"
                    stroke-dasharray="{circ:.1f}" stroke-dashoffset="{offset:.1f}"
                    stroke-linecap="round" transform="rotate(-90 {cx} {cy})"/>
            <text x="{cx}" y="{cy - 4}" text-anchor="middle"
                  font-family="JetBrains Mono,monospace" font-size="26" font-weight="700"
                  fill="{color}">{score:.0f}</text>
            <text x="{cx}" y="{cy + 16}" text-anchor="middle"
                  font-family="Inter,sans-serif" font-size="11" fill="#8b949e">/100</text>
        </svg>
        <div style="font-size:12px;color:{color};font-weight:600;margin-top:4px;">{label}</div>
    </div>"""


def _metric(label, val, unit, desc, color, prog_pct=None, disclaimer=""):
    prog = ""
    if prog_pct is not None:
        pct  = max(0.0, min(100.0, prog_pct))
        prog = f'<div class="prog-track"><div class="prog-fill" style="width:{pct:.1f}%;background:{color};"></div></div>'
    disc = f'<div class="pg-disc">{disclaimer}</div>' if disclaimer else ""
    return (
        f'<div class="pg-card">'
        f'<div class="pg-label">{label}</div>'
        f'<div class="pg-value" style="color:{color}">{val}<span class="pg-unit">&thinsp;{unit}</span></div>'
        f'{prog}'
        f'<div class="pg-desc">{desc}</div>'
        f'{disc}'
        f'</div>'
    )


# ─── WebRTC Video Processor ───────────────────────────────────────────────────

class PostureProcessor(VideoProcessorBase):
    def __init__(self):
        self.analyzer = PoseAnalyzer()
        self.metrics_queue = queue.Queue(maxsize=5)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")
        bgr = cv2.flip(bgr, 1)
        annotated, metrics = self.analyzer.analyze_frame(bgr)
        if metrics:
            if self.metrics_queue.full():
                try:
                    self.metrics_queue.get_nowait()
                except queue.Empty:
                    pass
            self.metrics_queue.put(metrics)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def get_latest_metrics(self):
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None


# ─── Session state ────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "logger":             None,
        "log_enabled":        True,
        "alert_cooldown":     30,
        "warn_thresh":        5,
        "consecutive_bad":    0,
        "last_alert_ts":      0.0,
        "alert_log":          [],
        "calibrating":        False,
        "calib_done":         False,
        "calib_frame_count":  0,
        "calib_triggered":    False,
        "good_streak_ts":     0.0,
        "no_face_since":      0.0,
        "show_roi":           False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🖥️ DeskPoseAI")
    st.markdown(
        '<div style="color:#8b949e;font-size:12px;margin-bottom:16px;">Masabaşı Duruş İzleme</div>',
        unsafe_allow_html=True)
    st.divider()

    # Kalibrasyon bölümü
    st.markdown('<div class="sb-title">Kalibrasyon</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:12px;color:#8b949e;margin-bottom:10px;line-height:1.5;">'
        'Dik oturuşunu baseline olarak kaydet. Tüm FHP skorları bu referansa göre hesaplanır.'
        '</div>',
        unsafe_allow_html=True)

    if st.session_state.calib_done:
        st.success("Kalibrasyon tamamlandı — baseline aktif")
    elif st.session_state.calibrating:
        st.info("Kalibrasyon devam ediyor...")
    else:
        st.caption("Henüz kalibrasyon yapılmadı — tahmini mod aktif")

    if st.button("Kalibre Et — Dik Otur ve Bekle", use_container_width=True,
                 disabled=st.session_state.calibrating):
        st.session_state.calibrating       = True
        st.session_state.calib_done        = False
        st.session_state.calib_frame_count = 0
        st.session_state.calib_triggered   = False

    if st.button("Kalibrasyonu Sıfırla", use_container_width=True):
        st.session_state.calibrating       = False
        st.session_state.calib_done        = False
        st.session_state.calib_frame_count = 0
        st.session_state.calib_triggered   = False

    st.divider()

    # Bildirim ayarları
    st.markdown('<div class="sb-title">Bildirim Ayarları</div>', unsafe_allow_html=True)
    st.session_state.alert_cooldown = st.slider(
        "Bildirim aralığı (sn)", 10, 120, 30, 5,
        help="İki uyarı arasındaki minimum bekleme süresi")
    st.session_state.warn_thresh = st.slider(
        "Uyarı eşiği (kare)", 2, 15, 5, 1,
        help="Kaç ardışık kötü kare sonrası uyarı verilsin")

    st.divider()

    # Loglama
    st.markdown('<div class="sb-title">Oturum Kaydı</div>', unsafe_allow_html=True)
    st.session_state.log_enabled = st.toggle("CSV olarak kaydet", value=True)
    st.session_state.show_roi    = st.toggle("ROI bölgelerini göster (rPPG)", value=False)
    st.divider()

    with st.expander("Ergonomi Eşikleri (ISO 9241 / NIOSH)"):
        st.markdown("""
        <table class="thresh-table">
        <tr><th>Metrik</th><th>🟢 İyi</th><th>🟡 Uyarı</th><th>🔴 Kritik</th></tr>
        <tr><td>FHP Skoru</td><td>&lt;25</td><td>25–50</td><td>&gt;50</td></tr>
        <tr><td>Baş Eğimi</td><td>&lt;5°</td><td>5–10°</td><td>&gt;10°</td></tr>
        <tr><td>Omuz Asim.</td><td>&lt;3%</td><td>3–6%</td><td>&gt;6%</td></tr>
        <tr><td>Göz Kırpma</td><td>≥10/dk</td><td>5–10/dk</td><td>&lt;5/dk</td></tr>
        <tr><td>Ekran Mes.</td><td>50–100cm</td><td>40–50cm</td><td>&lt;40cm</td></tr>
        <tr><td>Oturma</td><td>&lt;30dk</td><td>30–60dk</td><td>&gt;60dk</td></tr>
        <tr><td>Kalp Atışı</td><td>60–100</td><td>50–60/100–120</td><td>&lt;50/&gt;120</td></tr>
        </table>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown(
        '<div style="font-size:10px;color:#484f58;font-style:italic;">'
        'FHP Risk Skoru klinik CVA ölçümü değildir.</div>',
        unsafe_allow_html=True)


# ─── Layout ──────────────────────────────────────────────────────────────────

st.markdown(
    '<div style="font-size:20px;font-weight:700;letter-spacing:-0.02em;margin-bottom:2px;">'
    '🪑&nbsp; Duruş İzleme Paneli</div>'
    '<div style="font-size:11px;color:#8b949e;margin-bottom:12px;">'
    'Gerçek zamanlı duruş analizi ve sağlık metrikleri</div>',
    unsafe_allow_html=True)
st.divider()

col_cam, col_metrics = st.columns([3, 2], gap="large")

RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

with col_cam:
    ctx = webrtc_streamer(
        key="posture_guard",
        video_processor_factory=PostureProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )
    status_ph           = st.empty()
    calib_ph            = st.empty()
    feedback_ph         = st.empty()
    posture_alert_ph    = st.empty()
    stationary_alert_ph = st.empty()

with col_metrics:
    score_ph   = st.empty()
    metrics_ph = st.empty()
    session_ph = st.empty()
    alert_ph   = st.empty()


# ─── Canlı güncelleme ────────────────────────────────────────────────────────

if ctx.state.playing and ctx.video_processor:
    processor = ctx.video_processor

    # ── Kalibrasyon ──────────────────────────────────────────────────────────
    if st.session_state.calibrating and not st.session_state.calib_done:
        if not st.session_state.calib_triggered:
            processor.analyzer.reset_calibration()
            processor.analyzer.start_calibration()
            st.session_state.calib_triggered = True

        st.session_state.calib_frame_count = len(processor.analyzer._calib_buffer)
        progress  = min(st.session_state.calib_frame_count / 45, 1.0)
        pct       = int(progress * 100)
        secs_left = max(0, round(3 * (1 - progress)))

        with col_cam:
            calib_ph.markdown(
                f'<div class="status-bar status-calibrating">'
                f'🎯&nbsp; Dik otur, hareket etme...&nbsp;'
                f'<span style="font-size:15px;font-weight:700;">{pct}%</span>'
                f'&nbsp;({secs_left} sn kaldı)</div>',
                unsafe_allow_html=True)
            st.progress(progress)

        if processor.analyzer.baseline.valid:
            st.session_state.calibrating     = False
            st.session_state.calib_done      = True
            st.session_state.calib_triggered = False
            st.balloons()

    else:
        with col_cam:
            calib_ph.empty()
        if not st.session_state.calib_done and not st.session_state.calibrating:
            if processor.analyzer.baseline.valid:
                processor.analyzer.reset_calibration()
                
    processor.analyzer.show_roi = st.session_state.show_roi
    metrics = processor.get_latest_metrics()

    if metrics:
        now = time.time()
        st.session_state.no_face_since = 0.0

        if metrics.risk_level == "critical":
            posture_alert_ph.error("🚨 **Duruş Kritik** — Omurgana dikkat et, pozisyonunu düzelt!")
        elif metrics.risk_level == "warning":
            posture_alert_ph.warning("⚡ **Duruş Uyarısı** — Başını ve sırtını kontrol et.")
        else:
            posture_alert_ph.empty()

        stationary_alert_ph.empty()

        if metrics.risk_level != "good":
            st.session_state.consecutive_bad += 1
        else:
            st.session_state.consecutive_bad = 0

        if (st.session_state.consecutive_bad >= st.session_state.warn_thresh and
                now - st.session_state.last_alert_ts > st.session_state.alert_cooldown):
            st.session_state.last_alert_ts   = now
            st.session_state.consecutive_bad = 0
            st.session_state.alert_log.insert(0, {
                "ts":    time.strftime("%H:%M:%S"),
                "level": metrics.risk_level,
                "msg":   metrics.feedback[0] if metrics.feedback else "",
            })
            if len(st.session_state.alert_log) > 20:
                st.session_state.alert_log.pop()

        if st.session_state.log_enabled:
            if st.session_state.logger is None:
                st.session_state.logger = SessionLogger("logs")
                st.session_state.logger.start_session()
            st.session_state.logger.log(metrics)

    else:
        if st.session_state.no_face_since == 0.0:
            st.session_state.no_face_since = time.time()
        elif time.time() - st.session_state.no_face_since >= 2.0:
            posture_alert_ph.empty()
            stationary_alert_ph.empty()

        with col_cam:
            status_ph.markdown(
                '<div class="status-bar status-idle">'
                '👤&nbsp; Yüz/vücut algılanamadı — kameranın önüne geç'
                '</div>',
                unsafe_allow_html=True)
        metrics = None

    if metrics:
        risk        = metrics.risk_level
        icons       = {"good": "✅", "warning": "⚡", "critical": "🚨"}
        labels      = {
            "good":     "Duruş İyi",
            "warning":  "Uyarı — Duruşunu Düzelt",
            "critical": "Kritik — Hemen Düzelt",
        }
        calib_badge = (
            '<span class="badge badge-cal">CAL</span>' if metrics.calibration_active
            else '<span class="badge badge-est">EST</span>'
        )

        # ── Kamera altı durum çubuğu ─────────────────────────────────────────
        with col_cam:
            status_ph.markdown(
                f'<div class="status-bar status-{risk}">'
                f'{icons[risk]}&nbsp; {labels[risk]}{calib_badge}'
                f'</div>',
                unsafe_allow_html=True)

            # Feedback listesi
            fb_items = ""
            dot_color = {"good": "#3fb950", "warning": "#d29922", "critical": "#f85149"}[risk]
            for f in metrics.feedback:
                fb_items += (
                    f'<div class="fb-item">'
                    f'<span class="fb-dot" style="background:{dot_color};"></span>{f}'
                    f'</div>'
                )
            if fb_items:
                feedback_ph.markdown(
                    f'<div class="fb-list">{fb_items}</div>',
                    unsafe_allow_html=True)
            else:
                feedback_ph.empty()

        # ── Metrik renkleri ve değerleri ──────────────────────────────────────
        risk_color  = {"good": "#3fb950", "warning": "#d29922", "critical": "#f85149"}[risk]
        score       = processor.analyzer.get_posture_score()
        score_color = "#3fb950" if score >= 70 else "#d29922" if score >= 40 else "#f85149"

        fhp       = metrics.fhp_risk_score
        fhp_color = "#3fb950" if fhp < 25 else "#d29922" if fhp < 50 else "#f85149"
        fhp_label = "FHP Risk Skoru — Kalibre" if metrics.calibration_active else "FHP Risk Skoru — Tahmini"
        fhp_disc  = "Kişisel baseline'a göre" if metrics.calibration_active else "Kalibre et → daha doğru sonuç"

        # Kalp atışı
        hr = metrics.heart_rate
        if hr == -2.0:
            hr_color, hr_text, hr_unit, hr_desc, hr_prog = "#8b949e", "Hareket", "", "Sakin dur — ölçüm devam eder", None
        elif hr == -3.0:
            hr_color, hr_text, hr_unit, hr_desc, hr_prog = "#d29922", "Zayıf Sinyal", "", "İyi aydınlatma gerekli", None
        elif hr < 0:
            buf_pct = int(min(len(processor.analyzer._rppg_buffer) / 300 * 100, 100))
            hr_color, hr_text, hr_unit, hr_desc, hr_prog = "#8b949e", f"%{buf_pct}", "", "Yükleniyor... (~20 sn)", buf_pct
        elif hr < 60:
            hr_color, hr_text, hr_unit, hr_desc, hr_prog = "#d29922", f"{hr:.0f}", "BPM", "Düşük — dinlenme veya soğuk", (hr - 40) / 120 * 100
        elif hr <= 100:
            hr_color, hr_text, hr_unit, hr_desc, hr_prog = "#3fb950", f"{hr:.0f}", "BPM", "Normal aralık (60–100)", (hr - 40) / 120 * 100
        else:
            hr_color, hr_text, hr_unit, hr_desc, hr_prog = "#f85149", f"{hr:.0f}", "BPM", "Yüksek — stres/yorgunluk olabilir", min((hr - 40) / 120 * 100, 100)

        # Göz kırpma
        br = metrics.blink_rate
        if br < 0:
            blink_color, blink_text, blink_unit, blink_desc, blink_prog = "#8b949e", "Ölçülüyor", "", "60 sn sonra aktif", None
        elif br < 5:
            blink_color, blink_text, blink_unit, blink_desc, blink_prog = "#f85149", f"{br:.0f}", "/dk", "Çok az — CVS riski yüksek", br / 20 * 100
        elif br < 10:
            blink_color, blink_text, blink_unit, blink_desc, blink_prog = "#d29922", f"{br:.0f}", "/dk", "Az — gözlerini bilinçli kırp", br / 20 * 100
        else:
            blink_color, blink_text, blink_unit, blink_desc, blink_prog = "#3fb950", f"{br:.0f}", "/dk", "Normal (15–20/dk ideal)", min(br / 20 * 100, 100)

        # PERCLOS
        pc = metrics.perclos
        if pc < 0:
            perc_color, perc_text, perc_unit, perc_desc, perc_prog = "#8b949e", "Ölçülüyor", "", "~30 sn sonra aktif", None
        elif pc > 30:
            perc_color, perc_text, perc_unit, perc_desc, perc_prog = "#f85149", f"{pc:.1f}", "%", "Ciddi yorgunluk", pc
        elif pc > 15:
            perc_color, perc_text, perc_unit, perc_desc, perc_prog = "#d29922", f"{pc:.1f}", "%", "Orta yorgunluk", pc
        elif pc > 8:
            perc_color, perc_text, perc_unit, perc_desc, perc_prog = "#d29922", f"{pc:.1f}", "%", "Hafif yorgunluk", pc
        else:
            perc_color, perc_text, perc_unit, perc_desc, perc_prog = "#3fb950", f"{pc:.1f}", "%", "Normal uyanıklık", pc



        # Ekran mesafesi
        d = metrics.screen_distance
        if d < 0:
            dist_color, dist_text, dist_unit, dist_desc, dist_prog = "#8b949e", "Ölçülüyor", "", "Kalibrasyon sonrası daha doğru", None
        elif d < 40:
            dist_color, dist_text, dist_unit, dist_desc, dist_prog = "#f85149", f"{d:.0f}", "cm", "Çok yakın — geri çekil (min 40 cm)", d / 150 * 100
        elif d < 50:
            dist_color, dist_text, dist_unit, dist_desc, dist_prog = "#d29922", f"{d:.0f}", "cm", "Biraz yakın — ideal: 50–100 cm", d / 150 * 100
        elif d <= 100:
            dist_color, dist_text, dist_unit, dist_desc, dist_prog = "#3fb950", f"{d:.0f}", "cm", "İdeal mesafe (ISO 9241)", d / 150 * 100
        else:
            dist_color, dist_text, dist_unit, dist_desc, dist_prog = "#d29922", f"{d:.0f}", "cm", "Çok uzak — öne eğilme riski", min(d / 150 * 100, 100)

        # Hareketsizlik
        stat_min = processor.analyzer.session.stationary_minutes
        if stat_min >= 60:
            stat_color, stat_desc = "#f85149", "Kalk, 2 dk yürü"
        elif stat_min >= 30:
            stat_color, stat_desc = "#d29922", "Kısa mola ver, esne"
        else:
            stat_color, stat_desc = "#3fb950", "Normal"
        stat_prog = min(stat_min / 90 * 100, 100)

        # ── Metrik paneli ─────────────────────────────────────────────────────
        with col_metrics:
            score_ph.markdown(_score_ring(score, score_color), unsafe_allow_html=True)

            metrics_ph.markdown(
                _metric(fhp_label, f"{fhp:.0f}", "/100", fhp_color,
                        fhp_color, prog_pct=fhp, disclaimer=fhp_disc)
                + _metric("Kalp Atışı (rPPG)", hr_text, hr_unit, hr_desc,
                          hr_color, prog_pct=hr_prog)
                + _metric("Ekran Mesafesi", dist_text, dist_unit, dist_desc,
                          dist_color, prog_pct=dist_prog)
                + _metric("Göz Kırpma", blink_text, blink_unit, blink_desc,
                          blink_color, prog_pct=blink_prog)
                + _metric("PERCLOS (Göz Kapalı %)", perc_text, perc_unit, perc_desc,
                          perc_color, prog_pct=perc_prog)
                + _metric("Baş Eğim Açısı (Roll)",
                          str(metrics.head_tilt_angle), "°",
                          "arctan(Δy/Δx) — yana dönüşle karışabilir",
                          risk_color,
                          prog_pct=min(abs(metrics.head_tilt_angle) / 20 * 100, 100))
                + _metric("Omuz Asimetrisi",
                          f"{metrics.shoulder_asymmetry * 100:.1f}", "%",
                          "Sol/sağ omuz yükseklik farkı",
                          risk_color,
                          prog_pct=min(metrics.shoulder_asymmetry * 100 / 10 * 100, 100))
                + _metric("Hareketsizlik",
                          f"{stat_min:.1f}", "dk",
                          stat_desc, stat_color, prog_pct=stat_prog),
                unsafe_allow_html=True)

            # Oturum istatistikleri
            s     = processor.analyzer.session
            total = max(s.total_frames, 1)
            fps   = getattr(processor.analyzer, '_current_fps', 0.0)
            g_pct = s.good_frames / total * 100
            w_pct = s.warning_frames / total * 100
            c_pct = s.critical_frames / total * 100

            session_ph.markdown(f"""
            <div class="pg-card">
                <div class="pg-label">Oturum İstatistikleri</div>
                <div class="stat-chips">
                    <div class="stat-chip">
                        <span class="stat-chip-val" style="color:#3fb950">{g_pct:.0f}%</span>
                        <span class="stat-chip-lbl">İyi</span>
                    </div>
                    <div class="stat-chip">
                        <span class="stat-chip-val" style="color:#d29922">{w_pct:.0f}%</span>
                        <span class="stat-chip-lbl">Uyarı</span>
                    </div>
                    <div class="stat-chip">
                        <span class="stat-chip-val" style="color:#f85149">{c_pct:.0f}%</span>
                        <span class="stat-chip-lbl">Kritik</span>
                    </div>
                </div>
                <div style="margin-top:10px;font-size:11px;color:#8b949e;">
                    Ort. FHP&nbsp;<b>{s.avg_fhp_score:.1f}</b>
                    &nbsp;&nbsp;Ort. Eğim&nbsp;<b>{s.avg_tilt:.1f}°</b>
                    &nbsp;&nbsp;{s.total_frames} kare
                    &nbsp;&nbsp;⚡&nbsp;{fps:.1f} FPS
                </div>
            </div>""", unsafe_allow_html=True)

            # Son uyarılar
            if st.session_state.alert_log:
                rows = ""
                for a in st.session_state.alert_log[:5]:
                    c = "#f85149" if a["level"] == "critical" else "#d29922"
                    rows += (
                        f'<div class="alert-item">'
                        f'<span class="alert-time">{a["ts"]}</span>'
                        f'<span style="color:{c};">{a["msg"]}</span>'
                        f'</div>'
                    )
                alert_ph.markdown(
                    f'<div class="pg-card"><div class="pg-label">Son Uyarılar</div>{rows}</div>',
                    unsafe_allow_html=True)

elif not ctx.state.playing:
    with col_cam:
        status_ph.markdown(
            '<div class="status-bar status-idle">'
            '▶&nbsp; START butonu ile izlemeyi başlat'
            '</div>',
            unsafe_allow_html=True)
    with col_metrics:
        score_ph.markdown("""
        <div class="pg-card" style="text-align:center;padding:18px 16px 14px;">
            <div class="pg-label" style="margin-bottom:10px;">Oturum Skoru</div>
            <svg width="120" height="120" viewBox="0 0 128 128" style="display:block;margin:0 auto;">
                <circle cx="64" cy="64" r="50" fill="none" stroke="#21262d" stroke-width="10"/>
                <text x="64" y="60" text-anchor="middle"
                      font-family="JetBrains Mono,monospace" font-size="26" font-weight="700"
                      fill="#484f58">—</text>
                <text x="64" y="80" text-anchor="middle"
                      font-family="Inter,sans-serif" font-size="11" fill="#484f58">bekleniyor</text>
            </svg>
        </div>""", unsafe_allow_html=True)

if ctx.state.playing:
    time.sleep(0.2)
    st.rerun()
