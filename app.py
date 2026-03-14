"""
app.py
------
Interactive Streamlit dashboard for real-time bearing fault diagnosis.

Features:
  - Upload a raw .mat or .npy vibration signal
  - Visualize the raw signal, FFT spectrum, and envelope
  - Run fault diagnosis with the trained model
  - Display prediction confidence and SHAP feature importance
  - Show bearing characteristic frequencies overlaid on spectrum

Run:
    streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.io
from scipy.fft import rfft, rfftfreq
from scipy.signal import hilbert
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.features.feature_extraction import FeatureExtractor, BearingGeometry

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Bearing Fault Detection",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp {
        background-color: #0d1117;
    }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #e6edf3; }

    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-value { font-size: 2rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; color: #58a6ff; }

    .fault-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .normal { background: #0f4a31; color: #3fb950; border: 1px solid #3fb950; }
    .inner  { background: #4a1a0f; color: #ff7b72; border: 1px solid #ff7b72; }
    .outer  { background: #4a3a0f; color: #e3b341; border: 1px solid #e3b341; }
    .ball   { background: #1a0f4a; color: #d2a8ff; border: 1px solid #d2a8ff; }

    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8b949e;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES = ["Normal", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]
CLASS_COLORS = ["#3fb950", "#ff7b72", "#e3b341", "#d2a8ff"]
CLASS_CSS = ["normal", "inner", "outer", "ball"]
ICONS = ["✅", "🔴", "🟡", "🟣"]

PLOTLY_THEME = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3", family="IBM Plex Mono"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load trained XGBoost model if available."""
    model_path = Path("models/xgboost.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_resource
def get_extractor(fs: int, rpm: float):
    geometry = BearingGeometry(rpm=rpm)
    return FeatureExtractor(fs=fs, bearing=geometry), geometry

def load_signal(uploaded_file) -> np.ndarray | None:
    """Load signal from .mat, .npy, or .csv upload."""
    fname = uploaded_file.name
    if fname.endswith(".mat"):
        mat = scipy.io.loadmat(uploaded_file)
        for key in mat:
            if "DE_time" in key:
                return mat[key].squeeze().astype(np.float32)
        # Return first numeric array found
        for key in mat:
            if not key.startswith("_"):
                arr = mat[key]
                if isinstance(arr, np.ndarray) and arr.ndim in [1, 2]:
                    return arr.squeeze().astype(np.float32)
    elif fname.endswith(".npy"):
        return np.load(uploaded_file).squeeze().astype(np.float32)
    elif fname.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(uploaded_file, header=None)
        return df.iloc[:, 0].values.astype(np.float32)
    return None

def compute_fft(signal: np.ndarray, fs: int):
    N = len(signal)
    freqs = rfftfreq(N, d=1.0 / fs)
    mag = np.abs(rfft(signal)) / N
    return freqs, mag

def compute_envelope(signal: np.ndarray):
    analytic = hilbert(signal)
    return np.abs(analytic)

def generate_demo_signal(fault_type: int, fs: int = 12000, duration: float = 0.5) -> np.ndarray:
    """Generate a synthetic bearing signal for demo purposes."""
    t = np.linspace(0, duration, int(fs * duration))
    geometry = BearingGeometry()

    # Base signal
    signal = 0.5 * np.sin(2 * np.pi * geometry.fr * t)
    signal += 0.3 * np.sin(2 * np.pi * 2 * geometry.fr * t)
    signal += 0.1 * np.random.randn(len(t))

    if fault_type == 1:  # Inner Race
        fault_freq = geometry.bpfi
        signal += 1.2 * np.sin(2 * np.pi * fault_freq * t) * (1 + 0.5 * np.sin(2 * np.pi * geometry.fr * t))
        signal += 0.6 * np.sin(2 * np.pi * 2 * fault_freq * t)
    elif fault_type == 2:  # Outer Race
        fault_freq = geometry.bpfo
        signal += 1.4 * np.sin(2 * np.pi * fault_freq * t)
        signal += 0.7 * np.sin(2 * np.pi * 2 * fault_freq * t)
    elif fault_type == 3:  # Ball
        fault_freq = geometry.bsf
        signal += 0.9 * np.sin(2 * np.pi * fault_freq * t) * np.sin(2 * np.pi * geometry.ftf * t)

    return signal.astype(np.float32)

# ─── Plots ───────────────────────────────────────────────────────────────────

def plot_signal(signal: np.ndarray, fs: int) -> go.Figure:
    t = np.arange(len(signal)) / fs * 1000  # ms
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=t[:4000], y=signal[:4000],
        mode="lines", line=dict(color="#58a6ff", width=1),
        name="Vibration"
    ))
    fig.update_layout(
        title="Time Domain Signal", xaxis_title="Time (ms)", yaxis_title="Amplitude (g)",
        height=280, **PLOTLY_THEME
    )
    return fig

def plot_fft(freqs, mag, geometry: BearingGeometry, zoom_max: float = 6000) -> go.Figure:
    mask = freqs <= zoom_max
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=freqs[mask], y=mag[mask],
        mode="lines", line=dict(color="#3fb950", width=1),
        name="FFT Magnitude"
    ))

    # Overlay characteristic frequencies
    fault_freqs = {
        "BPFI": (geometry.bpfi, "#ff7b72"),
        "BPFO": (geometry.bpfo, "#e3b341"),
        "BSF":  (geometry.bsf, "#d2a8ff"),
        "FTF":  (geometry.ftf, "#79c0ff"),
    }
    for name, (freq, color) in fault_freqs.items():
        for h in [1, 2, 3]:
            fig.add_vline(
                x=freq * h, line=dict(color=color, dash="dash", width=1.5 if h == 1 else 0.7),
                annotation_text=f"{name}" if h == 1 else "",
                annotation_font=dict(color=color, size=10),
            )

    fig.update_layout(
        title="FFT Spectrum + Fault Frequencies", xaxis_title="Frequency (Hz)", yaxis_title="|X(f)|",
        height=300, **PLOTLY_THEME
    )
    return fig

def plot_envelope(signal: np.ndarray, fs: int, geometry: BearingGeometry) -> go.Figure:
    env = compute_envelope(signal[:4096])
    env_freqs, env_mag = compute_fft(env, fs)
    mask = env_freqs <= 1000

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=env_freqs[mask], y=env_mag[mask],
        mode="lines", line=dict(color="#e3b341", width=1.5),
        name="Envelope Spectrum", fill="tozeroy",
        fillcolor="rgba(227, 179, 65, 0.1)"
    ))

    for name, freq, color in [
        ("BPFI", geometry.bpfi, "#ff7b72"),
        ("BPFO", geometry.bpfo, "#e3b341"),
    ]:
        if freq <= 1000:
            fig.add_vline(x=freq, line=dict(color=color, dash="dash"),
                          annotation_text=name, annotation_font=dict(color=color))

    fig.update_layout(
        title="Envelope Spectrum (Hilbert Transform)", xaxis_title="Frequency (Hz)", yaxis_title="|Env(f)|",
        height=260, **PLOTLY_THEME
    )
    return fig

def plot_probabilities(probs: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=CLASS_NAMES, y=probs * 100,
        marker_color=CLASS_COLORS,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        title="Prediction Confidence",
        yaxis_title="Probability (%)", yaxis_range=[0, 115],
        height=300, **PLOTLY_THEME
    )
    return fig

def plot_feature_importance(feature_names, importances, top_n=15) -> go.Figure:
    idx = np.argsort(importances)[-top_n:]
    fig = go.Figure(go.Bar(
        x=importances[idx], y=[feature_names[i] for i in idx],
        orientation="h", marker_color="#58a6ff",
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance", height=400, **PLOTLY_THEME
    )
    return fig

# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    # ── Header ──
    st.markdown("# 🔩 Bearing Fault Detection")
    st.markdown("**Real-time predictive maintenance powered by signal processing & ML**")
    st.divider()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        fs = st.selectbox("Sampling Rate (Hz)", [12000, 48000], index=0)
        rpm = st.slider("Shaft Speed (RPM)", 1700, 1800, 1797)
        window_size = st.select_slider("Window Size", options=[1024, 2048, 4096], value=2048)

        st.divider()
        st.markdown("### 📥 Data Source")
        data_source = st.radio("Input", ["Upload Signal", "Demo Signal"])

        extractor, geometry = get_extractor(fs, rpm)

        if data_source == "Demo Signal":
            demo_fault = st.selectbox(
                "Simulate fault type",
                options=[0, 1, 2, 3],
                format_func=lambda x: ICONS[x] + " " + CLASS_NAMES[x],
            )

        st.divider()
        st.markdown("### 📐 Bearing Parameters")
        for k, v in geometry.summary().items():
            st.markdown(f"**{k}**: `{v} Hz`")

    # ── Signal Loading ──
    signal = None

    if data_source == "Upload Signal":
        uploaded = st.file_uploader(
            "Upload vibration signal (.mat, .npy, .csv)",
            type=["mat", "npy", "csv"]
        )
        if uploaded:
            with st.spinner("Loading signal..."):
                signal = load_signal(uploaded)
            if signal is None:
                st.error("Could not parse signal. Check file format.")
    else:
        signal = generate_demo_signal(demo_fault, fs=fs)
        st.info(f"🧪 Demo signal: **{CLASS_NAMES[demo_fault]}** (synthetic, for illustration)")

    if signal is None:
        # Landing state
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Upload a signal</div>
                <div class="metric-value">📂</div>
                <div class="metric-label">.mat · .npy · .csv</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Or try a demo</div>
                <div class="metric-value">🧪</div>
                <div class="metric-label">4 fault types</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Get diagnosis</div>
                <div class="metric-value">🔍</div>
                <div class="metric-label">in < 1 second</div>
            </div>
            """, unsafe_allow_html=True)
        return

    # ── Process Signal ──
    window = signal[:window_size]
    if len(window) < window_size:
        st.warning(f"Signal shorter than window ({len(signal)} < {window_size}). Padding with zeros.")
        window = np.pad(window, (0, window_size - len(window)))

    with st.spinner("🔬 Analyzing signal..."):
        # Extract features
        features = extractor.extract(window)
        X_feat = np.array(list(features.values()), dtype=np.float32).reshape(1, -1)

        # FFT
        freqs, mag = compute_fft(window, fs)

        # Model prediction
        model = load_model()
        if model is not None:
            proba = model.predict_proba(X_feat)[0]
            pred_class = int(np.argmax(proba))
        else:
            # Mock prediction when no model loaded
            proba = np.array([0.05, 0.05, 0.85, 0.05]) if data_source == "Demo Signal" and demo_fault == 2 \
                    else np.array([0.92, 0.03, 0.03, 0.02])
            pred_class = int(np.argmax(proba))
            st.warning("⚠ No trained model found. Showing mock prediction. Run `train.py --save` first.")

    # ── Results ──
    st.markdown('<p class="section-header">Diagnosis Result</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fault Class</div>
            <br>
            <span class="fault-badge {CLASS_CSS[pred_class]}">
                {ICONS[pred_class]} {CLASS_NAMES[pred_class]}
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{proba[pred_class]*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        rms = float(np.sqrt(np.mean(window**2)))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RMS (g)</div>
            <div class="metric-value">{rms:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        from scipy.stats import kurtosis
        kurt = float(kurtosis(window))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Kurtosis</div>
            <div class="metric-value">{kurt:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Signal Plots ──
    st.markdown('<p class="section-header">Signal Analysis</p>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.plotly_chart(plot_signal(window, fs), use_container_width=True)
        st.plotly_chart(plot_envelope(window, fs, geometry), use_container_width=True)
    with col_right:
        st.plotly_chart(plot_fft(freqs, mag, geometry), use_container_width=True)
        st.plotly_chart(plot_probabilities(proba), use_container_width=True)

    st.divider()

    # ── Feature Importance ──
    st.markdown('<p class="section-header">Feature Importance (Model)</p>', unsafe_allow_html=True)

    if model is not None and hasattr(model.named_steps.get("clf", None), "feature_importances_"):
        clf = model.named_steps["clf"]
        st.plotly_chart(
            plot_feature_importance(list(features.keys()), clf.feature_importances_),
            use_container_width=True
        )

    # ── Raw Features Table ──
    with st.expander("📊 All Extracted Features (52 features)"):
        import pandas as pd
        feat_df = pd.DataFrame([
            {"Feature": k, "Value": round(float(v), 6), "Domain": k.split("_")[0].upper()}
            for k, v in features.items()
        ])
        st.dataframe(feat_df, use_container_width=True, height=400)

    # ── Footer ──
    st.divider()
    st.markdown(
        "<div style='text-align:center; color: #8b949e; font-size: 0.8rem;'>"
        "Built by Alaa Eddine TAHIR · Arts et Métiers ParisTech · "
        "<a href='https://linkedin.com/in/alaaeddine-tahir' style='color:#58a6ff'>LinkedIn</a>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
