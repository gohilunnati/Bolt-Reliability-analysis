import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import plotly.figure_factory as ff
import os

# =============================================================================
# 0. EXTERNAL FILE LINK
# =============================================================================


# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Bolt Reliability Digital Twin by Unnati Gohil", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 32px; font-weight: 800; color: #0F172A; margin-bottom: 10px; }
    .status-safe { background-color: #DCFCE7; color: #14532D; padding: 15px; border-radius: 8px; border: 1px solid #86EFAC; }
    .status-danger { background-color: #FEE2E2; color: #7F1D1D; padding: 15px; border-radius: 8px; border: 1px solid #FCA5A5; }
    .css-card { background-color: #FFFFFF; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border: 1px solid #E5E7EB; }
</style>
""", unsafe_allow_html=True)

# Optional: show the URL in the sidebar as a clickable link


# =============================================================================
# 2. LOCAL MODEL LOADER (SIMPLE & ROBUST)
# =============================================================================
@st.cache_resource
def load_assets():
    # Make sure these filenames match exactly what you uploaded to GitHub!
    model_filename = 'bolt_dnn_precision_run.h5' 
    scaler_filename = 'scaler_precision_run.pkl'
    
    # 1. CHECK FILES
    if not os.path.exists(model_filename):
        st.error(f"❌ Critical Error: Model file '{model_filename}' not found in repo.")
        st.info(f"You can also store the model externally here: {FILE_URL}")  # <<< ADDED (optional)
        return None, None
        
    if not os.path.exists(scaler_filename):
        # Fallback names just in case
        if os.path.exists('dnn_scaler_precision_run.pkl'):
            scaler_filename = 'dnn_scaler_precision_run.pkl'
        else:
            st.error(f"❌ Critical Error: Scaler file '{scaler_filename}' not found in repo.")
            return None, None

    # 2. LOAD MODEL
    try:
        # Custom objects needed for Keras 3 saving format
        custom_objs = {'mse': tf.keras.losses.MeanSquaredError(), 'mae': tf.keras.metrics.MeanAbsoluteError()}
        model = tf.keras.models.load_model(model_filename, custom_objects=custom_objs, compile=False)
        model.compile(loss='mse', optimizer='adam')
    except Exception as e:
        st.error(f"❌ Error loading Keras model: {e}")
        return None, None

    # 3. LOAD SCALER
    try:
        scaler = joblib.load(scaler_filename)
    except Exception as e:
        st.error(f"❌ Error loading Scaler: {e}")
        return None, None
            
    return model, scaler

model, scaler = load_assets()

# =============================================================================
# 3. SIDEBAR & INPUTS
# =============================================================================
st.sidebar.header("🎛️ Parameters")

with st.sidebar.expander("1. Geometry", expanded=True):
    bolt_type = st.selectbox("Bolt Size", ["M8", "M10", "M12", "M14", "M16"], index=0)
    d_map = {"M8": 0.008, "M10": 0.010, "M12": 0.012, "M14": 0.014, "M16": 0.016}
    d_val = d_map[bolt_type]

with st.sidebar.expander("2. Assembly", expanded=True):
    def_torque = 18.0 if bolt_type == "M8" else (90.0 if bolt_type == "M12" else 50.0)
    torque = st.slider("Torque (Nm)", 10.0, 200.0, float(def_torque))
    fric_label = st.selectbox("Friction", ["Lubricated (0.12)", "Standard (0.18)", "Rough (0.30)"], index=1)
    mu_mean = 0.12 if "0.12" in fric_label else (0.30 if "0.30" in fric_label else 0.18)

with st.sidebar.expander("3. Loads", expanded=True):
    load_mean = st.number_input("Shear Load (N)", value=1800.0, step=100.0)

# =============================================================================
# 4. PREDICTION ENGINE
# =============================================================================
if model and scaler:
    n = 5000
    X_syn = np.column_stack([
        np.random.normal(torque, torque*0.05, n),
        np.random.lognormal(np.log(0.20), 0.15, n),
        np.random.lognormal(np.log(mu_mean), 0.22, n),
        np.random.normal(d_val, d_val*0.01, n),
        np.random.normal(d_val+0.0002, 1e-5, n),
        np.random.normal(0.006, 0.0002, n),
        np.random.normal(250, 15, n),
        np.random.normal(load_mean, 300, n)
    ])
    
    X_scaled = scaler.transform(X_syn)
    margins = model.predict(X_scaled, verbose=0).flatten()
    prob = np.sum(margins < 0) / n
    capacity = np.mean(X_syn[:, 2] * (X_syn[:, 0] / (X_syn[:, 1] * X_syn[:, 3])))
else:
    prob, capacity, margins = 0, 0, []

# =============================================================================
# 5. DASHBOARD
# =============================================================================
st.title("🔩 Bolt Reliability Digital Twin")

if prob < 0.001:
    st.markdown(f'<div class="status-safe">✅ DESIGN SAFE: Failure probability is negligible (< 0.1%).</div>', unsafe_allow_html=True)
    color = "#22C55E"
elif prob < 0.05:
    st.markdown(f'<div class="status-warn" style="background-color:#FEF9C3;padding:15px;border-radius:8px;">⚠️ MARGINAL RISK: Failure probability is {prob*100:.2f}%.</div>', unsafe_allow_html=True)
    color = "#F59E0B"
else:
    st.markdown(f'<div class="status-danger">❌ CRITICAL FAILURE: Probability {prob*100:.2f}% exceeds safety limits.</div>', unsafe_allow_html=True)
    color = "#EF4444"

st.write("")

c1, c2, c3 = st.columns(3)
c1.metric("Failure Probability", f"{prob*100:.4f}%")
c2.metric("Reliability Index", f"{(1-prob)*100:.4f}%")
c3.metric("Holding Capacity", f"{int(capacity)} N")

st.markdown("---")

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("##### Risk Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = prob * 100,
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': color},
                 'steps': [{'range': [0, 0.1], 'color': "#DCFCE7"}, {'range': [0.1, 100], 'color': "#FEE2E2"}]}
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    st.markdown("##### Physics Interference")
    if len(margins) > 0:
        stress = np.random.normal(load_mean, 300, 2000)
        strength = stress + np.random.choice(margins, 2000)
        fig_dist = ff.create_distplot([strength, stress], ['Strength', 'Stress'], colors=[color, '#64748B'], bin_size=100, show_rug=False)
        fig_dist.update_layout(height=300, margin=dict(l=20,r=20,t=30,b=20), showlegend=True)
        st.plotly_chart(fig_dist, use_container_width=True)
