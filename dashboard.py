import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import time
import random
from sklearn.linear_model import LinearRegression

# =========================
# LOAD SAVED MODELS
# =========================

clf = pickle.load(open("classifier.pkl", "rb"))
reg = pickle.load(open("regressor.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =========================
# SENSOR SIMULATION
# =========================

def generate_sensor_data():
    return {
        "air_temperature": random.uniform(280, 350),
        "process_temperature": random.uniform(290, 360),
        "rotational_speed": random.randint(1200, 2500),
        "torque": random.uniform(20, 80),
        "tool_wear": random.randint(0, 250),
        "machine_type": random.choice(["L", "M", "H"])
    }

# =========================
# UI
# =========================

st.title("⚙️ Industrial AI Monitoring Dashboard")

mode = st.sidebar.radio("Mode", ["Manual", "Real-Time"])

# =========================
# INPUT
# =========================

if mode == "Manual":
    air_temperature = st.sidebar.number_input("Air Temp", 300.0)
    process_temperature = st.sidebar.number_input("Process Temp", 310.0)
    rotational_speed = st.sidebar.number_input("RPM", 1500)
    torque = st.sidebar.number_input("Torque", 40.0)
    tool_wear = st.sidebar.number_input("Tool Wear", 100)
    machine_type = st.sidebar.selectbox("Type", ["L", "M", "H"])

else:
    vals = generate_sensor_data()
    air_temperature = vals["air_temperature"]
    process_temperature = vals["process_temperature"]
    rotational_speed = vals["rotational_speed"]
    torque = vals["torque"]
    tool_wear = vals["tool_wear"]
    machine_type = vals["machine_type"]

    st.sidebar.write("Live Data", vals)

# =========================
# ENCODING
# =========================

type_L, type_M, type_H = 0, 0, 0
if machine_type == "L":
    type_L = 1
elif machine_type == "M":
    type_M = 1
else:
    type_H = 1

input_data = np.array([[
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
    type_H,
    type_L,
    type_M
]])

input_scaled = scaler.transform(input_data)

# =========================
# PREDICTION
# =========================

failure = clf.predict(input_scaled)[0]
prob = clf.predict_proba(input_scaled)[0][1]
rul = reg.predict(input_scaled)[0]

# =========================
# ALERT SYSTEM
# =========================

if failure == 1:
    st.error("🚨 FAILURE DETECTED!")

    # Sound Alert
    st.audio("https://www.soundjay.com/button/beep-07.wav")

    # Blinking Effect
    for _ in range(2):
        st.warning("⚠️ CRITICAL ALERT ⚠️")
        time.sleep(0.3)
else:
    st.success("✅ Machine Healthy")

# =========================
# GAUGE CHART
# =========================

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prob * 100,
    title={'text': "Failure Risk (%)"},
    gauge={
        'axis': {'range': [0, 100]},
        'steps': [
            {'range': [0, 30], 'color': "green"},
            {'range': [30, 70], 'color': "yellow"},
            {'range': [70, 100], 'color': "red"}
        ]
    }
))

st.plotly_chart(gauge, use_container_width=True)

# =========================
# RUL DISPLAY
# =========================

st.subheader(f"Remaining Useful Life: {round(rul,2)} hours")

# =========================
# ANOMALY DETECTION
# =========================

if tool_wear > 200 or torque > 70:
    st.error("⚠️ Anomaly Detected!")

# =========================
# DATA LOGGING
# =========================

if "log" not in st.session_state:
    st.session_state.log = []

st.session_state.log.append({
    "wear": tool_wear,
    "temp": air_temperature,
    "torque": torque
})

df = np.array(st.session_state.log)

# =========================
# TREND LINE
# =========================

if len(st.session_state.log) > 5:

    x = np.arange(len(st.session_state.log)).reshape(-1, 1)
    y = np.array([i["wear"] for i in st.session_state.log])

    model = LinearRegression()
    model.fit(x, y)
    trend = model.predict(x)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(y=y, name="Actual Wear"))
    fig_trend.add_trace(go.Scatter(y=trend, name="Predicted Trend"))

    st.plotly_chart(fig_trend)

# =========================
# 3D GRAPH
# =========================

df_plot = {
    "temp": [i["temp"] for i in st.session_state.log],
    "torque": [i["torque"] for i in st.session_state.log],
    "wear": [i["wear"] for i in st.session_state.log]
}

fig3d = px.scatter_3d(
    df_plot,
    x="temp",
    y="torque",
    z="wear",
    title="3D Sensor Monitoring"
)

st.plotly_chart(fig3d)

# =========================
# AUTO REFRESH
# =========================

if mode == "Real-Time":
    time.sleep(2)
    st.rerun()