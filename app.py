import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="India AQI & HCHO Hotspot Dashboard", layout="wide")

st.title("🛰️ Satellite-Derived Surface AQI & HCHO Hotspot Dashboard")
st.markdown("ISRO BAH 2026 — Problem Statement 3: Columnar-to-Surface AQI Conversion & Formaldehyde Hotspot Detection")

# ---- Load trained model ----
@st.cache_resource
def load_model():
    return joblib.load("aqi_model_final.pkl")

model = load_model()

# ---- City reference data (extend this with more cities/coords as you collect more data) ----
CITY_COORDS = {
    'Ahmedabad': (23.03, 72.58), 'Bengaluru': (12.97, 77.59), 'Chennai': (13.08, 80.27),
    'Delhi': (28.70, 77.10), 'Hyderabad': (17.38, 78.48), 'Kolkata': (22.57, 88.36),
    'Mumbai': (19.07, 72.87), 'Nagpur': (21.15, 79.08), 'Nashik': (19.99, 73.79),
    'Pune': (18.52, 73.85)
}

def aqi_color(aqi):
    if aqi <= 50: return "green"
    elif aqi <= 100: return "yellow"
    elif aqi <= 200: return "orange"
    elif aqi <= 300: return "red"
    elif aqi <= 400: return "purple"
    else: return "darkred"

def aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

# ---- Sidebar: manual prediction / "what-if" tool ----
st.sidebar.header("🔍 Predict AQI for Any Location")
st.sidebar.markdown("Enter satellite + weather inputs (or use defaults) to predict surface AQI — including locations with no CPCB sensor.")

lat = st.sidebar.number_input("Latitude", value=21.0, min_value=6.0, max_value=37.0)
lon = st.sidebar.number_input("Longitude", value=78.0, min_value=68.0, max_value=97.0)
no2 = st.sidebar.number_input("NO2 column density", value=0.00009, format="%.6f")
so2 = st.sidebar.number_input("SO2 column density", value=0.00015, format="%.6f")
co = st.sidebar.number_input("CO column density", value=0.04, format="%.4f")
hcho = st.sidebar.number_input("HCHO column density", value=0.00022, format="%.6f")
temp = st.sidebar.number_input("Temperature (°C)", value=25.0)
wind = st.sidebar.number_input("Wind Speed (m/s)", value=2.5)
rh = st.sidebar.number_input("Relative Humidity (%)", value=60.0)
blh = st.sidebar.number_input("Boundary Layer Height (m)", value=400.0)

if st.sidebar.button("Predict AQI", type="primary"):
    X_input = pd.DataFrame([{
        'NO2': no2, 'SO2': so2, 'CO': co, 'HCHO': hcho,
        'Temp_C': temp, 'WindSpeed': wind, 'RH': rh, 'BLH': blh,
        'lat': lat, 'lon': lon
    }])
    pred = model.predict(X_input)[0]
    st.sidebar.success(f"Predicted AQI: {pred:.0f} ({aqi_category(pred)})")

# ---- Main: City AQI map ----
st.subheader("📍 Current Predicted AQI Across Cities")

# Demo values — replace with live satellite pulls in production
demo_data = []
for city, (clat, clon) in CITY_COORDS.items():
    X_demo = pd.DataFrame([{
        'NO2': 0.00009, 'SO2': 0.00015, 'CO': 0.04, 'HCHO': 0.00022,
        'Temp_C': 25.0, 'WindSpeed': 2.5, 'RH': 60.0, 'BLH': 400.0,
        'lat': clat, 'lon': clon
    }])
    pred_aqi = model.predict(X_demo)[0]
    demo_data.append({'City': city, 'lat': clat, 'lon': clon, 'AQI': pred_aqi})

demo_df = pd.DataFrame(demo_data)

m = folium.Map(location=[22.0, 79.0], zoom_start=5, tiles="CartoDB positron")

for _, row in demo_df.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=12,
        color=aqi_color(row['AQI']),
        fill=True,
        fill_color=aqi_color(row['AQI']),
        fill_opacity=0.8,
        popup=f"{row['City']}: AQI {row['AQI']:.0f} ({aqi_category(row['AQI'])})"
    ).add_to(m)

st_folium(m, width=1100, height=500)

st.dataframe(demo_df[['City', 'AQI']].sort_values('AQI', ascending=False), use_container_width=True)

# ---- HCHO Hotspot section ----
st.subheader("🔥 HCHO Hotspot — Biogenic vs Pyrogenic Comparison")
st.markdown("""
Based on Sentinel-5P HCHO + MODIS Fire data (Oct–Nov 2024):
""")

hcho_compare = pd.DataFrame({
    'Region': ['Punjab/Haryana (Indo-Gangetic Plain)', 'Western Ghats (forest belt)'],
    'Fire Locations Detected': [69, 3],
    'Background HCHO': [0.000221, 0.000159],
    'Fire-zone HCHO': [0.000242, None],
})
st.dataframe(hcho_compare, use_container_width=True)
st.caption("Statistically significant HCHO elevation at fire locations (p=0.032) confirms pyrogenic signal in the Indo-Gangetic Plain during stubble-burning season.")

st.markdown("---")
st.caption("Built for ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 3")
