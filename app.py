import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="India AQI & HCHO Hotspot Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LIVE DATA (GOOGLE EARTH ENGINE) ----------------
EE_AVAILABLE = False
try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    pass

def init_earth_engine():
    """Authenticate to Earth Engine using a service account stored in Streamlit secrets.
    Returns True if successful, False otherwise (dashboard falls back to cached data)."""
    if not EE_AVAILABLE:
        return False
    try:
        if "gee_service_account" not in st.secrets:
            return False
        sa_info = dict(st.secrets["gee_service_account"])
        credentials = ee.ServiceAccountCredentials(sa_info["client_email"], key_data=None)
        # ee needs the private key as a json string for key_data
        import json
        credentials = ee.ServiceAccountCredentials(sa_info["client_email"], key_data=json.dumps(sa_info))
        ee.Initialize(
    credentials,
    project="isro-hackathon-500120"
    )
        return True
    except Exception as e:
        st.session_state["ee_error"] = str(e)
        return False

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_city_values(lat, lon, city_name):
    """Pull the most recent available Sentinel-5P + ERA5 values for one city.
    Cached for 1 hour so repeated refresh clicks don't hammer the API."""
    point = ee.Geometry.Point([lon, lat])
    end = datetime.utcnow()
    start = end - timedelta(days=10)  # widen window since S5P has daily gaps from cloud cover

    def safe_mean(collection, band, scale=5000):
        img = collection.filterDate(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')) \
                         .filterBounds(point).select(band).mean()
        val = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=scale, maxPixels=1e9).get(band)
        return val.getInfo()

    no2 = safe_mean(ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2'), 'tropospheric_NO2_column_number_density')
    so2 = safe_mean(ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_SO2'), 'SO2_column_number_density')
    co = safe_mean(ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_CO'), 'CO_column_number_density')
    hcho = safe_mean(ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_HCHO'), 'tropospheric_HCHO_column_number_density')

    return {
        'NO2': no2, 'SO2': so2, 'CO': co, 'HCHO': hcho,
        'fetched_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    }

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background: linear-gradient(180deg, #0e1117 0%, #131722 100%); }
    h1 { font-weight: 800 !important; letter-spacing: -0.5px; }
    h2, h3 { font-weight: 700 !important; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 16px 18px;
    }
    div[data-testid="stMetric"] label { color: #9aa4b2 !important; }
    .badge {
        display:inline-block; padding: 4px 12px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; color: white; margin-right: 6px;
    }
    .hero {
        padding: 22px 26px; border-radius: 18px;
        background: radial-gradient(1200px 300px at 0% 0%, rgba(56,189,248,0.15), transparent),
                    rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 18px;
    }
    section[data-testid="stSidebar"] { background: #0b0e14; border-right: 1px solid rgba(255,255,255,0.06); }
</style>
""", unsafe_allow_html=True)

AQI_BANDS = [
    (0, 50, "#4ade80", "Good"),
    (51, 100, "#facc15", "Satisfactory"),
    (101, 200, "#fb923c", "Moderate"),
    (201, 300, "#f87171", "Poor"),
    (301, 400, "#a855f7", "Very Poor"),
    (401, 1000, "#7f1d1d", "Severe"),
]

def aqi_color(aqi):
    for lo, hi, color, _ in AQI_BANDS:
        if lo <= aqi <= hi:
            return color
    return "#7f1d1d"

def aqi_category(aqi):
    for lo, hi, _, label in AQI_BANDS:
        if lo <= aqi <= hi:
            return label
    return "Severe"

@st.cache_resource
def load_model():
    return joblib.load("aqi_model_final.pkl")

model = load_model()

# ---------------- REAL CITY SNAPSHOT DATA (latest observed values per city) ----------------
CITY_DATA = pd.DataFrame([
    {'City': 'Delhi', 'lat': 28.70, 'lon': 77.10, 'NO2': 1.42e-4, 'SO2': 1.46e-3, 'CO': 0.0411, 'HCHO': 2.84e-5, 'Temp_C': 11.5, 'WindSpeed': 2.43, 'RH': 85.7, 'BLH': 212.8, 'AQI': 173},
    {'City': 'Kolkata', 'lat': 22.57, 'lon': 88.36, 'NO2': 1.02e-4, 'SO2': 8.7e-5, 'CO': 0.0476, 'HCHO': 1.38e-4, 'Temp_C': 20.8, 'WindSpeed': 2.16, 'RH': 71.7, 'BLH': 394.2, 'AQI': 159},
    {'City': 'Mumbai', 'lat': 19.07, 'lon': 72.87, 'NO2': 2.08e-4, 'SO2': 1.62e-4, 'CO': 0.0483, 'HCHO': 3.88e-4, 'Temp_C': 26.8, 'WindSpeed': 2.34, 'RH': 68.9, 'BLH': 174.9, 'AQI': 121},
    {'City': 'Nagpur', 'lat': 21.15, 'lon': 79.08, 'NO2': 7.56e-5, 'SO2': 3.22e-4, 'CO': 0.0469, 'HCHO': 2.05e-4, 'Temp_C': 23.0, 'WindSpeed': 1.43, 'RH': 67.8, 'BLH': 411.2, 'AQI': 94},
    {'City': 'Chennai', 'lat': 13.08, 'lon': 80.27, 'NO2': 2.72e-5, 'SO2': 1.85e-4, 'CO': 0.0443, 'HCHO': 1.02e-4, 'Temp_C': 27.0, 'WindSpeed': 2.30, 'RH': 70.0, 'BLH': 702.8, 'AQI': 96},
    {'City': 'Nashik', 'lat': 19.99, 'lon': 73.79, 'NO2': 9.13e-5, 'SO2': 2.94e-4, 'CO': 0.0354, 'HCHO': 9.43e-5, 'Temp_C': 22.7, 'WindSpeed': 1.55, 'RH': 70.2, 'BLH': 463.9, 'AQI': 85},
    {'City': 'Pune', 'lat': 18.52, 'lon': 73.85, 'NO2': 8.09e-5, 'SO2': -1.9e-4, 'CO': 0.0361, 'HCHO': 4.65e-5, 'Temp_C': 22.8, 'WindSpeed': 2.02, 'RH': 62.4, 'BLH': 500.7, 'AQI': 87},
    {'City': 'Ahmedabad', 'lat': 23.03, 'lon': 72.58, 'NO2': 1.11e-4, 'SO2': 5.77e-5, 'CO': 0.0431, 'HCHO': 1.52e-4, 'Temp_C': 20.1, 'WindSpeed': 2.54, 'RH': 66.1, 'BLH': 284.9, 'AQI': 84},
    {'City': 'Hyderabad', 'lat': 17.38, 'lon': 78.48, 'NO2': 5.61e-5, 'SO2': -2.68e-4, 'CO': 0.0350, 'HCHO': 2.00e-4, 'Temp_C': 22.6, 'WindSpeed': 1.89, 'RH': 60.1, 'BLH': 470.5, 'AQI': 89},
    {'City': 'Bengaluru', 'lat': 12.97, 'lon': 77.59, 'NO2': 4.59e-5, 'SO2': -1.58e-4, 'CO': 0.0312, 'HCHO': 1.29e-4, 'Temp_C': 20.2, 'WindSpeed': 2.50, 'RH': 72.6, 'BLH': 491.4, 'AQI': 70},
])
CITY_DATA['Category'] = CITY_DATA['AQI'].apply(aqi_category)
CITY_DATA['Color'] = CITY_DATA['AQI'].apply(aqi_color)

# ---------------- LIVE REFRESH STATE ----------------
if 'live_data' not in st.session_state:
    st.session_state['live_data'] = None
if 'last_refresh' not in st.session_state:
    st.session_state['last_refresh'] = None
if 'ee_initialized' not in st.session_state:
    st.session_state['ee_initialized'] = init_earth_engine()
    
    st.write("EE initialized:", st.session_state['ee_initialized'])

if "ee_error" in st.session_state:
    st.error(st.session_state["ee_error"])



def refresh_live_data():

    st.write("✅ Refresh button clicked")

    if not st.session_state['ee_initialized']:
        st.error("Earth Engine NOT initialized")
        return

    st.success("Earth Engine initialized successfully")

    """Pulls live satellite values for all 10 cities and re-predicts AQI with the trained model.
    Falls back gracefully (keeps cached/static values) if Earth Engine is unreachable."""
    if not st.session_state['ee_initialized']:
        st.warning("Live satellite connection unavailable — showing last-known dataset values instead. "
                    "(Set up the GEE service account in Streamlit secrets to enable live refresh.)")
        return
    progress = st.progress(0, text="Connecting to Sentinel-5P...")
    live_rows = []
    try:
        for i, row in CITY_DATA.iterrows():
            progress.progress((i + 1) / len(CITY_DATA), text=f"Fetching live data for {row['City']}...")
            vals = fetch_live_city_values(row['lat'], row['lon'], row['City'])
            merged = row.to_dict()
            for k in ['NO2', 'SO2', 'CO', 'HCHO']:
                if vals.get(k) is not None:
                    merged[k] = vals[k]
            # weather (Temp_C, WindSpeed, RH, BLH) kept from latest known snapshot — ERA5 pull
            # is intentionally omitted from the on-demand path since CDS API requests take minutes,
            # not seconds, which would block the dashboard UI.
            X_live = pd.DataFrame([{f: merged[f] for f in
                ['NO2','SO2','CO','HCHO','Temp_C','WindSpeed','RH','BLH','lat','lon']}])
            merged['AQI'] = float(model.predict(X_live)[0])
            merged['Category'] = aqi_category(merged['AQI'])
            merged['Color'] = aqi_color(merged['AQI'])
            live_rows.append(merged)
        progress.empty()
        st.session_state['live_data'] = pd.DataFrame(live_rows)
        st.session_state['last_refresh'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        st.success("Live satellite data fetched and AQI re-predicted successfully.")
    except Exception as e:
        progress.empty()
        st.error(f"Live fetch failed ({e}) — showing last-known dataset values instead.")

# Use live data if available, otherwise the static snapshot
ACTIVE_DATA = st.session_state['live_data'] if st.session_state['live_data'] is not None else CITY_DATA
DATA_STATUS = "🟢 LIVE" if st.session_state['live_data'] is not None else "🟡 CACHED SNAPSHOT"

# ---------------- HERO HEADER ----------------
hero_col, btn_col = st.columns([5, 1])
with hero_col:
    refresh_note = f" · Last refreshed {st.session_state['last_refresh']}" if st.session_state['last_refresh'] else ""
    st.markdown(f"""
    <div class="hero">
      <span class="badge" style="background:#f59e0b; font-size:0.82rem; padding:5px 14px;">TEAM THRONES</span>
      <span class="badge" style="background:{'#16a34a' if DATA_STATUS.startswith('🟢') else '#71717a'};">{DATA_STATUS}{refresh_note}</span>
      <h1>🛰️ Satellite-Derived Surface AQI &amp; HCHO Hotspot Dashboard</h1>
      <p style="color:#9aa4b2; font-size:1.02rem; margin-top:-6px;">
      ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 3 · Columnar-to-Surface AQI Conversion &amp; Formaldehyde Source Attribution
      </p>
      <span class="badge" style="background:#1d4ed8;">Sentinel-5P / TROPOMI</span>
      <span class="badge" style="background:#0891b2;">ERA5 Reanalysis</span>
      <span class="badge" style="background:#7c3aed;">CPCB Ground Truth</span>
      <span class="badge" style="background:#16a34a;">XGBoost R²=0.87</span>
    </div>
    """, unsafe_allow_html=True)
with btn_col:
    st.write("")
    st.write("")
    if st.button("🔄 Refresh Live Data", use_container_width=True, type="primary"):
        refresh_live_data()

# ---------------- KPI ROW ----------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Cities Monitored", len(ACTIVE_DATA))
k2.metric("Avg Predicted AQI", f"{ACTIVE_DATA['AQI'].mean():.0f}")
k3.metric("Worst City", ACTIVE_DATA.loc[ACTIVE_DATA['AQI'].idxmax(), 'City'], f"AQI {ACTIVE_DATA['AQI'].max():.0f}")
k4.metric("Best City", ACTIVE_DATA.loc[ACTIVE_DATA['AQI'].idxmin(), 'City'], f"AQI {ACTIVE_DATA['AQI'].min():.0f}")

st.write("")
tab1, tab2, tab3 = st.tabs(["🗺️ AQI Map & Predictor", "📊 City Comparison", "🔥 HCHO Hotspot Analysis"])

# ===================== TAB 1: MAP + PREDICTOR =====================
with tab1:
    col_map, col_pred = st.columns([2.1, 1])

    with col_map:
        st.subheader("Live Surface AQI — National View")
        m = folium.Map(location=[22.0, 79.0], zoom_start=4.6, tiles="CartoDB dark_matter")
        for _, row in ACTIVE_DATA.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=10 + (row['AQI'] / 30),
                color=row['Color'],
                fill=True,
                fill_color=row['Color'],
                fill_opacity=0.85,
                weight=2,
                popup=folium.Popup(
                    f"<b>{row['City']}</b><br>AQI: {row['AQI']:.0f} ({row['Category']})<br>"
                    f"NO2: {row['NO2']:.2e}<br>HCHO: {row['HCHO']:.2e}", max_width=220
                ),
                tooltip=f"{row['City']}: {row['AQI']:.0f}"
            ).add_to(m)
        st_folium(m, width=720, height=480)

        legend_cols = st.columns(6)
        for i, (lo, hi, color, label) in enumerate(AQI_BANDS):
            legend_cols[i].markdown(
                f"<div style='text-align:center'><div style='background:{color};height:8px;border-radius:4px;'></div>"
                f"<span style='font-size:0.72rem;color:#9aa4b2;'>{label}</span></div>", unsafe_allow_html=True
            )

    with col_pred:
        st.subheader("🔍 Predict AQI — Any Location")
        st.caption("Works for rural/unmonitored areas with no CPCB sensor — the model only needs satellite + weather inputs.")

        lat = st.number_input("Latitude", value=21.0, min_value=6.0, max_value=37.0)
        lon = st.number_input("Longitude", value=78.0, min_value=68.0, max_value=97.0)
        with st.expander("Satellite & Weather Inputs", expanded=False):
            no2 = st.number_input("NO2 column density", value=0.00009, format="%.6f")
            so2 = st.number_input("SO2 column density", value=0.00015, format="%.6f")
            co = st.number_input("CO column density", value=0.04, format="%.4f")
            hcho_in = st.number_input("HCHO column density", value=0.00022, format="%.6f")
            temp = st.number_input("Temperature (°C)", value=25.0)
            wind = st.number_input("Wind Speed (m/s)", value=2.5)
            rh = st.number_input("Relative Humidity (%)", value=60.0)
            blh = st.number_input("Boundary Layer Height (m)", value=400.0)

        if st.button("Predict AQI", type="primary", use_container_width=True):
            X_input = pd.DataFrame([{
                'NO2': no2, 'SO2': so2, 'CO': co, 'HCHO': hcho_in,
                'Temp_C': temp, 'WindSpeed': wind, 'RH': rh, 'BLH': blh,
                'lat': lat, 'lon': lon
            }])
            pred = model.predict(X_input)[0]
            color = aqi_color(pred)
            cat = aqi_category(pred)
            st.markdown(f"""
            <div style="text-align:center; padding:22px; border-radius:16px; background:{color}22; border:1px solid {color};">
                <div style="font-size:0.85rem; color:#9aa4b2;">Predicted AQI</div>
                <div style="font-size:2.6rem; font-weight:800; color:{color};">{pred:.0f}</div>
                <div style="font-size:1rem; font-weight:600; color:{color};">{cat}</div>
            </div>
            """, unsafe_allow_html=True)

# ===================== TAB 2: CITY COMPARISON =====================
with tab2:
    st.subheader("AQI Ranking Across Cities")
    fig = px.bar(
        ACTIVE_DATA.sort_values('AQI', ascending=True),
        x='AQI', y='City', orientation='h', color='AQI',
        color_continuous_scale=['#4ade80', '#facc15', '#fb923c', '#f87171', '#7f1d1d'],
        text='AQI'
    )
    fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                       height=420, showlegend=False, coloraxis_showscale=False)
    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Pollutant Profile by City")
        melted = ACTIVE_DATA.melt(id_vars='City', value_vars=['NO2', 'SO2', 'CO', 'HCHO'], var_name='Pollutant', value_name='Value')
        fig2 = px.bar(melted, x='City', y='Value', color='Pollutant', barmode='group')
        fig2.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=380)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.subheader("Feature Importance (XGBoost Model)")
        importance_df = pd.DataFrame({
            'Feature': ['Temp_C', 'RH', 'NO2', 'CO', 'BLH', 'WindSpeed', 'HCHO', 'SO2'],
            'Importance': [0.287, 0.215, 0.180, 0.104, 0.106, 0.048, 0.038, 0.022]
        }).sort_values('Importance')
        fig3 = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
        fig3.update_traces(marker_color='#38bdf8')
        fig3.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=380)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Model Performance")
    perf_df = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'LightGBM'],
        'RMSE': [36.45, 25.53, 36.56],
        'MAE': [22.12, 17.45, 22.53],
        'R²': [0.729, 0.867, 0.728]
    })
    st.dataframe(perf_df.style.highlight_max(subset=['R²'], color='#16653433').highlight_min(subset=['RMSE','MAE'], color='#16653433'),
                 use_container_width=True, hide_index=True)
    st.caption("XGBoost selected as final model · Leave-One-Station-Out validation R² = 0.257 after adding lat/lon spatial features (up from -0.17 baseline).")

# ===================== TAB 3: HCHO HOTSPOT =====================
with tab3:
    st.subheader("🔥 Biogenic vs. Pyrogenic HCHO Source Attribution")
    st.caption("Sentinel-5P HCHO column density cross-referenced with MODIS/VIIRS active fire detections, October–November 2024 (peak stubble-burning season)")

    hcho_compare = pd.DataFrame({
        'Region': ['Punjab / Haryana\n(Indo-Gangetic Plain)', 'Western Ghats\n(forest belt)'],
        'Fire Locations Detected': [69, 3],
        'Background HCHO': [0.000221, 0.000159],
        'Fire-zone HCHO': [0.000242, None],
    })

    c1, c2 = st.columns([1, 1])
    with c1:
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(name='Background HCHO', x=hcho_compare['Region'], y=hcho_compare['Background HCHO'], marker_color='#38bdf8'))
        fig4.add_trace(go.Bar(name='Fire-zone HCHO', x=hcho_compare['Region'], y=hcho_compare['Fire-zone HCHO'], marker_color='#f87171'))
        fig4.update_layout(barmode='group', template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)', height=400, title="HCHO Column Density: Background vs Fire Zones")
        st.plotly_chart(fig4, use_container_width=True)

    with c2:
        fig5 = px.bar(hcho_compare, x='Region', y='Fire Locations Detected', color='Region',
                       color_discrete_sequence=['#f97316', '#22c55e'])
        fig5.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            height=400, showlegend=False, title="Active Fire Detections by Region")
        st.plotly_chart(fig5, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("HCHO elevation at fire sites", "+9.1%", "Punjab/Haryana")
    m2.metric("Statistical significance", "p = 0.032", "significant at 95% CI")
    m3.metric("Fire density ratio", "23×", "Punjab vs Western Ghats")

    st.markdown("""
    **Interpretation:** The Indo-Gangetic Plain shows 23× more active fire detections than the Western Ghats
    during peak stubble-burning season, with a statistically significant (p=0.032) elevation in HCHO at fire locations
    versus regional background — confirming a real **pyrogenic** signal. The Western Ghats' lower, fire-independent
    HCHO baseline represents the **biogenic** (vegetation-driven) signal, demonstrating the model's ability to
    disentangle the two source types as required by the problem statement.
    """)

st.markdown("---")
st.caption("Team THRONES · Built for ISRO Bharatiya Antariksh Hackathon 2026 · Problem Statement 3 · Data: Sentinel-5P TROPOMI, ERA5, CPCB, MODIS/VIIRS")
