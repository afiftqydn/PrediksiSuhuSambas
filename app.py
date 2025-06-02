import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from jcopml.utils import load_model  # Pastikan library jcopml terinstal

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Prediksi Cuaca Sambas",
    page_icon=None,  # Emoji dihapus, bisa diganti URL gambar jika ada
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# PATHS TO PICKLE FILES
# =========================
MODEL_PATH = "model/rf_cuaca.pkl"
DATA_SAMPLE_PATH = "dataset.pkl"
DESCRIPTIVE_STATS_PATH = "descriptive_stats.pkl"

# =========================
# ENHANCED CUSTOM CSS & MATERIAL ICONS LINK
# =========================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons'); /* Tambahan untuk Material Icons */
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Button styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2c99a3 0%, #2b8a94 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.6em 2em;
        border-radius: 12px;
        font-weight: 600;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(44, 153, 163, 0.3);
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #2b8a94 0%, #236f78 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(44, 153, 163, 0.4);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(145deg, #f8fafb 0%, #f0f2f6 100%);
        border-radius: 20px;
        margin: 15px 0 15px 8px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        padding: 25px;
        border: 1px solid rgba(44, 153, 163, 0.1);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] label {
        color: #1d2829 !important;
    }
    
    /* Main container */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f0f4f8 0%, #e8f0f2 100%);
    }
        
    .modern-card {
        background: white;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.12);
    }
    
    h1, h2, h3, h4 {
        color: #1d2829;
        font-weight: 600;
    }
    
    [data-testid="stMetric"] { 
        background: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(44, 153, 163, 0.1);
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2c99a3, #4db8c4);
    }
    
    .stExpander > div:first-child > details > summary { 
        background: linear-gradient(135deg, #2c99a3 0%, #4db8c4 100%);
        color: white !important;
        border-radius: 10px;
        padding: 10px 15px;
        font-weight: 600;
    }
    .stExpander > div:first-child > details > summary p { 
        color: white !important;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
        
    .weather-icon { /* Untuk Material Icons */
        font-size: 3em !important; /* Ukuran ikon bisa disesuaikan */
        line-height: 1;
        vertical-align: middle;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# UTILITY FUNCTIONS
# =========================
@st.cache_resource
def load_ml_model(path):
    try:
        return load_model(path)
    except FileNotFoundError:
        st.error(f"File model '{path}' tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Kesalahan memuat model: {e}")
        return None

@st.cache_data
def load_pickle_data(path, description):
    try:
        return pd.read_pickle(path)
    except FileNotFoundError:
        st.warning(f"File '{description}' ('{path}') tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Kesalahan memuat '{description}': {e}")
        return None

def get_weather_recommendation(temp, humidity, rainfall, sunshine):
    recommendations = []
    if rainfall > 10:
        recommendations.append("Bawa payung atau jas hujan & Hati-hati berkendara.")
    elif rainfall > 0:
        recommendations.append("Kemungkinan hujan ringan, siapkan payung.")
    if temp > 30:
        recommendations.append("Perbanyak minum & Gunakan pakaian ringan.")
    elif temp < 22:
        recommendations.append("Gunakan pakaian hangat.")
    if humidity > 85:
        recommendations.append("Kondisi sangat lembap, jaga ventilasi.")
    if sunshine > 8:
        recommendations.append("Gunakan tabir surya & kacamata hitam.")
    elif sunshine < 3 and rainfall == 0:
        recommendations.append("Hari ini mungkin berawan/mendung.")
    if not recommendations:
        recommendations.append("Kondisi cuaca tampak normal, nikmati harimu!")
    return recommendations


def create_weather_chart(temp_history_value):
    dates = [(datetime.now() - timedelta(days=x)).strftime("%d %b") for x in range(6, -1, -1)]
    base_temps = [temp_history_value - 3 + (i*0.5) + (x * (i%2 - 0.5)*2) for i, x in enumerate(range(7))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=base_temps, mode='lines+markers', name='Suhu',
                           line=dict(color='#2c99a3', width=3, shape='spline'),
                           marker=dict(size=8, color='#2c99a3', symbol='circle')))
    fig.update_layout(title_text="Tren Suhu (Data Simulasi)", title_x=0.5, yaxis_title="Suhu (°C)",
                      plot_bgcolor='rgba(255,255,255,0.8)', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(family="Inter", size=12, color="#333"), showlegend=False, height=350,
                      margin=dict(l=40, r=40, t=60, b=40))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(44,153,163,0.1)')
    return fig

# =========================
# LOAD MODEL
# =========================
if 'model' not in st.session_state:
    with st.spinner('Memuat model prediksi, mohon tunggu...'):
        st.session_state.model = load_ml_model(MODEL_PATH)
model = st.session_state.model

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("""
        <div class="modern-card" style="background: linear-gradient(135deg, #2c99a3 0%, #4db8c4 100%); padding: 25px; 
                     border-radius: 16px; text-align: center; box-shadow: 0 8px 25px rgba(44, 153, 163, 0.3);
                     margin-bottom: 25px;">
            <div style="font-size: 2.5em; margin-bottom: 10px;"><span class="material-icons" style="font-size: inherit; vertical-align: bottom;">thermostat</span></div>
            <h2 style="color: white; margin: 0; font-weight: 600;">Prediksi Cuaca</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-weight: 400;">Kabupaten Sambas</p>
        </div>""", unsafe_allow_html=True)
    
    # Opsi menu asli (tanpa nama ikon di tampilan)
    menu_options = ["Dashboard", "Analytics", "Prediksi"] 
    
    # st.selectbox sekarang menggunakan menu_options langsung
    menu = st.selectbox(
        "Navigasi Aplikasi:", 
        menu_options, 
        index=0, 
        key="main_menu_selectbox"
    )
    

# =========================
# MAIN CONTENT
# =========================

# Dashboard Section
if menu == "Dashboard": # Sekarang menu langsung berisi "Dashboard", "Analytics", atau "Prediksi"
    # ... (Sisa kode untuk Dashboard tetap sama) ...
    st.markdown("""
        <div class="fade-in-up" style="background: linear-gradient(135deg, #2c99a3 0%, #4db8c4 100%);
                         padding: 25px; border-radius: 16px; text-align: center; 
                         box-shadow: 0 8px 32px rgba(44, 153, 163, 0.3); margin-bottom: 30px;">
            <h1 style="color: white; margin: 0; font-size: 2.2em; font-weight: 700;"><span class="material-icons" style="font-size: 1.1em; vertical-align: middle; margin-right:10px;">dashboard</span>Dashboard Prediksi Cuaca</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 1.1em;">Ringkasan dan Analisis Sistem</p>
        </div>""", unsafe_allow_html=True)
    
    st.subheader("Statistik Cepat (Data Simulasi)")
    cols_metric = st.columns(4)
    metrics_data = [
        {"label": "Suhu Rata-rata", "value": "27.5°C", "delta": "0.2°C", "icon": "thermostat"},
        {"label": "Kelembapan", "value": "85%", "delta": "-1%", "icon": "water_drop"},
        {"label": "Curah Hujan", "value": "5.5mm", "delta": "0.5mm", "icon": "umbrella"},
        {"label": "Penyinaran", "value": "5.2 jam", "delta": "-0.3 jam", "icon": "wb_sunny"}
    ]
    for i, metric_item in enumerate(metrics_data): 
        with cols_metric[i]:
            st.metric(label=f"{metric_item['label']}", value=metric_item["value"], delta=metric_item["delta"])
    st.markdown("---")
    
    col_chart, col_accuracy = st.columns([2, 1])
    with col_chart:
        with st.container(border=True): 
            st.markdown("<h3 style='color: #2c99a3; margin-bottom: 15px;'><span class='material-icons' style='vertical-align: middle; margin-right: 5px;'>trending_up</span>Tren Suhu Terkini (Simulasi)</h3>", unsafe_allow_html=True)
            fig = create_weather_chart(27.5) 
            st.plotly_chart(fig, use_container_width=True)
    with col_accuracy:
        with st.container(border=True): 
            st.markdown("<h3 style='color: #2c99a3; margin-bottom: 15px;'><span class='material-icons' style='vertical-align: middle; margin-right: 5px;'>verified</span>Kinerja Model</h3>", unsafe_allow_html=True)
            accuracy_percentage = 74.00 
            st.progress(int(accuracy_percentage), text=f"R² Score (Data Uji): {accuracy_percentage:.2f}%")
            st.caption("Metrik evaluasi utama dari model Random Forest Regressor.")
            st.markdown(f"""
                <ul style='font-size: 0.9em; padding-left: 20px;'>
                    <li>MAE (Mean Absolute Error): 0.35 °C</li>
                    <li>RMSE (Root Mean Squared Error): 0.45 °C</li>
                </ul>
            """, unsafe_allow_html=True)
    st.markdown("---")

    with st.expander("Informasi Sistem Prediksi", expanded=False):
        st.markdown("<h3 style='color: #2c99a3;'><span class='material-icons' style='vertical-align: middle; margin-right: 5px;'>info</span>Tentang Sistem Prediksi</h3>", unsafe_allow_html=True)
        st.markdown("""<p style="font-size: 1em; line-height: 1.7; color: #444;">
Sistem <strong>Prediksi Cuaca Sambas</strong> menggunakan algoritma <strong>Random Forest Regressor</strong> yang telah dioptimalkan untuk memprediksi suhu rata-rata harian di Kabupaten Sambas berdasarkan parameter meteorologi.</p>""", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #2c99a3; margin-top: 20px;'>Fitur Utama Sistem:</h4>", unsafe_allow_html=True)
        
        cols_fitur_dashboard = st.columns(2) 
        fitur_utama_data = [
            {"title": "Model Akurat", "desc": "R² Score 74.00% (data uji)", "icon": "model_training"},
            {"title": "Prediksi Cepat", "desc": "Hasil prediksi instan", "icon": "bolt"},
            {"title": "Visualisasi Data", "desc": "Grafik tren suhu (simulasi)", "icon": "monitoring"},
            {"title": "Panduan Input", "desc": "Penjelasan parameter prediksi", "icon": "help_outline"}
        ]
        for i, fitur_item_dashboard in enumerate(fitur_utama_data): 
            with cols_fitur_dashboard[i % 2]:
                 st.markdown(f"""<div style="background: #f8fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #2c99a3; margin-bottom:10px; height: 100%;">
                                 <strong><span class="material-icons" style="font-size: 1.1em; vertical-align: bottom; margin-right: 5px;">{fitur_item_dashboard["icon"]}</span>{fitur_item_dashboard["title"]}</strong><br><small>{fitur_item_dashboard["desc"]}</small></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True) 

# Analytics Section
elif menu == "Analytics": 
    # ... (Sisa kode untuk Analytics tetap sama) ...
    st.markdown("""
        <div class="fade-in-up" style="background: linear-gradient(135deg, #2c99a3 0%, #4db8c4 100%); padding: 25px; 
                         border-radius: 16px; text-align: center; box-shadow: 0 8px 32px rgba(44, 153, 163, 0.3); 
                         margin-bottom: 30px;">
            <h1 style="color: white; margin: 0; font-size: 2.2em; font-weight: 700;"><span class="material-icons" style="font-size: 1.1em; vertical-align: middle; margin-right:10px;">bar_chart</span>Analisis Data Cuaca</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 1.1em;">Dataset Cuaca Kabupaten Sambas</p>
        </div>""", unsafe_allow_html=True)
    
    with st.expander("Informasi Detail Dataset", expanded=True):
        with st.container(border=True): 
            st.markdown("<h3 style='color: #2c99a3;'><span class='material-icons' style='vertical-align: middle; margin-right: 5px;'>description</span>Dataset Cuaca Historis</h3>", unsafe_allow_html=True)
            col_info_ds_main, col_param_ds_main = st.columns(2) 
            with col_info_ds_main:
                st.markdown("""<div style="background: #f0f8ff; padding: 15px; border-radius: 12px; border-left: 4px solid #2c99a3; height: 100%; margin-bottom:10px;">
                                <h4 style="color: #2c99a3; margin: 0 0 10px 0;">Informasi Umum</h4>
                                <p><strong>Sumber:</strong> <em>(BMKG Stasiun Meteorologi - Isi detail)</em></p>
                                <p><strong>Periode:</strong> <em>(01 Juli 2023 - 30 Juni 2024 - Isi detail)</em></p>
                                <p><strong>Lokasi:</strong> Kab. Sambas, Kalbar</p>
                                <p><strong>Data Bersih:</strong> 342 observasi</p></div>""", unsafe_allow_html=True)
            with col_param_ds_main:
                st.markdown("""<div style="background: #f0fff0; padding: 15px; border-radius: 12px; border-left: 4px solid #28a745; height: 100%; margin-bottom:10px;">
                                <h4 style="color: #28a745; margin: 0 0 10px 0;">Parameter Input Model</h4><small>
                                <p style="margin:0;">• RH_avg: Kelembapan (%)</p>
                                <p style="margin:0;">• RR: Curah hujan (mm)</p>
                                <p style="margin:0;">• ss: Penyinaran (jam)</p>
                                <p style="margin:0;">• ff_x: Kec. Angin Maks (m/s)</p>
                                <p style="margin:0;">• ddd_x: Arah Angin Maks (°)</p>
                                <p style="margin:0;">• ff_avg: Kec. Angin Rata2 (m/s)</p>
                                <p style="margin:0;">• ddd_car: Arah Angin Dominan</p></small></div>""", unsafe_allow_html=True)
            st.markdown("""<div style="background: #fff5f5; padding: 15px; border-radius: 12px; border-left: 4px solid #dc3545; margin-top: 10px;">
                            <h4 style="color: #dc3545; margin: 0 0 10px 0;">Target Prediksi</h4>
                            <p><strong>T_avg:</strong> Suhu udara rata-rata harian (°C)</p></div>""", unsafe_allow_html=True)
    st.markdown("---")
    
    col_data_sample_main, col_desc_stats_main = st.columns(2) 
    with col_data_sample_main:
        with st.container(border=True): 
            st.markdown("#### Sampel Data (5 Baris Pertama)")
            data_sample = load_pickle_data(DATA_SAMPLE_PATH, "sampel data")
            if data_sample is not None:
                st.dataframe(data_sample.head(), use_container_width=True, hide_index=True)
            else:
                st.info(f"File sampel data ('{DATA_SAMPLE_PATH}') tidak tersedia.")
    with col_desc_stats_main:
        with st.container(border=True): 
            st.markdown("#### Statistik Deskriptif Fitur Numerik")
            descriptive_stats = load_pickle_data(DESCRIPTIVE_STATS_PATH, "statistik deskriptif")
            if descriptive_stats is not None:
                st.dataframe(descriptive_stats.style.format("{:.2f}"), use_container_width=True)
            else:
                st.info(f"File statistik ('{DESCRIPTIVE_STATS_PATH}') tidak tersedia.")

# Prediction Section
elif menu == "Prediksi": 
    # ... (Sisa kode untuk Prediksi tetap sama, termasuk Panduan Parameter yang sudah diubah jadi 2 kolom) ...
    st.markdown("""
        <div class="fade-in-up" style="background: linear-gradient(135deg, #2c99a3 0%, #4db8c4 100%); padding: 25px; 
                         border-radius: 16px; text-align: center; box-shadow: 0 8px 32px rgba(44, 153, 163, 0.3); 
                         margin-bottom: 30px;">
            <h1 style="color: white; margin: 0; font-size: 2.2em; font-weight: 700;"><span class="material-icons" style="font-size: 1.1em; vertical-align: middle; margin-right:10px;">online_prediction</span>Prediksi Suhu Rata-rata Harian</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 1.1em;">Masukkan parameter cuaca untuk estimasi suhu</p>
        </div>""", unsafe_allow_html=True)
    
    with st.expander("Panduan Interpretasi Parameter Input", expanded=False):
        param_guide = {
            "RH_avg (%)": "Kelembapan udara. Tinggi (>85%) = lembap.",
            "RR (mm)": "Jumlah hujan. >5mm = potensi hujan.",
            "ss (jam)": "Durasi sinar matahari. >7 jam = cerah.",
            "ff_x (m/s)": "Kecepatan angin maks. >10 m/s = kencang.",
            "ddd_x (°)": "Arah angin maks. (0°=Utara).",
            "ff_avg (m/s)": "Kec. angin rata-rata.",
            "ddd_car": "Arah angin dominan (C=Tenang)."
        }
        param_items = list(param_guide.items())
        num_params = len(param_items)
        for i in range(0, num_params, 2):
            col_guide1, col_guide2 = st.columns(2) 
            with col_guide1:
                if i < num_params:
                    param_key_guide, param_desc_guide = param_items[i] 
                    st.markdown(f"<h5 style='color: #2c99a3; margin-top:15px; margin-bottom: 5px;'>{param_key_guide}</h5>", unsafe_allow_html=True)
                    st.caption(param_desc_guide)
            with col_guide2:
                if i + 1 < num_params:
                    param_key_guide, param_desc_guide = param_items[i+1] 
                    st.markdown(f"<h5 style='color: #2c99a3; margin-top:15px; margin-bottom: 5px;'>{param_key_guide}</h5>", unsafe_allow_html=True)
                    st.caption(param_desc_guide)
                else:
                    st.empty()
        st.markdown("""<div style="background: #e8f4fd; padding: 15px; border-radius: 12px; margin-top: 20px; border-left: 4px solid #2c99a3;">
                        <h5 style="color: #2c99a3; margin: 0 0 10px 0;">Tips</h5>
                        <ul style="margin:0; padding-left:20px; font-size:0.9em; line-height:1.6;">
                        <li>Gunakan data cuaca terkini untuk hasil prediksi akurat.</li>
                        <li>Kombinasi parameter mempengaruhi hasil.</li>
                        <li>Model dilatih untuk kondisi cuaca Kabupaten Sambas.</li></ul></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if model is None:
        st.error("Model prediksi tidak dapat dimuat. Mohon periksa konfigurasi.")
    else:
        with st.container(border=True): 
            st.markdown("<h3 style='color: #2c99a3; margin-bottom: 20px;'>Masukkan Parameter Cuaca</h3>", unsafe_allow_html=True)
            with st.form("prediction_form"):
                col_input1_pred, col_input2_pred, col_input3_pred = st.columns(3)
                with col_input1_pred:
                    RH_avg_input = st.number_input("Kelembapan rata-rata (%)", min_value=0.0, max_value=100.0, value=87.0, step=0.1, format="%.1f", help="Kelembapan udara (0-100%)", key="rh_avg_input")
                    RR_input = st.number_input("Curah hujan (mm)", min_value=0.0, max_value=200.0, value=5.0, step=0.1, format="%.1f", help="Jumlah curah hujan (milimeter)", key="rr_input")
                with col_input2_pred:
                    ss_input = st.number_input("Penyinaran matahari (jam)", min_value=0.0, max_value=24.0, value=4.5, step=0.1, format="%.1f", help="Durasi penyinaran (0-24 jam)", key="ss_input")
                    ff_avg_input = st.number_input("Kec. angin rata-rata (m/s)", min_value=0.0, max_value=50.0, value=1.0, step=0.1, format="%.1f", help="Kecepatan angin rata-rata (m/s)", key="ff_avg_input")
                with col_input3_pred:
                    ff_x_input = st.number_input("Kec. angin maksimum (m/s)", min_value=0.0, max_value=50.0, value=3.5, step=0.1, format="%.1f", help="Kecepatan angin maksimum (m/s)", key="ff_x_input")
                    ddd_x_input = st.number_input("Arah angin maks (°)", min_value=0, max_value=360, value=210, help="Arah angin saat kec. maks. (0-360°)", key="ddd_x_input")
                
                ddd_car_options = ['C', 'S', 'SE', 'E', 'NW', 'NE'] 
                ddd_car_input = st.selectbox("Arah angin dominan (ddd_car)", options=ddd_car_options, index=0, help="Arah angin yang paling sering terjadi", key="ddd_car_input")
                
                _, col_button_pred, _ = st.columns([1.2, 1, 1.2]) 
                with col_button_pred:
                    submit_button = st.form_submit_button(label="Prediksi Sekarang", use_container_width=True)
            
        if submit_button:
            with st.spinner('Memproses prediksi...'):
                time.sleep(0.5)
                input_data = pd.DataFrame({
                    'RH_avg': [RH_avg_input], 'RR': [RR_input], 'ss': [ss_input],
                    'ff_x': [ff_x_input], 'ddd_x': [ddd_x_input], 'ff_avg': [ff_avg_input],
                    'ddd_car': [ddd_car_input]
                })
                try:
                    predicted_t_avg = model.predict(input_data)[0]
                    predicted_t_avg = round(max(predicted_t_avg, 0), 1)
                    
                    weather_material_icon = "help_outline" 
                    if RR_input > 5 or (RH_avg_input > 85 and ss_input < 3):
                        weather_icon_html, weather_text, weather_color, weather_bg = "<span class='material-icons weather-icon'>umbrella</span>", "Berpotensi Hujan", "#4a90e2", "#e8f4f8"
                    elif (predicted_t_avg <= 23):
                        weather_icon_html, weather_text, weather_color, weather_bg = "<span class='material-icons weather-icon'>filter_drama</span>", "Sejuk / Berawan", "#7f8c8d", "#f0f4f8"
                    elif (RR_input == 0 and 60 <= RH_avg_input <= 85 and 3 <= ss_input <= 7) or (24 <= predicted_t_avg <= 28):
                        weather_icon_html, weather_text, weather_color, weather_bg = "<span class='material-icons weather-icon'>cloud</span>", "Berawan", "#f39c12", "#fff8e1"
                    else:
                        weather_icon_html, weather_text, weather_color, weather_bg = "<span class='material-icons weather-icon'>wb_sunny</span>", "Cerah", "#f1c40f", "#fffbe1"
                    
                    st.markdown("<h3 style='text-align:center; margin-top:25px; margin-bottom:-10px;'>Hasil Prediksi Cuaca</h3>", unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class="modern-card" style="background: {weather_bg}; border: 1px solid {weather_color}33; text-align: center; padding: 25px;">
                            {weather_icon_html}
                            <h2 style="color: {weather_color}; margin: 15px 0 8px 0; font-size: 2.2em;">{predicted_t_avg}°C</h2>
                            <h3 style="color: #333; margin: 0 0 20px 0; font-weight:500;">{weather_text}</h3>
                            <div style="background: rgba(255,255,255,0.8); padding: 15px; border-radius: 12px; margin-top: 15px; font-size: 0.9em; color: #555;">
                                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px;">
                                    <div><strong>Kelembapan:</strong> {RH_avg_input}%</div>
                                    <div><strong>Hujan:</strong> {RR_input} mm</div>
                                    <div><strong>Penyinaran:</strong> {ss_input} jam</div>
                                    <div><strong>Angin Rata-rata:</strong> {ff_avg_input} m/s</div>
                                </div></div></div>""", unsafe_allow_html=True)
                    
                    with st.container(border=True): 
                        col_fi, col_eval = st.columns(2)
                        with col_fi:
                            st.markdown("<h4 style='color: #2c99a3; margin-bottom: 10px;'><span class='material-icons' style='vertical-align: bottom; font-size: 1.1em; margin-right: 5px;'>star_rate</span>Feature Importance</h4>", unsafe_allow_html=True)
                            feature_importance_data = {
                                'Fitur': ['RH_avg', 'RR', 'ddd_x', 'ss', 'ff_x', 'ff_avg', 'ddd_car (C)'],
                                'Importance': [0.8123, 0.0763, 0.0453, 0.0346, 0.0199, 0.0099, 0.0015]
                            }
                            df_imp_display = pd.DataFrame(feature_importance_data)
                            st.dataframe(df_imp_display.style.format({'Importance': "{:.4f}"}), hide_index=True, use_container_width=True)
                            st.caption("<small>Berdasarkan <i>mean loss decrease</i> pada data latih.</small>", unsafe_allow_html=True)
                        with col_eval:
                            st.markdown("<h4 style='color: #2c99a3; margin-bottom: 10px;'><span class='material-icons' style='vertical-align: bottom; font-size: 1.1em; margin-right: 5px;'>assessment</span>Kinerja Model (Uji)</h4>", unsafe_allow_html=True)
                            st.markdown(f"""
                                <ul style='font-size: 0.9em; padding-left: 20px;'>
                                    <li><strong>R² Score:</strong> 0.7400 (74.00%)</li>
                                    <li><strong>MAE:</strong> 0.3525 °C</li>
                                    <li><strong>RMSE:</strong> 0.4468 °C</li>
                                </ul>
                            """, unsafe_allow_html=True)
                            st.caption("<small>Metrik evaluasi model pada data uji.</small>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")
# =========================
# FOOTER
# =========================
