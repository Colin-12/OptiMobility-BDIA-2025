import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine
import joblib
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="OptiMobility - Qualité de l'Air", page_icon="🌍", layout="wide")
st.title("🌍 Dashboard OptiMobility : Prédiction de la Pollution (Paris)")
st.markdown("Ce tableau de bord interactif utilise un modèle Machine Learning optimisé (XGBoost) pour anticiper la concentration de PM2.5.")

# --- CONNEXION À LA BASE DE DONNÉES ---
@st.cache_resource
def init_connection():
    db_uri = st.secrets["SUPABASE_URI"]
    if db_uri.startswith("postgres://"):
        db_uri = db_uri.replace("postgres://", "postgresql://", 1)
    return create_engine(db_uri)

engine = init_connection()

# --- RÉCUPÉRATION DES DONNÉES ---
@st.cache_data(ttl=600)
def get_recent_data():
    # On récupère les 25 dernières heures (nécessaire pour calculer pm25_H-24)
    query = "SELECT timestamp, pm2_5, pm10, no2, co FROM qualite_air ORDER BY timestamp DESC LIMIT 25;"
    df = pd.read_sql(query, engine)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp', ascending=True) 
    return df

df_recent = get_recent_data()

if df_recent.empty or len(df_recent) < 25:
    st.warning("Pas assez de données dans la base pour faire une prédiction (25h requises).")
else:
    # --- AFFICHAGE DES KPIS TEMPS RÉEL ---
    st.subheader("Situation Actuelle (Dernier relevé)")
    latest = df_recent.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PM 2.5", f"{latest['pm2_5']:.2f} µg/m³")
    col2.metric("PM 10", f"{latest['pm10']:.2f} µg/m³")
    col3.metric("NO2", f"{latest['no2']:.2f} µg/m³")
    col4.metric("CO", f"{latest['co']:.2f} µg/m³")

    # --- PRÉDICTION AVEC LE MODÈLE XGBOOST ---
    st.subheader("Prédiction de l'IA (Prochaine heure)")
    
    # Chemin vers ton modèle XGBoost
    model_path = os.path.join(os.path.dirname(__file__), '../models/modele_pollution_xgb.pkl')
    
    try:
        model = joblib.load(model_path)
        
        # Préparation des variables (Feature Engineering) exactement comme dans Colab
        next_hour_dt = latest['timestamp'] + pd.Timedelta(hours=1)
        
        features = pd.DataFrame([{
            'heure': next_hour_dt.hour,
            'jour_semaine': next_hour_dt.dayofweek,
            'mois': next_hour_dt.month,
            'pm25_H-1': latest['pm2_5'],
            'pm25_H-24': df_recent.iloc[0]['pm2_5'] # La valeur d'il y a 24h (index 0 sur 25 lignes)
        }])
        
        # Prédiction
        pred_value = model.predict(features)[0]
        
        st.success(f"Concentration PM2.5 prévue pour la prochaine heure : **{pred_value:.2f} µg/m³**")
        
        # --- VISUALISATION INTERACTIVE ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_recent['timestamp'], y=df_recent['pm2_5'], mode='lines+markers', name='Historique PM2.5', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=[latest['timestamp'], next_hour_dt], y=[latest['pm2_5'], pred_value], mode='lines+markers', name='Prédiction IA', line=dict(color='red', dash='dash')))
        fig.update_layout(title="Évolution et Prédiction des Particules Fines", xaxis_title="Heure", yaxis_title="PM 2.5 (µg/m³)")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle XGBoost : {e}")
