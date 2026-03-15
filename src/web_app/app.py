import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="OptiMobility - Qualité de l'Air", page_icon="🌍", layout="wide")
st.title("🌍 Dashboard OptiMobility : Prédiction de la Pollution (Paris)")
st.markdown("Ce tableau de bord interactif affiche les données en temps réel et utilise un modèle Deep Learning (LSTM) pour anticiper la concentration de PM2.5.")

# --- CONNEXION À LA BASE DE DONNÉES ---
# Streamlit Cloud utilise st.secrets pour cacher les mots de passe
@st.cache_resource
def init_connection():
    db_uri = st.secrets["SUPABASE_URI"]
    if db_uri.startswith("postgres://"):
        db_uri = db_uri.replace("postgres://", "postgresql://", 1)
    return create_engine(db_uri)

engine = init_connection()

# --- RÉCUPÉRATION DES DONNÉES ---
@st.cache_data(ttl=600) # Garde en cache pendant 10 min pour ne pas surcharger la BDD
def get_recent_data():
    # On récupère les 48 dernières lignes (puisque ton LSTM utilise SEQ_LENGTH = 48)
    query = "SELECT timestamp, pm2_5, pm10, no2, co FROM qualite_air ORDER BY timestamp DESC LIMIT 48;"
    df = pd.read_sql(query, engine)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp', ascending=True) # Remettre dans l'ordre chronologique
    return df

df_recent = get_recent_data()

if df_recent.empty or len(df_recent) < 48:
    st.warning("⚠️ Pas assez de données dans la base pour faire une prédiction (48h requises).")
else:
    # --- AFFICHAGE DES KPIS TEMPS RÉEL ---
    st.subheader("Situation Actuelle (Dernier relevé)")
    latest = df_recent.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PM 2.5", f"{latest['pm2_5']:.2f} µg/m³")
    col2.metric("PM 10", f"{latest['pm10']:.2f} µg/m³")
    col3.metric("NO2", f"{latest['no2']:.2f} µg/m³")
    col4.metric("CO", f"{latest['co']:.2f} µg/m³")

    # --- PRÉDICTION AVEC LE MODÈLE LSTM ---
    st.subheader("Prédiction de l'IA (Prochaine heure)")
    
    # Chargement du modèle
    model_path = os.path.join(os.path.dirname(__file__), '../models/modele_pollution_lstm_multivarie.keras')
    try:
        model = load_model(model_path)
        
        # Préparation des données pour le LSTM (On simule le MinMaxScaler de l'entraînement)
        scaler = MinMaxScaler()
        # Note : Dans un vrai projet, il faudrait charger le scaler sauvegardé (joblib) 
        # plutôt que de le refit ici, mais pour le MVP cela donnera une approximation visuelle
        scaled_data = scaler.fit_transform(df_recent[['pm2_5']].values) 
        
        # Création de la séquence 3D pour le LSTM
        X_pred = np.reshape(scaled_data, (1, 48, 1))
        
        # Prédiction
        pred_scaled = model.predict(X_pred)
        pred_value = scaler.inverse_transform(pred_scaled)[0][0]
        
        st.success(f"Concentration PM2.5 prévue pour la prochaine heure : **{pred_value:.2f} µg/m³**")
        
        # --- VISUALISATION INTERACTIVE (HISTORIQUE + PRÉDICTION) ---
        fig = go.Figure()
        # Ligne de l'historique
        fig.add_trace(go.Scatter(x=df_recent['timestamp'], y=df_recent['pm2_5'], mode='lines+markers', name='Historique PM2.5', line=dict(color='blue')))
        
        # Point de prédiction (1 heure plus tard)
        next_hour = df_recent['timestamp'].iloc[-1] + pd.Timedelta(hours=1)
        fig.add_trace(go.Scatter(x=[df_recent['timestamp'].iloc[-1], next_hour], 
                                 y=[latest['pm2_5'], pred_value], 
                                 mode='lines+markers', name='Prédiction IA', line=dict(color='red', dash='dash')))
        
        fig.update_layout(title="Évolution et Prédiction des Particules Fines", xaxis_title="Heure", yaxis_title="PM 2.5 (µg/m³)")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle ou de la prédiction : {e}")
