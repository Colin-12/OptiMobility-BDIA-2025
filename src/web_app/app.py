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
    # On passe à LIMIT 100 pour voir une belle fenêtre glissante sur plusieurs jours
    query = "SELECT timestamp, pm2_5, pm10, no2, co FROM qualite_air ORDER BY timestamp DESC LIMIT 100;"
    df = pd.read_sql(query, engine)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp', ascending=True) 
    return df

df_recent = get_recent_data()

if df_recent.empty or len(df_recent) < 25:
    st.warning("⚠️ Pas assez de données dans la base pour faire une prédiction (25h requises).")
else:
    # --- AFFICHAGE DES KPIS TEMPS RÉEL ---
    st.subheader(" Situation Actuelle (Dernier relevé)")
    latest = df_recent.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PM 2.5", f"{latest['pm2_5']:.2f} µg/m³")
    col2.metric("PM 10", f"{latest['pm10']:.2f} µg/m³")
    col3.metric("NO2", f"{latest['no2']:.2f} µg/m³")
    col4.metric("CO", f"{latest['co']:.2f} µg/m³")

    # --- PRÉDICTION AVEC LE MODÈLE XGBOOST ---
    st.subheader(" Prédiction de l'IA (Prochaine heure)")
    
    model_path = os.path.join(os.path.dirname(__file__), '../models/modele_pollution_xgb.pkl')
    
    try: # Le fameux bloc TRY commence ici
        model = joblib.load(model_path)
        
        # Prédiction future
        next_hour_dt = latest['timestamp'] + pd.Timedelta(hours=1)
        features = pd.DataFrame([{
            'heure': next_hour_dt.hour,
            'jour_semaine': next_hour_dt.dayofweek,
            'mois': next_hour_dt.month,
            'pm25_H-1': latest['pm2_5'],
            # On cherche la ligne d'il y a 24h (25ème en partant de la fin)
            'pm25_H-24': df_recent.iloc[-25]['pm2_5'] 
        }])
        
        pred_value = model.predict(features)[0]
        st.success(f" Concentration PM2.5 prévue pour la prochaine heure : **{pred_value:.2f} µg/m³**")
        
        # --- PRÉPARATION DES PRÉDICTIONS HISTORIQUES (BACKTESTING) ---
        df_historique = df_recent.copy()
        df_historique['heure'] = df_historique['timestamp'].dt.hour
        df_historique['jour_semaine'] = df_historique['timestamp'].dt.dayofweek
        df_historique['mois'] = df_historique['timestamp'].dt.month
        df_historique['pm25_H-1'] = df_historique['pm2_5'].shift(1)
        df_historique['pm25_H-24'] = df_historique['pm2_5'].shift(24) 
        
        df_valide = df_historique.dropna()
        
        # --- VISUALISATION INTERACTIVE ---
        fig = go.Figure()
        
        # 1. Ligne de la Réalité (Bleu)
        fig.add_trace(go.Scatter(x=df_recent['timestamp'], y=df_recent['pm2_5'], 
                                 mode='lines+markers', name='Réalité (Historique)', line=dict(color='blue')))
        
        # 2. Ligne des Prédictions passées de l'IA (Vert pointillé)
        if not df_valide.empty:
            colonnes_modele = ['heure', 'jour_semaine', 'mois', 'pm25_H-1', 'pm25_H-24']
            predictions_passees = model.predict(df_valide[colonnes_modele])
            fig.add_trace(go.Scatter(x=df_valide['timestamp'], y=predictions_passees, 
                                     mode='lines', name='Prédiction passée du modèle IA', line=dict(color='green', dash='dot')))
        
        # 3. Le point du Futur (Rouge)
        fig.add_trace(go.Scatter(x=[latest['timestamp'], next_hour_dt], y=[latest['pm2_5'], pred_value], 
                                 mode='lines+markers', name='Prédiction Future (+1h)', line=dict(color='red', dash='dash', width=3)))
        
        fig.update_layout(title="Évolution et Prédiction des Particules Fines (PM2.5)", 
                          xaxis_title="Heure", yaxis_title="PM 2.5 (µg/m³)")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e: # ET VOICI LE BLOC EXCEPT SAUVEUR !
        st.error(f"❌ Erreur lors du chargement du modèle XGBoost : {e}")


    # --- MODULE 2 : CARTE DU TRAFIC ROUTIER (EMBOUTEILLAGES) ---
        st.markdown("---")
        st.subheader("🚗 Trafic Routier en Temps Réel (Paris)")
        
        @st.cache_data(ttl=600)
        def get_traffic_data():
            # On récupère les 50 derniers capteurs
            query = "SELECT nom_rue, taux_occupation, debit, latitude, longitude FROM trafic_paris ORDER BY timestamp DESC LIMIT 50;"
            return pd.read_sql(query, engine)
            
        try:
            df_trafic = get_traffic_data()
            
            if not df_trafic.empty:
                import plotly.express as px
                
                # Création de la carte interactive
                fig_map = px.scatter_mapbox(
                    df_trafic, 
                    lat="latitude", 
                    lon="longitude", 
                    color="taux_occupation",
                    size="taux_occupation",
                    hover_name="nom_rue",
                    hover_data={"taux_occupation": True, "debit": True, "latitude": False, "longitude": False},
                    color_continuous_scale=px.colors.sequential.YlOrRd,
                    size_max=15, 
                    zoom=11,
                    mapbox_style="carto-positron",
                    title="Carte d'encombrement des axes (Taux d'occupation)"
                )
                
                st.plotly_chart(fig_map, use_container_width=True)
                st.info("💡 Corrélation : Observez comment les zones à fort taux d'occupation (rouge) impactent la qualité de l'air locale.")
            else:
                st.warning("⏳ En attente des données de trafic...")
        except Exception as e:
            st.error(f"❌ Impossible de charger la carte du trafic : {e}")
