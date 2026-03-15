import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine
import joblib
import os
import requests # <-- INDISPENSABLE POUR LES VELIBS

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="OptiMobility - Smart City", page_icon="🌍", layout="wide")
st.title(" OptiMobility : Éco-Mobilité & Prédiction (Paris)")
st.markdown("Ce tableau de bord fusionne l'IA, l'Open Data de la Ville de Paris et l'état du trafic pour optimiser vos déplacements.")

# --- MODULE 3 : INFO TRAFIC RATP (Bandeau d'alerte) ---
st.error(" **Info Trafic RATP en direct :** Trafic perturbé sur la Ligne 4 (Bagage abandonné) et la Ligne B (Panne électrique). Prévoyez des itinéraires de substitution.")

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
    query = "SELECT timestamp, pm2_5, pm10, no2, co FROM qualite_air ORDER BY timestamp DESC LIMIT 100;"
    df = pd.read_sql(query, engine)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values(by='timestamp', ascending=True) 

@st.cache_data(ttl=600)
def get_traffic_data():
    query = "SELECT nom_rue, taux_occupation, debit, latitude, longitude FROM trafic_paris ORDER BY timestamp DESC LIMIT 50;"
    return pd.read_sql(query, engine)

df_recent = get_recent_data()
try:
    df_trafic = get_traffic_data()
except:
    df_trafic = pd.DataFrame() # Sécurité si la table est vide

# --- CALCUL DU SCORE OPTIMOBILITY ---
moyenne_bouchons = df_trafic['taux_occupation'].mean() if not df_trafic.empty else 0
derniere_pollution = df_recent.iloc[-1]['pm2_5'] if not df_recent.empty else 0

st.markdown("---")
st.subheader(" Recommandation de Mobilité")
if derniere_pollution > 25 or moyenne_bouchons > 10:
    st.warning(f"⚠️ **Conditions Dégradées** (Pollution: {derniere_pollution:.1f} µg/m³ | Trafic: Dense). **Action recommandée :** Privilégiez le télétravail ou le métro.")
elif derniere_pollution > 15:
    st.info(f"🟡 **Conditions Moyennes** (Pollution modérée). **Action recommandée :** Transports en commun.")
else:
    st.success(f"🟢 **Conditions Optimales** (Air pur | Trafic fluide). **Action recommandée :** Mobilité douce (Marche, Vélo, Trottinette) !")

# --- MODULE 2 : CARTE GLOBALE (TRAFIC + VÉLIB') ---
st.markdown("---")
st.subheader("Carte : Embouteillages & Mobilité Douce")
st.markdown(" *Astuce : Cliquez sur la légende (en haut à gauche de la carte) pour masquer/afficher les vélos ou le trafic.*")

# 1. Requête en direct vers l'API Vélib'
df_velib = pd.DataFrame()
try:
    url_velib = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/records?limit=60"
    res = requests.get(url_velib).json().get('results', [])
    velibs = []
    for v in res:
        coords = v.get('coordonnees_geo')
        if coords:
            velibs.append({
                'Station': v.get('name'),
                'Vélos Dispos': v.get('numbikesavailable', 0) + v.get('ebike', 0),
                'lat': coords.get('lat'),
                'lon': coords.get('lon')
            })
    df_velib = pd.DataFrame(velibs)
except Exception as e:
    st.error("Impossible de joindre l'API Vélib'")

# 2. Création de la carte unifiée
fig_map = go.Figure()

# Calque 1 : Les Vélib' (Points Verts)
if not df_velib.empty:
    fig_map.add_trace(go.Scattermapbox(
        lat=df_velib['lat'], lon=df_velib['lon'], mode='markers',
        marker=go.scattermapbox.Marker(size=df_velib['Vélos Dispos'], sizemode='area', sizeref=1.5, sizemin=5, color='mediumseagreen'),
        text=df_velib['Station'] + "<br>Vélos dispos : " + df_velib['Vélos Dispos'].astype(str),
        hoverinfo='text', name="Stations Vélib'"
    ))

# Calque 2 : Le Trafic (Points Rouge/Orange)
if not df_trafic.empty:
    fig_map.add_trace(go.Scattermapbox(
        lat=df_trafic['latitude'], lon=df_trafic['longitude'], mode='markers',
        marker=go.scattermapbox.Marker(
            size=df_trafic['taux_occupation'], sizemode='area', sizeref=0.5, sizemin=6,
            color=df_trafic['taux_occupation'], colorscale='YlOrRd', showscale=True,
            colorbar=dict(title="Bouchons (%)", x=0.99)
        ),
        text=df_trafic['nom_rue'] + "<br>Taux d'occ : " + df_trafic['taux_occupation'].astype(str) + "%",
        hoverinfo='text', name="Trafic Routier"
    ))

# Paramétrage de la carte
fig_map.update_layout(
    mapbox_style="carto-positron", mapbox=dict(center=dict(lat=48.8566, lon=2.3522), zoom=11.5),
    margin={"r":0,"t":0,"l":0,"b":0}, height=600,
    legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(255, 255, 255, 0.8)")
)
st.plotly_chart(fig_map, use_container_width=True)

# --- MODULE 1 : QUALITÉ DE L'AIR & IA ---
st.markdown("---")
st.subheader(" Qualité de l'Air & Prévision (+1h)")

if df_recent.empty or len(df_recent) < 25:
    st.warning("⚠️ Pas assez de données dans la base pour faire une prédiction (25h requises).")
else:
    latest = df_recent.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PM 2.5", f"{latest['pm2_5']:.2f} µg/m³")
    col2.metric("PM 10", f"{latest['pm10']:.2f} µg/m³")
    col3.metric("NO2", f"{latest['no2']:.2f} µg/m³")
    col4.metric("CO", f"{latest['co']:.2f} µg/m³")

    model_path = os.path.join(os.path.dirname(__file__), '../models/modele_pollution_xgb.pkl')
    try:
        model = joblib.load(model_path)
        next_hour_dt = latest['timestamp'] + pd.Timedelta(hours=1)
        features = pd.DataFrame([{'heure': next_hour_dt.hour, 'jour_semaine': next_hour_dt.dayofweek,
                                  'mois': next_hour_dt.month, 'pm25_H-1': latest['pm2_5'],
                                  'pm25_H-24': df_recent.iloc[-25]['pm2_5']}])
        pred_value = model.predict(features)[0]
        st.success(f" Concentration PM2.5 prévue pour la prochaine heure : **{pred_value:.2f} µg/m³**")
        
        df_historique = df_recent.copy()
        df_historique['heure'], df_historique['jour_semaine'], df_historique['mois'] = df_historique['timestamp'].dt.hour, df_historique['timestamp'].dt.dayofweek, df_historique['timestamp'].dt.month
        df_historique['pm25_H-1'], df_historique['pm25_H-24'] = df_historique['pm2_5'].shift(1), df_historique['pm2_5'].shift(24) 
        df_valide = df_historique.dropna()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_recent['timestamp'], y=df_recent['pm2_5'], mode='lines+markers', name='Réalité (Historique)', line=dict(color='blue')))
        if not df_valide.empty:
            colonnes_modele = ['heure', 'jour_semaine', 'mois', 'pm25_H-1', 'pm25_H-24']
            predictions_passees = model.predict(df_valide[colonnes_modele])
            fig.add_trace(go.Scatter(x=df_valide['timestamp'], y=predictions_passees, mode='lines', name='Prédiction passée', line=dict(color='green', dash='dot')))
        fig.add_trace(go.Scatter(x=[latest['timestamp'], next_hour_dt], y=[latest['pm2_5'], pred_value], mode='lines+markers', name='Prédiction Future (+1h)', line=dict(color='red', dash='dash', width=3)))
        
        fig.update_layout(title="Évolution et Prédiction des Particules Fines (PM2.5)", xaxis_title="Heure", yaxis_title="PM 2.5 (µg/m³)")
        # --- MISE AU FORMAT FRANCAIS DE LA DATE ---
        fig.update_xaxes(tickformat="%d/%m %H:%M") 
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle XGBoost : {e}")
