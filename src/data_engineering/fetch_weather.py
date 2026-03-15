import os
import requests
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine

# --- 1. CONFIGURATION ---
# On utilise os.getenv() au lieu de userdata.get()
API_KEY = os.environ.get('OPENWEATHER_API_KEY')
DB_URI = os.environ.get('SUPABASE_URI')

if DB_URI and DB_URI.startswith("postgres://"):
    DB_URI = DB_URI.replace("postgres://", "postgresql://", 1)

LAT = 48.8566
LON = 2.3522

def fetch_air_quality():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()
        components = data['list'][0]['components']
        
        clean_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "city": "Paris",
            "aqi": data['list'][0]['main']['aqi'],
            "co": components.get('co'),
            "no2": components.get('no2'),
            "pm2_5": components.get('pm2_5'),
            "pm10": components.get('pm10')
        }
        
        df = pd.DataFrame([clean_data])
        print("✅ Extraction API réussie")
        return df
    except Exception as e:
        print(f"❌ Erreur API : {e}")
        return None

def load_to_supabase(df, table_name):
    try:
        engine = create_engine(DB_URI)
        df.to_sql(table_name, engine, if_exists='append', index=False)
        print(f"✅ Succès : Données insérées dans '{table_name}' !")
    except Exception as e:
        print(f"❌ Erreur BDD : {e}")

if __name__ == "__main__":
    df_pollution = fetch_air_quality()
    if df_pollution is not None:
        load_to_supabase(df_pollution, 'qualite_air')
