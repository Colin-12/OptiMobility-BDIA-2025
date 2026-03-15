import os
import requests
import pandas as pd
from sqlalchemy import create_engine

print("🚦 Extraction des données de trafic en temps réel (Open Data Paris)...")

url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptages-routiers-permanents/records?limit=50"
response = requests.get(url)

if response.status_code == 200:
    records = response.json().get('results', [])
    lignes_trafic = []
    
    for record in records:
        try:
            timestamp = record.get('t_1h')
            nom_rue = record.get('libelle', 'Rue inconnue')
            k_val = float(record.get('k') or 0.0)
            q_val = float(record.get('q') or 0.0)
            coords = record.get('geo_point_2d')
            
            if coords and timestamp:
                lignes_trafic.append({
                    'timestamp': pd.to_datetime(timestamp),
                    'nom_rue': nom_rue,
                    'taux_occupation': k_val,
                    'debit': q_val,
                    'latitude': coords.get('lat'),
                    'longitude': coords.get('lon')
                })
        except Exception:
            continue

    df_trafic = pd.DataFrame(lignes_trafic)
    
    if not df_trafic.empty:
        DB_URI = os.environ.get('SUPABASE_URI')
        if DB_URI.startswith("postgres://"):
            DB_URI = DB_URI.replace("postgres://", "postgresql://", 1)
            
        engine = create_engine(DB_URI)
        df_trafic.to_sql('trafic_paris', engine, if_exists='append', index=False)
        print(f"✅ {len(df_trafic)} relevés insérés.")
