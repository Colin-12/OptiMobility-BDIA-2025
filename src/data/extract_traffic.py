import requests
import pandas as pd
from sqlalchemy import create_engine
import os

def extract_and_save_traffic():
    print("🚦 Extraction du trafic parisien...")
    url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptages-routiers-permanents/records?limit=100"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        records = data.get('results', [])
        lignes = []
        for r in records:
            coords = r.get('geo_point_2d')
            if coords and r.get('t_1h'):
                lignes.append({
                    'timestamp': pd.to_datetime(r.get('t_1h')),
                    'nom_rue': r.get('libelle', 'Inconnu'),
                    'taux_occupation': float(r.get('k')) if r.get('k') is not None else 0.0,
                    'debit': float(r.get('q')) if r.get('q') is not None else 0.0,
                    'latitude': coords.get('lat'),
                    'longitude': coords.get('lon')
                })
        
        df = pd.DataFrame(lignes)
        if not df.empty:
            db_uri = os.environ.get('SUPABASE_URI')
            if db_uri.startswith("postgres://"):
                db_uri = db_uri.replace("postgres://", "postgresql://", 1)
            engine = create_engine(db_uri)
            df.to_sql('trafic_paris', engine, if_exists='append', index=False)
            print(f"✅ {len(df)} lignes ajoutées.")

if __name__ == "__main__":
    extract_and_save_traffic()
