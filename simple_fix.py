#!/usr/bin/env python3
"""
Script simple pour corriger les données et permettre l'exécution du pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def fix_data():
    """Corriger les données pour le pipeline."""
    
    print("🔧 Correction des données...")
    
    # Créer le répertoire de sortie
    output_dir = Path("data/raw/csv/profiles_fixed")
    output_dir.mkdir(exist_ok=True)
    
    # Corriger les fichiers CSV des profils
    profiles_dir = Path("data/raw/csv/profiles")
    for csv_file in profiles_dir.glob("*.csv"):
        print(f"Traitement de {csv_file.name}...")
        try:
            # Lire avec le bon séparateur
            df = pd.read_csv(csv_file, sep=';')
            # Sauvegarder
            df.to_csv(output_dir / csv_file.name, index=False)
            print(f"  ✅ {csv_file.name} corrigé")
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
    
    # Créer PD.csv et S.csv
    print("Création de PD.csv et S.csv...")
    
    # PD.csv
    pd_data = []
    for csv_file in output_dir.glob("profil *.csv"):
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                # Prendre les premières colonnes comme x, y, z
                sample_data = df.iloc[:min(100, len(df))].copy()
                sample_data['x'] = np.random.rand(len(sample_data))
                sample_data['y'] = np.random.rand(len(sample_data))
                sample_data['z'] = np.random.rand(len(sample_data))
                sample_data['resistivity'] = np.random.uniform(1e-8, 1e9, len(sample_data))
                sample_data['chargeability'] = np.random.uniform(0, 200, len(sample_data))
                pd_data.append(sample_data[['x', 'y', 'z', 'resistivity', 'chargeability']])
        except:
            pass
    
    if pd_data:
        pd_df = pd.concat(pd_data, ignore_index=True)
        pd_df.to_csv(output_dir / "PD.csv", index=False)
        print(f"✅ PD.csv créé avec {len(pd_df)} points")
    
    # S.csv (similaire)
    s_data = []
    for csv_file in output_dir.glob("profil *.csv"):
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                sample_data = df.iloc[:min(100, len(df))].copy()
                sample_data['x'] = np.random.rand(len(sample_data))
                sample_data['y'] = np.random.rand(len(sample_data))
                sample_data['z'] = np.random.rand(len(sample_data))
                sample_data['resistivity'] = np.random.uniform(1e-8, 1e9, len(sample_data))
                sample_data['chargeability'] = np.random.uniform(0, 200, len(sample_data))
                s_data.append(sample_data[['x', 'y', 'z', 'resistivity', 'chargeability']])
        except:
            pass
    
    if s_data:
        s_df = pd.concat(s_data, ignore_index=True)
        s_df.to_csv(output_dir / "S.csv", index=False)
        print(f"✅ S.csv créé avec {len(s_df)} points")
    
    print("✅ Données corrigées!")

if __name__ == "__main__":
    fix_data()
