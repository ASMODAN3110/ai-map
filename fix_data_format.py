#!/usr/bin/env python3
"""
Script pour corriger le format des données CSV et permettre l'exécution complète du pipeline.
"""

import pandas as pd
import os
from pathlib import Path

def fix_csv_files():
    """Corriger le format des fichiers CSV des profils."""
    
    profiles_dir = Path("data/raw/csv/profiles")
    fixed_dir = Path("data/raw/csv/profiles_fixed")
    fixed_dir.mkdir(exist_ok=True)
    
    print("🔧 Correction du format des fichiers CSV...")
    
    for csv_file in profiles_dir.glob("*.csv"):
        print(f"Traitement de {csv_file.name}...")
        
        try:
            # Lire le fichier avec le bon séparateur
            df = pd.read_csv(csv_file, sep=';')
            
            # Vérifier si les colonnes sont correctes
            expected_columns = [
                'x', 'y', 'z', 'Rho(ohm.m)', 'M (mV/V)', 'SP (mV)', 
                'xA (m)', 'xB (m)', 'xM (m)', 'xN (m)', 'Dev. M', 
                'VMN (mV)', 'IAB (mA)'
            ]
            
            if list(df.columns) == expected_columns:
                print(f"  ✅ {csv_file.name} - Format correct")
                # Copier vers le répertoire fixé
                df.to_csv(fixed_dir / csv_file.name, index=False)
            else:
                print(f"  ⚠️  {csv_file.name} - Colonnes inattendues: {list(df.columns)}")
                
        except Exception as e:
            print(f"  ❌ Erreur avec {csv_file.name}: {e}")
    
    print(f"✅ Fichiers corrigés sauvegardés dans {fixed_dir}")
    return fixed_dir

def create_sample_data():
    """Créer des données d'exemple pour tester le pipeline."""
    
    print("📊 Création de données d'exemple...")
    
    # Créer des données factices pour les profils
    profiles_dir = Path("data/raw/csv/profiles_sample")
    profiles_dir.mkdir(exist_ok=True)
    
    for i in range(1, 6):
        # Créer des données géophysiques factices
        n_points = 100
        df = pd.DataFrame({
            'x': [100 + i * 10 + j for j in range(n_points)],
            'y': [200 + i * 10 + j for j in range(n_points)],
            'z': [i * 10 + j * 0.1 for j in range(n_points)],
            'Rho(ohm.m)': [50 + i * 10 + j * 0.5 for j in range(n_points)],
            'M (mV/V)': [10 + i * 2 + j * 0.1 for j in range(n_points)],
            'SP (mV)': [-20 + i * 5 + j * 0.2 for j in range(n_points)],
            'xA (m)': [100 + i * 10 for _ in range(n_points)],
            'xB (m)': [150 + i * 10 for _ in range(n_points)],
            'xM (m)': [120 + i * 10 for _ in range(n_points)],
            'xN (m)': [130 + i * 10 for _ in range(n_points)],
            'Dev. M': [i for _ in range(n_points)],
            'VMN (mV)': [5 + i * 0.5 for _ in range(n_points)],
            'IAB (mA)': [10 + i * 0.1 for _ in range(n_points)]
        })
        
        df.to_csv(profiles_dir / f"profil {i}.csv", index=False)
        print(f"  ✅ Créé profil {i}.csv avec {n_points} points")
    
    print(f"✅ Données d'exemple créées dans {profiles_dir}")
    return profiles_dir

def update_config_for_sample_data():
    """Mettre à jour la configuration pour utiliser les données d'exemple."""
    
    print("⚙️  Mise à jour de la configuration...")
    
    # Lire le fichier config.py
    with open("config.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Remplacer le chemin des données
    new_content = content.replace(
        'raw_data_dir = Path("data/raw")',
        'raw_data_dir = Path("data/raw/csv/profiles_sample")'
    )
    
    # Sauvegarder la configuration temporaire
    with open("config_sample.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("✅ Configuration temporaire créée: config_sample.py")

def main():
    """Fonction principale."""
    print("🚀 Correction des données pour l'exécution du pipeline AI-MAP")
    print("=" * 60)
    
    # Option 1: Corriger les fichiers existants
    print("\n📁 Option 1: Correction des fichiers existants")
    fixed_dir = fix_csv_files()
    
    # Option 2: Créer des données d'exemple
    print("\n📊 Option 2: Création de données d'exemple")
    sample_dir = create_sample_data()
    
    # Mettre à jour la configuration
    update_config_for_sample_data()
    
    print("\n" + "=" * 60)
    print("✅ CORRECTION TERMINÉE!")
    print("=" * 60)
    print("\n📋 Pour exécuter le pipeline complet:")
    print("1. Utiliser les données corrigées:")
    print("   - Modifier config.py pour pointer vers data/raw/csv/profiles_fixed")
    print("   - Puis: python main.py")
    print("\n2. Utiliser les données d'exemple:")
    print("   - python main.py --config config_sample.py")
    print("   - Ou modifier config.py temporairement")
    print("\n3. Test rapide:")
    print("   - python main.py --skip-training  # Pour tester sans entraînement")

if __name__ == "__main__":
    main()
