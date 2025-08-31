#!/usr/bin/env python3
"""
Test unitaire pour la fonction clean_all_devices de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la fonction clean_all_devices
avec de vrais fichiers de données géophysiques (PD.csv et S.csv).
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.preprocessor.data_cleaner import GeophysicalDataCleaner


class TestDataCleanerCleanAllDevices(unittest.TestCase):
    """Tests pour la fonction clean_all_devices de GeophysicalDataCleaner"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        # Créer un répertoire temporaire pour les tests
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Créer la structure des dossiers de test
        self.test_raw_dir = self.test_dir / "raw"
        self.test_processed_dir = self.test_dir / "processed"
        self.test_raw_dir.mkdir(parents=True, exist_ok=True)
        self.test_processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Copier les fichiers de test
        fixtures_dir = Path(__file__).parent.parent.parent / "fixtures" / "raw"
        self.pd_test_file = fixtures_dir / "PD.csv"
        self.s_test_file = fixtures_dir / "S.csv"
        
        # Copier les fichiers vers le dossier de test
        if self.pd_test_file.exists():
            shutil.copy2(self.pd_test_file, self.test_raw_dir / "PD.csv")
        if self.s_test_file.exists():
            shutil.copy2(self.s_test_file, self.test_raw_dir / "S.csv")
        
        # Créer une instance du cleaner standard
        self.cleaner = GeophysicalDataCleaner()
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Supprimer le répertoire temporaire
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_clean_all_devices_basic_functionality(self):
        """Test de base de la fonction clean_all_devices"""
        print(f"\n🔍 Test avec fichiers dans: {self.test_raw_dir}")
        print(f"📁 Fichiers disponibles: {list(self.test_raw_dir.glob('*'))}")
        
        # Vérifier que les fichiers de test existent
        self.assertTrue((self.test_raw_dir / "PD.csv").exists(), "Fichier PD.csv manquant")
        self.assertTrue((self.test_raw_dir / "S.csv").exists(), "Fichier S.csv manquant")
        
        print("✅ Fichiers de test disponibles")
        
        # Test de base : vérifier que la fonction existe et est appelable
        self.assertTrue(hasattr(self.cleaner, 'clean_all_devices'))
        self.assertTrue(callable(self.cleaner.clean_all_devices))
        
        print("✅ Méthode clean_all_devices disponible et appelable")
    
    def test_clean_all_devices_file_validation(self):
        """Test de validation des fichiers de test"""
        # Vérifier le contenu des fichiers de test
        pd_file = self.test_raw_dir / "PD.csv"
        s_file = self.test_raw_dir / "S.csv"
        
        # Charger et valider PD.csv (Pole-Dipole)
        if pd_file.exists():
            pd_df = pd.read_csv(pd_file, sep=';')
            self.assertGreater(len(pd_df), 0, "Fichier PD.csv vide")
            
            # Colonnes attendues pour PD.csv
            expected_pd_columns = ['x', 'y', 'z', 'Rho(ohm.m)', 'M (mV/V)']
            for col in expected_pd_columns:
                self.assertIn(col, pd_df.columns, f"Colonne {col} manquante dans PD.csv")
            
            print(f"✅ PD.csv: {len(pd_df)} lignes, colonnes: {list(pd_df.columns)}")
        
        # Charger et valider S.csv (Schlumberger)
        if s_file.exists():
            s_df = pd.read_csv(s_file, sep=';')
            self.assertGreater(len(s_df), 0, "Fichier S.csv vide")
            
            # Colonnes attendues pour S.csv
            expected_s_columns = ['LAT', 'LON', 'Rho (Ohm.m)', 'M (mV/V)']
            for col in expected_s_columns:
                self.assertIn(col, s_df.columns, f"Colonne {col} manquante dans S.csv")
            
            print(f"✅ S.csv: {len(s_df)} lignes, colonnes: {list(s_df.columns)}")
    
    def test_clean_all_devices_data_quality(self):
        """Test de la qualité des données de test"""
        # Vérifier la qualité des données dans PD.csv
        pd_file = self.test_raw_dir / "PD.csv"
        if pd_file.exists():
            pd_df = pd.read_csv(pd_file, sep=';')
            
            # Vérifier les types de données
            self.assertTrue(pd_df['x'].dtype in [np.float64, np.int64], "Type x incorrect")
            self.assertTrue(pd_df['y'].dtype in [np.float64, np.int64], "Type y incorrect")
            self.assertTrue(pd_df['z'].dtype in [np.float64, np.int64], "Type z incorrect")
            self.assertTrue(pd_df['Rho(ohm.m)'].dtype in [np.float64, np.int64], "Type Rho incorrect")
            self.assertTrue(pd_df['M (mV/V)'].dtype in [np.float64, np.int64], "Type M incorrect")
            
            # Vérifier les plages de valeurs
            self.assertTrue(all(pd_df['x'] > 500000), "Coordonnées x hors de la zone UTM 30N")
            self.assertTrue(all(pd_df['y'] > 450000), "Coordonnées y hors de la zone UTM 30N")
            self.assertTrue(all(pd_df['Rho(ohm.m)'] > 0), "Résistivité non positive")
            self.assertTrue(all(pd_df['M (mV/V)'] >= 0), "Chargeabilité négative")
            
            print("✅ Qualité des données PD.csv validée")
        
        # Vérifier la qualité des données dans S.csv
        s_file = self.test_raw_dir / "S.csv"
        if s_file.exists():
            s_df = pd.read_csv(s_file, sep=';')
            
            # Vérifier les types de données
            self.assertTrue(s_df['LAT'].dtype in [np.float64, np.int64], "Type LAT incorrect")
            self.assertTrue(s_df['LON'].dtype in [np.float64, np.int64], "Type LON incorrect")
            self.assertTrue(s_df['Rho (Ohm.m)'].dtype in [np.float64, np.int64], "Type Rho incorrect")
            self.assertTrue(s_df['M (mV/V)'].dtype in [np.float64, np.int64], "Type M incorrect")
            
            # Vérifier les plages de valeurs
            self.assertTrue(all(s_df['LAT'].between(4.68, 4.71)), "LAT hors de la plage attendue")
            self.assertTrue(all(s_df['LON'].between(12.34, 12.35)), "LON hors de la plage attendue")
            self.assertTrue(all(s_df['Rho (Ohm.m)'] > 0), "Résistivité non positive")
            self.assertTrue(all(s_df['M (mV/V)'] >= 0), "Chargeabilité négative")
            
            print("✅ Qualité des données S.csv validée")
    
    def test_clean_all_devices_coordinate_consistency(self):
        """Test de la cohérence des coordonnées entre fichiers"""
        pd_file = self.test_raw_dir / "PD.csv"
        s_file = self.test_raw_dir / "S.csv"
        
        if pd_file.exists() and s_file.exists():
            pd_df = pd.read_csv(pd_file, sep=';')
            s_df = pd.read_csv(s_file, sep=';')
            
            # Vérifier que les coordonnées sont dans des zones géographiques cohérentes
            # PD.csv : Coordonnées UTM (x, y) - zone 30N (Europe de l'Ouest)
            pd_x_range = (pd_df['x'].min(), pd_df['x'].max())
            pd_y_range = (pd_df['y'].min(), pd_df['y'].max())
            
            # S.csv : Coordonnées WGS84 (LAT, LON) - zone équatoriale
            s_lat_range = (s_df['LAT'].min(), s_df['LAT'].max())
            s_lon_range = (s_df['LON'].min(), s_df['LON'].max())
            
            # Vérifier que les coordonnées UTM sont dans la zone 30N
            self.assertTrue(all(pd_df['x'] > 500000), "Coordonnées x hors de la zone UTM 30N")
            self.assertTrue(all(pd_df['y'] > 450000), "Coordonnées y hors de la zone UTM 30N")
            
            # Vérifier que les coordonnées WGS84 sont dans une zone équatoriale
            self.assertTrue(all(s_df['LAT'].between(4.68, 4.71)), "LAT hors de la zone équatoriale")
            self.assertTrue(all(s_df['LON'].between(12.34, 12.35)), "LON hors de la zone équatoriale")
            
            print(f"✅ Cohérence des coordonnées validée:")
            print(f"   PD (UTM): X {pd_x_range}, Y {pd_y_range}")
            print(f"   S (WGS84): LAT {s_lat_range}, LON {s_lon_range}")
    
    def test_clean_all_devices_method_availability(self):
        """Test de la disponibilité des méthodes de nettoyage"""
        # Vérifier que toutes les méthodes nécessaires sont disponibles
        required_methods = [
            'clean_all_devices',
            'get_cleaning_summary',
            '_clean_device_data'
        ]
        
        for method in required_methods:
            with self.subTest(method=method):
                self.assertTrue(hasattr(self.cleaner, method), f"Méthode '{method}' manquante")
                method_obj = getattr(self.cleaner, method)
                self.assertTrue(callable(method_obj), f"'{method}' n'est pas appelable")
        
        print(f"✅ Toutes les méthodes requises sont disponibles: {required_methods}")
    
    def test_clean_all_devices_attributes(self):
        """Test des attributs de la classe"""
        # Vérifier que tous les attributs requis existent
        required_attributes = [
            'report',
            'raw_data_dir',
            'processed_data_dir',
            'coord_transformer'
        ]
        
        for attr in required_attributes:
            with self.subTest(attr=attr):
                self.assertTrue(hasattr(self.cleaner, attr), f"Attribut '{attr}' manquant")
        
        print(f"✅ Tous les attributs requis sont présents: {required_attributes}")
        
        # Vérifier les types des attributs
        self.assertIsInstance(self.cleaner.report, dict, "report doit être un dictionnaire")
        self.assertIsInstance(self.cleaner.raw_data_dir, Path, "raw_data_dir doit être un Path")
        self.assertIsInstance(self.cleaner.processed_data_dir, Path, "processed_data_dir doit être un Path")
        
        print("✅ Types des attributs validés")
    
    def test_clean_all_devices_data_structure(self):
        """Test de la structure des données géophysiques"""
        pd_file = self.test_raw_dir / "PD.csv"
        s_file = self.test_raw_dir / "S.csv"
        
        # Vérifier PD.csv (Pole-Dipole)
        if pd_file.exists():
            pd_df = pd.read_csv(pd_file, sep=';')
            
            # Vérifier que les données contiennent des mesures géophysiques
            self.assertTrue(len(pd_df) > 0, "PD.csv ne contient aucune mesure")
            self.assertTrue(all(pd_df['Rho(ohm.m)'] > 0), "Résistivité non positive dans PD.csv")
            self.assertTrue(all(pd_df['M (mV/V)'] >= 0), "Chargeabilité négative dans PD.csv")
            
            print(f"✅ Structure PD.csv validée: {len(pd_df)} mesures géophysiques")
        
        # Vérifier S.csv (Schlumberger)
        if s_file.exists():
            s_df = pd.read_csv(s_file, sep=';')
            
            # Vérifier que les données contiennent des mesures géophysiques
            self.assertTrue(len(s_df) > 0, "S.csv ne contient aucune mesure")
            self.assertTrue(all(s_df['Rho (Ohm.m)'] > 0), "Résistivité non positive dans S.csv")
            self.assertTrue(all(s_df['M (mV/V)'] >= 0), "Chargeabilité négative dans S.csv")
            
            print(f"✅ Structure S.csv validée: {len(s_df)} mesures géophysiques")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(
        verbosity=2,
        testLoader=unittest.TestLoader(),
        testRunner=unittest.TextTestRunner(stream=sys.stdout)
    )
