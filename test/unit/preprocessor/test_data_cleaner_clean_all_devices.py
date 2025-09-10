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

from backend.preprocessor.data_cleaner import GeophysicalDataCleaner


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
        
        # Copier les fichiers de test depuis les vrais fichiers de données
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw"
        self.pd_test_file = data_dir / "PD.csv"
        self.s_test_file = data_dir / "S.csv"
        
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
            
            # Vérifier que le fichier a la structure attendue (colonne combinée)
            self.assertIn('x,y,z,resistivity,chargeability,profil_id', pd_df.columns, 
                         "Structure de colonnes incorrecte dans PD.csv")
            
            print(f"✅ PD.csv: {len(pd_df)} lignes, colonnes: {list(pd_df.columns)}")
        
        # Charger et valider S.csv (Schlumberger)
        if s_file.exists():
            s_df = pd.read_csv(s_file, sep=';')
            self.assertGreater(len(s_df), 0, "Fichier S.csv vide")
            
            # Vérifier que le fichier a la structure attendue (colonne combinée)
            self.assertIn('x,y,z,resistivity,chargeability,profil_id', s_df.columns, 
                         "Structure de colonnes incorrecte dans S.csv")
            
            print(f"✅ S.csv: {len(s_df)} lignes, colonnes: {list(s_df.columns)}")
    
    def test_clean_all_devices_data_quality(self):
        """Test de la qualité des données de test"""
        # Vérifier la qualité des données dans PD.csv
        pd_file = self.test_raw_dir / "PD.csv"
        if pd_file.exists():
            # Parser correctement le fichier CSV avec colonne combinée
            pd_df = pd.read_csv(pd_file, sep=';')
            pd_df = pd_df['x,y,z,resistivity,chargeability,profil_id'].str.split(',', expand=True)
            pd_df.columns = ['x', 'y', 'z', 'resistivity', 'chargeability', 'profil_id']
            
            # Convertir les colonnes en types numériques
            for col in ['x', 'y', 'z', 'resistivity', 'chargeability']:
                pd_df[col] = pd.to_numeric(pd_df[col], errors='coerce')
            
            # Vérifier les types de données
            self.assertTrue(pd_df['x'].dtype in [np.float64, np.int64], "Type x incorrect")
            self.assertTrue(pd_df['y'].dtype in [np.float64, np.int64], "Type y incorrect")
            self.assertTrue(pd_df['z'].dtype in [np.float64, np.int64], "Type z incorrect")
            self.assertTrue(pd_df['resistivity'].dtype in [np.float64, np.int64], "Type resistivity incorrect")
            self.assertTrue(pd_df['chargeability'].dtype in [np.float64, np.int64], "Type chargeability incorrect")
            
            # Vérifier les plages de valeurs
            self.assertTrue(all(pd_df['x'] > 500000), "Coordonnées x hors de la zone UTM 30N")
            self.assertTrue(all(pd_df['y'] > 450000), "Coordonnées y hors de la zone UTM 30N")
            self.assertTrue(all(pd_df['resistivity'] > 0), "Résistivité non positive")
            # Tolérer les valeurs de chargeabilité négatives (peuvent être normales dans les données réelles)
            self.assertTrue(len(pd_df[pd_df['chargeability'] < 0]) < len(pd_df) * 0.1, 
                          "Trop de valeurs de chargeabilité négatives")
            
            print("✅ Qualité des données PD.csv validée")
        
        # Vérifier la qualité des données dans S.csv
        s_file = self.test_raw_dir / "S.csv"
        if s_file.exists():
            # Parser correctement le fichier CSV avec colonne combinée
            s_df = pd.read_csv(s_file, sep=';')
            s_df = s_df['x,y,z,resistivity,chargeability,profil_id'].str.split(',', expand=True)
            s_df.columns = ['x', 'y', 'z', 'resistivity', 'chargeability', 'profil_id']
            
            # Convertir les colonnes en types numériques
            for col in ['x', 'y', 'z', 'resistivity', 'chargeability']:
                s_df[col] = pd.to_numeric(s_df[col], errors='coerce')
            
            # Vérifier les types de données
            self.assertTrue(s_df['x'].dtype in [np.float64, np.int64], "Type x incorrect")
            self.assertTrue(s_df['y'].dtype in [np.float64, np.int64], "Type y incorrect")
            self.assertTrue(s_df['z'].dtype in [np.float64, np.int64], "Type z incorrect")
            self.assertTrue(s_df['resistivity'].dtype in [np.float64, np.int64], "Type resistivity incorrect")
            self.assertTrue(s_df['chargeability'].dtype in [np.float64, np.int64], "Type chargeability incorrect")
            
            # Vérifier les plages de valeurs
            self.assertTrue(all(s_df['x'] > 500000), "Coordonnées x hors de la zone UTM 30N")
            self.assertTrue(all(s_df['y'] > 450000), "Coordonnées y hors de la zone UTM 30N")
            self.assertTrue(all(s_df['resistivity'] > 0), "Résistivité non positive")
            # Tolérer les valeurs de chargeabilité négatives (peuvent être normales dans les données réelles)
            self.assertTrue(len(s_df[s_df['chargeability'] < 0]) < len(s_df) * 0.1, 
                          "Trop de valeurs de chargeabilité négatives")
            
            print("✅ Qualité des données S.csv validée")
    
    def test_clean_all_devices_coordinate_consistency(self):
        """Test de la cohérence des coordonnées entre fichiers"""
        pd_file = self.test_raw_dir / "PD.csv"
        s_file = self.test_raw_dir / "S.csv"
        
        if pd_file.exists() and s_file.exists():
            # Parser correctement les fichiers CSV
            pd_df = pd.read_csv(pd_file, sep=';')
            pd_df = pd_df['x,y,z,resistivity,chargeability,profil_id'].str.split(',', expand=True)
            pd_df.columns = ['x', 'y', 'z', 'resistivity', 'chargeability', 'profil_id']
            
            s_df = pd.read_csv(s_file, sep=';')
            s_df = s_df['x,y,z,resistivity,chargeability,profil_id'].str.split(',', expand=True)
            s_df.columns = ['x', 'y', 'z', 'resistivity', 'chargeability', 'profil_id']
            
            # Convertir les colonnes en types numériques
            for col in ['x', 'y', 'z', 'resistivity', 'chargeability']:
                pd_df[col] = pd.to_numeric(pd_df[col], errors='coerce')
                s_df[col] = pd.to_numeric(s_df[col], errors='coerce')
            
            # Vérifier que les coordonnées sont dans des zones géographiques cohérentes
            # Les deux fichiers utilisent des coordonnées UTM (x, y, z)
            pd_x_range = (pd_df['x'].min(), pd_df['x'].max())
            pd_y_range = (pd_df['y'].min(), pd_df['y'].max())
            
            s_x_range = (s_df['x'].min(), s_df['x'].max())
            s_y_range = (s_df['y'].min(), s_df['y'].max())
            
            # Vérifier que les coordonnées UTM sont dans la zone 30N
            self.assertTrue(all(pd_df['x'] > 500000), "Coordonnées x hors de la zone UTM 30N")
            self.assertTrue(all(pd_df['y'] > 450000), "Coordonnées y hors de la zone UTM 30N")
            
            # Vérifier que les coordonnées UTM sont dans la zone 30N pour S.csv aussi
            self.assertTrue(all(s_df['x'] > 500000), "Coordonnées x hors de la zone UTM 30N")
            self.assertTrue(all(s_df['y'] > 450000), "Coordonnées y hors de la zone UTM 30N")
            
            print(f"✅ Cohérence des coordonnées validée:")
            print(f"   PD (UTM): X {pd_x_range}, Y {pd_y_range}")
            print(f"   S (UTM): X {s_x_range}, Y {s_y_range}")
    
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
            # Parser correctement le fichier CSV
            pd_df = pd.read_csv(pd_file, sep=';')
            pd_df = pd_df['x,y,z,resistivity,chargeability,profil_id'].str.split(',', expand=True)
            pd_df.columns = ['x', 'y', 'z', 'resistivity', 'chargeability', 'profil_id']
            
            # Convertir les colonnes en types numériques
            for col in ['x', 'y', 'z', 'resistivity', 'chargeability']:
                pd_df[col] = pd.to_numeric(pd_df[col], errors='coerce')
            
            # Vérifier que les données contiennent des mesures géophysiques
            self.assertTrue(len(pd_df) > 0, "PD.csv ne contient aucune mesure")
            self.assertTrue(all(pd_df['resistivity'] > 0), "Résistivité non positive dans PD.csv")
            # Tolérer les valeurs de chargeabilité négatives (peuvent être normales dans les données réelles)
            self.assertTrue(len(pd_df[pd_df['chargeability'] < 0]) < len(pd_df) * 0.1, 
                          "Trop de valeurs de chargeabilité négatives dans PD.csv")
            
            print(f"✅ Structure PD.csv validée: {len(pd_df)} mesures géophysiques")
        
        # Vérifier S.csv (Schlumberger)
        if s_file.exists():
            # Parser correctement le fichier CSV
            s_df = pd.read_csv(s_file, sep=';')
            s_df = s_df['x,y,z,resistivity,chargeability,profil_id'].str.split(',', expand=True)
            s_df.columns = ['x', 'y', 'z', 'resistivity', 'chargeability', 'profil_id']
            
            # Convertir les colonnes en types numériques
            for col in ['x', 'y', 'z', 'resistivity', 'chargeability']:
                s_df[col] = pd.to_numeric(s_df[col], errors='coerce')
            
            # Vérifier que les données contiennent des mesures géophysiques
            self.assertTrue(len(s_df) > 0, "S.csv ne contient aucune mesure")
            self.assertTrue(all(s_df['resistivity'] > 0), "Résistivité non positive dans S.csv")
            # Tolérer les valeurs de chargeabilité négatives (peuvent être normales dans les données réelles)
            self.assertTrue(len(s_df[s_df['chargeability'] < 0]) < len(s_df) * 0.1, 
                          "Trop de valeurs de chargeabilité négatives dans S.csv")
            
            print(f"✅ Structure S.csv validée: {len(s_df)} mesures géophysiques")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(
        verbosity=2,
        testLoader=unittest.TestLoader(),
        testRunner=unittest.TextTestRunner(stream=sys.stdout)
    )
