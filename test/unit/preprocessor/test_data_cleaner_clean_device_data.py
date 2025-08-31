#!/usr/bin/env python3
"""
Test unitaire pour la méthode _clean_device_data de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _clean_device_data
avec des fichiers CSV réels et des scénarios de test variés.
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.preprocessor.data_cleaner import GeophysicalDataCleaner


class TestDataCleanerCleanDeviceData(unittest.TestCase):
    """Tests pour la méthode _clean_device_data de GeophysicalDataCleaner"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        # Créer un répertoire temporaire pour les tests
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Créer la structure des dossiers de test
        self.test_raw_dir = self.test_dir / "raw"
        self.test_processed_dir = self.test_dir / "processed"
        self.test_raw_dir.mkdir(parents=True, exist_ok=True)
        self.test_processed_dir.mkdir(exist_ok=True)
        
        # Copier les fichiers de test depuis fixtures
        fixtures_dir = Path(__file__).parent.parent.parent / "fixtures" / "raw"
        self.pd_test_file = fixtures_dir / "PD.csv"
        self.s_test_file = fixtures_dir / "S.csv"
        
        # Copier les fichiers vers le dossier de test
        if self.pd_test_file.exists():
            shutil.copy2(self.pd_test_file, self.test_raw_dir / "PD.csv")
        if self.s_test_file.exists():
            shutil.copy2(self.s_test_file, self.test_raw_dir / "S.csv")
        
        # Créer une instance du cleaner avec des chemins de test
        with patch('src.preprocessor.data_cleaner.CONFIG') as mock_config:
            mock_config.paths.raw_data_dir = str(self.test_raw_dir)
            mock_config.paths.processed_data_dir = str(self.test_processed_dir)
            mock_config.geophysical_data.coordinate_systems = {
                'wgs84': "EPSG:4326",
                'utm_proj': "EPSG:32630"
            }
            self.cleaner = GeophysicalDataCleaner()
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Supprimer le répertoire temporaire
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_clean_device_data_pd_csv(self):
        """Test de nettoyage des données PD.csv (Pole-Dipole)"""
        device_name = "pole_dipole"
        raw_file = self.test_raw_dir / "PD.csv"
        
        # Vérifier que le fichier de test existe
        self.assertTrue(raw_file.exists(), "Fichier PD.csv manquant")
        
        # Appeler la méthode privée via reflection
        clean_path, report = self.cleaner._clean_device_data(device_name, raw_file)
        
        # Vérifications
        self.assertIsInstance(clean_path, Path)
        self.assertIsInstance(report, dict)
        
        # Vérifier que le fichier nettoyé a été créé
        self.assertTrue(clean_path.exists(), "Fichier nettoyé non créé")
        self.assertEqual(clean_path.name, f"{device_name}_cleaned.csv")
        
        # Vérifier le rapport
        self.assertIn("device", report)
        self.assertEqual(report["device"], device_name)
        self.assertIn("original_count", report)
        self.assertIn("cleaned_count", report)
        self.assertIn("removed_count", report)
        self.assertIn("clean_path", report)
        self.assertIn("coverage_area", report)
        self.assertIn("value_ranges", report)
        
        # Vérifier que les données ont été nettoyées
        cleaned_df = pd.read_csv(clean_path)
        self.assertGreater(len(cleaned_df), 0, "Données nettoyées vides")
        
        print(f"✅ PD.csv nettoyé: {len(cleaned_df)}/{report['original_count']} enregistrements conservés")
    
    def test_clean_device_data_s_csv(self):
        """Test de nettoyage des données S.csv (Schlumberger)"""
        device_name = "schlumberger"
        raw_file = self.test_raw_dir / "S.csv"
        
        # Vérifier que le fichier de test existe
        self.assertTrue(raw_file.exists(), "Fichier S.csv manquant")
        
        # Appeler la méthode privée via reflection
        clean_path, report = self.cleaner._clean_device_data(device_name, raw_file)
        
        # Vérifications
        self.assertIsInstance(clean_path, Path)
        self.assertIsInstance(report, dict)
        
        # Vérifier que le fichier nettoyé a été créé
        self.assertTrue(clean_path.exists(), "Fichier nettoyé non créé")
        self.assertEqual(clean_path.name, f"{device_name}_cleaned.csv")
        
        # Vérifier le rapport
        self.assertEqual(report["device"], device_name)
        self.assertGreater(report["original_count"], 0)
        self.assertGreater(report["cleaned_count"], 0)
        self.assertGreaterEqual(report["removed_count"], 0)
        
        # Vérifier que les données ont été nettoyées
        cleaned_df = pd.read_csv(clean_path)
        self.assertGreater(len(cleaned_df), 0, "Données nettoyées vides")
        
        print(f"✅ S.csv nettoyé: {len(cleaned_df)}/{report['original_count']} enregistrements conservés")
    
    def test_clean_device_data_skip_existing(self):
        """Test que le nettoyage est ignoré si le fichier existe déjà"""
        device_name = "pole_dipole"
        raw_file = self.test_raw_dir / "PD.csv"
        
        # Créer un fichier nettoyé factice
        fake_clean_file = self.test_processed_dir / f"{device_name}_cleaned.csv"
        fake_clean_file.write_text("fake_data")
        
        # Appeler la méthode
        clean_path, report = self.cleaner._clean_device_data(device_name, raw_file)
        
        # Vérifier que le fichier existant est retourné
        self.assertEqual(clean_path, fake_clean_file)
        self.assertEqual(report, {})  # Rapport vide car pas de nettoyage
        
        print("✅ Nettoyage ignoré pour fichier existant")
    
    def test_clean_device_data_invalid_csv(self):
        """Test avec un fichier CSV invalide"""
        device_name = "invalid_device"
        
        # Créer un fichier CSV invalide
        invalid_csv = self.test_raw_dir / "invalid.csv"
        invalid_csv.write_text("Ce n'est pas un CSV valide\nPas de séparateurs\n")
        
        # Appeler la méthode et vérifier qu'elle lève une exception
        with self.assertRaises(ValueError) as context:
            self.cleaner._clean_device_data(device_name, invalid_csv)
        
        # Vérifier le message d'erreur
        self.assertIn("n'est pas un CSV valide", str(context.exception))
        
        print("✅ Exception levée pour CSV invalide")
    
    def test_clean_device_data_missing_file(self):
        """Test avec un fichier manquant"""
        device_name = "missing_device"
        missing_file = self.test_raw_dir / "missing.csv"
        
        # Appeler la méthode et vérifier qu'elle lève une exception
        with self.assertRaises(ValueError) as context:
            self.cleaner._clean_device_data(device_name, missing_file)
        
        # Vérifier le message d'erreur
        self.assertIn("n'est pas un CSV valide", str(context.exception))
        
        print("✅ Exception levée pour fichier manquant")
    
    def test_clean_device_data_data_quality_improvement(self):
        """Test que la qualité des données s'améliore après nettoyage"""
        device_name = "pole_dipole"
        raw_file = self.test_raw_dir / "PD.csv"
        
        # Charger les données brutes
        raw_df = pd.read_csv(raw_file, sep=';')
        original_count = len(raw_df)
        
        # Nettoyer les données
        clean_path, report = self.cleaner._clean_device_data(device_name, raw_file)
        cleaned_df = pd.read_csv(clean_path)
        
        # Vérifier que le nettoyage a amélioré la qualité
        self.assertLessEqual(len(cleaned_df), original_count, "Le nettoyage devrait supprimer des données de mauvaise qualité")
        
        # Vérifier que les coordonnées sont numériques
        if 'x' in cleaned_df.columns and 'y' in cleaned_df.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['x']), "Coordonnées x non numériques après nettoyage")
            self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['y']), "Coordonnées y non numériques après nettoyage")
        
        # Vérifier que la résistivité est positive
        if 'Rho(ohm.m)' in cleaned_df.columns:
            self.assertTrue(all(cleaned_df['Rho(ohm.m)'] > 0), "Résistivité non positive après nettoyage")
        
        # Vérifier que la chargeabilité est non négative
        if 'M (mV/V)' in cleaned_df.columns:
            self.assertTrue(all(cleaned_df['M (mV/V)'] >= 0), "Chargeabilité négative après nettoyage")
        
        print(f"✅ Qualité des données améliorée: {len(cleaned_df)}/{original_count} enregistrements conservés")
    
    def test_clean_device_data_coordinate_transformation(self):
        """Test de la transformation des coordonnées"""
        device_name = "schlumberger"
        raw_file = self.test_raw_dir / "S.csv"
        
        # Charger les données brutes
        raw_df = pd.read_csv(raw_file, sep=';')
        
        # Vérifier que les coordonnées WGS84 sont présentes
        if 'LAT' in raw_df.columns and 'LON' in raw_df.columns:
            # Nettoyer les données
            clean_path, report = self.cleaner._clean_device_data(device_name, raw_file)
            cleaned_df = pd.read_csv(clean_path)
            
            # Vérifier que les coordonnées ont été transformées
            if 'x' in cleaned_df.columns and 'y' in cleaned_df.columns:
                # Les coordonnées UTM devraient être dans des plages raisonnables
                self.assertTrue(all(cleaned_df['x'] > 500000), "Coordonnées X UTM hors de la plage attendue")
                self.assertTrue(all(cleaned_df['y'] > 5000000), "Coordonnées Y UTM hors de la plage attendue")
                
                print("✅ Transformation des coordonnées WGS84 → UTM réussie")
            else:
                print("⚠️ Coordonnées UTM non trouvées après transformation")
    
    def test_clean_device_data_outlier_removal(self):
        """Test de la suppression des valeurs aberrantes"""
        device_name = "pole_dipole"
        raw_file = self.test_raw_dir / "PD.csv"
        
        # Charger les données brutes
        raw_df = pd.read_csv(raw_file, sep=';')
        
        # Nettoyer les données
        clean_path, report = self.cleaner._clean_device_data(device_name, raw_file)
        cleaned_df = pd.read_csv(clean_path)
        
        # Vérifier que des valeurs aberrantes ont été supprimées
        if 'Rho(ohm.m)' in cleaned_df.columns:
            # Calculer les quartiles des données nettoyées
            Q1 = cleaned_df['Rho(ohm.m)'].quantile(0.25)
            Q3 = cleaned_df['Rho(ohm.m)'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Vérifier qu'il n'y a plus de valeurs aberrantes extrêmes
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = cleaned_df[(cleaned_df['Rho(ohm.m)'] < lower_bound) | 
                                (cleaned_df['Rho(ohm.m)'] > upper_bound)]
            
            # Vérifier que le nombre de valeurs aberrantes a diminué (pas forcément à 0)
            original_outliers = raw_df[(raw_df['Rho(ohm.m)'] < lower_bound) | 
                                     (raw_df['Rho(ohm.m)'] > upper_bound)]
            
            self.assertLessEqual(len(outliers), len(original_outliers), 
                               "Le nombre de valeurs aberrantes devrait diminuer ou rester stable")
            
            print(f"✅ Valeurs aberrantes réduites: {len(outliers)}/{len(original_outliers)} restantes")
    
    def test_clean_device_data_report_accuracy(self):
        """Test de l'exactitude du rapport de nettoyage"""
        device_name = "pole_dipole"
        raw_file = self.test_raw_dir / "PD.csv"
        
        # Charger les données brutes
        raw_df = pd.read_csv(raw_file, sep=';')
        original_count = len(raw_df)
        
        # Nettoyer les données
        clean_path, report = self.cleaner._clean_device_data(device_name, raw_file)
        cleaned_df = pd.read_csv(clean_path)
        
        # Vérifier l'exactitude du rapport
        self.assertEqual(report["original_count"], original_count)
        self.assertEqual(report["cleaned_count"], len(cleaned_df))
        self.assertEqual(report["removed_count"], original_count - len(cleaned_df))
        self.assertEqual(report["clean_path"], str(clean_path))
        
        # Vérifier que le rapport est cohérent
        self.assertEqual(report["original_count"] - report["removed_count"], report["cleaned_count"])
        
        print("✅ Rapport de nettoyage exact et cohérent")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(
        verbosity=2,
        testLoader=unittest.TestLoader(),
        testRunner=unittest.TextTestRunner(stream=sys.stdout)
    )
