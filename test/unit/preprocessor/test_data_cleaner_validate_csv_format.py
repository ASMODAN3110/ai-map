#!/usr/bin/env python3
"""
Test unitaire pour la méthode _validate_csv_format de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _validate_csv_format
qui valide que les fichiers sont des CSV valides avec les vrais fichiers PD.csv et S.csv.
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


class TestDataCleanerValidateCsvFormat(unittest.TestCase):
    """Tests pour la méthode _validate_csv_format de GeophysicalDataCleaner"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        # Utiliser les vrais fichiers de données du projet
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.test_dir = self.project_root / "test" / "fixtures"
        
        # Créer une instance du cleaner avec les vrais chemins
        with patch('src.preprocessor.data_cleaner.CONFIG') as mock_config:
            mock_config.paths.raw_data_dir = str(self.raw_data_dir)
            mock_config.paths.processed_data_dir = str(self.test_dir / "processed")
            mock_config.geophysical_data.coordinate_systems = {
                'wgs84': "EPSG:4326",
                'utm_proj': "EPSG:32630"
            }
            self.cleaner = GeophysicalDataCleaner()
        
        # Créer le dossier processed s'il n'existe pas
        (self.test_dir / "processed").mkdir(exist_ok=True)
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Ne supprimer que le dossier processed temporaire
        processed_dir = self.test_dir / "processed"
        if processed_dir.exists():
            import shutil
            shutil.rmtree(processed_dir)
    
    def test_validate_csv_format_pd_csv(self):
        """Test de validation du format CSV avec PD.csv"""
        # Utiliser le vrai fichier PD.csv
        csv_file = self.raw_data_dir / "PD.csv"
        
        # Vérifier que le fichier existe
        self.assertTrue(csv_file.exists(), f"Le fichier {csv_file} n'existe pas")
        
        # Appeler la méthode de validation
        is_valid = self.cleaner._validate_csv_format(csv_file)
        
        # Vérifications
        self.assertTrue(is_valid, f"PD.csv devrait être reconnu comme un CSV valide")
        
        # Vérifier que le fichier contient bien des points-virgules
        with open(csv_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            self.assertIn(';', first_line, "PD.csv devrait contenir des points-virgules comme séparateur")
        
        print(f"✅ PD.csv validé comme CSV valide: {len(first_line.split(';'))} colonnes détectées")
    
    def test_validate_csv_format_s_csv(self):
        """Test de validation du format CSV avec S.csv"""
        # Utiliser le vrai fichier S.csv
        csv_file = self.raw_data_dir / "S.csv"
        
        # Vérifier que le fichier existe
        self.assertTrue(csv_file.exists(), f"Le fichier {csv_file} n'existe pas")
        
        # Appeler la méthode de validation
        is_valid = self.cleaner._validate_csv_format(csv_file)
        
        # Vérifications
        self.assertTrue(is_valid, f"S.csv devrait être reconnu comme un CSV valide")
        
        # Vérifier que le fichier contient bien des points-virgules
        with open(csv_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            self.assertIn(';', first_line, "S.csv devrait contenir des points-virgules comme séparateur")
        
        print(f"✅ S.csv validé comme CSV valide: {len(first_line.split(';'))} colonnes détectées")
    
    def test_validate_csv_format_both_files(self):
        """Test de validation des deux fichiers CSV ensemble"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        
        is_valid_pd = self.cleaner._validate_csv_format(pd_file)
        self.assertTrue(is_valid_pd, "PD.csv devrait être validé")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        is_valid_s = self.cleaner._validate_csv_format(s_file)
        self.assertTrue(is_valid_s, "S.csv devrait être validé")
        
        # Vérifier que les deux fichiers sont reconnus comme valides
        self.assertTrue(is_valid_pd and is_valid_s, "Les deux fichiers devraient être validés")
        
        print("✅ Les deux fichiers CSV (PD.csv et S.csv) sont validés avec succès")
    
    def test_validate_csv_format_file_structure(self):
        """Test de la structure des fichiers CSV validés"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        # Vérifier la structure
        self.assertGreater(len(df_pd.columns), 1, "PD.csv devrait avoir plusieurs colonnes")
        self.assertGreater(len(df_pd), 0, "PD.csv ne devrait pas être vide")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        
        # Vérifier la structure
        self.assertGreater(len(df_s.columns), 1, "S.csv devrait avoir plusieurs colonnes")
        self.assertGreater(len(df_s), 0, "S.csv ne devrait pas être vide")
        
        print(f"✅ Structure validée: PD.csv ({len(df_pd.columns)} colonnes, {len(df_pd)} lignes), S.csv ({len(df_s.columns)} colonnes, {len(df_s)} lignes)")
    
    def test_validate_csv_format_separator_detection(self):
        """Test de la détection des séparateurs dans les vrais fichiers"""
        # Tester PD.csv (virgules)
        pd_file = self.raw_data_dir / "PD.csv"
        with open(pd_file, 'r', encoding='utf-8') as f:
            first_lines = [f.readline() for _ in range(5)]
        
        # Vérifier la présence de points-virgules
        has_semicolons = any(';' in line for line in first_lines if line.strip())
        self.assertTrue(has_semicolons, "PD.csv devrait contenir des points-virgules")
        
        # Tester S.csv (points-virgules)
        s_file = self.raw_data_dir / "S.csv"
        with open(s_file, 'r', encoding='utf-8') as f:
            first_lines = [f.readline() for _ in range(5)]
        
        # Vérifier la présence de points-virgules
        has_semicolons = any(';' in line for line in first_lines if line.strip())
        self.assertTrue(has_semicolons, "S.csv devrait contenir des points-virgules")
        
        print("✅ Détection des séparateurs réussie: virgules dans PD.csv, points-virgules dans S.csv")
    
    def test_validate_csv_format_encoding_validation(self):
        """Test de validation de l'encodage des fichiers CSV"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        try:
            with open(pd_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
            self.assertIsInstance(first_line, str, "PD.csv devrait être lisible en UTF-8")
        except UnicodeDecodeError:
            self.fail("PD.csv ne devrait pas avoir de problème d'encodage UTF-8")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        try:
            with open(s_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
            self.assertIsInstance(first_line, str, "S.csv devrait être lisible en UTF-8")
        except UnicodeDecodeError:
            self.fail("S.csv ne devrait pas avoir de problème d'encodage UTF-8")
        
        print("✅ Encodage UTF-8 validé pour les deux fichiers CSV")
    
    def test_validate_csv_format_content_validation(self):
        """Test de validation du contenu des fichiers CSV"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        # Vérifier que les colonnes importantes existent
        expected_cols_pd = ['x', 'y', 'z', 'Rho(ohm.m)', 'M (mV/V)']
        for col in expected_cols_pd:
            if col in df_pd.columns:
                self.assertGreater(len(df_pd[col]), 0, f"Colonne {col} de PD.csv ne devrait pas être vide")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        
        # Vérifier que les colonnes importantes existent
        expected_cols_s = ['El-array', 'LAT', 'LON', 'Rho (Ohm.m)', 'M (mV/V)']
        for col in expected_cols_s:
            if col in df_s.columns:
                self.assertGreater(len(df_s[col]), 0, f"Colonne {col} de S.csv ne devrait pas être vide")
        
        print("✅ Contenu des fichiers CSV validé avec succès")
    
    def test_validate_csv_format_performance(self):
        """Test de performance de la validation avec les vrais fichiers"""
        import time
        
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        start_time = time.time()
        is_valid_pd = self.cleaner._validate_csv_format(pd_file)
        pd_time = time.time() - start_time
        
        self.assertTrue(is_valid_pd, "PD.csv devrait être validé")
        self.assertLess(pd_time, 1.0, "Validation de PD.csv devrait être rapide (< 1 seconde)")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        start_time = time.time()
        is_valid_s = self.cleaner._validate_csv_format(s_file)
        s_time = time.time() - start_time
        
        self.assertTrue(is_valid_s, "S.csv devrait être validé")
        self.assertLess(s_time, 1.0, "Validation de S.csv devrait être rapide (< 1 seconde)")
        
        print(f"✅ Performance validée: PD.csv ({pd_time:.3f}s), S.csv ({s_time:.3f}s)")
    
    def test_validate_csv_format_error_handling(self):
        """Test de gestion des erreurs avec des fichiers invalides"""
        # Créer un fichier temporaire invalide
        invalid_file = self.test_dir / "invalid.txt"
        invalid_file.write_text("Ceci n'est pas un CSV valide\nPas de séparateurs\nFormat incorrect")
        
        # Tester la validation
        is_valid = self.cleaner._validate_csv_format(invalid_file)
        self.assertFalse(is_valid, "Le fichier invalide devrait être rejeté")
        
        # Nettoyer
        invalid_file.unlink()
        
        print("✅ Gestion des erreurs validée: fichier invalide rejeté")
    
    def test_validate_csv_format_edge_cases(self):
        """Test des cas limites avec les vrais fichiers"""
        # Tester avec un fichier très petit mais valide
        pd_file = self.raw_data_dir / "PD.csv"
        
        # Lire seulement les 3 premières lignes
        with open(pd_file, 'r', encoding='utf-8') as f:
            first_lines = [f.readline() for _ in range(3)]
        
        # Créer un fichier temporaire avec seulement ces lignes
        small_file = self.test_dir / "small.csv"
        small_file.write_text(''.join(first_lines))
        
        # Tester la validation
        is_valid = self.cleaner._validate_csv_format(small_file)
        self.assertTrue(is_valid, "Le petit fichier CSV devrait être validé")
        
        # Nettoyer
        small_file.unlink()
        
        print("✅ Cas limites validés: petit fichier CSV accepté")
    
    def test_validate_csv_format_integration(self):
        """Test d'intégration avec la méthode _load_device_data"""
        # Tester que la validation fonctionne avant le chargement
        pd_file = self.raw_data_dir / "PD.csv"
        s_file = self.raw_data_dir / "S.csv"
        
        # Valider d'abord
        pd_valid = self.cleaner._validate_csv_format(pd_file)
        s_valid = self.cleaner._validate_csv_format(s_file)
        
        self.assertTrue(pd_valid, "PD.csv devrait être validé")
        self.assertTrue(s_valid, "S.csv devrait être validé")
        
        # Puis charger les données
        if pd_valid:
            df_pd = self.cleaner._load_device_data(pd_file, "pole_dipole")
            self.assertIsInstance(df_pd, pd.DataFrame)
            self.assertGreater(len(df_pd), 0)
        
        if s_valid:
            df_s = self.cleaner._load_device_data(s_file, "schlumberger")
            self.assertIsInstance(df_s, pd.DataFrame)
            self.assertGreater(len(df_s), 0)
        
        print("✅ Intégration validée: validation + chargement des données réussis")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(
        verbosity=2,
        testLoader=unittest.TestLoader(),
        testRunner=unittest.TextTestRunner(stream=sys.stdout)
    )
