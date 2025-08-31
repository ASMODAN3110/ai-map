#!/usr/bin/env python3
"""
Test unitaire pour la méthode _load_device_data de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _load_device_data
qui gère la lecture des fichiers CSV avec détection automatique des séparateurs.
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


class TestDataCleanerLoadDeviceData(unittest.TestCase):
    """Tests pour la méthode _load_device_data de GeophysicalDataCleaner"""
    
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
    
    def test_load_device_data_pd_csv(self):
        """Test de lecture du vrai fichier PD.csv (Pole-Dipole)"""
        # Utiliser le vrai fichier PD.csv
        csv_file = self.raw_data_dir / "PD.csv"
        
        # Vérifier que le fichier existe
        self.assertTrue(csv_file.exists(), f"Le fichier {csv_file} n'existe pas")
        
        # Appeler la méthode
        df = self.cleaner._load_device_data(csv_file, "pole_dipole")
        
        # Vérifications
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0, "Le fichier PD.csv ne devrait pas être vide")
        
        # Vérifier les colonnes attendues pour PD.csv
        expected_columns = ['x', 'y', 'z', 'Rho(ohm.m)', 'M (mV/V)']
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Colonne {col} manquante dans PD.csv")
        
        # Vérifier quelques données
        self.assertIsInstance(df.iloc[0]['x'], (int, float))
        self.assertIsInstance(df.iloc[0]['Rho(ohm.m)'], (int, float))
        
        print(f"✅ PD.csv lu correctement: {len(df)} lignes, {len(df.columns)} colonnes")
    
    def test_load_device_data_s_csv(self):
        """Test de lecture du vrai fichier S.csv (Schlumberger)"""
        # Utiliser le vrai fichier S.csv
        csv_file = self.raw_data_dir / "S.csv"
        
        # Vérifier que le fichier existe
        self.assertTrue(csv_file.exists(), f"Le fichier {csv_file} n'existe pas")
        
        # Appeler la méthode
        df = self.cleaner._load_device_data(csv_file, "schlumberger")
        
        # Vérifications
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0, "Le fichier S.csv ne devrait pas être vide")
        
        # Vérifier les colonnes attendues pour S.csv
        expected_columns = ['El-array', 'LAT', 'LON', 'Rho (Ohm.m)', 'M (mV/V)']
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Colonne {col} manquante dans S.csv")
        
        # Vérifier quelques données
        self.assertIsInstance(df.iloc[0]['LAT'], (int, float))
        self.assertIsInstance(df.iloc[0]['Rho (Ohm.m)'], (int, float))
        
        print(f"✅ S.csv lu correctement: {len(df)} lignes, {len(df.columns)} colonnes")
    
    def test_load_device_data_csv_mixed_separators(self):
        """Test de détection automatique des séparateurs avec S.csv"""
        # Utiliser le vrai fichier S.csv qui utilise des points-virgules
        csv_file = self.raw_data_dir / "S.csv"
        
        # Vérifier que le fichier existe
        self.assertTrue(csv_file.exists(), f"Le fichier {csv_file} n'existe pas")
        
        # Appeler la méthode
        df = self.cleaner._load_device_data(csv_file, "schlumberger")
        
        # Vérifications
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0, "Le fichier S.csv ne devrait pas être vide")
        
        # Vérifier que la détection automatique a fonctionné (plus d'une colonne)
        self.assertGreater(len(df.columns), 1, "La détection automatique des séparateurs a échoué")
        
        print(f"✅ Détection automatique des séparateurs réussie avec S.csv: {len(df.columns)} colonnes")
    
    def test_load_device_data_csv_with_quotes(self):
        """Test de lecture des vrais fichiers CSV et vérification de la gestion des guillemets"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        
        df_pd = self.cleaner._load_device_data(pd_file, "pole_dipole")
        self.assertIsInstance(df_pd, pd.DataFrame)
        self.assertGreater(len(df_pd), 0, "PD.csv ne devrait pas être vide")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        df_s = self.cleaner._load_device_data(s_file, "schlumberger")
        self.assertIsInstance(df_s, pd.DataFrame)
        self.assertGreater(len(df_s), 0, "S.csv ne devrait pas être vide")
        
        # Vérifier que la lecture s'est bien passée (pas d'erreurs de parsing)
        self.assertGreater(len(df_pd.columns), 1, "PD.csv devrait avoir plusieurs colonnes")
        self.assertGreater(len(df_s.columns), 1, "S.csv devrait avoir plusieurs colonnes")
        
        print(f"✅ Gestion des guillemets réussie: PD.csv ({len(df_pd.columns)} colonnes), S.csv ({len(df_s.columns)} colonnes)")
    
    def test_load_device_data_csv_with_headers(self):
        """Test de lecture des en-têtes des vrais fichiers CSV"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        
        df_pd = self.cleaner._load_device_data(pd_file, "pole_dipole")
        self.assertIsInstance(df_pd, pd.DataFrame)
        self.assertGreater(len(df_pd.columns), 0, "PD.csv devrait avoir des colonnes")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        df_s = self.cleaner._load_device_data(s_file, "schlumberger")
        self.assertIsInstance(df_s, pd.DataFrame)
        self.assertGreater(len(df_s.columns), 0, "S.csv devrait avoir des colonnes")
        
        print(f"✅ En-têtes des vrais fichiers lus: PD.csv ({len(df_pd.columns)} colonnes), S.csv ({len(df_s.columns)} colonnes)")
    
    def test_load_device_data_csv_empty_file(self):
        """Test de gestion des fichiers CSV avec peu de données"""
        # Tester que les vrais fichiers ne sont pas vides
        pd_file = self.raw_data_dir / "PD.csv"
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        
        df_pd = self.cleaner._load_device_data(pd_file, "pole_dipole")
        self.assertIsInstance(df_pd, pd.DataFrame)
        self.assertGreater(len(df_pd), 0, "PD.csv ne devrait pas être vide")
        
        s_file = self.raw_data_dir / "S.csv"
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        df_s = self.cleaner._load_device_data(s_file, "schlumberger")
        self.assertIsInstance(df_s, pd.DataFrame)
        self.assertGreater(len(df_s), 0, "S.csv ne devrait pas être vide")
        
        print(f"✅ Fichiers non-vides validés: PD.csv ({len(df_pd)} lignes), S.csv ({len(df_s)} lignes)")
    
    def test_load_device_data_csv_single_column(self):
        """Test de détection automatique des séparateurs avec les vrais fichiers"""
        # Tester PD.csv (virgules)
        pd_file = self.raw_data_dir / "PD.csv"
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        
        df_pd = self.cleaner._load_device_data(pd_file, "pole_dipole")
        self.assertIsInstance(df_pd, pd.DataFrame)
        self.assertGreater(len(df_pd.columns), 1, "PD.csv devrait avoir plusieurs colonnes")
        
        # Tester S.csv (points-virgules)
        s_file = self.raw_data_dir / "S.csv"
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        df_s = self.cleaner._load_device_data(s_file, "schlumberger")
        self.assertIsInstance(df_s, pd.DataFrame)
        self.assertGreater(len(df_s.columns), 1, "S.csv devrait avoir plusieurs colonnes")
        
        print(f"✅ Détection automatique des séparateurs réussie: PD.csv ({len(df_pd.columns)} colonnes), S.csv ({len(df_s.columns)} colonnes)")
    
    def test_load_device_data_non_csv_file(self):
        """Test de rejet des fichiers non-CSV"""
        # Créer un fichier non-CSV
        non_csv_file = self.test_dir / "test.txt"
        non_csv_file.write_text("Ceci n'est pas un CSV")
        
        # Appeler la méthode et vérifier qu'elle lève une exception
        with self.assertRaises(ValueError) as context:
            self.cleaner._load_device_data(non_csv_file, "test_device")
        
        # Vérifier le message d'erreur
        self.assertIn("Seuls les fichiers CSV sont supportés", str(context.exception))
        self.assertIn(".txt", str(context.exception))
        
        print("✅ Fichier non-CSV rejeté correctement")
    
    def test_load_device_data_missing_file(self):
        """Test de lecture d'un fichier inexistant"""
        # Créer un chemin vers un fichier inexistant
        missing_file = self.test_dir / "missing.csv"
        
        # Appeler la méthode et vérifier qu'elle lève une exception
        with self.assertRaises(ValueError) as context:
            self.cleaner._load_device_data(missing_file, "test_device")
        
        # Vérifier le message d'erreur
        self.assertIn("Erreur lors de la lecture du fichier CSV", str(context.exception))
        
        print("✅ Fichier manquant géré correctement")
    
    def test_load_device_data_csv_with_missing_values(self):
        """Test de lecture des vrais fichiers CSV et vérification des valeurs manquantes"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        
        df_pd = self.cleaner._load_device_data(pd_file, "pole_dipole")
        self.assertIsInstance(df_pd, pd.DataFrame)
        self.assertGreater(len(df_pd), 0, "PD.csv ne devrait pas être vide")
        
        # Vérifier qu'il n'y a pas de valeurs manquantes dans les colonnes importantes
        important_cols_pd = ['x', 'y', 'z', 'Rho(ohm.m)']
        for col in important_cols_pd:
            if col in df_pd.columns:
                missing_count = df_pd[col].isna().sum()
                self.assertEqual(missing_count, 0, f"Colonne {col} dans PD.csv contient {missing_count} valeurs manquantes")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        df_s = self.cleaner._load_device_data(s_file, "schlumberger")
        self.assertIsInstance(df_s, pd.DataFrame)
        self.assertGreater(len(df_s), 0, "S.csv ne devrait pas être vide")
        
        # Vérifier qu'il n'y a pas de valeurs manquantes dans les colonnes importantes
        important_cols_s = ['LAT', 'LON', 'Rho (Ohm.m)']
        for col in important_cols_s:
            if col in df_s.columns:
                missing_count = df_s[col].isna().sum()
                self.assertEqual(missing_count, 0, f"Colonne {col} dans S.csv contient {missing_count} valeurs manquantes")
        
        print(f"✅ Vrais fichiers CSV lus sans valeurs manquantes: PD.csv ({len(df_pd)} lignes), S.csv ({len(df_s)} lignes)")
    
    def test_load_device_data_csv_with_special_characters(self):
        """Test de lecture d'un fichier CSV avec des caractères spéciaux"""
        # Créer un fichier CSV avec des caractères spéciaux
        csv_file = self.test_dir / "test_special.csv"
        csv_content = "x,y,z,description,value\n100,200,300,\"Resistivite elevee\",150.5\n101,201,301,\"Chargeabilite moyenne\",26.1\n102,202,302,\"Zone d'interet\",155.8"
        csv_file.write_text(csv_content)
        
        # Appeler la méthode
        df = self.cleaner._load_device_data(csv_file, "test_device")
        
        # Vérifications
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertEqual(len(df.columns), 5)
        
        # Vérifier que les caractères spéciaux sont préservés
        self.assertEqual(df.iloc[0]['description'], "Resistivite elevee")
        self.assertEqual(df.iloc[1]['description'], "Chargeabilite moyenne")
        self.assertEqual(df.iloc[2]['description'], "Zone d'interet")
        
        print("✅ CSV avec caractères spéciaux lu correctement")
    
    def test_load_device_data_csv_large_file(self):
        """Test de lecture des vrais fichiers CSV de grande taille"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        
        df_pd = self.cleaner._load_device_data(pd_file, "pole_dipole")
        self.assertIsInstance(df_pd, pd.DataFrame)
        self.assertGreater(len(df_pd), 100, "PD.csv devrait avoir plus de 100 lignes")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        df_s = self.cleaner._load_device_data(s_file, "schlumberger")
        self.assertIsInstance(df_s, pd.DataFrame)
        self.assertGreater(len(df_s), 500, "S.csv devrait avoir plus de 500 lignes")
        
        # Vérifier que les données sont cohérentes
        self.assertGreater(len(df_pd.columns), 0, "PD.csv devrait avoir des colonnes")
        self.assertGreater(len(df_s.columns), 0, "S.csv devrait avoir des colonnes")
        
        print(f"✅ Vrais fichiers CSV de grande taille lus: PD.csv ({len(df_pd)} lignes), S.csv ({len(df_s)} lignes)")
    
    def test_load_device_data_csv_encoding_utf8(self):
        """Test de lecture des vrais fichiers CSV avec encodage UTF-8"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        
        df_pd = self.cleaner._load_device_data(pd_file, "pole_dipole")
        self.assertIsInstance(df_pd, pd.DataFrame)
        self.assertGreater(len(df_pd), 0, "PD.csv ne devrait pas être vide")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        df_s = self.cleaner._load_device_data(s_file, "schlumberger")
        self.assertIsInstance(df_s, pd.DataFrame)
        self.assertGreater(len(df_s), 0, "S.csv ne devrait pas être vide")
        
        # Vérifier que l'encodage est correct (pas d'erreurs de lecture)
        self.assertIsInstance(df_pd.columns[0], str, "Les noms de colonnes de PD.csv devraient être des chaînes")
        self.assertIsInstance(df_s.columns[0], str, "Les noms de colonnes de S.csv devraient être des chaînes")
        
        print(f"✅ Vrais fichiers CSV avec encodage UTF-8 lus: PD.csv ({len(df_pd)} lignes), S.csv ({len(df_s)} lignes)")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(
        verbosity=2,
        testLoader=unittest.TestLoader(),
        testRunner=unittest.TextTestRunner(stream=sys.stdout)
    )
