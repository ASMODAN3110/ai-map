#!/usr/bin/env python3
"""
Test unitaire pour la méthode _validate_columns de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _validate_columns
qui valide que les colonnes requises sont présentes dans les vrais fichiers PD.csv et S.csv.
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


class TestDataCleanerValidateColumns(unittest.TestCase):
    """Tests pour la méthode _validate_columns de GeophysicalDataCleaner"""
    
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
            # Mock des colonnes requises
            mock_config.geophysical_data.required_columns = [
                'x', 'y', 'z', 'Rho(ohm.m)', 'M (mV/V)', 'LAT', 'LON', 'El-array'
            ]
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
    
    def test_validate_columns_pd_csv(self):
        """Test de validation des colonnes avec PD.csv"""
        # Charger PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        
        df_pd = pd.read_csv(pd_file, sep=';')
        self.assertIsInstance(df_pd, pd.DataFrame)
        self.assertGreater(len(df_pd), 0, "PD.csv ne devrait pas être vide")
        
        # Appeler la méthode de validation
        validated_df = self.cleaner._validate_columns(df_pd, "pole_dipole")
        
        # Vérifications
        self.assertIsInstance(validated_df, pd.DataFrame)
        self.assertEqual(len(validated_df), len(df_pd), "Le nombre de lignes ne devrait pas changer")
        self.assertEqual(len(validated_df.columns), len(df_pd.columns), "Le nombre de colonnes ne devrait pas changer")
        
        # Vérifier que les colonnes importantes sont présentes
        expected_cols_pd = ['x', 'y', 'z', 'Rho(ohm.m)', 'M (mV/V)']
        for col in expected_cols_pd:
            self.assertIn(col, validated_df.columns, f"Colonne {col} manquante dans PD.csv")
        
        print(f"✅ Colonnes de PD.csv validées: {len(validated_df.columns)} colonnes, {len(validated_df)} lignes")
    
    def test_validate_columns_s_csv(self):
        """Test de validation des colonnes avec S.csv"""
        # Charger S.csv
        s_file = self.raw_data_dir / "S.csv"
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        df_s = pd.read_csv(s_file, sep=';')
        self.assertIsInstance(df_s, pd.DataFrame)
        self.assertGreater(len(df_s), 0, "S.csv ne devrait pas être vide")
        
        # Appeler la méthode de validation
        validated_df = self.cleaner._validate_columns(df_s, "schlumberger")
        
        # Vérifications
        self.assertIsInstance(validated_df, pd.DataFrame)
        self.assertEqual(len(validated_df), len(df_s), "Le nombre de lignes ne devrait pas changer")
        self.assertEqual(len(validated_df.columns), len(df_s.columns), "Le nombre de colonnes ne devrait pas changer")
        
        # Vérifier que les colonnes importantes sont présentes
        expected_cols_s = ['El-array', 'LAT', 'LON', 'Rho (Ohm.m)', 'M (mV/V)']
        for col in expected_cols_s:
            self.assertIn(col, validated_df.columns, f"Colonne {col} manquante dans S.csv")
        
        print(f"✅ Colonnes de S.csv validées: {len(validated_df.columns)} colonnes, {len(validated_df)} lignes")
    
    def test_validate_columns_both_files(self):
        """Test de validation des colonnes des deux fichiers CSV ensemble"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        validated_pd = self.cleaner._validate_columns(df_pd, "pole_dipole")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        validated_s = self.cleaner._validate_columns(df_s, "schlumberger")
        
        # Vérifier que les deux validations ont réussi
        self.assertIsInstance(validated_pd, pd.DataFrame)
        self.assertIsInstance(validated_s, pd.DataFrame)
        
        # Vérifier que les données sont préservées
        self.assertEqual(len(validated_pd), len(df_pd))
        self.assertEqual(len(validated_s), len(df_s))
        
        print(f"✅ Colonnes des deux fichiers validées: PD.csv ({len(validated_pd.columns)} colonnes), S.csv ({len(validated_s.columns)} colonnes)")
    
    def test_validate_columns_coordinate_columns(self):
        """Test de validation des colonnes de coordonnées"""
        # Tester PD.csv (coordonnées x, y, z)
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        validated_pd = self.cleaner._validate_columns(df_pd, "pole_dipole")
        
        # Vérifier les colonnes de coordonnées
        coord_cols_pd = ['x', 'y', 'z']
        for col in coord_cols_pd:
            self.assertIn(col, validated_pd.columns, f"Colonne de coordonnée {col} manquante dans PD.csv")
            # Vérifier que les coordonnées sont numériques
            self.assertTrue(pd.api.types.is_numeric_dtype(validated_pd[col]), f"Colonne {col} devrait être numérique")
        
        # Tester S.csv (coordonnées LAT, LON)
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        validated_s = self.cleaner._validate_columns(df_s, "schlumberger")
        
        # Vérifier les colonnes de coordonnées
        coord_cols_s = ['LAT', 'LON']
        for col in coord_cols_s:
            self.assertIn(col, validated_s.columns, f"Colonne de coordonnée {col} manquante dans S.csv")
            # Vérifier que les coordonnées sont numériques
            self.assertTrue(pd.api.types.is_numeric_dtype(validated_s[col]), f"Colonne {col} devrait être numérique")
        
        print("✅ Colonnes de coordonnées validées pour les deux fichiers")
    
    def test_validate_columns_measurement_columns(self):
        """Test de validation des colonnes de mesures géophysiques"""
        # Tester PD.csv (résistivité et chargeabilité)
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        validated_pd = self.cleaner._validate_columns(df_pd, "pole_dipole")
        
        # Vérifier les colonnes de mesures
        measurement_cols_pd = ['Rho(ohm.m)', 'M (mV/V)']
        for col in measurement_cols_pd:
            self.assertIn(col, validated_pd.columns, f"Colonne de mesure {col} manquante dans PD.csv")
            # Vérifier que les mesures sont numériques
            self.assertTrue(pd.api.types.is_numeric_dtype(validated_pd[col]), f"Colonne {col} devrait être numérique")
            # Vérifier que les valeurs sont positives (pour la résistivité et chargeabilité)
            if col == 'Rho(ohm.m)':
                self.assertTrue((validated_pd[col] > 0).all(), f"Toutes les valeurs de {col} devraient être positives")
        
        # Tester S.csv (résistivité et chargeabilité)
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        validated_s = self.cleaner._validate_columns(df_s, "schlumberger")
        
        # Vérifier les colonnes de mesures
        measurement_cols_s = ['Rho (Ohm.m)', 'M (mV/V)']
        for col in measurement_cols_s:
            self.assertIn(col, validated_s.columns, f"Colonne de mesure {col} manquante dans S.csv")
            # Vérifier que les mesures sont numériques
            self.assertTrue(pd.api.types.is_numeric_dtype(validated_s[col]), f"Colonne {col} devrait être numérique")
            # Vérifier que les valeurs sont positives
            if col == 'Rho (Ohm.m)':
                self.assertTrue((validated_s[col] > 0).all(), f"Toutes les valeurs de {col} devraient être positives")
        
        print("✅ Colonnes de mesures géophysiques validées pour les deux fichiers")
    
    def test_validate_columns_data_integrity(self):
        """Test de l'intégrité des données après validation"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd_original = pd.read_csv(pd_file, sep=';')
        validated_pd = self.cleaner._validate_columns(df_pd_original, "pole_dipole")
        
        # Vérifier que les données sont préservées
        for col in df_pd_original.columns:
            if col in validated_pd.columns:
                pd.testing.assert_series_equal(
                    df_pd_original[col], 
                    validated_pd[col], 
                    check_names=False,
                    check_dtype=False
                )
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s_original = pd.read_csv(s_file, sep=';')
        validated_s = self.cleaner._validate_columns(df_s_original, "schlumberger")
        
        # Vérifier que les données sont préservées
        for col in df_s_original.columns:
            if col in validated_s.columns:
                pd.testing.assert_series_equal(
                    df_s_original[col], 
                    validated_s[col], 
                    check_names=False,
                    check_dtype=False
                )
        
        print("✅ Intégrité des données préservée après validation des colonnes")
    
    def test_validate_columns_missing_columns_handling(self):
        """Test de gestion des colonnes manquantes"""
        # Créer un DataFrame avec des colonnes manquantes
        incomplete_df = pd.DataFrame({
            'x': [100, 200, 300],
            'y': [400, 500, 600],
            # Colonnes manquantes: 'z', 'Rho(ohm.m)', 'M (mV/V)'
        })
        
        # Appeler la méthode de validation
        validated_df = self.cleaner._validate_columns(incomplete_df, "test_device")
        
        # Vérifier que le DataFrame est retourné même avec des colonnes manquantes
        self.assertIsInstance(validated_df, pd.DataFrame)
        self.assertEqual(len(validated_df), len(incomplete_df))
        
        # Vérifier que les colonnes existantes sont préservées
        self.assertIn('x', validated_df.columns)
        self.assertIn('y', validated_df.columns)
        
        print("✅ Gestion des colonnes manquantes validée")
    
    def test_validate_columns_empty_dataframe(self):
        """Test de validation avec un DataFrame vide"""
        # Créer un DataFrame vide
        empty_df = pd.DataFrame()
        
        # Appeler la méthode de validation
        validated_df = self.cleaner._validate_columns(empty_df, "test_device")
        
        # Vérifier que le DataFrame vide est retourné
        self.assertIsInstance(validated_df, pd.DataFrame)
        self.assertEqual(len(validated_df), 0)
        self.assertEqual(len(validated_df.columns), 0)
        
        print("✅ Validation avec DataFrame vide réussie")
    
    def test_validate_columns_single_column_dataframe(self):
        """Test de validation avec un DataFrame à une seule colonne"""
        # Créer un DataFrame avec une seule colonne
        single_col_df = pd.DataFrame({
            'x': [100, 200, 300]
        })
        
        # Appeler la méthode de validation
        validated_df = self.cleaner._validate_columns(single_col_df, "test_device")
        
        # Vérifier que le DataFrame est retourné
        self.assertIsInstance(validated_df, pd.DataFrame)
        self.assertEqual(len(validated_df), 3)
        self.assertEqual(len(validated_df.columns), 1)
        self.assertIn('x', validated_df.columns)
        
        print("✅ Validation avec DataFrame à une seule colonne réussie")
    
    def test_validate_columns_performance(self):
        """Test de performance de la validation des colonnes"""
        import time
        
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        start_time = time.time()
        validated_pd = self.cleaner._validate_columns(df_pd, "pole_dipole")
        pd_time = time.time() - start_time
        
        self.assertIsInstance(validated_pd, pd.DataFrame)
        self.assertLess(pd_time, 1.0, "Validation des colonnes de PD.csv devrait être rapide (< 1 seconde)")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        
        start_time = time.time()
        validated_s = self.cleaner._validate_columns(df_s, "schlumberger")
        s_time = time.time() - start_time
        
        self.assertIsInstance(validated_s, pd.DataFrame)
        self.assertLess(s_time, 1.0, "Validation des colonnes de S.csv devrait être rapide (< 1 seconde)")
        
        print(f"✅ Performance validée: PD.csv ({pd_time:.3f}s), S.csv ({s_time:.3f}s)")
    
    def test_validate_columns_integration_with_load_device_data(self):
        """Test d'intégration avec _load_device_data"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = self.cleaner._load_device_data(pd_file, "pole_dipole")
        validated_pd = self.cleaner._validate_columns(df_pd, "pole_dipole")
        
        self.assertIsInstance(validated_pd, pd.DataFrame)
        self.assertEqual(len(validated_pd), len(df_pd))
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = self.cleaner._load_device_data(s_file, "schlumberger")
        validated_s = self.cleaner._validate_columns(df_s, "schlumberger")
        
        self.assertIsInstance(validated_s, pd.DataFrame)
        self.assertEqual(len(validated_s), len(df_s))
        
        print("✅ Intégration avec _load_device_data réussie")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(
        verbosity=2,
        testLoader=unittest.TestLoader(),
        testRunner=unittest.TextTestRunner(stream=sys.stdout)
    )
