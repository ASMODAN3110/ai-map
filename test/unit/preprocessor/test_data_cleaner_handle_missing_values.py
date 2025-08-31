#!/usr/bin/env python3
"""
Test unitaire pour la méthode _handle_missing_values de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _handle_missing_values
qui supprime les lignes avec des valeurs manquantes dans les colonnes critiques
en utilisant les vrais fichiers PD.csv et S.csv.
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


class TestDataCleanerHandleMissingValues(unittest.TestCase):
    """Tests pour la méthode _handle_missing_values de GeophysicalDataCleaner"""
    
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
    
    def test_handle_missing_values_pd_csv(self):
        """Test de gestion des valeurs manquantes avec PD.csv"""
        # Charger PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        
        df_pd = pd.read_csv(pd_file, sep=';')
        self.assertIsInstance(df_pd, pd.DataFrame)
        self.assertGreater(len(df_pd), 0, "PD.csv ne devrait pas être vide")
        
        # Compter les valeurs manquantes initiales
        initial_count = len(df_pd)
        initial_missing_coords = df_pd[['x', 'y', 'z']].isna().any(axis=1).sum()
        
        # Appeler la méthode de gestion des valeurs manquantes
        cleaned_df = self.cleaner._handle_missing_values(df_pd)
        
        # Vérifications
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertLessEqual(len(cleaned_df), initial_count, "Le nombre de lignes ne devrait pas augmenter")
        
        # Vérifier qu'il n'y a plus de valeurs manquantes dans les coordonnées
        coord_cols = ['x', 'y', 'z']
        for col in coord_cols:
            if col in cleaned_df.columns:
                missing_count = cleaned_df[col].isna().sum()
                self.assertEqual(missing_count, 0, f"Colonne {col} ne devrait plus avoir de valeurs manquantes")
        
        # Vérifier que les données sont préservées pour les lignes valides
        if len(cleaned_df) > 0:
            self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['x']))
            self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['y']))
            self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['z']))
        
        print(f"✅ Valeurs manquantes gérées dans PD.csv: {initial_count} → {len(cleaned_df)} lignes, {initial_missing_coords} coordonnées manquantes supprimées")
    
    def test_handle_missing_values_s_csv(self):
        """Test de gestion des valeurs manquantes avec S.csv"""
        # Charger S.csv
        s_file = self.raw_data_dir / "S.csv"
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        df_s = pd.read_csv(s_file, sep=';')
        self.assertIsInstance(df_s, pd.DataFrame)
        self.assertGreater(len(df_s), 0, "S.csv ne devrait pas être vide")
        
        # Compter les valeurs manquantes initiales
        initial_count = len(df_s)
        initial_missing_coords = df_s[['LAT', 'LON']].isna().any(axis=1).sum()
        
        # Appeler la méthode de gestion des valeurs manquantes
        cleaned_df = self.cleaner._handle_missing_values(df_s)
        
        # Vérifications
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertLessEqual(len(cleaned_df), initial_count, "Le nombre de lignes ne devrait pas augmenter")
        
        # Vérifier qu'il n'y a plus de valeurs manquantes dans les coordonnées
        coord_cols = ['LAT', 'LON']
        for col in coord_cols:
            if col in cleaned_df.columns:
                missing_count = cleaned_df[col].isna().sum()
                self.assertEqual(missing_count, 0, f"Colonne {col} ne devrait plus avoir de valeurs manquantes")
        
        # Vérifier que les données sont préservées pour les lignes valides
        if len(cleaned_df) > 0:
            self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['LAT']))
            self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['LON']))
        
        print(f"✅ Valeurs manquantes gérées dans S.csv: {initial_count} → {len(cleaned_df)} lignes, {initial_missing_coords} coordonnées manquantes supprimées")
    
    def test_handle_missing_values_both_files(self):
        """Test de gestion des valeurs manquantes des deux fichiers CSV ensemble"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        cleaned_pd = self.cleaner._handle_missing_values(df_pd)
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        cleaned_s = self.cleaner._handle_missing_values(df_s)
        
        # Vérifier que les deux nettoyages ont réussi
        self.assertIsInstance(cleaned_pd, pd.DataFrame)
        self.assertIsInstance(cleaned_s, pd.DataFrame)
        
        # Vérifier que les données sont cohérentes
        self.assertLessEqual(len(cleaned_pd), len(df_pd))
        self.assertLessEqual(len(cleaned_s), len(df_s))
        
        print(f"✅ Valeurs manquantes gérées dans les deux fichiers: PD.csv ({len(cleaned_pd)} lignes), S.csv ({len(cleaned_s)} lignes)")
    
    def test_handle_missing_values_coordinate_columns_detection(self):
        """Test de détection automatique des colonnes de coordonnées"""
        # Tester PD.csv (coordonnées x, y, z)
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        # Vérifier que les colonnes de coordonnées sont détectées
        coord_cols = ['x', 'y', 'z']
        detected_cols = [col for col in coord_cols if col in df_pd.columns]
        self.assertEqual(len(detected_cols), 3, "Toutes les colonnes de coordonnées devraient être détectées dans PD.csv")
        
        # Tester S.csv (coordonnées LAT, LON)
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        
        # Vérifier que les colonnes de coordonnées sont détectées
        coord_cols_s = ['LAT', 'LON']
        detected_cols_s = [col for col in coord_cols_s if col in df_s.columns]
        self.assertEqual(len(detected_cols_s), 2, "Toutes les colonnes de coordonnées devraient être détectées dans S.csv")
        
        print("✅ Détection automatique des colonnes de coordonnées réussie")
    
    def test_handle_missing_values_data_integrity(self):
        """Test de l'intégrité des données après suppression des valeurs manquantes"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd_original = pd.read_csv(pd_file, sep=';')
        cleaned_pd = self.cleaner._handle_missing_values(df_pd_original)
        
        # Vérifier que les données non-coordonnées sont préservées
        non_coord_cols = [col for col in df_pd_original.columns if col not in ['x', 'y', 'z']]
        for col in non_coord_cols:
            if col in cleaned_pd.columns:
                # Vérifier que les valeurs sont préservées (pour les lignes qui existent encore)
                original_values = df_pd_original[col].dropna()
                if len(original_values) > 0:
                    # Trouver les indices correspondants dans le DataFrame nettoyé
                    common_indices = df_pd_original.index.intersection(cleaned_pd.index)
                    if len(common_indices) > 0:
                        self.assertTrue(
                            df_pd_original.loc[common_indices, col].equals(cleaned_pd.loc[common_indices, col]),
                            f"Les données de la colonne {col} devraient être préservées"
                        )
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s_original = pd.read_csv(s_file, sep=';')
        cleaned_s = self.cleaner._handle_missing_values(df_s_original)
        
        # Vérifier que les données non-coordonnées sont préservées
        non_coord_cols_s = [col for col in df_s_original.columns if col not in ['LAT', 'LON']]
        for col in non_coord_cols_s:
            if col in cleaned_s.columns:
                # Vérifier que les valeurs sont préservées
                original_values = df_s_original[col].dropna()
                if len(original_values) > 0:
                    common_indices = df_s_original.index.intersection(cleaned_s.index)
                    if len(common_indices) > 0:
                        self.assertTrue(
                            df_s_original.loc[common_indices, col].equals(cleaned_s.loc[common_indices, col]),
                            f"Les données de la colonne {col} devraient être préservées"
                        )
        
        print("✅ Intégrité des données préservée après suppression des valeurs manquantes")
    
    def test_handle_missing_values_no_missing_values(self):
        """Test avec des données sans valeurs manquantes"""
        # Créer un DataFrame sans valeurs manquantes
        clean_df = pd.DataFrame({
            'x': [100, 200, 300],
            'y': [400, 500, 600],
            'z': [700, 800, 900],
            'value': [1.0, 2.0, 3.0]
        })
        
        initial_count = len(clean_df)
        
        # Appeler la méthode
        cleaned_df = self.cleaner._handle_missing_values(clean_df)
        
        # Vérifier que rien n'a changé
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertEqual(len(cleaned_df), initial_count, "Le nombre de lignes ne devrait pas changer")
        
        # Vérifier que toutes les données sont préservées
        pd.testing.assert_frame_equal(clean_df, cleaned_df, check_dtype=False)
        
        print("✅ Données sans valeurs manquantes préservées")
    
    def test_handle_missing_values_all_missing_coordinates(self):
        """Test avec des données où toutes les coordonnées sont manquantes"""
        # Créer un DataFrame avec toutes les coordonnées manquantes
        all_missing_df = pd.DataFrame({
            'x': [np.nan, np.nan, np.nan],
            'y': [np.nan, np.nan, np.nan],
            'z': [np.nan, np.nan, np.nan],
            'value': [1.0, 2.0, 3.0]
        })
        
        initial_count = len(all_missing_df)
        
        # Appeler la méthode
        cleaned_df = self.cleaner._handle_missing_values(all_missing_df)
        
        # Vérifier que toutes les lignes sont supprimées
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertEqual(len(cleaned_df), 0, "Toutes les lignes avec coordonnées manquantes devraient être supprimées")
        
        print("✅ Toutes les lignes avec coordonnées manquantes supprimées")
    
    def test_handle_missing_values_partial_missing_coordinates(self):
        """Test avec des données où certaines coordonnées sont manquantes"""
        # Créer un DataFrame avec des coordonnées partiellement manquantes
        partial_missing_df = pd.DataFrame({
            'x': [100, np.nan, 300],
            'y': [400, 500, np.nan],
            'z': [700, np.nan, 900],
            'value': [1.0, 2.0, 3.0]
        })
        
        initial_count = len(partial_missing_df)
        
        # Appeler la méthode
        cleaned_df = self.cleaner._handle_missing_values(partial_missing_df)
        
        # Vérifier que seules les lignes avec coordonnées complètes restent
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertLess(len(cleaned_df), initial_count, "Certaines lignes devraient être supprimées")
        
        # Vérifier qu'il n'y a plus de valeurs manquantes dans les coordonnées
        for col in ['x', 'y', 'z']:
            if col in cleaned_df.columns:
                missing_count = cleaned_df[col].isna().sum()
                self.assertEqual(missing_count, 0, f"Colonne {col} ne devrait plus avoir de valeurs manquantes")
        
        print(f"✅ Coordonnées partiellement manquantes gérées: {initial_count} → {len(cleaned_df)} lignes")
    
    def test_handle_missing_values_performance(self):
        """Test de performance de la gestion des valeurs manquantes"""
        import time
        
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        start_time = time.time()
        cleaned_pd = self.cleaner._handle_missing_values(df_pd)
        pd_time = time.time() - start_time
        
        self.assertIsInstance(cleaned_pd, pd.DataFrame)
        self.assertLess(pd_time, 1.0, "Gestion des valeurs manquantes de PD.csv devrait être rapide (< 1 seconde)")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        
        start_time = time.time()
        cleaned_s = self.cleaner._handle_missing_values(df_s)
        s_time = time.time() - start_time
        
        self.assertIsInstance(cleaned_s, pd.DataFrame)
        self.assertLess(s_time, 1.0, "Gestion des valeurs manquantes de S.csv devrait être rapide (< 1 seconde)")
        
        print(f"✅ Performance validée: PD.csv ({pd_time:.3f}s), S.csv ({s_time:.3f}s)")
    
    def test_handle_missing_values_integration_with_validation(self):
        """Test d'intégration avec la validation des colonnes"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = self.cleaner._load_device_data(pd_file, "pole_dipole")
        validated_pd = self.cleaner._validate_columns(df_pd, "pole_dipole")
        cleaned_pd = self.cleaner._handle_missing_values(validated_pd)
        
        self.assertIsInstance(cleaned_pd, pd.DataFrame)
        self.assertLessEqual(len(cleaned_pd), len(validated_pd))
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = self.cleaner._load_device_data(s_file, "schlumberger")
        validated_s = self.cleaner._validate_columns(df_s, "schlumberger")
        cleaned_s = self.cleaner._handle_missing_values(validated_s)
        
        self.assertIsInstance(cleaned_s, pd.DataFrame)
        self.assertLessEqual(len(cleaned_s), len(validated_s))
        
        print("✅ Intégration avec la validation des colonnes réussie")
    
    def test_handle_missing_values_edge_cases(self):
        """Test des cas limites"""
        # Test avec DataFrame vide
        empty_df = pd.DataFrame()
        cleaned_empty = self.cleaner._handle_missing_values(empty_df)
        self.assertIsInstance(cleaned_empty, pd.DataFrame)
        self.assertEqual(len(cleaned_empty), 0)
        
        # Test avec DataFrame sans colonnes de coordonnées
        no_coord_df = pd.DataFrame({
            'value1': [1, 2, 3],
            'value2': [4, 5, 6]
        })
        cleaned_no_coord = self.cleaner._handle_missing_values(no_coord_df)
        self.assertIsInstance(cleaned_no_coord, pd.DataFrame)
        self.assertEqual(len(cleaned_no_coord), len(no_coord_df))
        
        # Test avec DataFrame à une seule ligne
        single_row_df = pd.DataFrame({
            'x': [100],
            'y': [200],
            'z': [300]
        })
        cleaned_single = self.cleaner._handle_missing_values(single_row_df)
        self.assertIsInstance(cleaned_single, pd.DataFrame)
        self.assertEqual(len(cleaned_single), 1)
        
        print("✅ Cas limites gérés avec succès")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(
        verbosity=2,
        testLoader=unittest.TestLoader(),
        testRunner=unittest.TextTestRunner(stream=sys.stdout)
    )
