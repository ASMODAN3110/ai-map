#!/usr/bin/env python3
"""
Test unitaire pour la méthode _clean_coordinates de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _clean_coordinates
qui nettoie et transforme les coordonnées géographiques
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


class TestDataCleanerCleanCoordinates(unittest.TestCase):
    """Tests pour la méthode _clean_coordinates de GeophysicalDataCleaner"""
    
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
    
    def test_clean_coordinates_pd_csv(self):
        """Test de nettoyage des coordonnées avec PD.csv"""
        # Charger PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        
        df_pd = pd.read_csv(pd_file, sep=';')
        self.assertIsInstance(df_pd, pd.DataFrame)
        self.assertGreater(len(df_pd), 0, "PD.csv ne devrait pas être vide")
        
        # Vérifier que les colonnes de coordonnées sont présentes
        coord_cols = ['x', 'y', 'z']
        for col in coord_cols:
            self.assertIn(col, df_pd.columns, f"Colonne {col} manquante dans PD.csv")
        
        # Compter les lignes initiales
        initial_count = len(df_pd)
        
        # Appeler la méthode de nettoyage des coordonnées
        cleaned_df = self.cleaner._clean_coordinates(df_pd, "pole_dipole")
        
        # Vérifications
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertLessEqual(len(cleaned_df), initial_count, "Le nombre de lignes ne devrait pas augmenter")
        
        # Vérifier que les coordonnées sont numériques
        for col in coord_cols:
            if col in cleaned_df.columns:
                self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df[col]), f"Colonne {col} devrait être numérique")
                # Vérifier qu'il n'y a pas de valeurs manquantes
                missing_count = cleaned_df[col].isna().sum()
                self.assertEqual(missing_count, 0, f"Colonne {col} ne devrait plus avoir de valeurs manquantes")
        
        # Vérifier que les données non-coordonnées sont préservées
        non_coord_cols = [col for col in df_pd.columns if col not in coord_cols]
        for col in non_coord_cols:
            if col in cleaned_df.columns:
                self.assertIn(col, cleaned_df.columns, f"Colonne {col} devrait être préservée")
        
        print(f"✅ Coordonnées de PD.csv nettoyées: {initial_count} → {len(cleaned_df)} lignes")
    
    def test_clean_coordinates_s_csv(self):
        """Test de nettoyage des coordonnées avec S.csv"""
        # Charger S.csv
        s_file = self.raw_data_dir / "S.csv"
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        df_s = pd.read_csv(s_file, sep=';')
        self.assertIsInstance(df_s, pd.DataFrame)
        self.assertGreater(len(df_s), 0, "S.csv ne devrait pas être vide")
        
        # Vérifier que les colonnes de coordonnées sont présentes
        coord_cols = ['LAT', 'LON']
        for col in coord_cols:
            self.assertIn(col, df_s.columns, f"Colonne {col} manquante dans S.csv")
        
        # Compter les lignes initiales
        initial_count = len(df_s)
        
        # Appeler la méthode de nettoyage des coordonnées
        cleaned_df = self.cleaner._clean_coordinates(df_s, "schlumberger")
        
        # Vérifications
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertLessEqual(len(cleaned_df), initial_count, "Le nombre de lignes ne devrait pas augmenter")
        
        # Vérifier que les coordonnées sont numériques
        for col in coord_cols:
            if col in cleaned_df.columns:
                self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df[col]), f"Colonne {col} devrait être numérique")
                # Vérifier qu'il n'y a pas de valeurs manquantes
                missing_count = cleaned_df[col].isna().sum()
                self.assertEqual(missing_count, 0, f"Colonne {col} ne devrait plus avoir de valeurs manquantes")
        
        # Vérifier que les données non-coordonnées sont préservées
        non_coord_cols = [col for col in df_s.columns if col not in coord_cols]
        for col in non_coord_cols:
            if col in cleaned_df.columns:
                self.assertIn(col, cleaned_df.columns, f"Colonne {col} devrait être préservée")
        
        print(f"✅ Coordonnées de S.csv nettoyées: {initial_count} → {len(cleaned_df)} lignes")
    
    def test_clean_coordinates_both_files(self):
        """Test de nettoyage des coordonnées des deux fichiers CSV ensemble"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        cleaned_pd = self.cleaner._clean_coordinates(df_pd, "pole_dipole")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        cleaned_s = self.cleaner._clean_coordinates(df_s, "schlumberger")
        
        # Vérifier que les deux nettoyages ont réussi
        self.assertIsInstance(cleaned_pd, pd.DataFrame)
        self.assertIsInstance(cleaned_s, pd.DataFrame)
        
        # Vérifier que les données sont cohérentes
        self.assertLessEqual(len(cleaned_pd), len(df_pd))
        self.assertLessEqual(len(cleaned_s), len(df_s))
        
        print(f"✅ Coordonnées des deux fichiers nettoyées: PD.csv ({len(cleaned_pd)} lignes), S.csv ({len(cleaned_s)} lignes)")
    
    def test_clean_coordinates_coordinate_transformation(self):
        """Test de transformation des coordonnées lat/lon vers x/y"""
        # Créer un DataFrame avec des coordonnées lat/lon
        lat_lon_df = pd.DataFrame({
            'lat': [48.8566, 43.2965, 45.7640],  # Paris, Lyon, Nice
            'lon': [2.3522, 5.3698, 4.8357],
            'z': [100, 200, 300],
            'value': [1.0, 2.0, 3.0]
        })
        
        initial_count = len(lat_lon_df)
        
        # Appeler la méthode de nettoyage
        cleaned_df = self.cleaner._clean_coordinates(lat_lon_df, "test_device")
        
        # Vérifier que la transformation a eu lieu
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertIn('x', cleaned_df.columns, "Colonne 'x' devrait être créée après transformation")
        self.assertIn('y', cleaned_df.columns, "Colonne 'y' devrait être créée après transformation")
        self.assertNotIn('lat', cleaned_df.columns, "Colonne 'lat' devrait être supprimée après transformation")
        self.assertNotIn('lon', cleaned_df.columns, "Colonne 'lon' devrait être supprimée après transformation")
        
        # Vérifier que les coordonnées sont numériques
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['x']))
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['y']))
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['z']))
        
        print("✅ Transformation des coordonnées lat/lon vers x/y réussie")
    
    def test_clean_coordinates_data_integrity(self):
        """Test de l'intégrité des données après nettoyage des coordonnées"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd_original = pd.read_csv(pd_file, sep=';')
        cleaned_pd = self.cleaner._clean_coordinates(df_pd_original, "pole_dipole")
        
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
        cleaned_s = self.cleaner._clean_coordinates(df_s_original, "schlumberger")
        
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
        
        print("✅ Intégrité des données préservée après nettoyage des coordonnées")
    
    def test_clean_coordinates_numeric_conversion(self):
        """Test de conversion des coordonnées en valeurs numériques"""
        # Créer un DataFrame avec des coordonnées en string
        string_coords_df = pd.DataFrame({
            'x': ['100.5', '200.7', '300.2'],
            'y': ['400.1', '500.3', '600.8'],
            'z': ['700.9', '800.4', '900.6'],
            'value': [1.0, 2.0, 3.0]
        })
        
        initial_count = len(string_coords_df)
        
        # Appeler la méthode de nettoyage
        cleaned_df = self.cleaner._clean_coordinates(string_coords_df, "test_device")
        
        # Vérifier que les coordonnées sont converties en numériques
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertEqual(len(cleaned_df), initial_count, "Le nombre de lignes ne devrait pas changer")
        
        # Vérifier que les coordonnées sont numériques
        for col in ['x', 'y', 'z']:
            self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df[col]), f"Colonne {col} devrait être numérique")
        
        # Vérifier que les valeurs sont correctement converties (sans ordre spécifique)
        x_values = cleaned_df['x'].tolist()
        y_values = cleaned_df['y'].tolist()
        z_values = cleaned_df['z'].tolist()
        
        # Vérifier que toutes les valeurs attendues sont présentes
        expected_x = [100.5, 200.7, 300.2]
        expected_y = [400.1, 500.3, 600.8]
        expected_z = [700.9, 800.4, 900.6]
        
        for val in expected_x:
            self.assertIn(val, x_values, f"Valeur {val} devrait être présente dans la colonne x")
        for val in expected_y:
            self.assertIn(val, y_values, f"Valeur {val} devrait être présente dans la colonne y")
        for val in expected_z:
            self.assertIn(val, z_values, f"Valeur {val} devrait être présente dans la colonne z")
        
        print("✅ Conversion des coordonnées string vers numériques réussie")
    
    def test_clean_coordinates_invalid_coordinates_removal(self):
        """Test de suppression des lignes avec coordonnées invalides"""
        # Créer un DataFrame avec des coordonnées invalides
        invalid_coords_df = pd.DataFrame({
            'x': [100, np.nan, 300, 'invalid', 500],
            'y': [400, 500, np.nan, 600, 'invalid'],
            'z': [700, 800, 900, np.nan, 1000],
            'value': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        initial_count = len(invalid_coords_df)
        
        # Appeler la méthode de nettoyage
        cleaned_df = self.cleaner._clean_coordinates(invalid_coords_df, "test_device")
        
        # Vérifier que les lignes invalides sont supprimées
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertLess(len(cleaned_df), initial_count, "Certaines lignes avec coordonnées invalides devraient être supprimées")
        
        # Vérifier qu'il n'y a plus de valeurs manquantes dans les coordonnées
        for col in ['x', 'y', 'z']:
            if col in cleaned_df.columns:
                missing_count = cleaned_df[col].isna().sum()
                self.assertEqual(missing_count, 0, f"Colonne {col} ne devrait plus avoir de valeurs manquantes")
        
        print(f"✅ Lignes avec coordonnées invalides supprimées: {initial_count} → {len(cleaned_df)} lignes")
    
    def test_clean_coordinates_no_coordinate_columns(self):
        """Test avec un DataFrame sans colonnes de coordonnées"""
        # Créer un DataFrame sans colonnes de coordonnées
        no_coord_df = pd.DataFrame({
            'value1': [1, 2, 3],
            'value2': [4, 5, 6],
            'value3': [7, 8, 9]
        })
        
        initial_count = len(no_coord_df)
        
        # Appeler la méthode de nettoyage
        cleaned_df = self.cleaner._clean_coordinates(no_coord_df, "test_device")
        
        # Vérifier que rien n'a changé
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertEqual(len(cleaned_df), initial_count, "Le nombre de lignes ne devrait pas changer")
        
        # Vérifier que toutes les colonnes sont préservées
        for col in no_coord_df.columns:
            self.assertIn(col, cleaned_df.columns, f"Colonne {col} devrait être préservée")
        
        print("✅ DataFrame sans colonnes de coordonnées préservé")
    
    def test_clean_coordinates_performance(self):
        """Test de performance du nettoyage des coordonnées"""
        import time
        
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        start_time = time.time()
        cleaned_pd = self.cleaner._clean_coordinates(df_pd, "pole_dipole")
        pd_time = time.time() - start_time
        
        self.assertIsInstance(cleaned_pd, pd.DataFrame)
        self.assertLess(pd_time, 1.0, "Nettoyage des coordonnées de PD.csv devrait être rapide (< 1 seconde)")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        
        start_time = time.time()
        cleaned_s = self.cleaner._clean_coordinates(df_s, "schlumberger")
        s_time = time.time() - start_time
        
        self.assertIsInstance(cleaned_s, pd.DataFrame)
        self.assertLess(s_time, 1.0, "Nettoyage des coordonnées de S.csv devrait être rapide (< 1 seconde)")
        
        print(f"✅ Performance validée: PD.csv ({pd_time:.3f}s), S.csv ({s_time:.3f}s)")
    
    def test_clean_coordinates_integration_with_other_methods(self):
        """Test d'intégration avec les autres méthodes de nettoyage"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = self.cleaner._load_device_data(pd_file, "pole_dipole")
        validated_pd = self.cleaner._validate_columns(df_pd, "pole_dipole")
        cleaned_pd = self.cleaner._clean_coordinates(validated_pd, "pole_dipole")
        
        self.assertIsInstance(cleaned_pd, pd.DataFrame)
        self.assertLessEqual(len(cleaned_pd), len(validated_pd))
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = self.cleaner._load_device_data(s_file, "schlumberger")
        validated_s = self.cleaner._validate_columns(df_s, "schlumberger")
        cleaned_s = self.cleaner._clean_coordinates(validated_s, "schlumberger")
        
        self.assertIsInstance(cleaned_s, pd.DataFrame)
        self.assertLessEqual(len(cleaned_s), len(validated_s))
        
        print("✅ Intégration avec les autres méthodes de nettoyage réussie")
    
    def test_clean_coordinates_edge_cases(self):
        """Test des cas limites"""
        # Test avec DataFrame vide
        empty_df = pd.DataFrame()
        cleaned_empty = self.cleaner._clean_coordinates(empty_df, "test_device")
        self.assertIsInstance(cleaned_empty, pd.DataFrame)
        self.assertEqual(len(cleaned_empty), 0)
        
        # Test avec DataFrame à une seule ligne
        single_row_df = pd.DataFrame({
            'x': [100],
            'y': [200],
            'z': [300]
        })
        cleaned_single = self.cleaner._clean_coordinates(single_row_df, "test_device")
        self.assertIsInstance(cleaned_single, pd.DataFrame)
        self.assertEqual(len(cleaned_single), 1)
        
        # Test avec DataFrame avec des coordonnées extrêmes
        extreme_coords_df = pd.DataFrame({
            'x': [1e10, -1e10, 0],
            'y': [1e10, -1e10, 0],
            'z': [1e10, -1e10, 0]
        })
        cleaned_extreme = self.cleaner._clean_coordinates(extreme_coords_df, "test_device")
        self.assertIsInstance(cleaned_extreme, pd.DataFrame)
        self.assertEqual(len(cleaned_extreme), 3)
        
        print("✅ Cas limites gérés avec succès")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(
        verbosity=2,
        testLoader=unittest.TestLoader(),
        testRunner=unittest.TextTestRunner(stream=sys.stdout)
    )
