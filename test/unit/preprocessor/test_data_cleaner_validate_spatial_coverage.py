#!/usr/bin/env python3
"""
Test unitaire pour la méthode _validate_spatial_coverage de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _validate_spatial_coverage
qui valide que les données couvrent la zone spatiale attendue
en utilisant les vrais fichiers PD.csv et S.csv.
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile
import time
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.preprocessor.data_cleaner import GeophysicalDataCleaner


class TestDataCleanerValidateSpatialCoverage(unittest.TestCase):
    """Tests pour la méthode _validate_spatial_coverage de GeophysicalDataCleaner"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        # Utiliser les vrais fichiers de données du projet
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.test_dir = self.project_root / "test" / "fixtures"
        
        # Créer une instance du cleaner avec les vrais chemins
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
    
    def test_validate_spatial_coverage_pd_csv(self):
        """Test de validation de la couverture spatiale avec PD.csv"""
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
        
        # Appeler la méthode de validation de la couverture spatiale
        validated_df = self.cleaner._validate_spatial_coverage(df_pd, "pole_dipole")
        
        # Vérifications
        self.assertIsInstance(validated_df, pd.DataFrame)
        self.assertEqual(len(validated_df), initial_count, "Le nombre de lignes ne devrait pas changer")
        
        # Vérifier que les coordonnées sont préservées
        for col in coord_cols:
            self.assertIn(col, validated_df.columns, f"Colonne {col} devrait être préservée")
        
        # Vérifier que les données non-coordonnées sont préservées
        non_coord_cols = [col for col in df_pd.columns if col not in coord_cols]
        for col in non_coord_cols:
            self.assertIn(col, validated_df.columns, f"Colonne {col} devrait être préservée")
        
        print(f"✅ Couverture spatiale de PD.csv validée: {initial_count} lignes")
    
    def test_validate_spatial_coverage_s_csv(self):
        """Test de validation de la couverture spatiale avec S.csv"""
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
        
        # Appeler la méthode de validation de la couverture spatiale
        validated_df = self.cleaner._validate_spatial_coverage(df_s, "schlumberger")
        
        # Vérifications
        self.assertIsInstance(validated_df, pd.DataFrame)
        self.assertEqual(len(validated_df), initial_count, "Le nombre de lignes ne devrait pas changer")
        
        # Vérifier que les coordonnées sont préservées
        for col in coord_cols:
            self.assertIn(col, validated_df.columns, f"Colonne {col} devrait être préservée")
        
        # Vérifier que les données non-coordonnées sont préservées
        non_coord_cols = [col for col in df_s.columns if col not in coord_cols]
        for col in non_coord_cols:
            self.assertIn(col, validated_df.columns, f"Colonne {col} devrait être préservée")
        
        print(f"✅ Couverture spatiale de S.csv validée: {initial_count} lignes")
    
    def test_validate_spatial_coverage_both_files(self):
        """Test de validation de la couverture spatiale des deux fichiers CSV ensemble"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        validated_pd = self.cleaner._validate_spatial_coverage(df_pd, "pole_dipole")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        validated_s = self.cleaner._validate_spatial_coverage(df_s, "schlumberger")
        
        # Vérifier que les deux validations ont réussi
        self.assertIsInstance(validated_pd, pd.DataFrame)
        self.assertIsInstance(validated_s, pd.DataFrame)
        
        # Vérifier que les données sont cohérentes
        self.assertEqual(len(validated_pd), len(df_pd))
        self.assertEqual(len(validated_s), len(df_s))
        
        print(f"✅ Couverture spatiale des deux fichiers validée: PD.csv ({len(validated_pd)} lignes), S.csv ({len(validated_s)} lignes)")
    
    def test_validate_spatial_coverage_coordinate_calculation(self):
        """Test de calcul des coordonnées et de la couverture spatiale"""
        # Créer un DataFrame avec des coordonnées connues
        test_df = pd.DataFrame({
            'x': [0, 10, 20, 30, 40, 50],
            'y': [0, 5, 10, 15, 20, 25],
            'z': [0, 1, 2, 3, 4, 5],
            'value': ['A', 'B', 'C', 'D', 'E', 'F']
        })
        
        initial_count = len(test_df)
        
        # Appeler la méthode de validation de la couverture spatiale
        validated_df = self.cleaner._validate_spatial_coverage(test_df, "pole_dipole")
        
        # Vérifier que le DataFrame a été traité
        self.assertIsInstance(validated_df, pd.DataFrame)
        self.assertEqual(len(validated_df), initial_count)
        
        # Vérifier que les coordonnées sont préservées
        self.assertIn('x', validated_df.columns)
        self.assertIn('y', validated_df.columns)
        self.assertIn('z', validated_df.columns)
        
        # Vérifier que les autres colonnes sont préservées
        self.assertIn('value', validated_df.columns)
        
        print(f"✅ Calcul des coordonnées et couverture spatiale validé: {initial_count} lignes")
    
    def test_validate_spatial_coverage_no_coordinate_columns(self):
        """Test avec un DataFrame sans colonnes de coordonnées"""
        # Créer un DataFrame sans colonnes de coordonnées
        no_coord_df = pd.DataFrame({
            'value1': [1, 2, 3, 4, 5],
            'value2': [6, 7, 8, 9, 10],
            'value3': [11, 12, 13, 14, 15]
        })
        
        initial_count = len(no_coord_df)
        
        # Appeler la méthode de validation de la couverture spatiale
        validated_df = self.cleaner._validate_spatial_coverage(no_coord_df, "pole_dipole")
        
        # Vérifier que rien n'a changé
        self.assertIsInstance(validated_df, pd.DataFrame)
        self.assertEqual(len(validated_df), initial_count, "Le nombre de lignes ne devrait pas changer")
        
        # Vérifier que toutes les colonnes sont préservées
        for col in no_coord_df.columns:
            self.assertIn(col, validated_df.columns, f"Colonne {col} devrait être préservée")
        
        print("✅ DataFrame sans colonnes de coordonnées préservé")
    
    def test_validate_spatial_coverage_data_integrity(self):
        """Test de l'intégrité des données après validation de la couverture spatiale"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd_original = pd.read_csv(pd_file, sep=';')
        validated_pd = self.cleaner._validate_spatial_coverage(df_pd_original, "pole_dipole")
        
        # Vérifier que toutes les données sont préservées
        for col in df_pd_original.columns:
            self.assertIn(col, validated_pd.columns, f"Colonne {col} devrait être préservée")
            # Vérifier que les valeurs sont identiques
            pd.testing.assert_series_equal(
                df_pd_original[col], 
                validated_pd[col], 
                check_names=False,
                check_dtype=False
            )
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s_original = pd.read_csv(s_file, sep=';')
        validated_s = self.cleaner._validate_spatial_coverage(df_s_original, "schlumberger")
        
        # Vérifier que toutes les données sont préservées
        for col in df_s_original.columns:
            self.assertIn(col, validated_s.columns, f"Colonne {col} devrait être préservée")
            # Vérifier que les valeurs sont identiques
            pd.testing.assert_series_equal(
                df_s_original[col], 
                validated_s[col], 
                check_names=False,
                check_dtype=False
            )
        
        print("✅ Intégrité des données préservée après validation de la couverture spatiale")
    
    def test_validate_spatial_coverage_coordinate_ranges(self):
        """Test des plages de coordonnées et de la couverture spatiale"""
        # Tester PD.csv avec coordonnées cartésiennes
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        # Vérifier que les coordonnées sont numériques
        self.assertTrue(pd.api.types.is_numeric_dtype(df_pd['x']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df_pd['y']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df_pd['z']))
        
        # Calculer les plages de coordonnées
        x_range = df_pd['x'].max() - df_pd['x'].min()
        y_range = df_pd['y'].max() - df_pd['y'].min()
        z_range = df_pd['z'].max() - df_pd['z'].min()
        
        # Vérifier que les plages sont cohérentes
        self.assertGreater(x_range, 0, "La plage x devrait être positive")
        self.assertGreater(y_range, 0, "La plage y devrait être positive")
        self.assertGreater(z_range, 0, "La plage z devrait être positive")
        
        # Appeler la méthode de validation
        validated_df = self.cleaner._validate_spatial_coverage(df_pd, "pole_dipole")
        
        # Vérifier que les plages sont préservées
        validated_x_range = validated_df['x'].max() - validated_df['x'].min()
        validated_y_range = validated_df['y'].max() - validated_df['y'].min()
        validated_z_range = validated_df['z'].max() - validated_df['z'].min()
        
        self.assertEqual(x_range, validated_x_range, "La plage x devrait être préservée")
        self.assertEqual(y_range, validated_y_range, "La plage y devrait être préservée")
        self.assertEqual(z_range, validated_z_range, "La plage z devrait être préservée")
        
        print(f"✅ Plages de coordonnées validées: x={x_range:.1f}m, y={y_range:.1f}m, z={z_range:.1f}m")
    
    def test_validate_spatial_coverage_performance(self):
        """Test de performance de la validation de la couverture spatiale"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        start_time = time.time()
        validated_pd = self.cleaner._validate_spatial_coverage(df_pd, "pole_dipole")
        pd_time = time.time() - start_time
        
        self.assertIsInstance(validated_pd, pd.DataFrame)
        self.assertLess(pd_time, 1.0, "Validation de la couverture spatiale de PD.csv devrait être rapide (< 1 seconde)")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        
        start_time = time.time()
        validated_s = self.cleaner._validate_spatial_coverage(df_s, "schlumberger")
        s_time = time.time() - start_time
        
        self.assertIsInstance(validated_s, pd.DataFrame)
        self.assertLess(s_time, 1.0, "Validation de la couverture spatiale de S.csv devrait être rapide (< 1 seconde)")
        
        print(f"✅ Performance validée: PD.csv ({pd_time:.3f}s), S.csv ({s_time:.3f}s)")
    
    def test_validate_spatial_coverage_integration_with_cleaner(self):
        """Test d'intégration avec le processus de nettoyage complet"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = self.cleaner._load_device_data(pd_file, "pole_dipole")
        validated_pd = self.cleaner._validate_columns(df_pd, "pole_dipole")
        spatial_validated_pd = self.cleaner._validate_spatial_coverage(validated_pd, "pole_dipole")
        
        self.assertIsInstance(spatial_validated_pd, pd.DataFrame)
        self.assertEqual(len(spatial_validated_pd), len(validated_pd))
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = self.cleaner._load_device_data(s_file, "schlumberger")
        validated_s = self.cleaner._validate_columns(df_s, "schlumberger")
        spatial_validated_s = self.cleaner._validate_spatial_coverage(validated_s, "schlumberger")
        
        self.assertIsInstance(spatial_validated_s, pd.DataFrame)
        self.assertEqual(len(spatial_validated_s), len(validated_s))
        
        print("✅ Intégration avec le processus de nettoyage complet réussie")
    
    def test_validate_spatial_coverage_edge_cases(self):
        """Test des cas limites"""
        # Test avec DataFrame vide
        empty_df = pd.DataFrame()
        validated_empty = self.cleaner._validate_spatial_coverage(empty_df, "pole_dipole")
        self.assertIsInstance(validated_empty, pd.DataFrame)
        self.assertEqual(len(validated_empty), 0)
        
        # Test avec DataFrame à une seule ligne
        single_row_df = pd.DataFrame({
            'x': [100.0],
            'y': [200.0],
            'z': [300.0]
        })
        validated_single = self.cleaner._validate_spatial_coverage(single_row_df, "pole_dipole")
        self.assertIsInstance(validated_single, pd.DataFrame)
        self.assertEqual(len(validated_single), 1)
        
        # Test avec DataFrame avec des coordonnées identiques
        identical_coords_df = pd.DataFrame({
            'x': [100.0, 100.0, 100.0],
            'y': [200.0, 200.0, 200.0],
            'z': [300.0, 300.0, 300.0]
        })
        validated_identical = self.cleaner._validate_spatial_coverage(identical_coords_df, "pole_dipole")
        self.assertIsInstance(validated_identical, pd.DataFrame)
        self.assertEqual(len(validated_identical), len(identical_coords_df))
        
        print("✅ Cas limites gérés avec succès")
    
    def test_validate_spatial_coverage_coordinate_systems(self):
        """Test de validation avec différents systèmes de coordonnées"""
        # Tester PD.csv (coordonnées cartésiennes UTM)
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        # PD.csv utilise des coordonnées cartésiennes (x, y, z)
        self.assertIn('x', df_pd.columns)
        self.assertIn('y', df_pd.columns)
        self.assertIn('z', df_pd.columns)
        
        validated_pd = self.cleaner._validate_spatial_coverage(df_pd, "pole_dipole")
        self.assertIsInstance(validated_pd, pd.DataFrame)
        
        # Tester S.csv (coordonnées géographiques LAT/LON)
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        
        # S.csv utilise des coordonnées géographiques (LAT, LON)
        self.assertIn('LAT', df_s.columns)
        self.assertIn('LON', df_s.columns)
        
        validated_s = self.cleaner._validate_spatial_coverage(df_s, "schlumberger")
        self.assertIsInstance(validated_s, pd.DataFrame)
        
        print("✅ Validation avec différents systèmes de coordonnées réussie")
    
    def test_validate_spatial_coverage_logging(self):
        """Test de la journalisation lors de la validation de la couverture spatiale"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        # Appeler la méthode avec un appareil configuré
        validated_pd = self.cleaner._validate_spatial_coverage(df_pd, "pole_dipole")
        
        # Vérifier que le DataFrame a été traité
        self.assertIsInstance(validated_pd, pd.DataFrame)
        self.assertEqual(len(validated_pd), len(df_pd))
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        
        # Appeler la méthode avec un appareil configuré
        validated_s = self.cleaner._validate_spatial_coverage(df_s, "schlumberger")
        
        # Vérifier que le DataFrame a été traité
        self.assertIsInstance(validated_s, pd.DataFrame)
        self.assertEqual(len(validated_s), len(df_s))
        
        print("✅ Journalisation lors de la validation de la couverture spatiale testée")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(
        verbosity=2,
        testLoader=unittest.TestLoader(),
        testRunner=unittest.TextTestRunner(stream=sys.stdout)
    )
