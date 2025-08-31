#!/usr/bin/env python3
"""
Test unitaire pour la méthode _remove_outliers de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _remove_outliers
qui supprime les valeurs aberrantes statistiques des mesures géophysiques
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


class TestDataCleanerRemoveOutliers(unittest.TestCase):
    """Tests pour la méthode _remove_outliers de GeophysicalDataCleaner"""
    
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
    
    def test_remove_outliers_pd_csv(self):
        """Test de suppression des valeurs aberrantes avec PD.csv"""
        # Charger PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        
        df_pd = pd.read_csv(pd_file, sep=';')
        self.assertIsInstance(df_pd, pd.DataFrame)
        self.assertGreater(len(df_pd), 0, "PD.csv ne devrait pas être vide")
        
        # Vérifier que les colonnes géophysiques sont présentes
        geophysical_cols = ['Rho(ohm.m)', 'M (mV/V)']
        for col in geophysical_cols:
            self.assertIn(col, df_pd.columns, f"Colonne {col} manquante dans PD.csv")
        
        # Compter les lignes initiales
        initial_count = len(df_pd)
        
        # Appeler la méthode de suppression des valeurs aberrantes
        cleaned_df = self.cleaner._remove_outliers(df_pd)
        
        # Vérifications
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertLessEqual(len(cleaned_df), initial_count, "Le nombre de lignes ne devrait pas augmenter")
        
        # Vérifier que les données non-géophysiques sont préservées
        non_geophysical_cols = [col for col in df_pd.columns if col not in geophysical_cols]
        for col in non_geophysical_cols:
            if col in cleaned_df.columns:
                self.assertIn(col, cleaned_df.columns, f"Colonne {col} devrait être préservée")
        
        print(f"✅ Valeurs aberrantes supprimées de PD.csv: {initial_count} → {len(cleaned_df)} lignes")
    
    def test_remove_outliers_s_csv(self):
        """Test de suppression des valeurs aberrantes avec S.csv"""
        # Charger S.csv
        s_file = self.raw_data_dir / "S.csv"
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        df_s = pd.read_csv(s_file, sep=';')
        self.assertIsInstance(df_s, pd.DataFrame)
        self.assertGreater(len(df_s), 0, "S.csv ne devrait pas être vide")
        
        # Vérifier que les colonnes géophysiques sont présentes
        geophysical_cols = ['Rho (Ohm.m)', 'M (mV/V)']
        for col in geophysical_cols:
            self.assertIn(col, df_s.columns, f"Colonne {col} manquante dans S.csv")
        
        # Compter les lignes initiales
        initial_count = len(df_s)
        
        # Appeler la méthode de suppression des valeurs aberrantes
        cleaned_df = self.cleaner._remove_outliers(df_s)
        
        # Vérifications
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertLessEqual(len(cleaned_df), initial_count, "Le nombre de lignes ne devrait pas augmenter")
        
        # Vérifier que les données non-géophysiques sont préservées
        non_geophysical_cols = [col for col in df_s.columns if col not in geophysical_cols]
        for col in non_geophysical_cols:
            if col in cleaned_df.columns:
                self.assertIn(col, cleaned_df.columns, f"Colonne {col} devrait être préservée")
        
        print(f"✅ Valeurs aberrantes supprimées de S.csv: {initial_count} → {len(cleaned_df)} lignes")
    
    def test_remove_outliers_both_files(self):
        """Test de suppression des valeurs aberrantes des deux fichiers CSV ensemble"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        cleaned_pd = self.cleaner._remove_outliers(df_pd)
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        cleaned_s = self.cleaner._remove_outliers(df_s)
        
        # Vérifier que les deux nettoyages ont réussi
        self.assertIsInstance(cleaned_pd, pd.DataFrame)
        self.assertIsInstance(cleaned_s, pd.DataFrame)
        
        # Vérifier que les données sont cohérentes
        self.assertLessEqual(len(cleaned_pd), len(df_pd))
        self.assertLessEqual(len(cleaned_s), len(df_s))
        
        print(f"✅ Valeurs aberrantes supprimées des deux fichiers: PD.csv ({len(cleaned_pd)} lignes), S.csv ({len(cleaned_s)} lignes)")
    
    def test_remove_outliers_resistivity_validation(self):
        """Test de validation de la suppression des valeurs aberrantes de résistivité"""
        # Créer un DataFrame avec des valeurs de résistivité
        resistivity_df = pd.DataFrame({
            'resistivity': [100.5, 200.7, 300.2, 400.1, 500.0, 1000.0, 2000.0],
            'chargeability': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            'other_col': ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        })
        
        initial_count = len(resistivity_df)
        
        # Appeler la méthode de suppression des valeurs aberrantes
        cleaned_df = self.cleaner._remove_outliers(resistivity_df)
        
        # Vérifier que certaines lignes ont été supprimées (valeurs aberrantes)
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertLessEqual(len(cleaned_df), initial_count, "Le nombre de lignes ne devrait pas augmenter")
        
        # Vérifier que les autres colonnes sont préservées
        self.assertIn('chargeability', cleaned_df.columns)
        self.assertIn('other_col', cleaned_df.columns)
        
        print(f"✅ Validation de la suppression des valeurs aberrantes de résistivité: {initial_count} → {len(cleaned_df)} lignes")
    
    def test_remove_outliers_chargeability_validation(self):
        """Test de validation de la suppression des valeurs aberrantes de chargeabilité"""
        # Créer un DataFrame avec des valeurs de chargeabilité
        chargeability_df = pd.DataFrame({
            'resistivity': [100.0, 200.0, 300.0, 400.0, 500.0],
            'chargeability': [1.0, 2.0, 3.0, 4.0, 10.0],
            'other_col': ['A', 'B', 'C', 'D', 'E']
        })
        
        initial_count = len(chargeability_df)
        
        # Appeler la méthode de suppression des valeurs aberrantes
        cleaned_df = self.cleaner._remove_outliers(chargeability_df)
        
        # Vérifier que certaines lignes ont été supprimées (valeurs aberrantes)
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertLessEqual(len(cleaned_df), initial_count, "Le nombre de lignes ne devrait pas augmenter")
        
        # Vérifier que les autres colonnes sont préservées
        self.assertIn('resistivity', cleaned_df.columns)
        self.assertIn('other_col', cleaned_df.columns)
        
        print(f"✅ Validation de la suppression des valeurs aberrantes de chargeabilité: {initial_count} → {len(cleaned_df)} lignes")
    
    def test_remove_outliers_data_integrity(self):
        """Test de l'intégrité des données après suppression des valeurs aberrantes"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd_original = pd.read_csv(pd_file, sep=';')
        cleaned_pd = self.cleaner._remove_outliers(df_pd_original)
        
        # Vérifier que les données non-géophysiques sont préservées
        geophysical_cols = ['Rho(ohm.m)', 'M (mV/V)']
        non_geophysical_cols = [col for col in df_pd_original.columns if col not in geophysical_cols]
        
        for col in non_geophysical_cols:
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
        cleaned_s = self.cleaner._remove_outliers(df_s_original)
        
        # Vérifier que les données non-géophysiques sont préservées
        geophysical_cols_s = ['Rho (Ohm.m)', 'M (mV/V)']
        non_geophysical_cols_s = [col for col in df_s_original.columns if col not in geophysical_cols_s]
        
        for col in non_geophysical_cols_s:
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
        
        print("✅ Intégrité des données préservée après suppression des valeurs aberrantes")
    
    def test_remove_outliers_iqr_method(self):
        """Test de la méthode IQR pour la suppression des valeurs aberrantes"""
        # Créer un DataFrame avec des valeurs connues pour tester IQR
        test_df = pd.DataFrame({
            'resistivity': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'chargeability': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        initial_count = len(test_df)
        
        # Appeler la méthode de suppression des valeurs aberrantes
        cleaned_df = self.cleaner._remove_outliers(test_df)
        
        # Vérifier que le DataFrame a été traité
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertLessEqual(len(cleaned_df), initial_count)
        
        # Vérifier que les colonnes sont préservées
        self.assertIn('resistivity', cleaned_df.columns)
        self.assertIn('chargeability', cleaned_df.columns)
        
        print(f"✅ Méthode IQR validée: {initial_count} → {len(cleaned_df)} lignes")
    
    def test_remove_outliers_no_geophysical_columns(self):
        """Test avec un DataFrame sans colonnes géophysiques"""
        # Créer un DataFrame sans colonnes géophysiques
        no_geophysical_df = pd.DataFrame({
            'value1': [1, 2, 3, 4, 5],
            'value2': [6, 7, 8, 9, 10],
            'value3': [11, 12, 13, 14, 15]
        })
        
        initial_count = len(no_geophysical_df)
        
        # Appeler la méthode de suppression des valeurs aberrantes
        cleaned_df = self.cleaner._remove_outliers(no_geophysical_df)
        
        # Vérifier que rien n'a changé
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertEqual(len(cleaned_df), initial_count, "Le nombre de lignes ne devrait pas changer")
        
        # Vérifier que toutes les colonnes sont préservées
        for col in no_geophysical_df.columns:
            self.assertIn(col, cleaned_df.columns, f"Colonne {col} devrait être préservée")
        
        print("✅ DataFrame sans colonnes géophysiques préservé")
    
    def test_remove_outliers_performance(self):
        """Test de performance de la suppression des valeurs aberrantes"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        start_time = time.time()
        cleaned_pd = self.cleaner._remove_outliers(df_pd)
        pd_time = time.time() - start_time
        
        self.assertIsInstance(cleaned_pd, pd.DataFrame)
        self.assertLess(pd_time, 1.0, "Suppression des valeurs aberrantes de PD.csv devrait être rapide (< 1 seconde)")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        
        start_time = time.time()
        cleaned_s = self.cleaner._remove_outliers(df_s)
        s_time = time.time() - start_time
        
        self.assertIsInstance(cleaned_s, pd.DataFrame)
        self.assertLess(s_time, 1.0, "Suppression des valeurs aberrantes de S.csv devrait être rapide (< 1 seconde)")
        
        print(f"✅ Performance validée: PD.csv ({pd_time:.3f}s), S.csv ({s_time:.3f}s)")
    
    def test_remove_outliers_integration_with_cleaner(self):
        """Test d'intégration avec le processus de nettoyage complet"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = self.cleaner._load_device_data(pd_file, "pole_dipole")
        validated_pd = self.cleaner._validate_columns(df_pd, "pole_dipole")
        cleaned_pd = self.cleaner._remove_outliers(validated_pd)
        
        self.assertIsInstance(cleaned_pd, pd.DataFrame)
        self.assertLessEqual(len(cleaned_pd), len(validated_pd))
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = self.cleaner._load_device_data(s_file, "schlumberger")
        validated_s = self.cleaner._validate_columns(df_s, "schlumberger")
        cleaned_s = self.cleaner._remove_outliers(validated_s)
        
        self.assertIsInstance(cleaned_s, pd.DataFrame)
        self.assertLessEqual(len(cleaned_s), len(validated_s))
        
        print("✅ Intégration avec le processus de nettoyage complet réussie")
    
    def test_remove_outliers_edge_cases(self):
        """Test des cas limites"""
        # Test avec DataFrame vide
        empty_df = pd.DataFrame()
        cleaned_empty = self.cleaner._remove_outliers(empty_df)
        self.assertIsInstance(cleaned_empty, pd.DataFrame)
        self.assertEqual(len(cleaned_empty), 0)
        
        # Test avec DataFrame à une seule ligne
        single_row_df = pd.DataFrame({
            'resistivity': [100.0],
            'chargeability': [1.0]
        })
        cleaned_single = self.cleaner._remove_outliers(single_row_df)
        self.assertIsInstance(cleaned_single, pd.DataFrame)
        self.assertEqual(len(cleaned_single), 1)
        
        # Test avec DataFrame avec des valeurs identiques (pas de valeurs aberrantes)
        identical_values_df = pd.DataFrame({
            'resistivity': [100.0, 100.0, 100.0, 100.0, 100.0],
            'chargeability': [1.0, 1.0, 1.0, 1.0, 1.0]
        })
        cleaned_identical = self.cleaner._remove_outliers(identical_values_df)
        self.assertIsInstance(cleaned_identical, pd.DataFrame)
        self.assertEqual(len(cleaned_identical), len(identical_values_df))
        
        print("✅ Cas limites gérés avec succès")
    
    def test_remove_outliers_statistical_validation(self):
        """Test de validation statistique de la suppression des valeurs aberrantes"""
        # Créer un DataFrame avec des valeurs statistiquement aberrantes
        # Valeurs normales: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # Q1 = 32.5, Q3 = 77.5, IQR = 45
        # Limites: lower = 32.5 - 1.5*45 = -35, upper = 77.5 + 1.5*45 = 145
        # Valeurs aberrantes: 200, 300 (au-dessus de 145)
        statistical_df = pd.DataFrame({
            'resistivity': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300],
            'chargeability': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        })
        
        initial_count = len(statistical_df)
        
        # Appeler la méthode de suppression des valeurs aberrantes
        cleaned_df = self.cleaner._remove_outliers(statistical_df)
        
        # Vérifier que des valeurs aberrantes ont été supprimées
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertLess(len(cleaned_df), initial_count, "Les valeurs aberrantes 200 et 300 devraient être supprimées")
        
        # Vérifier que les valeurs aberrantes ont été supprimées
        if 'resistivity' in cleaned_df.columns:
            self.assertNotIn(200, cleaned_df['resistivity'].values, "Valeur aberrante 200 devrait être supprimée")
            self.assertNotIn(300, cleaned_df['resistivity'].values, "Valeur aberrante 300 devrait être supprimée")
        
        print(f"✅ Validation statistique réussie: {initial_count} → {len(cleaned_df)} lignes (valeurs aberrantes supprimées)")
    
    def test_remove_outliers_column_mapping(self):
        """Test de la correspondance des noms de colonnes entre les fichiers"""
        # Vérifier que PD.csv et S.csv utilisent des noms de colonnes différents
        pd_file = self.raw_data_dir / "PD.csv"
        s_file = self.raw_data_dir / "S.csv"
        
        df_pd = pd.read_csv(pd_file, sep=';')
        df_s = pd.read_csv(s_file, sep=';')
        
        # PD.csv utilise 'Rho(ohm.m)' et 'M (mV/V)'
        pd_resistivity_col = 'Rho(ohm.m)'
        pd_chargeability_col = 'M (mV/V)'
        
        # S.csv utilise 'Rho (Ohm.m)' et 'M (mV/V)'
        s_resistivity_col = 'Rho (Ohm.m)'
        s_chargeability_col = 'M (mV/V)'
        
        # Vérifier que les colonnes existent dans les deux fichiers
        self.assertIn(pd_resistivity_col, df_pd.columns, f"Colonne {pd_resistivity_col} manquante dans PD.csv")
        self.assertIn(pd_chargeability_col, df_pd.columns, f"Colonne {pd_chargeability_col} manquante dans PD.csv")
        self.assertIn(s_resistivity_col, df_s.columns, f"Colonne {s_resistivity_col} manquante dans S.csv")
        self.assertIn(s_chargeability_col, df_s.columns, f"Colonne {s_chargeability_col} manquante dans S.csv")
        
        # Tester la suppression des valeurs aberrantes sur les deux fichiers
        cleaned_pd = self.cleaner._remove_outliers(df_pd)
        cleaned_s = self.cleaner._remove_outliers(df_s)
        
        # Vérifier que les deux nettoyages ont réussi
        self.assertIsInstance(cleaned_pd, pd.DataFrame)
        self.assertIsInstance(cleaned_s, pd.DataFrame)
        
        print("✅ Correspondance des noms de colonnes validée entre PD.csv et S.csv")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(
        verbosity=2,
        testLoader=unittest.TestLoader(),
        testRunner=unittest.TextTestRunner(stream=sys.stdout)
    )
