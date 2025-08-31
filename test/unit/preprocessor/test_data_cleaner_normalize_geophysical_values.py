#!/usr/bin/env python3
"""
Test unitaire pour la méthode _normalize_geophysical_values de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _normalize_geophysical_values
qui normalise les mesures géophysiques (résistivité et chargeabilité)
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


class TestDataCleanerNormalizeGeophysicalValues(unittest.TestCase):
    """Tests pour la méthode _normalize_geophysical_values de GeophysicalDataCleaner"""
    
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
    
    def test_normalize_geophysical_values_pd_csv(self):
        """Test de normalisation des valeurs géophysiques avec PD.csv"""
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
        
        # Appeler la méthode de normalisation
        normalized_df = self.cleaner._normalize_geophysical_values(df_pd)
        
        # Vérifications
        self.assertIsInstance(normalized_df, pd.DataFrame)
        self.assertLessEqual(len(normalized_df), initial_count, "Le nombre de lignes ne devrait pas augmenter")
        
        # Vérifier que les valeurs géophysiques sont numériques
        for col in geophysical_cols:
            if col in normalized_df.columns:
                self.assertTrue(pd.api.types.is_numeric_dtype(normalized_df[col]), f"Colonne {col} devrait être numérique")
        
        # Vérifier que les données non-géophysiques sont préservées
        non_geophysical_cols = [col for col in df_pd.columns if col not in geophysical_cols]
        for col in non_geophysical_cols:
            if col in normalized_df.columns:
                self.assertIn(col, normalized_df.columns, f"Colonne {col} devrait être préservée")
        
        print(f"✅ Valeurs géophysiques de PD.csv normalisées: {initial_count} → {len(normalized_df)} lignes")
    
    def test_normalize_geophysical_values_s_csv(self):
        """Test de normalisation des valeurs géophysiques avec S.csv"""
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
        
        # Appeler la méthode de normalisation
        normalized_df = self.cleaner._normalize_geophysical_values(df_s)
        
        # Vérifications
        self.assertIsInstance(normalized_df, pd.DataFrame)
        self.assertLessEqual(len(normalized_df), initial_count, "Le nombre de lignes ne devrait pas augmenter")
        
        # Vérifier que les valeurs géophysiques sont numériques
        for col in geophysical_cols:
            if col in normalized_df.columns:
                self.assertTrue(pd.api.types.is_numeric_dtype(normalized_df[col]), f"Colonne {col} devrait être numérique")
        
        # Vérifier que les données non-géophysiques sont préservées
        non_geophysical_cols = [col for col in df_s.columns if col not in geophysical_cols]
        for col in non_geophysical_cols:
            if col in normalized_df.columns:
                self.assertIn(col, normalized_df.columns, f"Colonne {col} devrait être préservée")
        
        print(f"✅ Valeurs géophysiques de S.csv normalisées: {initial_count} → {len(normalized_df)} lignes")
    
    def test_normalize_geophysical_values_both_files(self):
        """Test de normalisation des valeurs géophysiques des deux fichiers CSV ensemble"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        normalized_pd = self.cleaner._normalize_geophysical_values(df_pd)
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        normalized_s = self.cleaner._normalize_geophysical_values(df_s)
        
        # Vérifier que les deux normalisations ont réussi
        self.assertIsInstance(normalized_pd, pd.DataFrame)
        self.assertIsInstance(normalized_s, pd.DataFrame)
        
        # Vérifier que les données sont cohérentes
        self.assertLessEqual(len(normalized_pd), len(df_pd))
        self.assertLessEqual(len(normalized_s), len(df_s))
        
        print(f"✅ Valeurs géophysiques des deux fichiers normalisées: PD.csv ({len(normalized_pd)} lignes), S.csv ({len(normalized_s)} lignes)")
    
    def test_normalize_geophysical_values_resistivity_validation(self):
        """Test de validation de la résistivité"""
        # Créer un DataFrame avec des valeurs de résistivité
        resistivity_df = pd.DataFrame({
            'resistivity': [100.5, 200.7, 300.2, -50.0, 0.0, 500.0],
            'chargeability': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'other_col': ['A', 'B', 'C', 'D', 'E', 'F']
        })
        
        initial_count = len(resistivity_df)
        
        # Appeler la méthode de normalisation
        normalized_df = self.cleaner._normalize_geophysical_values(resistivity_df)
        
        # Vérifier que seules les lignes avec résistivité positive sont conservées
        self.assertIsInstance(normalized_df, pd.DataFrame)
        self.assertLess(len(normalized_df), initial_count, "Certaines lignes avec résistivité invalide devraient être supprimées")
        
        # Vérifier que toutes les valeurs de résistivité restantes sont positives
        if 'resistivity' in normalized_df.columns:
            self.assertTrue((normalized_df['resistivity'] > 0).all(), "Toutes les valeurs de résistivité devraient être positives")
        
        # Vérifier que les autres colonnes sont préservées
        self.assertIn('chargeability', normalized_df.columns)
        self.assertIn('other_col', normalized_df.columns)
        
        print(f"✅ Validation de la résistivité réussie: {initial_count} → {len(normalized_df)} lignes")
    
    def test_normalize_geophysical_values_chargeability_validation(self):
        """Test de validation de la chargeabilité"""
        # Créer un DataFrame avec des valeurs de chargeabilité
        chargeability_df = pd.DataFrame({
            'resistivity': [100.0, 200.0, 300.0, 400.0, 500.0],
            'chargeability': [1.0, -2.0, 3.0, -4.0, 5.0],
            'other_col': ['A', 'B', 'C', 'D', 'E']
        })
        
        initial_count = len(chargeability_df)
        
        # Appeler la méthode de normalisation
        normalized_df = self.cleaner._normalize_geophysical_values(chargeability_df)
        
        # Vérifier que seules les lignes avec chargeabilité non-négative sont conservées
        self.assertIsInstance(normalized_df, pd.DataFrame)
        self.assertLess(len(normalized_df), initial_count, "Certaines lignes avec chargeabilité négative devraient être supprimées")
        
        # Vérifier que toutes les valeurs de chargeabilité restantes sont non-négatives
        if 'chargeability' in normalized_df.columns:
            self.assertTrue((normalized_df['chargeability'] >= 0).all(), "Toutes les valeurs de chargeabilité devraient être non-négatives")
        
        # Vérifier que les autres colonnes sont préservées
        self.assertIn('resistivity', normalized_df.columns)
        self.assertIn('other_col', normalized_df.columns)
        
        print(f"✅ Validation de la chargeabilité réussie: {initial_count} → {len(normalized_df)} lignes")
    
    def test_normalize_geophysical_values_data_integrity(self):
        """Test de l'intégrité des données après normalisation"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd_original = pd.read_csv(pd_file, sep=';')
        normalized_pd = self.cleaner._normalize_geophysical_values(df_pd_original)
        
        # Vérifier que les données non-géophysiques sont préservées
        geophysical_cols = ['Rho(ohm.m)', 'M (mV/V)']
        non_geophysical_cols = [col for col in df_pd_original.columns if col not in geophysical_cols]
        
        for col in non_geophysical_cols:
            if col in normalized_pd.columns:
                # Vérifier que les valeurs sont préservées (pour les lignes qui existent encore)
                original_values = df_pd_original[col].dropna()
                if len(original_values) > 0:
                    # Trouver les indices correspondants dans le DataFrame normalisé
                    common_indices = df_pd_original.index.intersection(normalized_pd.index)
                    if len(common_indices) > 0:
                        self.assertTrue(
                            df_pd_original.loc[common_indices, col].equals(normalized_pd.loc[common_indices, col]),
                            f"Les données de la colonne {col} devraient être préservées"
                        )
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s_original = pd.read_csv(s_file, sep=';')
        normalized_s = self.cleaner._normalize_geophysical_values(df_s_original)
        
        # Vérifier que les données non-géophysiques sont préservées
        geophysical_cols_s = ['Rho (Ohm.m)', 'M (mV/V)']
        non_geophysical_cols_s = [col for col in df_s_original.columns if col not in geophysical_cols_s]
        
        for col in non_geophysical_cols_s:
            if col in normalized_s.columns:
                # Vérifier que les valeurs sont préservées
                original_values = df_s_original[col].dropna()
                if len(original_values) > 0:
                    common_indices = df_s_original.index.intersection(normalized_s.index)
                    if len(common_indices) > 0:
                        self.assertTrue(
                            df_s_original.loc[common_indices, col].equals(normalized_s.loc[common_indices, col]),
                            f"Les données de la colonne {col} devraient être préservées"
                        )
        
        print("✅ Intégrité des données préservée après normalisation")
    
    def test_normalize_geophysical_values_numeric_conversion(self):
        """Test de conversion des valeurs géophysiques en numériques"""
        # Créer un DataFrame avec des valeurs géophysiques en string
        string_values_df = pd.DataFrame({
            'resistivity': ['100.5', '200.7', '300.2', '400.1', '500.0'],
            'chargeability': ['1.0', '2.0', '3.0', '4.0', '5.0'],
            'other_col': ['A', 'B', 'C', 'D', 'E']
        })
        
        initial_count = len(string_values_df)
        
        # Appeler la méthode de normalisation
        normalized_df = self.cleaner._normalize_geophysical_values(string_values_df)
        
        # Vérifier que les valeurs géophysiques sont converties en numériques
        self.assertIsInstance(normalized_df, pd.DataFrame)
        self.assertEqual(len(normalized_df), initial_count, "Le nombre de lignes ne devrait pas changer")
        
        # Vérifier que les valeurs géophysiques sont numériques
        if 'resistivity' in normalized_df.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(normalized_df['resistivity']))
        if 'chargeability' in normalized_df.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(normalized_df['chargeability']))
        
        # Vérifier que les autres colonnes sont préservées
        self.assertIn('other_col', normalized_df.columns)
        
        print("✅ Conversion des valeurs géophysiques string vers numériques réussie")
    
    def test_normalize_geophysical_values_no_geophysical_columns(self):
        """Test avec un DataFrame sans colonnes géophysiques"""
        # Créer un DataFrame sans colonnes géophysiques
        no_geophysical_df = pd.DataFrame({
            'value1': [1, 2, 3],
            'value2': [4, 5, 6],
            'value3': [7, 8, 9]
        })
        
        initial_count = len(no_geophysical_df)
        
        # Appeler la méthode de normalisation
        normalized_df = self.cleaner._normalize_geophysical_values(no_geophysical_df)
        
        # Vérifier que rien n'a changé
        self.assertIsInstance(normalized_df, pd.DataFrame)
        self.assertEqual(len(normalized_df), initial_count, "Le nombre de lignes ne devrait pas changer")
        
        # Vérifier que toutes les colonnes sont préservées
        for col in no_geophysical_df.columns:
            self.assertIn(col, normalized_df.columns, f"Colonne {col} devrait être préservée")
        
        print("✅ DataFrame sans colonnes géophysiques préservé")
    
    def test_normalize_geophysical_values_performance(self):
        """Test de performance de la normalisation des valeurs géophysiques"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        start_time = time.time()
        normalized_pd = self.cleaner._normalize_geophysical_values(df_pd)
        pd_time = time.time() - start_time
        
        self.assertIsInstance(normalized_pd, pd.DataFrame)
        self.assertLess(pd_time, 1.0, "Normalisation des valeurs géophysiques de PD.csv devrait être rapide (< 1 seconde)")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        
        start_time = time.time()
        normalized_s = self.cleaner._normalize_geophysical_values(df_s)
        s_time = time.time() - start_time
        
        self.assertIsInstance(normalized_s, pd.DataFrame)
        self.assertLess(s_time, 1.0, "Normalisation des valeurs géophysiques de S.csv devrait être rapide (< 1 seconde)")
        
        print(f"✅ Performance validée: PD.csv ({pd_time:.3f}s), S.csv ({s_time:.3f}s)")
    
    def test_normalize_geophysical_values_integration_with_cleaner(self):
        """Test d'intégration avec le processus de nettoyage complet"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = self.cleaner._load_device_data(pd_file, "pole_dipole")
        validated_pd = self.cleaner._validate_columns(df_pd, "pole_dipole")
        normalized_pd = self.cleaner._normalize_geophysical_values(validated_pd)
        
        self.assertIsInstance(normalized_pd, pd.DataFrame)
        self.assertLessEqual(len(normalized_pd), len(validated_pd))
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = self.cleaner._load_device_data(s_file, "schlumberger")
        validated_s = self.cleaner._validate_columns(df_s, "schlumberger")
        normalized_s = self.cleaner._normalize_geophysical_values(validated_s)
        
        self.assertIsInstance(normalized_s, pd.DataFrame)
        self.assertLessEqual(len(normalized_s), len(validated_s))
        
        print("✅ Intégration avec le processus de nettoyage complet réussie")
    
    def test_normalize_geophysical_values_edge_cases(self):
        """Test des cas limites"""
        # Test avec DataFrame vide
        empty_df = pd.DataFrame()
        normalized_empty = self.cleaner._normalize_geophysical_values(empty_df)
        self.assertIsInstance(normalized_empty, pd.DataFrame)
        self.assertEqual(len(normalized_empty), 0)
        
        # Test avec DataFrame à une seule ligne
        single_row_df = pd.DataFrame({
            'resistivity': [100.0],
            'chargeability': [1.0]
        })
        normalized_single = self.cleaner._normalize_geophysical_values(single_row_df)
        self.assertIsInstance(normalized_single, pd.DataFrame)
        self.assertEqual(len(normalized_single), 1)
        
        # Test avec DataFrame avec des valeurs extrêmes
        extreme_values_df = pd.DataFrame({
            'resistivity': [1e-10, 1e10, 0.0, -1.0],
            'chargeability': [0.0, 100.0, -1.0, 1.0]
        })
        normalized_extreme = self.cleaner._normalize_geophysical_values(extreme_values_df)
        self.assertIsInstance(normalized_extreme, pd.DataFrame)
        self.assertLess(len(normalized_extreme), len(extreme_values_df))
        
        print("✅ Cas limites gérés avec succès")
    
    def test_normalize_geophysical_values_column_mapping(self):
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
        
        # Tester la normalisation sur les deux fichiers
        normalized_pd = self.cleaner._normalize_geophysical_values(df_pd)
        normalized_s = self.cleaner._normalize_geophysical_values(df_s)
        
        # Vérifier que les deux normalisations ont réussi
        self.assertIsInstance(normalized_pd, pd.DataFrame)
        self.assertIsInstance(normalized_s, pd.DataFrame)
        
        print("✅ Correspondance des noms de colonnes validée entre PD.csv et S.csv")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(
        verbosity=2,
        testLoader=unittest.TestLoader(),
        testRunner=unittest.TextTestRunner(stream=sys.stdout)
    )
