#!/usr/bin/env python3
"""
Test unitaire pour la méthode _generate_synthetic_geophysical_data de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _generate_synthetic_geophysical_data
avec des paramètres réalistes et des données synthétiques.
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.preprocessor.data_cleaner import GeophysicalDataCleaner


class TestDataCleanerGenerateSyntheticGeophysicalDataRealData(unittest.TestCase):
    """Tests pour la méthode _generate_synthetic_geophysical_data avec données réalistes"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        # Utiliser les vrais fichiers de données du projet
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.test_dir = self.project_root / "test" / "fixtures"
        
        # Créer une instance du cleaner avec les vrais chemins
        with patch('backend.preprocessor.data_cleaner.CONFIG') as mock_config:
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
            shutil.rmtree(processed_dir)
    
    def test_generate_synthetic_geophysical_data_pole_dipole(self):
        """Test de génération de données géophysiques synthétiques pour Pole-Dipole"""
        num_samples = 1000
        device_type = "pole_dipole"
        
        # Appeler la méthode privée via reflection
        synthetic_df = self.cleaner._generate_synthetic_geophysical_data(num_samples, device_type)
        
        # Vérifications
        self.assertIsInstance(synthetic_df, pd.DataFrame)
        self.assertEqual(len(synthetic_df), num_samples)
        
        # Vérifier les colonnes requises
        required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        for col in required_columns:
            self.assertIn(col, synthetic_df.columns, f"Colonne {col} manquante")
        
        # Vérifier que les données ne sont pas vides
        for col in required_columns:
            self.assertFalse(synthetic_df[col].isna().all(), f"Colonne {col} ne devrait pas être entièrement NaN")
        
        # Vérifier les plages de valeurs
        self.assertGreater(synthetic_df['x'].min(), 0, "Coordonnées X devraient être > 0")
        self.assertGreater(synthetic_df['y'].min(), 0, "Coordonnées Y devraient être > 0")
        self.assertGreater(synthetic_df['z'].min(), 0, "Coordonnées Z devraient être > 0")
        self.assertGreater(synthetic_df['resistivity'].min(), 0, "Résistivité devrait être > 0")
        self.assertGreater(synthetic_df['chargeability'].min(), 0, "Chargeabilité devrait être > 0")
        
        print(f"✅ Données géophysiques synthétiques Pole-Dipole générées:")
        print(f"   Nombre d'échantillons: {len(synthetic_df)}")
        print(f"   Colonnes: {list(synthetic_df.columns)}")
        print(f"   X: {synthetic_df['x'].min():.2f} à {synthetic_df['x'].max():.2f}")
        print(f"   Y: {synthetic_df['y'].min():.2f} à {synthetic_df['y'].max():.2f}")
        print(f"   Z: {synthetic_df['z'].min():.2f} à {synthetic_df['z'].max():.2f}")
        print(f"   Résistivité: {synthetic_df['resistivity'].min():.2f} à {synthetic_df['resistivity'].max():.2f}")
        print(f"   Chargeabilité: {synthetic_df['chargeability'].min():.2f} à {synthetic_df['chargeability'].max():.2f}")
    
    def test_generate_synthetic_geophysical_data_schlumberger(self):
        """Test de génération de données géophysiques synthétiques pour Schlumberger"""
        num_samples = 500
        device_type = "schlumberger"
        
        # Appeler la méthode privée via reflection
        synthetic_df = self.cleaner._generate_synthetic_geophysical_data(num_samples, device_type)
        
        # Vérifications
        self.assertIsInstance(synthetic_df, pd.DataFrame)
        self.assertEqual(len(synthetic_df), num_samples)
        
        # Vérifier les colonnes requises
        required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
        for col in required_columns:
            self.assertIn(col, synthetic_df.columns, f"Colonne {col} manquante")
        
        # Vérifier que les données ne sont pas vides
        for col in required_columns:
            self.assertFalse(synthetic_df[col].isna().all(), f"Colonne {col} ne devrait pas être entièrement NaN")
        
        # Vérifier les plages de valeurs
        self.assertGreater(synthetic_df['x'].min(), 0, "Coordonnées X devraient être > 0")
        self.assertGreater(synthetic_df['y'].min(), 0, "Coordonnées Y devraient être > 0")
        self.assertGreater(synthetic_df['z'].min(), 0, "Coordonnées Z devraient être > 0")
        self.assertGreater(synthetic_df['resistivity'].min(), 0, "Résistivité devrait être > 0")
        self.assertGreater(synthetic_df['chargeability'].min(), 0, "Chargeabilité devrait être > 0")
        
        print(f"✅ Données géophysiques synthétiques Schlumberger générées:")
        print(f"   Nombre d'échantillons: {len(synthetic_df)}")
        print(f"   Colonnes: {list(synthetic_df.columns)}")
        print(f"   X: {synthetic_df['x'].min():.2f} à {synthetic_df['x'].max():.2f}")
        print(f"   Y: {synthetic_df['y'].min():.2f} à {synthetic_df['y'].max():.2f}")
        print(f"   Z: {synthetic_df['z'].min():.2f} à {synthetic_df['z'].max():.2f}")
        print(f"   Résistivité: {synthetic_df['resistivity'].min():.2f} à {synthetic_df['resistivity'].max():.2f}")
        print(f"   Chargeabilité: {synthetic_df['chargeability'].min():.2f} à {synthetic_df['chargeability'].max():.2f}")
    
    def test_generate_synthetic_geophysical_data_different_sample_sizes(self):
        """Test de génération avec différentes tailles d'échantillons"""
        device_type = "pole_dipole"
        sample_sizes = [10, 100, 500, 1000]
        
        for num_samples in sample_sizes:
            with self.subTest(num_samples=num_samples):
                synthetic_df = self.cleaner._generate_synthetic_geophysical_data(num_samples, device_type)
                
                # Vérifications
                self.assertIsInstance(synthetic_df, pd.DataFrame)
                self.assertEqual(len(synthetic_df), num_samples)
                
                # Vérifier que toutes les colonnes sont présentes
                required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
                for col in required_columns:
                    self.assertIn(col, synthetic_df.columns)
                
                print(f"✅ {num_samples} échantillons géophysiques générés correctement")
    
    def test_generate_synthetic_geophysical_data_data_types(self):
        """Test des types de données générées"""
        num_samples = 1000
        device_type = "pole_dipole"
        
        synthetic_df = self.cleaner._generate_synthetic_geophysical_data(num_samples, device_type)
        
        # Vérifier les types de données
        self.assertTrue(pd.api.types.is_numeric_dtype(synthetic_df['x']), "X devrait être numérique")
        self.assertTrue(pd.api.types.is_numeric_dtype(synthetic_df['y']), "Y devrait être numérique")
        self.assertTrue(pd.api.types.is_numeric_dtype(synthetic_df['z']), "Z devrait être numérique")
        self.assertTrue(pd.api.types.is_numeric_dtype(synthetic_df['resistivity']), "Résistivité devrait être numérique")
        self.assertTrue(pd.api.types.is_numeric_dtype(synthetic_df['chargeability']), "Chargeabilité devrait être numérique")
        
        # Vérifier qu'il n'y a pas de NaN
        for col in synthetic_df.columns:
            self.assertFalse(synthetic_df[col].isna().any(), f"Colonne {col} ne devrait pas contenir de NaN")
        
        print(f"✅ Types de données vérifiés:")
        print(f"   Types: {synthetic_df.dtypes.to_dict()}")
    
    def test_generate_synthetic_geophysical_data_statistical_properties(self):
        """Test des propriétés statistiques des données générées"""
        num_samples = 1000
        device_type = "pole_dipole"
        
        synthetic_df = self.cleaner._generate_synthetic_geophysical_data(num_samples, device_type)
        
        # Vérifier les propriétés statistiques
        for col in ['x', 'y', 'z', 'resistivity', 'chargeability']:
            col_data = synthetic_df[col]
            
            # Vérifier que les valeurs sont dans des plages raisonnables
            self.assertGreater(col_data.min(), 0, f"{col}: valeurs trop faibles")
            
            # Limites spécifiques par type de colonne (basées sur la méthode _generate_synthetic_geophysical_data)
            if col == 'x':
                # Coordonnées UTM X : plage 500000-510000
                self.assertLess(col_data.max(), 510000, f"{col}: valeurs trop élevées pour coordonnées UTM X")
                self.assertGreater(col_data.min(), 500000, f"{col}: valeurs trop faibles pour coordonnées UTM X")
            elif col == 'y':
                # Coordonnées UTM Y : plage 450000-460000
                self.assertLess(col_data.max(), 460000, f"{col}: valeurs trop élevées pour coordonnées UTM Y")
                self.assertGreater(col_data.min(), 450000, f"{col}: valeurs trop faibles pour coordonnées UTM Y")
            elif col == 'z':
                # Profondeur : plage 500-600
                self.assertLess(col_data.max(), 600, f"{col}: valeurs trop élevées pour profondeur")
                self.assertGreater(col_data.min(), 500, f"{col}: valeurs trop faibles pour profondeur")
            elif col == 'resistivity':
                # Résistivité : distribution lognormale, plage 0-200
                self.assertLess(col_data.max(), 200, f"{col}: valeurs trop élevées pour résistivité")
            elif col == 'chargeability':
                # Chargeabilité : distribution exponentielle, plage 0-400
                self.assertLess(col_data.max(), 400, f"{col}: valeurs trop élevées pour chargeabilité")
            
            # Vérifier la variance (doit être > 0)
            variance = col_data.var()
            self.assertGreater(variance, 0, f"{col}: variance devrait être > 0")
            
            # Vérifier que les valeurs ne sont pas toutes identiques
            unique_values = col_data.nunique()
            self.assertGreater(unique_values, 1, f"{col}: devrait avoir plus d'une valeur unique")
            
            print(f"   {col}: Min={col_data.min():.2f}, Max={col_data.max():.2f}, Mean={col_data.mean():.2f}, Std={col_data.std():.2f}")
    
    def test_generate_synthetic_geophysical_data_consistency(self):
        """Test de cohérence de la génération"""
        num_samples = 500
        device_type = "schlumberger"
        
        # Générer les données deux fois
        df1 = self.cleaner._generate_synthetic_geophysical_data(num_samples, device_type)
        df2 = self.cleaner._generate_synthetic_geophysical_data(num_samples, device_type)
        
        # Vérifier que les structures sont identiques
        self.assertEqual(df1.shape, df2.shape)
        self.assertEqual(list(df1.columns), list(df2.columns))
        
        # Les données devraient être différentes (génération aléatoire)
        for col in df1.columns:
            self.assertFalse(df1[col].equals(df2[col]), f"Colonne {col}: les données devraient être différentes")
        
        print(f"✅ Cohérence de la génération vérifiée")
    
    def test_generate_synthetic_geophysical_data_device_specific_characteristics(self):
        """Test des caractéristiques spécifiques par dispositif"""
        num_samples = 1000
        
        # Tester les deux types de dispositifs
        device_types = ["pole_dipole", "schlumberger"]
        
        for device_type in device_types:
            with self.subTest(device_type=device_type):
                synthetic_df = self.cleaner._generate_synthetic_geophysical_data(num_samples, device_type)
                
                # Vérifier que les données sont générées
                self.assertEqual(len(synthetic_df), num_samples)
                
                # Vérifier que toutes les colonnes sont présentes
                required_columns = ['x', 'y', 'z', 'resistivity', 'chargeability']
                for col in required_columns:
                    self.assertIn(col, synthetic_df.columns)
                
                # Vérifier que les données ne sont pas vides
                for col in required_columns:
                    self.assertFalse(synthetic_df[col].isna().all())
                
                print(f"✅ Caractéristiques {device_type} vérifiées")
    
    def test_generate_synthetic_geophysical_data_edge_cases(self):
        """Test des cas limites"""
        device_type = "pole_dipole"
        
        # Test avec un seul échantillon
        df_single = self.cleaner._generate_synthetic_geophysical_data(1, device_type)
        self.assertEqual(len(df_single), 1)
        
        # Test avec un grand nombre d'échantillons
        df_large = self.cleaner._generate_synthetic_geophysical_data(10000, device_type)
        self.assertEqual(len(df_large), 10000)
        
        print(f"✅ Cas limites gérés correctement:")
        print(f"   1 échantillon: {df_single.shape}")
        print(f"   10000 échantillons: {df_large.shape}")
    
    def test_generate_synthetic_geophysical_data_geophysical_realism(self):
        """Test du réalisme géophysique des données générées"""
        num_samples = 1000
        device_type = "pole_dipole"
        
        synthetic_df = self.cleaner._generate_synthetic_geophysical_data(num_samples, device_type)
        
        # Vérifier que les valeurs géophysiques sont réalistes
        # Résistivité: généralement entre 1 et 10000 Ω⋅m
        resistivity = synthetic_df['resistivity']
        self.assertGreater(resistivity.min(), 0.1, "Résistivité trop faible")
        self.assertLess(resistivity.max(), 50000, "Résistivité trop élevée")
        
        # Chargeabilité: généralement entre 0 et 100 mV/V
        chargeability = synthetic_df['chargeability']
        self.assertGreaterEqual(chargeability.min(), 0, "Chargeabilité ne devrait pas être négative")
        self.assertLess(chargeability.max(), 1000, "Chargeabilité trop élevée")
        
        # Coordonnées: devraient être dans des plages raisonnables
        x_coords = synthetic_df['x']
        y_coords = synthetic_df['y']
        z_coords = synthetic_df['z']
        
        self.assertGreater(x_coords.min(), 0, "Coordonnées X devraient être > 0")
        self.assertGreater(y_coords.min(), 0, "Coordonnées Y devraient être > 0")
        self.assertGreater(z_coords.min(), 0, "Coordonnées Z devraient être > 0")
        
        print(f"✅ Réalisme géophysique vérifié:")
        print(f"   Résistivité: {resistivity.min():.2f} à {resistivity.max():.2f} Ω⋅m")
        print(f"   Chargeabilité: {chargeability.min():.2f} à {chargeability.max():.2f} mV/V")
        print(f"   Coordonnées X: {x_coords.min():.2f} à {x_coords.max():.2f}")
        print(f"   Coordonnées Y: {y_coords.min():.2f} à {y_coords.max():.2f}")
        print(f"   Coordonnées Z: {z_coords.min():.2f} à {z_coords.max():.2f}")
    
    def test_generate_synthetic_geophysical_data_correlation_analysis(self):
        """Test de l'analyse de corrélation des données générées"""
        num_samples = 1000
        device_type = "pole_dipole"
        
        synthetic_df = self.cleaner._generate_synthetic_geophysical_data(num_samples, device_type)
        
        # Calculer les corrélations
        correlations = synthetic_df[['x', 'y', 'z', 'resistivity', 'chargeability']].corr()
        
        # Vérifier que les corrélations sont dans des plages raisonnables
        for i in range(len(correlations.columns)):
            for j in range(len(correlations.columns)):
                if i != j:  # Pas d'auto-corrélation
                    corr_value = correlations.iloc[i, j]
                    self.assertGreaterEqual(corr_value, -1, f"Corrélation {i},{j} trop faible")
                    self.assertLessEqual(corr_value, 1, f"Corrélation {i},{j} trop élevée")
        
        print(f"✅ Analyse de corrélation vérifiée:")
        print(f"   Matrice de corrélation calculée: {correlations.shape}")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(verbosity=2)
