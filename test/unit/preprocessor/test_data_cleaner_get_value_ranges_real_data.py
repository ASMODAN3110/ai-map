#!/usr/bin/env python3
"""
Test unitaire pour la méthode _get_value_ranges de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _get_value_ranges
avec des données réelles extraites des fichiers PD.csv et S.csv.
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


class TestDataCleanerGetValueRangesRealData(unittest.TestCase):
    """Tests pour la méthode _get_value_ranges avec données réelles"""
    
    def setUp(self):
        """Configuration avant chaque test avec données réelles"""
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
        
        # Charger les vraies données pour les tests
        self.pd_file = self.raw_data_dir / "PD.csv"
        self.s_file = self.raw_data_dir / "S.csv"
        
        if self.pd_file.exists():
            self.pd_df = pd.read_csv(self.pd_file, sep=';')
        else:
            self.pd_df = None
            
        if self.s_file.exists():
            self.s_df = pd.read_csv(self.s_file, sep=';')
        else:
            self.s_df = None
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Ne supprimer que le dossier processed temporaire
        processed_dir = self.test_dir / "processed"
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
    
    def test_get_value_ranges_pd_csv_real(self):
        """Test de calcul des plages de valeurs avec les vraies données PD.csv"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Appeler la méthode de calcul des plages de valeurs
        value_ranges = self.cleaner._get_value_ranges(self.pd_df)
        
        # Vérifications
        self.assertIsInstance(value_ranges, dict)
        
        # Vérifier les colonnes de résistivité si elles existent
        resistivity_cols = ['Rho(ohm.m)', 'resistivity']
        for col in resistivity_cols:
            if col in self.pd_df.columns:
                self.assertIn('resistivity', value_ranges)
                resistivity_range = value_ranges['resistivity']
                self.assertIn('min', resistivity_range)
                self.assertIn('max', resistivity_range)
                self.assertIn('mean', resistivity_range)
                
                # Vérifier les valeurs
                self.assertLess(resistivity_range['min'], resistivity_range['max'])
                self.assertLessEqual(resistivity_range['min'], resistivity_range['mean'])
                self.assertLessEqual(resistivity_range['mean'], resistivity_range['max'])
                
                # Vérifier la cohérence avec les données réelles
                real_min = self.pd_df[col].min()
                real_max = self.pd_df[col].max()
                real_mean = self.pd_df[col].mean()
                
                self.assertAlmostEqual(resistivity_range['min'], real_min, places=2)
                self.assertAlmostEqual(resistivity_range['max'], real_max, places=2)
                self.assertAlmostEqual(resistivity_range['mean'], real_mean, places=2)
                
                print(f"✅ Plages de résistivité PD.csv:")
                print(f"   Min: {resistivity_range['min']:.2f} Ω⋅m")
                print(f"   Max: {resistivity_range['max']:.2f} Ω⋅m")
                print(f"   Moyenne: {resistivity_range['mean']:.2f} Ω⋅m")
                break
        
        # Vérifier les colonnes de chargeabilité si elles existent
        chargeability_cols = ['M (mV/V)', 'chargeability']
        for col in chargeability_cols:
            if col in self.pd_df.columns:
                self.assertIn('chargeability', value_ranges)
                chargeability_range = value_ranges['chargeability']
                self.assertIn('min', chargeability_range)
                self.assertIn('max', chargeability_range)
                self.assertIn('mean', chargeability_range)
                
                # Vérifier les valeurs
                self.assertLess(chargeability_range['min'], chargeability_range['max'])
                self.assertLessEqual(chargeability_range['min'], chargeability_range['mean'])
                self.assertLessEqual(chargeability_range['mean'], chargeability_range['max'])
                
                # Vérifier la cohérence avec les données réelles
                real_min = self.pd_df[col].min()
                real_max = self.pd_df[col].max()
                real_mean = self.pd_df[col].mean()
                
                self.assertAlmostEqual(chargeability_range['min'], real_min, places=2)
                self.assertAlmostEqual(chargeability_range['max'], real_max, places=2)
                self.assertAlmostEqual(chargeability_range['mean'], real_mean, places=2)
                
                print(f"✅ Plages de chargeabilité PD.csv:")
                print(f"   Min: {chargeability_range['min']:.2f} mV/V")
                print(f"   Max: {chargeability_range['max']:.2f} mV/V")
                print(f"   Moyenne: {chargeability_range['mean']:.2f} mV/V")
                break
    
    def test_get_value_ranges_s_csv_real(self):
        """Test de calcul des plages de valeurs avec les vraies données S.csv"""
        if self.s_df is None:
            self.skipTest("Fichier S.csv non trouvé")
        
        # Appeler la méthode de calcul des plages de valeurs
        value_ranges = self.cleaner._get_value_ranges(self.s_df)
        
        # Vérifications
        self.assertIsInstance(value_ranges, dict)
        
        # Vérifier les colonnes de résistivité si elles existent
        resistivity_cols = ['Rho (Ohm.m)', 'resistivity']
        for col in resistivity_cols:
            if col in self.s_df.columns:
                self.assertIn('resistivity', value_ranges)
                resistivity_range = value_ranges['resistivity']
                self.assertIn('min', resistivity_range)
                self.assertIn('max', resistivity_range)
                self.assertIn('mean', resistivity_range)
                
                # Vérifier les valeurs
                self.assertLess(resistivity_range['min'], resistivity_range['max'])
                self.assertLessEqual(resistivity_range['min'], resistivity_range['mean'])
                self.assertLessEqual(resistivity_range['mean'], resistivity_range['max'])
                
                print(f"✅ Plages de résistivité S.csv:")
                print(f"   Min: {resistivity_range['min']:.2f} Ω⋅m")
                print(f"   Max: {resistivity_range['max']:.2f} Ω⋅m")
                print(f"   Moyenne: {resistivity_range['mean']:.2f} Ω⋅m")
                break
        
        # Vérifier les colonnes de chargeabilité si elles existent
        chargeability_cols = ['M (mV/V)', 'chargeability']
        for col in chargeability_cols:
            if col in self.s_df.columns:
                self.assertIn('chargeability', value_ranges)
                chargeability_range = value_ranges['chargeability']
                self.assertIn('min', chargeability_range)
                self.assertIn('max', chargeability_range)
                self.assertIn('mean', chargeability_range)
                
                # Vérifier les valeurs
                self.assertLess(chargeability_range['min'], chargeability_range['max'])
                self.assertLessEqual(chargeability_range['min'], chargeability_range['mean'])
                self.assertLessEqual(chargeability_range['mean'], chargeability_range['max'])
                
                print(f"✅ Plages de chargeabilité S.csv:")
                print(f"   Min: {chargeability_range['min']:.2f} mV/V")
                print(f"   Max: {chargeability_range['max']:.2f} mV/V")
                print(f"   Moyenne: {chargeability_range['mean']:.2f} mV/V")
                break
    
    def test_get_value_ranges_empty_dataframe(self):
        """Test de calcul des plages de valeurs avec un DataFrame vide"""
        empty_df = pd.DataFrame(columns=['resistivity', 'chargeability'])
        
        # Appeler la méthode de calcul des plages de valeurs
        value_ranges = self.cleaner._get_value_ranges(empty_df)
        
        # Vérifications
        self.assertIsInstance(value_ranges, dict)
        # La méthode retourne un dictionnaire avec les colonnes même si le DataFrame est vide
        self.assertEqual(len(value_ranges), 2, "Plages de valeurs devrait contenir 2 colonnes même pour un DataFrame vide")
        
        # Vérifier que les valeurs sont NaN pour un DataFrame vide
        for col in ['resistivity', 'chargeability']:
            self.assertIn(col, value_ranges)
            self.assertTrue(pd.isna(value_ranges[col]['min']))
            self.assertTrue(pd.isna(value_ranges[col]['max']))
            self.assertTrue(pd.isna(value_ranges[col]['mean']))
        
        print(f"✅ Plages de valeurs DataFrame vide gérées correctement")
    
    def test_get_value_ranges_missing_columns(self):
        """Test de calcul des plages de valeurs avec des colonnes manquantes"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Créer un DataFrame sans colonnes géophysiques
        df_no_geophysical = self.pd_df.drop(columns=['Rho(ohm.m)', 'M (mV/V)'], errors='ignore')
        
        # Appeler la méthode de calcul des plages de valeurs
        value_ranges = self.cleaner._get_value_ranges(df_no_geophysical)
        
        # Vérifications
        self.assertIsInstance(value_ranges, dict)
        self.assertEqual(len(value_ranges), 0, "Plages de valeurs devrait être vide sans colonnes géophysiques")
        
        print(f"✅ Plages de valeurs sans colonnes géophysiques gérées correctement")
    
    def test_get_value_ranges_with_nan_values(self):
        """Test de calcul des plages de valeurs avec des valeurs NaN"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Créer un DataFrame avec des NaN
        df_with_nan = self.pd_df.copy()
        if 'Rho(ohm.m)' in df_with_nan.columns:
            df_with_nan.loc[0, 'Rho(ohm.m)'] = np.nan
            df_with_nan.loc[1, 'Rho(ohm.m)'] = np.nan
        
        # Appeler la méthode de calcul des plages de valeurs
        value_ranges = self.cleaner._get_value_ranges(df_with_nan)
        
        # Vérifications
        self.assertIsInstance(value_ranges, dict)
        
        # Vérifier que les NaN sont gérés correctement
        if 'resistivity' in value_ranges:
            resistivity_range = value_ranges['resistivity']
            self.assertFalse(np.isnan(resistivity_range['min']))
            self.assertFalse(np.isnan(resistivity_range['max']))
            self.assertFalse(np.isnan(resistivity_range['mean']))
        
        print(f"✅ Plages de valeurs avec NaN gérées correctement")
    
    def test_get_value_ranges_precision(self):
        """Test de précision du calcul des plages de valeurs"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Appeler la méthode de calcul des plages de valeurs
        value_ranges = self.cleaner._get_value_ranges(self.pd_df)
        
        # Vérifier la précision pour la résistivité
        if 'resistivity' in value_ranges:
            resistivity_range = value_ranges['resistivity']
            
            # Vérifier que les calculs sont précis
            real_min = self.pd_df['Rho(ohm.m)'].min()
            real_max = self.pd_df['Rho(ohm.m)'].max()
            real_mean = self.pd_df['Rho(ohm.m)'].mean()
            
            self.assertAlmostEqual(resistivity_range['min'], real_min, places=10)
            self.assertAlmostEqual(resistivity_range['max'], real_max, places=10)
            self.assertAlmostEqual(resistivity_range['mean'], real_mean, places=10)
        
        # Vérifier la précision pour la chargeabilité
        if 'chargeability' in value_ranges:
            chargeability_range = value_ranges['chargeability']
            
            # Vérifier que les calculs sont précis
            real_min = self.pd_df['M (mV/V)'].min()
            real_max = self.pd_df['M (mV/V)'].max()
            real_mean = self.pd_df['M (mV/V)'].mean()
            
            self.assertAlmostEqual(chargeability_range['min'], real_min, places=10)
            self.assertAlmostEqual(chargeability_range['max'], real_max, places=10)
            self.assertAlmostEqual(chargeability_range['mean'], real_mean, places=10)
        
        print(f"✅ Précision du calcul des plages de valeurs vérifiée (10 décimales)")
    
    def test_get_value_ranges_statistical_properties(self):
        """Test des propriétés statistiques des plages de valeurs"""
        if self.pd_df is None:
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Appeler la méthode de calcul des plages de valeurs
        value_ranges = self.cleaner._get_value_ranges(self.pd_df)
        
        # Vérifier les propriétés statistiques
        for param, range_info in value_ranges.items():
            with self.subTest(parameter=param):
                # Vérifier que min ≤ mean ≤ max
                self.assertLessEqual(range_info['min'], range_info['mean'])
                self.assertLessEqual(range_info['mean'], range_info['max'])
                
                # Vérifier que les valeurs sont numériques
                self.assertIsInstance(range_info['min'], (int, float))
                self.assertIsInstance(range_info['max'], (int, float))
                self.assertIsInstance(range_info['mean'], (int, float))
                
                # Vérifier que les valeurs ne sont pas NaN
                self.assertFalse(np.isnan(range_info['min']))
                self.assertFalse(np.isnan(range_info['max']))
                self.assertFalse(np.isnan(range_info['mean']))
                
                print(f"✅ Propriétés statistiques {param}: min={range_info['min']:.2f}, mean={range_info['mean']:.2f}, max={range_info['max']:.2f}")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(verbosity=2)
