#!/usr/bin/env python3
"""
Test unitaire pour la méthode _get_value_ranges de GeophysicalDataCleaner

Ce test vérifie le bon fonctionnement de la méthode _get_value_ranges
qui calcule les plages de valeurs géophysiques (résistivité et chargeabilité)
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


class TestDataCleanerGetValueRanges(unittest.TestCase):
    """Tests pour la méthode _get_value_ranges de GeophysicalDataCleaner"""
    
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
    
    def test_get_value_ranges_pd_csv(self):
        """Test de calcul des plages de valeurs avec PD.csv"""
        # Charger PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        self.assertTrue(pd_file.exists(), f"Le fichier {pd_file} n'existe pas")
        
        df_pd = pd.read_csv(pd_file, sep=';')
        self.assertIsInstance(df_pd, pd.DataFrame)
        self.assertGreater(len(df_pd), 0, "PD.csv ne devrait pas être vide")
        
        # Vérifier que les colonnes géophysiques sont présentes
        # PD.csv utilise 'Rho(ohm.m)' et 'M (mV/V)'
        self.assertIn('Rho(ohm.m)', df_pd.columns, "Colonne 'Rho(ohm.m)' manquante dans PD.csv")
        self.assertIn('M (mV/V)', df_pd.columns, "Colonne 'M (mV/V)' manquante dans PD.csv")
        
        # Créer un DataFrame avec les noms de colonnes standardisés pour le test
        df_pd_test = df_pd.copy()
        df_pd_test['resistivity'] = df_pd_test['Rho(ohm.m)']
        df_pd_test['chargeability'] = df_pd_test['M (mV/V)']
        
        # Appeler la méthode de calcul des plages de valeurs
        value_ranges = self.cleaner._get_value_ranges(df_pd_test)
        
        # Vérifications
        self.assertIsInstance(value_ranges, dict)
        self.assertGreater(len(value_ranges), 0, "Les plages de valeurs ne devraient pas être vides")
        
        # Vérifier que les clés sont présentes
        self.assertIn('resistivity', value_ranges, "Clé 'resistivity' manquante dans les plages de valeurs")
        self.assertIn('chargeability', value_ranges, "Clé 'chargeability' manquante dans les plages de valeurs")
        
        # Vérifier la structure de la résistivité
        resistivity_data = value_ranges['resistivity']
        self.assertIn('min', resistivity_data, "Clé 'min' manquante pour la résistivité")
        self.assertIn('max', resistivity_data, "Clé 'max' manquante pour la résistivité")
        self.assertIn('mean', resistivity_data, "Clé 'mean' manquante pour la résistivité")
        
        # Vérifier la structure de la chargeabilité
        chargeability_data = value_ranges['chargeability']
        self.assertIn('min', chargeability_data, "Clé 'min' manquante pour la chargeabilité")
        self.assertIn('max', chargeability_data, "Clé 'max' manquante pour la chargeabilité")
        self.assertIn('mean', chargeability_data, "Clé 'mean' manquante pour la chargeabilité")
        
        # Vérifier que les valeurs sont numériques
        for key in ['min', 'max', 'mean']:
            self.assertTrue(np.issubdtype(type(resistivity_data[key]), np.number), 
                          f"Valeur {key} de la résistivité devrait être numérique")
            self.assertTrue(np.issubdtype(type(chargeability_data[key]), np.number), 
                          f"Valeur {key} de la chargeabilité devrait être numérique")
        
        # Vérifier la cohérence des valeurs
        self.assertLessEqual(resistivity_data['min'], resistivity_data['max'], 
                            "min de résistivité devrait être <= max")
        self.assertLessEqual(chargeability_data['min'], chargeability_data['max'], 
                            "min de chargeabilité devrait être <= max")
        
        self.assertLessEqual(resistivity_data['min'], resistivity_data['mean'], 
                            "min de résistivité devrait être <= mean")
        self.assertGreaterEqual(resistivity_data['max'], resistivity_data['mean'], 
                               "max de résistivité devrait être >= mean")
        
        self.assertLessEqual(chargeability_data['min'], chargeability_data['mean'], 
                            "min de chargeabilité devrait être <= mean")
        self.assertGreaterEqual(chargeability_data['max'], chargeability_data['mean'], 
                               "max de chargeabilité devrait être >= mean")
        
        print(f"✅ Plages de valeurs de PD.csv calculées:")
        print(f"   Résistivité: {resistivity_data['min']:.2f} - {resistivity_data['max']:.2f} (moy: {resistivity_data['mean']:.2f})")
        print(f"   Chargeabilité: {chargeability_data['min']:.4f} - {chargeability_data['max']:.4f} (moy: {chargeability_data['mean']:.4f})")
    
    def test_get_value_ranges_s_csv(self):
        """Test de calcul des plages de valeurs avec S.csv"""
        # Charger S.csv
        s_file = self.raw_data_dir / "S.csv"
        self.assertTrue(s_file.exists(), f"Le fichier {s_file} n'existe pas")
        
        df_s = pd.read_csv(s_file, sep=';')
        self.assertIsInstance(df_s, pd.DataFrame)
        self.assertGreater(len(df_s), 0, "S.csv ne devrait pas être vide")
        
        # Vérifier que les colonnes géophysiques sont présentes
        # S.csv utilise 'Rho (Ohm.m)' et 'M (mV/V)'
        self.assertIn('Rho (Ohm.m)', df_s.columns, "Colonne 'Rho (Ohm.m)' manquante dans S.csv")
        self.assertIn('M (mV/V)', df_s.columns, "Colonne 'M (mV/V)' manquante dans S.csv")
        
        # Créer un DataFrame avec les noms de colonnes standardisés pour le test
        df_s_test = df_s.copy()
        df_s_test['resistivity'] = df_s_test['Rho (Ohm.m)']
        df_s_test['chargeability'] = df_s_test['M (mV/V)']
        
        # Appeler la méthode de calcul des plages de valeurs
        value_ranges = self.cleaner._get_value_ranges(df_s_test)
        
        # Vérifications
        self.assertIsInstance(value_ranges, dict)
        self.assertGreater(len(value_ranges), 0, "Les plages de valeurs ne devraient pas être vides")
        
        # Vérifier que les clés sont présentes
        self.assertIn('resistivity', value_ranges, "Clé 'resistivity' manquante dans les plages de valeurs")
        self.assertIn('chargeability', value_ranges, "Clé 'chargeability' manquante dans les plages de valeurs")
        
        # Vérifier la structure de la résistivité
        resistivity_data = value_ranges['resistivity']
        self.assertIn('min', resistivity_data, "Clé 'min' manquante pour la résistivité")
        self.assertIn('max', resistivity_data, "Clé 'max' manquante pour la résistivité")
        self.assertIn('mean', resistivity_data, "Clé 'mean' manquante pour la résistivité")
        
        # Vérifier la structure de la chargeabilité
        chargeability_data = value_ranges['chargeability']
        self.assertIn('min', chargeability_data, "Clé 'min' manquante pour la chargeabilité")
        self.assertIn('max', chargeability_data, "Clé 'max' manquante pour la chargeabilité")
        self.assertIn('mean', chargeability_data, "Clé 'mean' manquante pour la chargeabilité")
        
        # Vérifier que les valeurs sont numériques
        for key in ['min', 'max', 'mean']:
            self.assertTrue(np.issubdtype(type(resistivity_data[key]), np.number), 
                          f"Valeur {key} de la résistivité devrait être numérique")
            self.assertTrue(np.issubdtype(type(chargeability_data[key]), np.number), 
                          f"Valeur {key} de la chargeabilité devrait être numérique")
        
        # Vérifier la cohérence des valeurs
        self.assertLessEqual(resistivity_data['min'], resistivity_data['max'], 
                            "min de résistivité devrait être <= max")
        self.assertLessEqual(chargeability_data['min'], chargeability_data['max'], 
                            "min de chargeabilité devrait être <= max")
        
        self.assertLessEqual(resistivity_data['min'], resistivity_data['mean'], 
                            "min de résistivité devrait être <= mean")
        self.assertGreaterEqual(resistivity_data['max'], resistivity_data['mean'], 
                               "max de résistivité devrait être >= mean")
        
        self.assertLessEqual(chargeability_data['min'], chargeability_data['mean'], 
                            "min de chargeabilité devrait être <= mean")
        self.assertGreaterEqual(chargeability_data['max'], chargeability_data['mean'], 
                               "max de chargeabilité devrait être >= mean")
        
        print(f"✅ Plages de valeurs de S.csv calculées:")
        print(f"   Résistivité: {resistivity_data['min']:.2f} - {resistivity_data['max']:.2f} (moy: {resistivity_data['mean']:.2f})")
        print(f"   Chargeabilité: {chargeability_data['min']:.4f} - {chargeability_data['max']:.4f} (moy: {chargeability_data['mean']:.4f})")
    
    def test_get_value_ranges_both_files(self):
        """Test de calcul des plages de valeurs des deux fichiers CSV ensemble"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        df_pd_test = df_pd.copy()
        df_pd_test['resistivity'] = df_pd_test['Rho(ohm.m)']
        df_pd_test['chargeability'] = df_pd_test['M (mV/V)']
        ranges_pd = self.cleaner._get_value_ranges(df_pd_test)
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        df_s_test = df_s.copy()
        df_s_test['resistivity'] = df_s_test['Rho (Ohm.m)']
        df_s_test['chargeability'] = df_s_test['M (mV/V)']
        ranges_s = self.cleaner._get_value_ranges(df_s_test)
        
        # Vérifier que les deux calculs ont réussi
        self.assertIsInstance(ranges_pd, dict)
        self.assertIsInstance(ranges_s, dict)
        
        # Les deux fichiers devraient avoir des plages de valeurs
        self.assertGreater(len(ranges_pd), 0, "PD.csv devrait avoir des plages de valeurs calculées")
        self.assertGreater(len(ranges_s), 0, "S.csv devrait avoir des plages de valeurs calculées")
        
        # Vérifier que les deux ont les mêmes clés
        self.assertEqual(set(ranges_pd.keys()), set(ranges_s.keys()), 
                        "Les deux fichiers devraient avoir les mêmes types de mesures")
        
        # Comparer les plages de valeurs
        pd_resistivity = ranges_pd['resistivity']
        s_resistivity = ranges_s['resistivity']
        pd_chargeability = ranges_pd['chargeability']
        s_chargeability = ranges_s['chargeability']
        
        print(f"✅ Plages de valeurs comparées:")
        print(f"   PD.csv - Résistivité: {pd_resistivity['min']:.2f} - {pd_resistivity['max']:.2f}")
        print(f"   S.csv  - Résistivité: {s_resistivity['min']:.2f} - {s_resistivity['max']:.2f}")
        print(f"   PD.csv - Chargeabilité: {pd_chargeability['min']:.4f} - {pd_chargeability['max']:.4f}")
        print(f"   S.csv  - Chargeabilité: {s_chargeability['min']:.4f} - {s_chargeability['max']:.4f}")
    
    def test_get_value_ranges_data_validation(self):
        """Test de validation des données et calcul des plages de valeurs"""
        # Créer un DataFrame avec des valeurs géophysiques connues
        test_df = pd.DataFrame({
            'resistivity': [10.0, 20.0, 30.0, 40.0, 50.0],
            'chargeability': [0.001, 0.002, 0.003, 0.004, 0.005],
            'x': [0, 10, 20, 30, 40],
            'y': [0, 5, 10, 15, 20]
        })
        
        # Appeler la méthode de calcul des plages de valeurs
        value_ranges = self.cleaner._get_value_ranges(test_df)
        
        # Vérifier que les plages de valeurs ont été calculées
        self.assertIsInstance(value_ranges, dict)
        self.assertGreater(len(value_ranges), 0)
        
        # Vérifier les valeurs attendues
        self.assertEqual(value_ranges['resistivity']['min'], 10.0)
        self.assertEqual(value_ranges['resistivity']['max'], 50.0)
        self.assertEqual(value_ranges['resistivity']['mean'], 30.0)
        
        self.assertEqual(value_ranges['chargeability']['min'], 0.001)
        self.assertEqual(value_ranges['chargeability']['max'], 0.005)
        self.assertEqual(value_ranges['chargeability']['mean'], 0.003)
        
        print(f"✅ Validation des données et calcul des plages de valeurs: Résistivité {value_ranges['resistivity']['min']}-{value_ranges['resistivity']['max']}, Chargeabilité {value_ranges['chargeability']['min']}-{value_ranges['chargeability']['max']}")
    
    def test_get_value_ranges_no_geophysical_columns(self):
        """Test avec un DataFrame sans colonnes géophysiques"""
        # Créer un DataFrame sans colonnes géophysiques
        no_geo_df = pd.DataFrame({
            'value1': [1, 2, 3, 4, 5],
            'value2': [6, 7, 8, 9, 10],
            'value3': [11, 12, 13, 14, 15]
        })
        
        # Appeler la méthode de calcul des plages de valeurs
        value_ranges = self.cleaner._get_value_ranges(no_geo_df)
        
        # Vérifier que les plages de valeurs sont vides
        self.assertIsInstance(value_ranges, dict)
        self.assertEqual(len(value_ranges), 0, "Les plages de valeurs devraient être vides sans colonnes géophysiques")
        
        print("✅ DataFrame sans colonnes géophysiques traité correctement")
    
    def test_get_value_ranges_data_integrity(self):
        """Test de l'intégrité des données après calcul des plages de valeurs"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd_original = pd.read_csv(pd_file, sep=';')
        
        # Sauvegarder les valeurs originales
        original_resistivity_min = df_pd_original['Rho(ohm.m)'].min()
        original_resistivity_max = df_pd_original['Rho(ohm.m)'].max()
        original_chargeability_min = df_pd_original['M (mV/V)'].min()
        original_chargeability_max = df_pd_original['M (mV/V)'].max()
        
        # Créer un DataFrame avec les noms de colonnes standardisés pour le test
        df_pd_test = df_pd_original.copy()
        df_pd_test['resistivity'] = df_pd_test['Rho(ohm.m)']
        df_pd_test['chargeability'] = df_pd_test['M (mV/V)']
        
        # Appeler la méthode de calcul des plages de valeurs
        value_ranges = self.cleaner._get_value_ranges(df_pd_test)
        
        # Vérifier que les données originales n'ont pas été modifiées
        self.assertEqual(df_pd_original['Rho(ohm.m)'].min(), original_resistivity_min, 
                        "min de résistivité ne devrait pas être modifié")
        self.assertEqual(df_pd_original['Rho(ohm.m)'].max(), original_resistivity_max, 
                        "max de résistivité ne devrait pas être modifié")
        self.assertEqual(df_pd_original['M (mV/V)'].min(), original_chargeability_min, 
                        "min de chargeabilité ne devrait pas être modifié")
        self.assertEqual(df_pd_original['M (mV/V)'].max(), original_chargeability_max, 
                        "max de chargeabilité ne devrait pas être modifié")
        
        # Vérifier que les plages de valeurs correspondent aux données originales
        self.assertEqual(value_ranges['resistivity']['min'], original_resistivity_min)
        self.assertEqual(value_ranges['resistivity']['max'], original_resistivity_max)
        self.assertEqual(value_ranges['chargeability']['min'], original_chargeability_min)
        self.assertEqual(value_ranges['chargeability']['max'], original_chargeability_max)
        
        print("✅ Intégrité des données préservée après calcul des plages de valeurs")
    
    def test_get_value_ranges_value_ranges(self):
        """Test des plages de valeurs géophysiques et calcul des statistiques"""
        # Tester PD.csv avec mesures géophysiques
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        # Vérifier que les mesures sont numériques
        self.assertTrue(pd.api.types.is_numeric_dtype(df_pd['Rho(ohm.m)']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df_pd['M (mV/V)']))
        
        # Calculer manuellement les plages de valeurs
        manual_resistivity_min = df_pd['Rho(ohm.m)'].min()
        manual_resistivity_max = df_pd['Rho(ohm.m)'].max()
        manual_resistivity_mean = df_pd['Rho(ohm.m)'].mean()
        
        manual_chargeability_min = df_pd['M (mV/V)'].min()
        manual_chargeability_max = df_pd['M (mV/V)'].max()
        manual_chargeability_mean = df_pd['M (mV/V)'].mean()
        
        # Créer un DataFrame avec les noms de colonnes standardisés pour le test
        df_pd_test = df_pd.copy()
        df_pd_test['resistivity'] = df_pd_test['Rho(ohm.m)']
        df_pd_test['chargeability'] = df_pd_test['M (mV/V)']
        
        # Appeler la méthode de calcul des plages de valeurs
        value_ranges = self.cleaner._get_value_ranges(df_pd_test)
        
        # Vérifier que les calculs correspondent
        self.assertEqual(value_ranges['resistivity']['min'], manual_resistivity_min)
        self.assertEqual(value_ranges['resistivity']['max'], manual_resistivity_max)
        self.assertEqual(value_ranges['resistivity']['mean'], manual_resistivity_mean)
        
        self.assertEqual(value_ranges['chargeability']['min'], manual_chargeability_min)
        self.assertEqual(value_ranges['chargeability']['max'], manual_chargeability_max)
        self.assertEqual(value_ranges['chargeability']['mean'], manual_chargeability_mean)
        
        print(f"✅ Plages de valeurs géophysiques validées:")
        print(f"   Résistivité: {manual_resistivity_min:.2f} - {manual_resistivity_max:.2f} (moy: {manual_resistivity_mean:.2f})")
        print(f"   Chargeabilité: {manual_chargeability_min:.4f} - {manual_chargeability_max:.4f} (moy: {manual_chargeability_mean:.4f})")
    
    def test_get_value_ranges_performance(self):
        """Test de performance du calcul des plages de valeurs"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        df_pd_test = df_pd.copy()
        df_pd_test['resistivity'] = df_pd_test['Rho(ohm.m)']
        df_pd_test['chargeability'] = df_pd_test['M (mV/V)']
        
        start_time = time.time()
        ranges_pd = self.cleaner._get_value_ranges(df_pd_test)
        pd_time = time.time() - start_time
        
        self.assertIsInstance(ranges_pd, dict)
        self.assertLess(pd_time, 1.0, "Calcul des plages de valeurs de PD.csv devrait être rapide (< 1 seconde)")
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        df_s_test = df_s.copy()
        df_s_test['resistivity'] = df_s_test['Rho (Ohm.m)']
        df_s_test['chargeability'] = df_s_test['M (mV/V)']
        
        start_time = time.time()
        ranges_s = self.cleaner._get_value_ranges(df_s_test)
        s_time = time.time() - start_time
        
        self.assertIsInstance(ranges_s, dict)
        self.assertLess(s_time, 1.0, "Calcul des plages de valeurs de S.csv devrait être rapide (< 1 seconde)")
        
        print(f"✅ Performance validée: PD.csv ({pd_time:.3f}s), S.csv ({s_time:.3f}s)")
    
    def test_get_value_ranges_integration_with_cleaner(self):
        """Test d'intégration avec le processus de nettoyage complet"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = self.cleaner._load_device_data(pd_file, "pole_dipole")
        validated_pd = self.cleaner._validate_columns(df_pd, "pole_dipole")
        # Créer un DataFrame avec les noms de colonnes standardisés pour le test
        validated_pd_test = validated_pd.copy()
        validated_pd_test['resistivity'] = validated_pd_test['Rho(ohm.m)']
        validated_pd_test['chargeability'] = validated_pd_test['M (mV/V)']
        ranges_pd = self.cleaner._get_value_ranges(validated_pd_test)
        
        self.assertIsInstance(ranges_pd, dict)
        self.assertGreater(len(ranges_pd), 0)
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = self.cleaner._load_device_data(s_file, "schlumberger")
        validated_s = self.cleaner._validate_columns(df_s, "schlumberger")
        # Créer un DataFrame avec les noms de colonnes standardisés pour le test
        validated_s_test = validated_s.copy()
        validated_s_test['resistivity'] = validated_s_test['Rho (Ohm.m)']
        validated_s_test['chargeability'] = validated_s_test['M (mV/V)']
        ranges_s = self.cleaner._get_value_ranges(validated_s_test)
        
        self.assertIsInstance(ranges_s, dict)
        self.assertGreater(len(ranges_s), 0)
        
        print("✅ Intégration avec le processus de nettoyage complet réussie")
    
    def test_get_value_ranges_edge_cases(self):
        """Test des cas limites"""
        # Test avec DataFrame vide
        empty_df = pd.DataFrame()
        ranges_empty = self.cleaner._get_value_ranges(empty_df)
        self.assertIsInstance(ranges_empty, dict)
        self.assertEqual(len(ranges_empty), 0)
        
        # Test avec DataFrame à une seule ligne
        single_row_df = pd.DataFrame({
            'resistivity': [100.0],
            'chargeability': [0.001]
        })
        ranges_single = self.cleaner._get_value_ranges(single_row_df)
        self.assertIsInstance(ranges_single, dict)
        self.assertGreater(len(ranges_single), 0)
        
        # Vérifier que min = max = mean pour un seul point
        self.assertEqual(ranges_single['resistivity']['min'], ranges_single['resistivity']['max'])
        self.assertEqual(ranges_single['resistivity']['min'], ranges_single['resistivity']['mean'])
        self.assertEqual(ranges_single['chargeability']['min'], ranges_single['chargeability']['max'])
        self.assertEqual(ranges_single['chargeability']['min'], ranges_single['chargeability']['mean'])
        
        # Test avec DataFrame avec des valeurs identiques
        identical_values_df = pd.DataFrame({
            'resistivity': [100.0, 100.0, 100.0],
            'chargeability': [0.001, 0.001, 0.001]
        })
        ranges_identical = self.cleaner._get_value_ranges(identical_values_df)
        self.assertIsInstance(ranges_identical, dict)
        self.assertGreater(len(ranges_identical), 0)
        
        # Vérifier que min = max = mean pour des valeurs identiques
        self.assertEqual(ranges_identical['resistivity']['min'], ranges_identical['resistivity']['max'])
        self.assertEqual(ranges_identical['resistivity']['min'], ranges_identical['resistivity']['mean'])
        self.assertEqual(ranges_identical['chargeability']['min'], ranges_identical['chargeability']['max'])
        self.assertEqual(ranges_identical['chargeability']['min'], ranges_identical['chargeability']['mean'])
        
        print("✅ Cas limites gérés avec succès")
    
    def test_get_value_ranges_column_mapping(self):
        """Test de la correspondance des colonnes entre les fichiers"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        # PD.csv utilise 'Rho(ohm.m)' et 'M (mV/V)'
        self.assertIn('Rho(ohm.m)', df_pd.columns)
        self.assertIn('M (mV/V)', df_pd.columns)
        
        # Créer un DataFrame avec les noms de colonnes standardisés pour le test
        df_pd_test = df_pd.copy()
        df_pd_test['resistivity'] = df_pd_test['Rho(ohm.m)']
        df_pd_test['chargeability'] = df_pd_test['M (mV/V)']
        
        ranges_pd = self.cleaner._get_value_ranges(df_pd_test)
        self.assertIsInstance(ranges_pd, dict)
        self.assertGreater(len(ranges_pd), 0)
        
        # Tester S.csv
        s_file = self.raw_data_dir / "S.csv"
        df_s = pd.read_csv(s_file, sep=';')
        
        # S.csv utilise 'Rho (Ohm.m)' et 'M (mV/V)'
        self.assertIn('Rho (Ohm.m)', df_s.columns)
        self.assertIn('M (mV/V)', df_s.columns)
        
        # Créer un DataFrame avec les noms de colonnes standardisés pour le test
        df_s_test = df_s.copy()
        df_s_test['resistivity'] = df_s_test['Rho (Ohm.m)']
        df_s_test['chargeability'] = df_s_test['M (mV/V)']
        
        ranges_s = self.cleaner._get_value_ranges(df_s_test)
        self.assertIsInstance(ranges_s, dict)
        self.assertGreater(len(ranges_s), 0)
        
        print("✅ Correspondance des colonnes validée entre PD.csv et S.csv")
    
    def test_get_value_ranges_return_structure(self):
        """Test de la structure de retour des plages de valeurs"""
        # Tester PD.csv
        pd_file = self.raw_data_dir / "PD.csv"
        df_pd = pd.read_csv(pd_file, sep=';')
        
        # Créer un DataFrame avec les noms de colonnes standardisés pour le test
        df_pd_test = df_pd.copy()
        df_pd_test['resistivity'] = df_pd_test['Rho(ohm.m)']
        df_pd_test['chargeability'] = df_pd_test['M (mV/V)']
        
        value_ranges = self.cleaner._get_value_ranges(df_pd_test)
        
        # Vérifier la structure du dictionnaire retourné
        self.assertIsInstance(value_ranges, dict)
        self.assertGreater(len(value_ranges), 0)
        
        # Vérifier que toutes les clés requises sont présentes
        required_keys = ['resistivity', 'chargeability']
        for key in required_keys:
            self.assertIn(key, value_ranges, f"Clé {key} manquante dans les plages de valeurs")
        
        # Vérifier la structure de chaque mesure
        for measure_key in required_keys:
            measure_data = value_ranges[measure_key]
            measure_keys = ['min', 'max', 'mean']
            for key in measure_keys:
                self.assertIn(key, measure_data, f"Clé {key} manquante pour {measure_key}")
        
        # Vérifier que les valeurs sont cohérentes
        for measure_key in required_keys:
            measure_data = value_ranges[measure_key]
            self.assertLessEqual(measure_data['min'], measure_data['max'], 
                                f"min de {measure_key} devrait être <= max")
            self.assertLessEqual(measure_data['min'], measure_data['mean'], 
                                f"min de {measure_key} devrait être <= mean")
            self.assertGreaterEqual(measure_data['max'], measure_data['mean'], 
                                   f"max de {measure_key} devrait être >= mean")
        
        print("✅ Structure de retour des plages de valeurs validée")


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(
        verbosity=2,
        testLoader=unittest.TestLoader(),
        testRunner=unittest.TextTestRunner(stream=sys.stdout)
    )
