#!/usr/bin/env python3
"""
Tests unitaires pour GeophysicalDataProcessor.

Teste toutes les méthodes en utilisant les données réelles PD.csv et S.csv.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import sys
import os
from unittest.mock import patch, MagicMock

# Ajouter le répertoire racine au path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_processor import GeophysicalDataProcessor
from src.utils.logger import logger


class TestGeophysicalDataProcessor(unittest.TestCase):
    """Tests pour GeophysicalDataProcessor avec données réelles."""
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = GeophysicalDataProcessor()
        
        # Charger les données réelles
        self.pd_data = self.load_pd_data()
        self.s_data = self.load_s_data()
        
        # Créer des fichiers temporaires pour simuler les données nettoyées
        self.create_temp_clean_files()
        
        # Mock de la configuration
        self.mock_config()
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        shutil.rmtree(self.temp_dir)
    
    def load_pd_data(self) -> pd.DataFrame:
        """Charger les données réelles de PD.csv."""
        pd_file = Path("data/raw/PD.csv")
        if not pd_file.exists():
            self.skipTest("Fichier PD.csv non trouvé")
        
        # Charger avec le bon séparateur
        df = pd.read_csv(pd_file, sep=';', decimal=',')
        
        # Nettoyer et convertir les colonnes numériques
        numeric_columns = ['Rho(ohm.m)', 'M (mV/V)', 'SP (mV)', 'xA (m)', 'xB (m)', 'xM (m)', 'xN (m)']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Supprimer les lignes avec des valeurs manquantes
        df = df.dropna(subset=['Rho(ohm.m)', 'M (mV/V)', 'SP (mV)'])
        
        # Renommer les colonnes pour correspondre au format attendu
        df = df.rename(columns={
            'Rho(ohm.m)': 'resistivity',
            'M (mV/V)': 'chargeability',
            'SP (mV)': 'spontaneous_potential',
            'xA (m)': 'xA',
            'xB (m)': 'xB',
            'xM (m)': 'xM',
            'xN (m)': 'xN'
        })
        
        # Ajouter les coordonnées X, Y (utiliser xA comme X et une valeur fixe pour Y)
        df['x'] = df['xA']
        df['y'] = 0.0  # Coordonnée Y fixe pour les tests
        
        logger.info(f"Données PD chargées: {len(df)} enregistrements")
        return df
    
    def load_s_data(self) -> pd.DataFrame:
        """Charger les données réelles de S.csv."""
        s_file = Path("data/raw/S.csv")
        if not s_file.exists():
            self.skipTest("Fichier S.csv non trouvé")
        
        # Charger avec le bon séparateur
        df = pd.read_csv(s_file, sep=';', decimal=',')
        
        # Nettoyer et convertir les colonnes numériques
        numeric_columns = ['Rho (Ohm.m)', 'M (mV/V)', 'SP (mV) ', 'xA(m)', 'xB(m)', 'xM(m)', 'xN(m)']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Supprimer les lignes avec des valeurs manquantes
        df = df.dropna(subset=['Rho (Ohm.m)', 'M (mV/V)', 'SP (mV) '])
        
        # Renommer les colonnes pour correspondre au format attendu
        df = df.rename(columns={
            'Rho (Ohm.m)': 'resistivity',
            'M (mV/V)': 'chargeability',
            'SP (mV) ': 'spontaneous_potential',
            'xA(m)': 'xA',
            'xB(m)': 'xB',
            'xM(m)': 'xM',
            'xN(m)': 'xN'
        })
        
        # Ajouter les coordonnées X, Y (utiliser xA comme X et une valeur fixe pour Y)
        df['x'] = df['xA']
        df['y'] = 1.0  # Coordonnée Y différente de PD pour les tests
        
        logger.info(f"Données S chargées: {len(df)} enregistrements")
        return df
    
    def create_temp_clean_files(self):
        """Créer des fichiers temporaires de données nettoyées."""
        # Créer le répertoire processed
        processed_dir = Path(self.temp_dir) / "processed"
        processed_dir.mkdir()
        
        # Sauvegarder les données nettoyées
        self.pd_data.to_csv(processed_dir / "PD_cleaned.csv", index=False)
        self.s_data.to_csv(processed_dir / "S_cleaned.csv", index=False)
        
        logger.info(f"Fichiers temporaires créés dans: {processed_dir}")
    
    def mock_config(self):
        """Mock de la configuration pour les tests."""
        mock_config = MagicMock()
        mock_config.paths.processed_data_dir = str(Path(self.temp_dir) / "processed")
        mock_config.geophysical_data.devices = {"PD": {}, "S": {}}
        mock_config.processing.grid_2d = (32, 32)  # Grille 32x32 pour les tests
        mock_config.processing.grid_3d = (16, 32, 32)  # Volume 16x32x32 pour les tests
        
        # Patcher la configuration
        self.config_patcher = patch('src.data.data_processor.CONFIG', mock_config)
        self.config_patcher.start()
    
    def test_initialization(self):
        """Tester l'initialisation de la classe."""
        processor = GeophysicalDataProcessor()
        
        # Vérifier que les attributs sont initialisés
        self.assertIsInstance(processor.scalers, dict)
        self.assertIsInstance(processor.spatial_grids, dict)
        self.assertIsInstance(processor.device_data, dict)
        
        # Vérifier que les dictionnaires sont vides
        self.assertEqual(len(processor.scalers), 0)
        self.assertEqual(len(processor.spatial_grids), 0)
        self.assertEqual(len(processor.device_data), 0)
    
    def test_load_and_validate_success(self):
        """Tester le chargement et la validation des données avec succès."""
        # Charger les données
        device_data = self.processor.load_and_validate()
        
        # Vérifier que les données ont été chargées
        self.assertIn("PD", device_data)
        self.assertIn("S", device_data)
        
        # Vérifier le contenu des DataFrames
        pd_df = device_data["PD"]
        s_df = device_data["S"]
        
        self.assertIsInstance(pd_df, pd.DataFrame)
        self.assertIsInstance(s_df, pd.DataFrame)
        
        # Vérifier que les colonnes essentielles sont présentes
        required_columns = ['x', 'y', 'resistivity', 'chargeability']
        for col in required_columns:
            self.assertIn(col, pd_df.columns)
            self.assertIn(col, s_df.columns)
        
        # Vérifier que les données ne sont pas vides
        self.assertGreater(len(pd_df), 0)
        self.assertGreater(len(s_df), 0)
        
        # Vérifier que les coordonnées sont numériques
        self.assertTrue(pd_df['x'].dtype in ['float64', 'int64'])
        self.assertTrue(pd_df['y'].dtype in ['float64', 'int64'])
        self.assertTrue(pd_df['resistivity'].dtype in ['float64', 'int64'])
        self.assertTrue(pd_df['chargeability'].dtype in ['float64', 'int64'])
    
    def test_load_and_validate_missing_files(self):
        """Tester le comportement avec des fichiers manquants."""
        # Créer un processeur avec une configuration différente
        processor = GeophysicalDataProcessor()
        
        # Mock avec des dispositifs inexistants
        with patch('src.data.data_processor.CONFIG') as mock_config:
            mock_config.paths.processed_data_dir = "/chemin/inexistant"
            mock_config.geophysical_data.devices = {"INEXISTANT": {}}
            
            # Charger les données (ne devrait pas planter)
            device_data = processor.load_and_validate()
            
            # Vérifier que le dictionnaire est vide
            self.assertEqual(len(device_data), 0)
    
    def test_create_2d_grid_basic(self):
        """Tester la création d'une grille 2D de base."""
        # Charger d'abord les données
        self.processor.load_and_validate()
        
        # Créer une grille 2D pour PD
        grid = self.processor._create_2d_grid(self.processor.device_data["PD"], "PD")
        
        # Vérifier la forme de la grille
        expected_shape = (32, 32, 4)  # Selon la configuration mock
        self.assertEqual(grid.shape, expected_shape)
        
        # Vérifier que la grille n'est pas vide
        self.assertFalse(np.all(grid == 0))
        
        # Vérifier que les coordonnées X, Y sont présentes
        self.assertTrue(np.any(grid[:, :, 2] != 0))  # Coordonnée X
        self.assertTrue(np.any(grid[:, :, 3] != 0))  # Coordonnée Y
    
    def test_create_2d_grid_with_resistivity(self):
        """Tester la création d'une grille 2D avec données de résistivité."""
        # Charger les données
        self.processor.load_and_validate()
        
        # Créer une grille 2D
        grid = self.processor._create_2d_grid(self.processor.device_data["PD"], "PD")
        
        # Vérifier que la résistivité est présente
        resistivity_channel = grid[:, :, 0]
        self.assertTrue(np.any(resistivity_channel != 0))
        
        # Vérifier que les valeurs sont dans une plage raisonnable
        # (résistivité géophysique typique : 1-10000 ohm.m)
        self.assertTrue(np.all(resistivity_channel >= 0))  # Résistivité positive
        self.assertTrue(np.all(resistivity_channel < 1e6))  # Pas de valeurs aberrantes
    
    def test_create_2d_grid_with_chargeability(self):
        """Tester la création d'une grille 2D avec données de chargeabilité."""
        # Charger les données
        self.processor.load_and_validate()
        
        # Créer une grille 2D
        grid = self.processor._create_2d_grid(self.processor.device_data["PD"], "PD")
        
        # Vérifier que la chargeabilité est présente
        chargeability_channel = grid[:, :, 1]
        self.assertTrue(np.any(chargeability_channel != 0))
        
        # Vérifier que les valeurs sont dans une plage raisonnable
        # (chargeabilité typique : 0-100 mV/V)
        self.assertTrue(np.all(chargeability_channel >= 0))  # Chargeabilité positive
        self.assertTrue(np.all(chargeability_channel < 1000))  # Pas de valeurs aberrantes
    
    def test_create_spatial_grids(self):
        """Tester la création des grilles spatiales pour tous les dispositifs."""
        # Charger les données
        self.processor.load_and_validate()
        
        # Créer les grilles spatiales
        spatial_grids = self.processor.create_spatial_grids()
        
        # Vérifier que les grilles ont été créées
        self.assertIn("PD", spatial_grids)
        self.assertIn("S", spatial_grids)
        
        # Vérifier la forme des grilles
        pd_grid = spatial_grids["PD"]
        s_grid = spatial_grids["S"]
        
        expected_shape = (32, 32, 4)
        self.assertEqual(pd_grid.shape, expected_shape)
        self.assertEqual(s_grid.shape, expected_shape)
        
        # Vérifier que les grilles sont différentes (données différentes)
        self.assertFalse(np.array_equal(pd_grid, s_grid))
    
    def test_create_multi_device_tensor(self):
        """Tester la création du tenseur multi-dispositifs."""
        # Charger les données et créer les grilles
        self.processor.load_and_validate()
        self.processor.create_spatial_grids()
        
        # Créer le tenseur multi-dispositifs
        multi_device_tensor = self.processor.create_multi_device_tensor()
        
        # Vérifier la forme du tenseur
        expected_shape = (2, 32, 32, 4)  # 2 dispositifs, 32x32, 4 canaux
        self.assertEqual(multi_device_tensor.shape, expected_shape)
        
        # Vérifier que le tenseur n'est pas vide
        self.assertFalse(np.all(multi_device_tensor == 0))
        
        # Vérifier que les deux dispositifs sont présents
        pd_slice = multi_device_tensor[0]  # Premier dispositif
        s_slice = multi_device_tensor[1]   # Deuxième dispositif
        
        self.assertFalse(np.array_equal(pd_slice, s_slice))  # Données différentes
    
    def test_create_multi_device_tensor_no_data(self):
        """Tester la création du tenseur avec aucune donnée."""
        # Créer un processeur vide
        processor = GeophysicalDataProcessor()
        
        # Mock avec aucun dispositif
        with patch('src.data.data_processor.CONFIG') as mock_config:
            mock_config.geophysical_data.devices = {}
            mock_config.processing.grid_2d = (32, 32)
            
            # Créer le tenseur (devrait créer un tenseur factice)
            multi_device_tensor = processor.create_multi_device_tensor()
            
            # Vérifier que le tenseur factice a été créé
            expected_shape = (1, 32, 32, 4)
            self.assertEqual(multi_device_tensor.shape, expected_shape)
            
            # Vérifier que le tenseur est vide
            self.assertTrue(np.all(multi_device_tensor == 0))
    
    def test_create_3d_volume(self):
        """Tester la création du volume 3D."""
        # Charger les données et créer les grilles
        self.processor.load_and_validate()
        self.processor.create_spatial_grids()
        
        # Créer le volume 3D
        volume = self.processor.create_3d_volume()
        
        # Vérifier la forme du volume
        expected_shape = (16, 32, 32, 4)  # 16 profondeurs, 32x32, 4 canaux
        self.assertEqual(volume.shape, expected_shape)
        
        # Vérifier que le volume n'est pas vide
        self.assertFalse(np.all(volume == 0))
        
        # Vérifier que la profondeur est correctement étendue
        # (toutes les couches de profondeur devraient être identiques)
        first_layer = volume[0]
        last_layer = volume[-1]
        self.assertTrue(np.array_equal(first_layer, last_layer))
    
    def test_split_data_basic(self):
        """Tester la division basique des données."""
        # Créer des données de test
        test_tensor = np.random.rand(100, 32, 32, 4)
        test_labels = np.random.randint(0, 2, 100)
        
        # Diviser les données
        x_train, x_test = self.processor.split_data(test_tensor, test_labels)
        
        # Vérifier les tailles
        expected_train_size = int(100 * 0.8)  # 80%
        expected_test_size = 100 - expected_train_size  # 20%
        
        self.assertEqual(len(x_train), expected_train_size)
        self.assertEqual(len(x_test), expected_test_size)
        
        # Vérifier que les données sont différentes
        self.assertFalse(np.array_equal(x_train, x_test))
    
    def test_split_data_small_dataset(self):
        """Tester la division avec un petit ensemble de données."""
        # Créer un petit ensemble de données
        small_tensor = np.random.rand(5, 32, 32, 4)
        
        # Diviser les données
        x_train, x_test = self.processor.split_data(small_tensor)
        
        # Vérifier les tailles
        expected_train_size = int(5 * 0.8)  # 4
        expected_test_size = 5 - expected_train_size  # 1
        
        self.assertEqual(len(x_train), expected_train_size)
        self.assertEqual(len(x_test), expected_test_size)
    
    def test_get_data_summary(self):
        """Tester l'obtention du résumé des données."""
        # Charger les données et créer les grilles
        self.processor.load_and_validate()
        self.processor.create_spatial_grids()
        
        # Obtenir le résumé
        summary = self.processor.get_data_summary()
        
        # Vérifier la structure du résumé
        required_keys = ['devices_processed', 'spatial_grids_created', 'scalers_created', 'device_details']
        for key in required_keys:
            self.assertIn(key, summary)
        
        # Vérifier les valeurs
        self.assertEqual(summary['devices_processed'], 2)  # PD et S
        self.assertEqual(summary['spatial_grids_created'], 2)
        self.assertEqual(summary['scalers_created'], 0)  # Pas de scalers créés
        
        # Vérifier les détails des dispositifs
        self.assertIn('PD', summary['device_details'])
        self.assertIn('S', summary['device_details'])
        
        # Vérifier les détails de PD
        pd_details = summary['device_details']['PD']
        self.assertIn('record_count', pd_details)
        self.assertIn('spatial_coverage', pd_details)
        self.assertGreater(pd_details['record_count'], 0)
        self.assertGreater(pd_details['spatial_coverage']['x_range'], 0)
    
    def test_data_processing_pipeline_integration(self):
        """Tester le pipeline complet de traitement des données."""
        # 1. Charger et valider
        device_data = self.processor.load_and_validate()
        self.assertEqual(len(device_data), 2)
        
        # 2. Créer les grilles spatiales
        spatial_grids = self.processor.create_spatial_grids()
        self.assertEqual(len(spatial_grids), 2)
        
        # 3. Créer le tenseur multi-dispositifs
        multi_device_tensor = self.processor.create_multi_device_tensor()
        self.assertEqual(multi_device_tensor.shape, (2, 32, 32, 4))
        
        # 4. Créer le volume 3D
        volume = self.processor.create_3d_volume()
        self.assertEqual(volume.shape, (16, 32, 32, 4))
        
        # 5. Diviser les données
        x_train, x_test = self.processor.split_data(multi_device_tensor)
        self.assertGreater(len(x_train), 0)
        self.assertGreater(len(x_test), 0)
        
        # 6. Obtenir le résumé
        summary = self.processor.get_data_summary()
        self.assertEqual(summary['devices_processed'], 2)
        self.assertEqual(summary['spatial_grids_created'], 2)
    
    def test_error_handling_invalid_data(self):
        """Tester la gestion des erreurs avec des données invalides."""
        # Créer un DataFrame avec des données invalides
        invalid_df = pd.DataFrame({
            'x': ['invalid', 'data', 'here'],
            'y': [1, 2, 3],
            'resistivity': [100, 200, 300]
        })
        
        # Tenter de créer une grille (devrait gérer l'erreur gracieusement)
        try:
            grid = self.processor._create_2d_grid(invalid_df, "TEST")
            # Si on arrive ici, vérifier que la grille a été créée malgré les erreurs
            self.assertIsInstance(grid, np.ndarray)
        except Exception as e:
            # Si une erreur est levée, vérifier qu'elle est appropriée
            self.assertIsInstance(e, (ValueError, TypeError))
    
    def test_spatial_interpolation_quality(self):
        """Tester la qualité de l'interpolation spatiale."""
        # Charger les données
        self.processor.load_and_validate()
        
        # Créer une grille 2D
        grid = self.processor._create_2d_grid(self.processor.device_data["PD"], "PD")
        
        # Vérifier que l'interpolation préserve les valeurs originales
        # (au moins pour les points qui correspondent exactement)
        original_data = self.processor.device_data["PD"]
        
        # Trouver quelques points originaux dans la grille
        x_coords = original_data['x'].values
        y_coords = original_data['y'].values
        
        # Vérifier que les valeurs sont cohérentes
        for i in range(min(5, len(original_data))):  # Tester les 5 premiers points
            x, y = x_coords[i], y_coords[i]
            
            # Trouver la position dans la grille
            x_idx = int((x - original_data['x'].min()) / (original_data['x'].max() - original_data['x'].min()) * 31)
            y_idx = int((y - original_data['y'].min()) / (original_data['y'].max() - original_data['y'].min()) * 31)
            
            # Vérifier que les indices sont dans les limites
            if 0 <= x_idx < 32 and 0 <= y_idx < 32:
                # La valeur dans la grille devrait être proche de la valeur originale
                grid_resistivity = grid[x_idx, y_idx, 0]
                original_resistivity = original_data.iloc[i]['resistivity']
                
                # Tolérance de 10% pour l'interpolation
                tolerance = original_resistivity * 0.1
                self.assertAlmostEqual(grid_resistivity, original_resistivity, delta=tolerance)


if __name__ == "__main__":
    unittest.main(verbosity=2)
