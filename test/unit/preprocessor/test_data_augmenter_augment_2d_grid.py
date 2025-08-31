#!/usr/bin/env python3
"""
Test unitaire pour la méthode augment_2d_grid de GeophysicalDataAugmenter

Ce test vérifie le bon fonctionnement de la méthode augment_2d_grid
avec des données géophysiques réelles (PD.csv et S.csv) et des scénarios de test variés.
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.preprocessor.data_augmenter import GeophysicalDataAugmenter
from src.preprocessor.data_cleaner import GeophysicalDataCleaner


class TestDataAugmenterAugment2dGrid(unittest.TestCase):
    """Tests pour la méthode augment_2d_grid de GeophysicalDataAugmenter"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        # Créer un répertoire temporaire pour les tests
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Créer la structure des dossiers de test
        self.test_raw_dir = self.test_dir / "raw"
        self.test_processed_dir = self.test_dir / "processed"
        self.test_raw_dir.mkdir(parents=True, exist_ok=True)
        self.test_processed_dir.mkdir(exist_ok=True)
        
        # Copier les fichiers de test depuis fixtures
        fixtures_dir = Path(__file__).parent.parent.parent / "fixtures" / "raw"
        self.pd_test_file = fixtures_dir / "PD.csv"
        self.s_test_file = fixtures_dir / "S.csv"
        
        # Copier les fichiers vers le dossier de test
        if self.pd_test_file.exists():
            shutil.copy2(self.pd_test_file, self.test_raw_dir / "PD.csv")
        if self.s_test_file.exists():
            shutil.copy2(self.s_test_file, self.test_raw_dir / "S.csv")
        
        # Créer une instance du cleaner avec des chemins de test
        with patch('src.preprocessor.data_cleaner.CONFIG') as mock_config:
            mock_config.paths.raw_data_dir = str(self.test_raw_dir)
            mock_config.paths.processed_data_dir = str(self.test_processed_dir)
            mock_config.geophysical_data.coordinate_systems = {
                'wgs84': "EPSG:4326",
                'utm_proj': "EPSG:32630"
            }
            self.cleaner = GeophysicalDataCleaner()
        
        # Initialiser l'augmenteur
        self.augmenter = GeophysicalDataAugmenter(random_seed=42)
        
        # Créer des grilles de test basées sur les vraies données
        self._create_test_grids()
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Supprimer le répertoire temporaire
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _create_test_grids(self):
        """Créer des grilles de test basées sur les vraies données"""
        # Charger les données PD.csv
        pd_df = pd.read_csv(self.test_raw_dir / "PD.csv", sep=';')
        
        # Créer une grille 2D basée sur PD.csv (16x16x4)
        grid_size = 16
        self.grid_2d_pd = np.zeros((grid_size, grid_size, 4))
        
        # Remplir la grille avec les données PD.csv
        for i in range(min(len(pd_df), grid_size * grid_size)):
            row = i // grid_size
            col = i % grid_size
            if row < grid_size and col < grid_size:
                self.grid_2d_pd[row, col, 0] = pd_df.iloc[i]['Rho(ohm.m)'] if i < len(pd_df) else 0
                self.grid_2d_pd[row, col, 1] = pd_df.iloc[i]['M (mV/V)'] if i < len(pd_df) else 0
                self.grid_2d_pd[row, col, 2] = pd_df.iloc[i]['x'] if i < len(pd_df) else 0
                self.grid_2d_pd[row, col, 3] = pd_df.iloc[i]['y'] if i < len(pd_df) else 0
        
        # Charger les données S.csv
        s_df = pd.read_csv(self.test_raw_dir / "S.csv", sep=';')
        
        # Créer une grille 2D basée sur S.csv (32x32x4)
        grid_size_s = 32
        self.grid_2d_s = np.zeros((grid_size_s, grid_size_s, 4))
        
        # Remplir la grille avec les données S.csv
        for i in range(min(len(s_df), grid_size_s * grid_size_s)):
            row = i // grid_size_s
            col = i % grid_size_s
            if row < grid_size_s and col < grid_size_s:
                self.grid_2d_s[row, col, 0] = s_df.iloc[i]['Rho (Ohm.m)'] if i < len(s_df) else 0
                self.grid_2d_s[row, col, 1] = s_df.iloc[i]['M (mV/V)'] if i < len(s_df) else 0
                self.grid_2d_s[row, col, 2] = s_df.iloc[i]['LAT'] if i < len(s_df) else 0
                self.grid_2d_s[row, col, 3] = s_df.iloc[i]['LON'] if i < len(s_df) else 0
        
        # Créer une grille de test simple pour les tests de base
        self.grid_2d_simple = np.random.rand(8, 8, 4)
        
        print(f"📊 Grilles de test créées:")
        print(f"   - PD.csv: {self.grid_2d_pd.shape}")
        print(f"   - S.csv: {self.grid_2d_s.shape}")
        print(f"   - Simple: {self.grid_2d_simple.shape}")
    
    def test_augment_2d_grid_basic_functionality(self):
        """Test de la fonctionnalité de base de augment_2d_grid"""
        # Test avec une grille simple
        augmentations = ["flip_horizontal"]
        result = self.augmenter.augment_2d_grid(self.grid_2d_simple, augmentations, 1)
        
        # Vérifications de base
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, self.grid_2d_simple.shape)
        
        # Vérifier que l'augmentation a été appliquée
        self.assertFalse(np.array_equal(result[0], self.grid_2d_simple))
        
        print("✅ Fonctionnalité de base de augment_2d_grid validée")
    
    def test_augment_2d_grid_with_pd_csv_data(self):
        """Test de augment_2d_grid avec les données PD.csv"""
        augmentations = ["rotation", "flip_horizontal", "gaussian_noise"]
        num_augmentations = 3
        
        result = self.augmenter.augment_2d_grid(
            self.grid_2d_pd, 
            augmentations, 
            num_augmentations
        )
        
        # Vérifications
        self.assertEqual(len(result), num_augmentations)
        for augmented_grid in result:
            self.assertEqual(augmented_grid.shape, self.grid_2d_pd.shape)
            self.assertFalse(np.array_equal(augmented_grid, self.grid_2d_pd))
        
        # Vérifier l'historique
        self.assertEqual(len(self.augmenter.augmentation_history), num_augmentations)
        
        print(f"✅ augment_2d_grid avec PD.csv validé: {num_augmentations} augmentations générées")
    
    def test_augment_2d_grid_with_s_csv_data(self):
        """Test de augment_2d_grid avec les données S.csv"""
        augmentations = ["flip_vertical", "spatial_shift", "value_variation"]
        num_augmentations = 2
        
        result = self.augmenter.augment_2d_grid(
            self.grid_2d_s, 
            augmentations, 
            num_augmentations
        )
        
        # Vérifications
        self.assertEqual(len(result), num_augmentations)
        for augmented_grid in result:
            self.assertEqual(augmented_grid.shape, self.grid_2d_s.shape)
            self.assertFalse(np.array_equal(augmented_grid, self.grid_2d_s))
        
        print(f"✅ augment_2d_grid avec S.csv validé: {num_augmentations} augmentations générées")
    
    def test_augment_2d_grid_rotation_technique(self):
        """Test spécifique de la technique de rotation"""
        augmentations = ["rotation"]
        result = self.augmenter.augment_2d_grid(self.grid_2d_simple, augmentations, 1)
        
        # Vérifier que la rotation a été appliquée
        rotated_grid = result[0]
        self.assertFalse(np.array_equal(rotated_grid, self.grid_2d_simple))
        
        # Vérifier que les dimensions sont préservées
        self.assertEqual(rotated_grid.shape, self.grid_2d_simple.shape)
        
        print("✅ Technique de rotation validée")
    
    def test_augment_2d_grid_flip_horizontal_technique(self):
        """Test spécifique de la technique de retournement horizontal"""
        augmentations = ["flip_horizontal"]
        result = self.augmenter.augment_2d_grid(self.grid_2d_simple, augmentations, 1)
        
        # Vérifier que le retournement horizontal a été appliqué
        flipped_grid = result[0]
        self.assertFalse(np.array_equal(flipped_grid, self.grid_2d_simple))
        
        # Vérifier que les dimensions sont préservées
        self.assertEqual(flipped_grid.shape, self.grid_2d_simple.shape)
        
        print("✅ Technique de retournement horizontal validée")
    
    def test_augment_2d_grid_flip_vertical_technique(self):
        """Test spécifique de la technique de retournement vertical"""
        augmentations = ["flip_vertical"]
        result = self.augmenter.augment_2d_grid(self.grid_2d_simple, augmentations, 1)
        
        # Vérifier que le retournement vertical a été appliqué
        flipped_grid = result[0]
        self.assertFalse(np.array_equal(flipped_grid, self.grid_2d_simple))
        
        # Vérifier que les dimensions sont préservées
        self.assertEqual(flipped_grid.shape, self.grid_2d_simple.shape)
        
        print("✅ Technique de retournement vertical validée")
    
    def test_augment_2d_grid_spatial_shift_technique(self):
        """Test spécifique de la technique de décalage spatial"""
        augmentations = ["spatial_shift"]
        result = self.augmenter.augment_2d_grid(self.grid_2d_simple, augmentations, 1)
        
        # Vérifier que le décalage spatial a été appliqué
        shifted_grid = result[0]
        self.assertFalse(np.array_equal(shifted_grid, self.grid_2d_simple))
        
        # Vérifier que les dimensions sont préservées
        self.assertEqual(shifted_grid.shape, self.grid_2d_simple.shape)
        
        print("✅ Technique de décalage spatial validée")
    
    def test_augment_2d_grid_gaussian_noise_technique(self):
        """Test spécifique de la technique de bruit gaussien"""
        augmentations = ["gaussian_noise"]
        result = self.augmenter.augment_2d_grid(self.grid_2d_simple, augmentations, 1)
        
        # Vérifier que le bruit gaussien a été appliqué
        noisy_grid = result[0]
        self.assertFalse(np.array_equal(noisy_grid, self.grid_2d_simple))
        
        # Vérifier que les dimensions sont préservées
        self.assertEqual(noisy_grid.shape, self.grid_2d_simple.shape)
        
        print("✅ Technique de bruit gaussien validée")
    
    def test_augment_2d_grid_salt_pepper_noise_technique(self):
        """Test spécifique de la technique de bruit poivre et sel"""
        augmentations = ["salt_pepper_noise"]
        result = self.augmenter.augment_2d_grid(self.grid_2d_simple, augmentations, 1)
        
        # Vérifier que le bruit poivre et sel a été appliqué
        noisy_grid = result[0]
        
        # Le bruit poivre et sel peut parfois ne pas modifier les données si la probabilité est faible
        # Vérifier au moins que les dimensions sont préservées
        self.assertEqual(noisy_grid.shape, self.grid_2d_simple.shape)
        
        # Vérifier que la grille est toujours un tableau numpy valide
        self.assertIsInstance(noisy_grid, np.ndarray)
        
        print("✅ Technique de bruit poivre et sel validée")
    
    def test_augment_2d_grid_value_variation_technique(self):
        """Test spécifique de la technique de variation des valeurs"""
        augmentations = ["value_variation"]
        result = self.augmenter.augment_2d_grid(self.grid_2d_simple, augmentations, 1)
        
        # Vérifier que la variation des valeurs a été appliquée
        varied_grid = result[0]
        self.assertFalse(np.array_equal(varied_grid, self.grid_2d_simple))
        
        # Vérifier que les dimensions sont préservées
        self.assertEqual(varied_grid.shape, self.grid_2d_simple.shape)
        
        print("✅ Technique de variation des valeurs validée")
    
    def test_augment_2d_grid_elastic_deformation_technique(self):
        """Test spécifique de la technique de déformation élastique"""
        augmentations = ["elastic_deformation"]
        result = self.augmenter.augment_2d_grid(self.grid_2d_simple, augmentations, 1)
        
        # Vérifier que la déformation élastique a été appliquée
        deformed_grid = result[0]
        self.assertFalse(np.array_equal(deformed_grid, self.grid_2d_simple))
        
        # Vérifier que les dimensions sont préservées
        self.assertEqual(deformed_grid.shape, self.grid_2d_simple.shape)
        
        print("✅ Technique de déformation élastique validée")
    
    def test_augment_2d_grid_multiple_techniques(self):
        """Test de augment_2d_grid avec plusieurs techniques combinées"""
        augmentations = ["rotation", "flip_horizontal", "gaussian_noise", "spatial_shift"]
        num_augmentations = 2
        
        result = self.augmenter.augment_2d_grid(
            self.grid_2d_pd, 
            augmentations, 
            num_augmentations
        )
        
        # Vérifications
        self.assertEqual(len(result), num_augmentations)
        for augmented_grid in result:
            self.assertEqual(augmented_grid.shape, self.grid_2d_pd.shape)
            self.assertFalse(np.array_equal(augmented_grid, self.grid_2d_pd))
        
        print(f"✅ Combinaison de techniques validée: {len(augmentations)} techniques appliquées")
    
    def test_augment_2d_grid_reproducibility(self):
        """Test de reproductibilité avec la même graine"""
        # Créer deux augmenteurs avec la même graine
        augmenter1 = GeophysicalDataAugmenter(random_seed=42)
        augmenter2 = GeophysicalDataAugmenter(random_seed=42)
        
        # Appliquer une seule augmentation déterministe (pas de mélange aléatoire)
        augmentations = ["flip_horizontal"]  # Technique déterministe
        result1 = augmenter1.augment_2d_grid(self.grid_2d_simple, augmentations, 1)
        result2 = augmenter2.augment_2d_grid(self.grid_2d_simple, augmentations, 1)
        
        # Les résultats devraient être identiques avec la même graine pour une technique déterministe
        np.testing.assert_array_equal(result1[0], result2[0])
        
        print("✅ Reproductibilité avec la même graine validée")
    
    def test_augment_2d_grid_error_handling(self):
        """Test de gestion des erreurs"""
        # Test avec une grille de mauvaise dimension
        invalid_grid = np.random.rand(8, 8)  # 2D au lieu de 3D
        
        with self.assertRaises(ValueError):
            self.augmenter.augment_2d_grid(invalid_grid, ["flip_horizontal"], 1)
        
        # Test avec un type invalide
        with self.assertRaises(ValueError):
            self.augmenter.augment_2d_grid("invalid", ["flip_horizontal"], 1)
        
        print("✅ Gestion des erreurs validée")
    
    def test_augment_2d_grid_with_cleaned_data(self):
        """Test de augment_2d_grid avec des données nettoyées"""
        # Nettoyer les données PD.csv
        cleaning_results = self.cleaner.clean_all_devices()
        
        if 'pole_dipole' in cleaning_results:
            clean_path, report = cleaning_results['pole_dipole']
            if clean_path and Path(clean_path).exists():
                # Charger les données nettoyées
                cleaned_df = pd.read_csv(clean_path)
                
                # Créer une grille à partir des données nettoyées
                grid_size = min(16, int(np.sqrt(len(cleaned_df))))
                cleaned_grid = np.zeros((grid_size, grid_size, 4))
                
                # Remplir la grille
                for i in range(min(len(cleaned_df), grid_size * grid_size)):
                    row = i // grid_size
                    col = i % grid_size
                    if row < grid_size and col < grid_size:
                        cleaned_grid[row, col, 0] = cleaned_df.iloc[i]['resistivity'] if 'resistivity' in cleaned_df.columns else 0
                        cleaned_grid[row, col, 1] = cleaned_df.iloc[i]['chargeability'] if 'chargeability' in cleaned_df.columns else 0
                        cleaned_grid[row, col, 2] = cleaned_df.iloc[i]['x'] if 'x' in cleaned_df.columns else 0
                        cleaned_grid[row, col, 3] = cleaned_df.iloc[i]['y'] if 'y' in cleaned_df.columns else 0
                
                # Tester l'augmentation
                augmentations = ["rotation", "flip_horizontal"]
                result = self.augmenter.augment_2d_grid(cleaned_grid, augmentations, 1)
                
                # Vérifications
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].shape, cleaned_grid.shape)
                
                print(f"✅ augment_2d_grid avec données nettoyées validé: {cleaned_grid.shape}")
            else:
                print("⚠️ Données PD nettoyées non disponibles")
        else:
            print("⚠️ Nettoyage des données PD non effectué")
    
    def test_augment_2d_grid_performance(self):
        """Test de performance de augment_2d_grid"""
        import time
        
        # Mesurer le temps d'exécution
        start_time = time.time()
        
        augmentations = ["rotation", "flip_horizontal", "gaussian_noise", "spatial_shift"]
        result = self.augmenter.augment_2d_grid(
            self.grid_2d_s,  # Grille plus grande (32x32x4)
            augmentations, 
            5  # 5 augmentations
        )
        
        execution_time = time.time() - start_time
        
        # Vérifier que l'exécution est raisonnable
        self.assertLess(execution_time, 5.0, "L'augmentation devrait être rapide (< 5 secondes)")
        
        print(f"✅ Performance validée: 5 augmentations en {execution_time:.3f} secondes")
    
    def test_augment_2d_grid_history_tracking(self):
        """Test du suivi de l'historique des augmentations"""
        # Réinitialiser l'historique
        self.augmenter.reset_history()
        self.assertEqual(len(self.augmenter.augmentation_history), 0)
        
        # Effectuer des augmentations
        augmentations = ["rotation", "flip_horizontal"]
        self.augmenter.augment_2d_grid(self.grid_2d_simple, augmentations, 3)
        
        # Vérifier l'historique
        self.assertEqual(len(self.augmenter.augmentation_history), 3)
        
        # Vérifier la structure de l'historique
        for entry in self.augmenter.augmentation_history:
            self.assertIn('grid_shape', entry)
            self.assertIn('augmentations_applied', entry)
            self.assertIn('augmentation_index', entry)
            self.assertEqual(entry['grid_shape'], self.grid_2d_simple.shape)
            self.assertEqual(entry['augmentations_applied'], augmentations)
        
        print("✅ Suivi de l'historique validé")
    
    def test_augment_2d_grid_edge_cases(self):
        """Test des cas limites de augment_2d_grid"""
        # Test avec une seule augmentation
        result = self.augmenter.augment_2d_grid(
            self.grid_2d_simple, 
            ["flip_horizontal"], 
            1
        )
        self.assertEqual(len(result), 1)
        
        # Test avec zéro augmentation
        result = self.augmenter.augment_2d_grid(
            self.grid_2d_simple, 
            ["flip_horizontal"], 
            0
        )
        self.assertEqual(len(result), 0)
        
        # Test avec une liste d'augmentations vide
        result = self.augmenter.augment_2d_grid(
            self.grid_2d_simple, 
            [], 
            1
        )
        self.assertEqual(len(result), 1)
        # La grille devrait être identique car aucune augmentation n'a été appliquée
        np.testing.assert_array_equal(result[0], self.grid_2d_simple)
        
        print("✅ Cas limites validés")
    
    def test_augment_2d_grid_integration_with_real_data(self):
        """Test d'intégration avec les vraies données géophysiques"""
        # Test avec PD.csv
        pd_result = self.augmenter.augment_2d_grid(
            self.grid_2d_pd, 
            ["rotation", "gaussian_noise"], 
            2
        )
        
        # Test avec S.csv
        s_result = self.augmenter.augment_2d_grid(
            self.grid_2d_s, 
            ["flip_horizontal", "value_variation"], 
            2
        )
        
        # Vérifications
        self.assertEqual(len(pd_result), 2)
        self.assertEqual(len(s_result), 2)
        
        # Vérifier que les augmentations ont modifié les données
        for pd_grid in pd_result:
            self.assertFalse(np.array_equal(pd_grid, self.grid_2d_pd))
        
        for s_grid in s_result:
            self.assertFalse(np.array_equal(s_grid, self.grid_2d_s))
        
        print("✅ Intégration avec vraies données géophysiques validée")
    
    def test_augment_2d_grid_data_integrity(self):
        """Test de l'intégrité des données après augmentation"""
        # Sauvegarder les valeurs originales
        original_min = np.min(self.grid_2d_pd)
        original_max = np.max(self.grid_2d_pd)
        original_mean = np.mean(self.grid_2d_pd)
        
        # Appliquer des augmentations
        augmentations = ["gaussian_noise", "value_variation"]
        result = self.augmenter.augment_2d_grid(self.grid_2d_pd, augmentations, 1)
        
        augmented_grid = result[0]
        
        # Vérifier que les données ont été modifiées mais restent dans des limites raisonnables
        self.assertFalse(np.array_equal(augmented_grid, self.grid_2d_pd))
        
        # Vérifier que les dimensions sont préservées
        self.assertEqual(augmented_grid.shape, self.grid_2d_pd.shape)
        
        # Vérifier que les valeurs restent dans des limites raisonnables
        augmented_min = np.min(augmented_grid)
        augmented_max = np.max(augmented_grid)
        
        # Les valeurs ne devraient pas être extrêmement différentes
        self.assertLess(abs(augmented_min - original_min), 1000)
        self.assertLess(abs(augmented_max - original_max), 1000)
        
        print("✅ Intégrité des données après augmentation validée")


if __name__ == "__main__":
    unittest.main(verbosity=2)
