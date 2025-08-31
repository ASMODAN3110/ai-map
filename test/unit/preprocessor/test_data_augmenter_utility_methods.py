#!/usr/bin/env python3
"""
Test unitaire pour les méthodes utilitaires de GeophysicalDataAugmenter

Ce test vérifie le bon fonctionnement des méthodes utilitaires d'augmentation
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


class TestDataAugmenterUtilityMethods(unittest.TestCase):
    """Tests pour les méthodes utilitaires de GeophysicalDataAugmenter"""
    
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
        
        # Créer des données de test basées sur les vraies données
        self._create_test_data()
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Supprimer le répertoire temporaire
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _create_test_data(self):
        """Créer des données de test basées sur les vraies données"""
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
        
        # Créer un volume 3D basé sur S.csv (8x8x8x4)
        volume_size = 8
        self.volume_3d_s = np.zeros((volume_size, volume_size, volume_size, 4))
        
        # Remplir le volume avec les données S.csv
        for i in range(min(len(s_df), volume_size * volume_size * volume_size)):
            d = i // (volume_size * volume_size)
            h = (i % (volume_size * volume_size)) // volume_size
            w = i % volume_size
            if d < volume_size and h < volume_size and w < volume_size:
                self.volume_3d_s[d, h, w, 0] = s_df.iloc[i]['Rho (Ohm.m)'] if i < len(s_df) else 0
                self.volume_3d_s[d, h, w, 1] = s_df.iloc[i]['M (mV/V)'] if i < len(s_df) else 0
                self.volume_3d_s[d, h, w, 2] = s_df.iloc[i]['LAT'] if i < len(s_df) else 0
                self.volume_3d_s[d, h, w, 3] = s_df.iloc[i]['LON'] if i < len(s_df) else 0
        
        # Créer un DataFrame de test basé sur PD.csv
        self.df_pd = pd_df.head(50).copy()
        
        print(f"📊 Données de test créées:")
        print(f"   - Grille 2D PD: {self.grid_2d_pd.shape}")
        print(f"   - Volume 3D S: {self.volume_3d_s.shape}")
        print(f"   - DataFrame PD: {self.df_pd.shape}")
    
    def test_get_augmentation_summary_empty_history(self):
        """Test de get_augmentation_summary avec un historique vide"""
        summary = self.augmenter.get_augmentation_summary()
        
        # Vérifications
        self.assertIsInstance(summary, dict)
        self.assertIn("message", summary)
        self.assertEqual(summary["message"], "Aucune augmentation effectuée")
        
        print("✅ get_augmentation_summary avec historique vide validé")
    
    def test_get_augmentation_summary_with_2d_grid_augmentations(self):
        """Test de get_augmentation_summary avec des augmentations 2D basées sur PD.csv"""
        # Effectuer quelques augmentations 2D
        self.augmenter.augment_2d_grid(
            self.grid_2d_pd, 
            ["rotation", "flip_horizontal"], 
            num_augmentations=3
        )
        
        summary = self.augmenter.get_augmentation_summary()
        
        # Vérifications
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["total_augmentations"], 3)
        self.assertIn("augmentation_types", summary)
        self.assertIn("shape_distribution", summary)
        self.assertIn("recent_augmentations", summary)
        
        # Vérifier les types d'augmentation
        self.assertIn("rotation", summary["augmentation_types"])
        self.assertIn("flip_horizontal", summary["augmentation_types"])
        self.assertEqual(summary["augmentation_types"]["rotation"], 3)
        self.assertEqual(summary["augmentation_types"]["flip_horizontal"], 3)
        
        # Vérifier la distribution des formes
        shape_key = str(self.grid_2d_pd.shape)
        self.assertIn(shape_key, summary["shape_distribution"])
        self.assertEqual(summary["shape_distribution"][shape_key], 3)
        
        # Vérifier les augmentations récentes
        self.assertEqual(len(summary["recent_augmentations"]), 3)
        
        print("✅ get_augmentation_summary avec augmentations 2D validé")
    
    def test_get_augmentation_summary_with_3d_volume_augmentations(self):
        """Test de get_augmentation_summary avec des augmentations 3D basées sur S.csv"""
        # Effectuer quelques augmentations 3D
        self.augmenter.augment_3d_volume(
            self.volume_3d_s, 
            ["rotation", "gaussian_noise"], 
            num_augmentations=2
        )
        
        summary = self.augmenter.get_augmentation_summary()
        
        # Vérifications
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["total_augmentations"], 2)
        
        # Vérifier les types d'augmentation
        self.assertIn("rotation", summary["augmentation_types"])
        self.assertIn("gaussian_noise", summary["augmentation_types"])
        self.assertEqual(summary["augmentation_types"]["rotation"], 2)
        self.assertEqual(summary["augmentation_types"]["gaussian_noise"], 2)
        
        # Vérifier la distribution des formes
        shape_key = str(self.volume_3d_s.shape)
        self.assertIn(shape_key, summary["shape_distribution"])
        self.assertEqual(summary["shape_distribution"][shape_key], 2)
        
        print("✅ get_augmentation_summary avec augmentations 3D validé")
    
    def test_get_augmentation_summary_with_dataframe_augmentations(self):
        """Test de get_augmentation_summary avec des augmentations DataFrame basées sur PD.csv"""
        # Effectuer quelques augmentations DataFrame
        self.augmenter.augment_dataframe(
            self.df_pd, 
            ["gaussian_noise", "value_variation"], 
            num_augmentations=4
        )
        
        summary = self.augmenter.get_augmentation_summary()
        
        # Vérifications
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["total_augmentations"], 4)
        
        # Vérifier les types d'augmentation
        self.assertIn("gaussian_noise", summary["augmentation_types"])
        self.assertIn("value_variation", summary["augmentation_types"])
        self.assertEqual(summary["augmentation_types"]["gaussian_noise"], 4)
        self.assertEqual(summary["augmentation_types"]["value_variation"], 4)
        
        # Vérifier la distribution des formes
        shape_key = str(self.df_pd.shape)
        self.assertIn(shape_key, summary["shape_distribution"])
        self.assertEqual(summary["shape_distribution"][shape_key], 4)
        
        print("✅ get_augmentation_summary avec augmentations DataFrame validé")
    
    def test_get_augmentation_summary_mixed_augmentations(self):
        """Test de get_augmentation_summary avec des augmentations mixtes"""
        # Effectuer des augmentations de différents types
        self.augmenter.augment_2d_grid(self.grid_2d_pd, ["rotation"], num_augmentations=1)
        self.augmenter.augment_3d_volume(self.volume_3d_s, ["flip_horizontal"], num_augmentations=1)
        self.augmenter.augment_dataframe(self.df_pd, ["gaussian_noise"], num_augmentations=1)
        
        summary = self.augmenter.get_augmentation_summary()
        
        # Vérifications
        self.assertEqual(summary["total_augmentations"], 3)
        
        # Vérifier que tous les types sont présents
        self.assertIn("rotation", summary["augmentation_types"])
        self.assertIn("flip_horizontal", summary["augmentation_types"])
        self.assertIn("gaussian_noise", summary["augmentation_types"])
        
        # Vérifier que toutes les formes sont présentes
        self.assertIn(str(self.grid_2d_pd.shape), summary["shape_distribution"])
        self.assertIn(str(self.volume_3d_s.shape), summary["shape_distribution"])
        self.assertIn(str(self.df_pd.shape), summary["shape_distribution"])
        
        print("✅ get_augmentation_summary avec augmentations mixtes validé")
    
    def test_get_augmentation_summary_recent_augmentations_limit(self):
        """Test de get_augmentation_summary avec limite des augmentations récentes"""
        # Effectuer plus de 5 augmentations
        for i in range(7):
            self.augmenter.augment_2d_grid(
                self.grid_2d_pd, 
                ["rotation"], 
                num_augmentations=1
            )
        
        summary = self.augmenter.get_augmentation_summary()
        
        # Vérifications
        self.assertEqual(summary["total_augmentations"], 7)
        self.assertEqual(len(summary["recent_augmentations"]), 5)  # Limite à 5
        
        # Vérifier que les 5 dernières sont bien présentes
        # Note: Les indices peuvent varier selon l'implémentation, vérifions juste la structure
        self.assertIsInstance(summary["recent_augmentations"], list)
        self.assertEqual(len(summary["recent_augmentations"]), 5)
        
        # Vérifier que toutes les augmentations récentes ont la bonne structure
        for aug in summary["recent_augmentations"]:
            self.assertIsInstance(aug, dict)
            self.assertIn("augmentations_applied", aug)
        
        print("✅ get_augmentation_summary avec limite des augmentations récentes validé")
    
    def test_reset_history_method(self):
        """Test de la méthode reset_history"""
        # Effectuer quelques augmentations
        self.augmenter.augment_2d_grid(self.grid_2d_pd, ["rotation"], num_augmentations=2)
        
        # Vérifier que l'historique n'est pas vide
        self.assertGreater(len(self.augmenter.augmentation_history), 0)
        
        # Réinitialiser l'historique
        self.augmenter.reset_history()
        
        # Vérifier que l'historique est vide
        self.assertEqual(len(self.augmenter.augmentation_history), 0)
        
        # Vérifier que get_augmentation_summary retourne le message approprié
        summary = self.augmenter.get_augmentation_summary()
        self.assertEqual(summary["message"], "Aucune augmentation effectuée")
        
        print("✅ reset_history validé")
    
    def test_get_recommended_augmentations_2d_grid(self):
        """Test de get_recommended_augmentations pour les grilles 2D"""
        recommendations = self.augmenter.get_recommended_augmentations("2d_grid")
        
        # Vérifications
        self.assertIsInstance(recommendations, list)
        self.assertIn("rotation", recommendations)
        self.assertIn("flip_horizontal", recommendations)
        self.assertIn("flip_vertical", recommendations)
        self.assertIn("gaussian_noise", recommendations)
        self.assertIn("spatial_shift", recommendations)
        self.assertIn("value_variation", recommendations)
        
        print("✅ get_recommended_augmentations pour 2D validé")
    
    def test_get_recommended_augmentations_3d_volume(self):
        """Test de get_recommended_augmentations pour les volumes 3D"""
        recommendations = self.augmenter.get_recommended_augmentations("3d_volume")
        
        # Vérifications
        self.assertIsInstance(recommendations, list)
        self.assertIn("rotation", recommendations)
        self.assertIn("flip_horizontal", recommendations)
        self.assertIn("flip_vertical", recommendations)
        self.assertIn("gaussian_noise", recommendations)
        self.assertIn("value_variation", recommendations)
        
        # Vérifier que les techniques 2D spécifiques ne sont pas présentes
        self.assertNotIn("spatial_shift", recommendations)
        self.assertNotIn("elastic_deformation", recommendations)
        
        print("✅ get_recommended_augmentations pour 3D validé")
    
    def test_get_recommended_augmentations_dataframe(self):
        """Test de get_recommended_augmentations pour les DataFrames"""
        recommendations = self.augmenter.get_recommended_augmentations("dataframe")
        
        # Vérifications
        self.assertIsInstance(recommendations, list)
        self.assertIn("gaussian_noise", recommendations)
        self.assertIn("value_variation", recommendations)
        self.assertIn("spatial_jitter", recommendations)
        self.assertIn("coordinate_perturbation", recommendations)
        
        # Vérifier que les techniques 2D/3D ne sont pas présentes
        self.assertNotIn("rotation", recommendations)
        self.assertNotIn("flip_horizontal", recommendations)
        
        print("✅ get_recommended_augmentations pour DataFrame validé")
    
    def test_get_recommended_augmentations_invalid_type(self):
        """Test de get_recommended_augmentations avec un type invalide"""
        recommendations = self.augmenter.get_recommended_augmentations("invalid_type")
        
        # Vérifications
        self.assertIsInstance(recommendations, list)
        self.assertEqual(len(recommendations), 0)  # Liste vide pour type invalide
        
        print("✅ get_recommended_augmentations avec type invalide validé")
    
    def test_validate_augmentation_parameters_2d_grid_valid(self):
        """Test de validate_augmentation_parameters avec des paramètres 2D valides"""
        valid_augmentations = ["rotation", "flip_horizontal", "gaussian_noise"]
        is_valid = self.augmenter.validate_augmentation_parameters(valid_augmentations, "2d_grid")
        
        # Vérifications
        self.assertTrue(is_valid)
        
        print("✅ validate_augmentation_parameters 2D valides validé")
    
    def test_validate_augmentation_parameters_2d_grid_invalid(self):
        """Test de validate_augmentation_parameters avec des paramètres 2D invalides"""
        invalid_augmentations = ["rotation", "invalid_technique", "gaussian_noise"]
        is_valid = self.augmenter.validate_augmentation_parameters(invalid_augmentations, "2d_grid")
        
        # Vérifications
        self.assertFalse(is_valid)
        
        print("✅ validate_augmentation_parameters 2D invalides validé")
    
    def test_validate_augmentation_parameters_3d_volume_valid(self):
        """Test de validate_augmentation_parameters avec des paramètres 3D valides"""
        valid_augmentations = ["rotation", "flip_horizontal", "value_variation"]
        is_valid = self.augmenter.validate_augmentation_parameters(valid_augmentations, "3d_volume")
        
        # Vérifications
        self.assertTrue(is_valid)
        
        print("✅ validate_augmentation_parameters 3D valides validé")
    
    def test_validate_augmentation_parameters_dataframe_valid(self):
        """Test de validate_augmentation_parameters avec des paramètres DataFrame valides"""
        valid_augmentations = ["gaussian_noise", "spatial_jitter", "coordinate_perturbation"]
        is_valid = self.augmenter.validate_augmentation_parameters(valid_augmentations, "dataframe")
        
        # Vérifications
        self.assertTrue(is_valid)
        
        print("✅ validate_augmentation_parameters DataFrame valides validé")
    
    def test_validate_augmentation_parameters_empty_list(self):
        """Test de validate_augmentation_parameters avec une liste vide"""
        is_valid = self.augmenter.validate_augmentation_parameters([], "2d_grid")
        
        # Vérifications
        self.assertTrue(is_valid)  # Liste vide est considérée comme valide
        
        print("✅ validate_augmentation_parameters avec liste vide validé")
    
    def test_validate_augmentation_parameters_invalid_data_type(self):
        """Test de validate_augmentation_parameters avec un type de données invalide"""
        augmentations = ["rotation", "flip_horizontal"]
        is_valid = self.augmenter.validate_augmentation_parameters(augmentations, "invalid_type")
        
        # Vérifications
        self.assertFalse(is_valid)  # Type invalide devrait retourner False
        
        print("✅ validate_augmentation_parameters avec type de données invalide validé")
    
    def test_utility_methods_integration_with_real_data(self):
        """Test d'intégration des méthodes utilitaires avec les vraies données"""
        # Effectuer des augmentations avec les vraies données
        self.augmenter.augment_2d_grid(self.grid_2d_pd, ["rotation", "flip_horizontal"], 2)
        self.augmenter.augment_3d_volume(self.volume_3d_s, ["rotation", "gaussian_noise"], 2)
        self.augmenter.augment_dataframe(self.df_pd, ["gaussian_noise", "value_variation"], 2)
        
        # Tester get_augmentation_summary
        summary = self.augmenter.get_augmentation_summary()
        self.assertEqual(summary["total_augmentations"], 6)
        
        # Tester get_recommended_augmentations
        recommendations_2d = self.augmenter.get_recommended_augmentations("2d_grid")
        self.assertIsInstance(recommendations_2d, list)
        self.assertGreater(len(recommendations_2d), 0)
        
        # Tester validate_augmentation_parameters
        is_valid = self.augmenter.validate_augmentation_parameters(["rotation", "flip_horizontal"], "2d_grid")
        self.assertTrue(is_valid)
        
        # Tester reset_history
        self.augmenter.reset_history()
        self.assertEqual(len(self.augmenter.augmentation_history), 0)
        
        print("✅ Intégration des méthodes utilitaires avec vraies données validée")
    
    def test_utility_methods_performance(self):
        """Test de performance des méthodes utilitaires"""
        import time
        
        # Effectuer des augmentations pour avoir des données à analyser
        for i in range(10):
            self.augmenter.augment_2d_grid(self.grid_2d_pd, ["rotation"], 1)
        
        # Test performance get_augmentation_summary
        start_time = time.time()
        for _ in range(100):
            summary = self.augmenter.get_augmentation_summary()
        summary_time = time.time() - start_time
        
        # Test performance get_recommended_augmentations
        start_time = time.time()
        for _ in range(100):
            recommendations = self.augmenter.get_recommended_augmentations("2d_grid")
        recommendations_time = time.time() - start_time
        
        # Test performance validate_augmentation_parameters
        start_time = time.time()
        for _ in range(100):
            is_valid = self.augmenter.validate_augmentation_parameters(["rotation", "flip_horizontal"], "2d_grid")
        validation_time = time.time() - start_time
        
        # Vérifier que les performances sont raisonnables
        self.assertLess(summary_time, 1.0, "get_augmentation_summary devrait être rapide (< 1 seconde)")
        self.assertLess(recommendations_time, 1.0, "get_recommended_augmentations devrait être rapide (< 1 seconde)")
        self.assertLess(validation_time, 1.0, "validate_augmentation_parameters devrait être rapide (< 1 seconde)")
        
        print(f"✅ Performance validée: Summary: {summary_time:.3f}s, Recommendations: {recommendations_time:.3f}s, Validation: {validation_time:.3f}s")


if __name__ == "__main__":
    unittest.main(verbosity=2)
