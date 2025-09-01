#!/usr/bin/env python3
"""
Tests unitaires pour les méthodes utilitaires de GeophysicalTrainer.

Teste get_augmentation_summary, validate_augmentations_for_data_type et reset_training_history.
"""

import unittest
import numpy as np
import torch
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

from src.model.geophysical_trainer import GeophysicalTrainer
from src.preprocessor.data_augmenter import GeophysicalDataAugmenter
from src.utils.logger import logger


class TestGeophysicalTrainerUtilityMethods(unittest.TestCase):
    """Tests pour les méthodes utilitaires de GeophysicalTrainer."""
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Créer un augmenter de test
        self.augmenter = GeophysicalDataAugmenter()
        
        # Créer un entraîneur de test
        self.trainer = GeophysicalTrainer(self.augmenter, device="cpu")
        
        # Créer un historique d'entraînement de test
        self.original_training_history = {
            "epochs": [0, 1, 2, 3, 4],
            "train_loss": [0.8, 0.7, 0.6, 0.5, 0.4],
            "val_loss": [0.85, 0.75, 0.65, 0.55, 0.45],
            "train_accuracy": [45.0, 55.0, 65.0, 75.0, 85.0],
            "val_accuracy": [40.0, 50.0, 60.0, 70.0, 80.0]
        }
        
        # Appliquer l'historique de test
        self.trainer.training_history = self.original_training_history.copy()
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        shutil.rmtree(self.temp_dir)
    
    def test_get_augmentation_summary_basic_functionality(self):
        """Tester la fonctionnalité de base de get_augmentation_summary."""
        # Mock de la méthode get_augmentation_summary de l'augmenter
        mock_summary = {
            "total_augmentations": 150,
            "techniques_used": ["gaussian_noise", "rotation", "flip_horizontal"],
            "success_rate": 0.95
        }
        
        with patch.object(self.augmenter, 'get_augmentation_summary', return_value=mock_summary):
            result = self.trainer.get_augmentation_summary()
            
            # Vérifier que le résultat est correct
            self.assertEqual(result, mock_summary)
            self.assertIn("total_augmentations", result)
            self.assertIn("techniques_used", result)
            self.assertIn("success_rate", result)
    
    def test_get_augmentation_summary_empty_result(self):
        """Tester get_augmentation_summary avec un résultat vide."""
        empty_summary = {}
        
        with patch.object(self.augmenter, 'get_augmentation_summary', return_value=empty_summary):
            result = self.trainer.get_augmentation_summary()
            
            # Vérifier que le résultat est vide
            self.assertEqual(result, {})
            self.assertIsInstance(result, dict)
    
    def test_get_augmentation_summary_complex_result(self):
        """Tester get_augmentation_summary avec un résultat complexe."""
        complex_summary = {
            "augmentations_by_type": {
                "2d": {"rotation": 45, "flip_horizontal": 30, "gaussian_noise": 25},
                "3d": {"rotation": 20, "gaussian_noise": 15},
                "dataframe": {"gaussian_noise": 15}
            },
            "performance_metrics": {
                "average_time": 0.15,
                "memory_usage": "2.3GB",
                "quality_score": 0.87
            }
        }
        
        with patch.object(self.augmenter, 'get_augmentation_summary', return_value=complex_summary):
            result = self.trainer.get_augmentation_summary()
            
            # Vérifier la structure complexe
            self.assertEqual(result, complex_summary)
            self.assertIn("augmentations_by_type", result)
            self.assertIn("performance_metrics", result)
            self.assertIn("2d", result["augmentations_by_type"])
            self.assertIn("3d", result["augmentations_by_type"])
            self.assertIn("dataframe", result["augmentations_by_type"])
    
    def test_validate_augmentations_for_data_type_2d_valid(self):
        """Tester validate_augmentations_for_data_type avec des augmentations 2D valides."""
        valid_augmentations = ["rotation", "flip_horizontal", "gaussian_noise"]
        
        result = self.trainer.validate_augmentations_for_data_type(valid_augmentations, "2d")
        
        # Vérifier que la validation réussit
        self.assertTrue(result)
    
    def test_validate_augmentations_for_data_type_2d_invalid(self):
        """Tester validate_augmentations_for_data_type avec des augmentations 2D invalides."""
        invalid_augmentations = ["rotation", "invalid_technique", "gaussian_noise"]
        
        result = self.trainer.validate_augmentations_for_data_type(invalid_augmentations, "2d")
        
        # Vérifier que la validation échoue
        self.assertFalse(result)
    
    def test_validate_augmentations_for_data_type_3d_valid(self):
        """Tester validate_augmentations_for_data_type avec des augmentations 3D valides."""
        valid_augmentations = ["rotation", "gaussian_noise", "value_variation"]
        
        result = self.trainer.validate_augmentations_for_data_type(valid_augmentations, "3d")
        
        # Vérifier que la validation réussit
        self.assertTrue(result)
    
    def test_validate_augmentations_for_data_type_3d_invalid(self):
        """Tester validate_augmentations_for_data_type avec des augmentations 3D invalides."""
        invalid_augmentations = ["elastic_deformation", "spatial_shift"]  # Techniques 2D uniquement
        
        result = self.trainer.validate_augmentations_for_data_type(invalid_augmentations, "3d")
        
        # Vérifier que la validation échoue
        self.assertFalse(result)
    
    def test_validate_augmentations_for_data_type_dataframe_valid(self):
        """Tester validate_augmentations_for_data_type avec des augmentations DataFrame valides."""
        valid_augmentations = ["gaussian_noise", "value_variation", "spatial_jitter"]
        
        result = self.trainer.validate_augmentations_for_data_type(valid_augmentations, "dataframe")
        
        # Vérifier que la validation réussit
        self.assertTrue(result)
    
    def test_validate_augmentations_for_data_type_dataframe_invalid(self):
        """Tester validate_augmentations_for_data_type avec des augmentations DataFrame invalides."""
        invalid_augmentations = ["rotation", "flip_horizontal"]  # Techniques 2D uniquement
        
        result = self.trainer.validate_augmentations_for_data_type(invalid_augmentations, "dataframe")
        
        # Vérifier que la validation échoue
        self.assertFalse(result)
    
    def test_validate_augmentations_for_data_type_unsupported_type(self):
        """Tester validate_augmentations_for_data_type avec un type de données non supporté."""
        augmentations = ["gaussian_noise"]
        
        # Vérifier que ValueError est levé pour un type non supporté
        with self.assertRaises(ValueError) as context:
            self.trainer.validate_augmentations_for_data_type(augmentations, "unsupported_type")
        
        # Vérifier le message d'erreur
        self.assertIn("Type de données non supporté", str(context.exception))
    
    def test_validate_augmentations_for_data_type_empty_list(self):
        """Tester validate_augmentations_for_data_type avec une liste vide."""
        empty_augmentations = []
        
        result = self.trainer.validate_augmentations_for_data_type(empty_augmentations, "2d")
        
        # Vérifier que la validation réussit avec une liste vide
        self.assertTrue(result)
    
    def test_validate_augmentations_for_data_type_mixed_valid_invalid(self):
        """Tester validate_augmentations_for_data_type avec un mélange d'augmentations valides et invalides."""
        mixed_augmentations = ["rotation", "valid_technique", "gaussian_noise", "invalid_technique"]
        
        result = self.trainer.validate_augmentations_for_data_type(mixed_augmentations, "2d")
        
        # Vérifier que la validation échoue à cause des techniques invalides
        self.assertFalse(result)
    
    def test_reset_training_history_basic_functionality(self):
        """Tester la fonctionnalité de base de reset_training_history."""
        # Vérifier que l'historique existe avant la réinitialisation
        self.assertIsNotNone(self.trainer.training_history)
        self.assertIn("epochs", self.trainer.training_history)
        self.assertIn("train_loss", self.trainer.training_history)
        self.assertIn("val_loss", self.trainer.training_history)
        self.assertIn("train_accuracy", self.trainer.training_history)
        self.assertIn("val_accuracy", self.trainer.training_history)
        
        # Réinitialiser l'historique
        self.trainer.reset_training_history()
        
        # Vérifier que l'historique a été réinitialisé
        expected_empty_history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "epochs": []
        }
        
        self.assertEqual(self.trainer.training_history, expected_empty_history)
    
    def test_reset_training_history_preserves_structure(self):
        """Tester que reset_training_history préserve la structure de l'historique."""
        # Réinitialiser l'historique
        self.trainer.reset_training_history()
        
        # Vérifier que toutes les clés sont présentes
        required_keys = ["train_loss", "val_loss", "train_accuracy", "val_accuracy", "epochs"]
        for key in required_keys:
            self.assertIn(key, self.trainer.training_history)
            self.assertIsInstance(self.trainer.training_history[key], list)
            self.assertEqual(len(self.trainer.training_history[key]), 0)
    
    def test_reset_training_history_multiple_calls(self):
        """Tester que reset_training_history fonctionne correctement lors d'appels multiples."""
        # Première réinitialisation
        self.trainer.reset_training_history()
        
        # Vérifier que l'historique est vide après la première réinitialisation
        for key in self.trainer.training_history:
            self.assertEqual(len(self.trainer.training_history[key]), 0)
        
        # Ajouter des données
        self.trainer.training_history["epochs"].append(0)
        self.trainer.training_history["train_loss"].append(0.5)
        
        # Vérifier que les données ont été ajoutées
        self.assertEqual(len(self.trainer.training_history["epochs"]), 1)
        self.assertEqual(len(self.trainer.training_history["train_loss"]), 1)
        
        # Deuxième réinitialisation
        self.trainer.reset_training_history()
        
        # Vérifier que l'historique est vide après la deuxième réinitialisation
        for key in self.trainer.training_history:
            self.assertEqual(len(self.trainer.training_history[key]), 0)
    
    def test_reset_training_history_after_training(self):
        """Tester reset_training_history après un entraînement simulé."""
        # Simuler un entraînement en ajoutant des données
        self.trainer.training_history["epochs"] = [0, 1, 2, 3, 4]
        self.trainer.training_history["train_loss"] = [0.8, 0.7, 0.6, 0.5, 0.4]
        self.trainer.training_history["val_loss"] = [0.85, 0.75, 0.65, 0.55, 0.45]
        self.trainer.training_history["train_accuracy"] = [45.0, 55.0, 65.0, 75.0, 85.0]
        self.trainer.training_history["val_accuracy"] = [40.0, 50.0, 60.0, 70.0, 80.0]
        
        # Vérifier que l'historique contient des données
        self.assertGreater(len(self.trainer.training_history["epochs"]), 0)
        self.assertGreater(len(self.trainer.training_history["train_loss"]), 0)
        
        # Réinitialiser l'historique
        self.trainer.reset_training_history()
        
        # Vérifier que l'historique est vide
        for key in self.trainer.training_history:
            self.assertEqual(len(self.trainer.training_history[key]), 0)
    
    def test_reset_training_history_logging(self):
        """Tester que reset_training_history génère les logs appropriés."""
        with patch('src.utils.logger.logger.info') as mock_logger:
            self.trainer.reset_training_history()
            
            # Vérifier que le log a été appelé
            mock_logger.assert_called_once_with("Historique d'entraînement réinitialisé")
    
    def test_utility_methods_integration(self):
        """Tester l'intégration des méthodes utilitaires ensemble."""
        # 1. Valider des augmentations
        valid_augmentations = ["gaussian_noise", "value_variation"]
        is_valid = self.trainer.validate_augmentations_for_data_type(valid_augmentations, "dataframe")
        self.assertTrue(is_valid)
        
        # 2. Obtenir le résumé des augmentations
        mock_summary = {"total": 100, "success": 95}
        with patch.object(self.augmenter, 'get_augmentation_summary', return_value=mock_summary):
            summary = self.trainer.get_augmentation_summary()
            self.assertEqual(summary, mock_summary)
        
        # 3. Réinitialiser l'historique
        self.trainer.reset_training_history()
        self.assertEqual(len(self.trainer.training_history["epochs"]), 0)
        
        # 4. Vérifier que l'historique est dans un état valide
        required_keys = ["train_loss", "val_loss", "train_accuracy", "val_accuracy", "epochs"]
        for key in required_keys:
            self.assertIn(key, self.trainer.training_history)
            self.assertIsInstance(self.trainer.training_history[key], list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
