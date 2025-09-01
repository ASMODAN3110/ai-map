#!/usr/bin/env python3
"""
Tests unitaires pour la méthode load_model de GeophysicalTrainer.

Teste le chargement des modèles sauvegardés, la restauration de l'historique et la gestion des erreurs.
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
import json

# Ajouter le répertoire racine au path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.geophysical_trainer import GeophysicalTrainer, GeophysicalCNN2D
from src.preprocessor.data_augmenter import GeophysicalDataAugmenter
from src.utils.logger import logger


class TestGeophysicalTrainerLoadModel(unittest.TestCase):
    """Tests pour la méthode load_model de GeophysicalTrainer."""
    
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
        
        # Créer un modèle de test simple
        self.test_model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        # Ajouter des attributs au modèle pour tester la configuration
        self.test_model.input_channels = 10
        self.test_model.num_classes = 2
        self.test_model.grid_size = (32, 32)
        self.test_model.volume_size = (32, 32, 32)
        self.test_model.input_features = 10
        
        # Créer un checkpoint de test
        self.test_checkpoint = {
            'model_state_dict': self.test_model.state_dict(),
            'training_history': self.original_training_history,
            'model_config': {
                'input_channels': 10,
                'num_classes': 2,
                'grid_size': (32, 32),
                'volume_size': (32, 32, 32),
                'input_features': 10
            }
        }
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_model_basic_functionality(self):
        """Tester le chargement de base du modèle."""
        # Sauvegarder d'abord un modèle
        save_path = os.path.join(self.temp_dir, "test_model.pth")
        torch.save(self.test_checkpoint, save_path)
        
        # Créer un nouveau modèle vide
        new_model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        # Charger le modèle
        loaded_model = self.trainer.load_model(new_model, save_path)
        
        # Vérifier que le modèle a été chargé
        self.assertIsInstance(loaded_model, torch.nn.Module)
        
        # Vérifier que l'historique d'entraînement a été restauré
        self.assertEqual(self.trainer.training_history, self.original_training_history)
    
    def test_load_model_state_dict_restoration(self):
        """Tester la restauration du state_dict du modèle."""
        # Sauvegarder d'abord un modèle
        save_path = os.path.join(self.temp_dir, "test_state_dict.pth")
        torch.save(self.test_checkpoint, save_path)
        
        # Créer un nouveau modèle avec des poids aléatoires
        new_model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        # Sauvegarder les poids originaux pour comparaison
        original_weights = {}
        for name, param in new_model.named_parameters():
            original_weights[name] = param.clone()
        
        # Charger le modèle
        loaded_model = self.trainer.load_model(new_model, save_path)
        
        # Vérifier que les poids ont été modifiés (chargés)
        for name, param in loaded_model.named_parameters():
            if name in original_weights:
                # Les poids doivent être différents des poids aléatoires originaux
                self.assertFalse(torch.equal(param, original_weights[name]))
    
    def test_load_model_training_history_restoration(self):
        """Tester la restauration de l'historique d'entraînement."""
        # Sauvegarder d'abord un modèle
        save_path = os.path.join(self.temp_dir, "test_history.pth")
        torch.save(self.test_checkpoint, save_path)
        
        # Vider l'historique d'entraînement du trainer
        self.trainer.training_history = {}
        
        # Créer un nouveau modèle
        new_model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        # Charger le modèle
        self.trainer.load_model(new_model, save_path)
        
        # Vérifier que l'historique d'entraînement a été restauré
        self.assertEqual(self.trainer.training_history, self.original_training_history)
        
        # Vérifier chaque clé individuellement
        for key in self.original_training_history:
            self.assertIn(key, self.trainer.training_history)
            self.assertEqual(self.trainer.training_history[key], self.original_training_history[key])
    
    def test_load_model_return_value(self):
        """Tester que la méthode retourne bien le modèle chargé."""
        # Sauvegarder d'abord un modèle
        save_path = os.path.join(self.temp_dir, "test_return.pth")
        torch.save(self.test_checkpoint, save_path)
        
        # Créer un nouveau modèle
        new_model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        # Charger le modèle
        returned_model = self.trainer.load_model(new_model, save_path)
        
        # Vérifier que le modèle retourné est le même que celui passé en paramètre
        self.assertIs(returned_model, new_model)
    
    def test_load_model_device_mapping(self):
        """Tester le mapping du device lors du chargement."""
        # Sauvegarder d'abord un modèle
        save_path = os.path.join(self.temp_dir, "test_device.pth")
        torch.save(self.test_checkpoint, save_path)
        
        # Créer un nouveau modèle
        new_model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        # Charger le modèle
        loaded_model = self.trainer.load_model(new_model, save_path)
        
        # Vérifier que le modèle est sur le bon device
        for param in loaded_model.parameters():
            self.assertEqual(param.device.type, 'cpu')  # device="cpu" dans setUp
    
    def test_load_model_file_not_found(self):
        """Tester la gestion d'un fichier inexistant."""
        # Chemin vers un fichier inexistant
        non_existent_path = os.path.join(self.temp_dir, "non_existent_model.pth")
        
        # Créer un nouveau modèle
        new_model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        # Tester que la méthode lève une exception pour un fichier inexistant
        with self.assertRaises(FileNotFoundError):
            self.trainer.load_model(new_model, non_existent_path)
    
    def test_load_model_corrupted_checkpoint(self):
        """Tester la gestion d'un checkpoint corrompu."""
        # Créer un fichier corrompu
        corrupted_path = os.path.join(self.temp_dir, "corrupted_model.pth")
        with open(corrupted_path, 'w') as f:
            f.write("Ceci n'est pas un checkpoint valide")
        
        # Créer un nouveau modèle
        new_model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        # Tester que la méthode lève une exception pour un checkpoint corrompu
        with self.assertRaises(Exception):  # Peut être RuntimeError, pickle.UnpicklingError, etc.
            self.trainer.load_model(new_model, corrupted_path)
    
    def test_load_model_missing_checkpoint_keys(self):
        """Tester la gestion d'un checkpoint avec des clés manquantes."""
        # Créer un checkpoint incomplet
        incomplete_checkpoint = {
            'model_state_dict': self.test_model.state_dict()
            # 'training_history' et 'model_config' manquent
        }
        
        # Sauvegarder le checkpoint incomplet
        save_path = os.path.join(self.temp_dir, "incomplete_model.pth")
        torch.save(incomplete_checkpoint, save_path)
        
        # Créer un nouveau modèle
        new_model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        # Tester que la méthode lève une exception pour un checkpoint incomplet
        with self.assertRaises(KeyError):
            self.trainer.load_model(new_model, save_path)
    
    def test_load_model_logging(self):
        """Tester le logging lors du chargement."""
        # Sauvegarder d'abord un modèle
        save_path = os.path.join(self.temp_dir, "test_logging.pth")
        torch.save(self.test_checkpoint, save_path)
        
        # Créer un nouveau modèle
        new_model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        # Mock du logger pour capturer les appels
        with patch('src.utils.logger.logger.info') as mock_logger:
            # Charger le modèle
            self.trainer.load_model(new_model, save_path)
            
            # Vérifier que le logging a été effectué
            mock_logger.assert_called_once_with(f"Modèle chargé: {save_path}")
    
    def test_load_model_preserve_model_structure(self):
        """Tester que la structure du modèle est préservée après chargement."""
        # Sauvegarder d'abord un modèle
        save_path = os.path.join(self.temp_dir, "test_structure.pth")
        torch.save(self.test_checkpoint, save_path)
        
        # Créer un nouveau modèle avec la même structure
        new_model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        # Sauvegarder la structure originale
        original_modules = list(new_model.modules())
        original_named_modules = dict(new_model.named_modules())
        
        # Charger le modèle
        loaded_model = self.trainer.load_model(new_model, save_path)
        
        # Vérifier que la structure est préservée
        loaded_modules = list(loaded_model.modules())
        loaded_named_modules = dict(loaded_model.named_modules())
        
        self.assertEqual(len(original_modules), len(loaded_modules))
        self.assertEqual(set(original_named_modules.keys()), set(loaded_named_modules.keys()))
    
    def test_load_model_complex_model(self):
        """Tester le chargement d'un modèle plus complexe."""
        # Créer un modèle plus complexe
        complex_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 10)
        )
        
        # Créer un checkpoint pour ce modèle
        complex_checkpoint = {
            'model_state_dict': complex_model.state_dict(),
            'training_history': self.original_training_history,
            'model_config': {
                'input_channels': 3,
                'num_classes': 10,
                'grid_size': (64, 64),
                'volume_size': (64, 64, 64),
                'input_features': 3
            }
        }
        
        # Sauvegarder le modèle complexe
        save_path = os.path.join(self.temp_dir, "test_complex_model.pth")
        torch.save(complex_checkpoint, save_path)
        
        # Créer un nouveau modèle complexe vide
        new_complex_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 10)
        )
        
        # Charger le modèle
        loaded_complex_model = self.trainer.load_model(new_complex_model, save_path)
        
        # Vérifier que le modèle a été chargé
        self.assertIsInstance(loaded_complex_model, torch.nn.Module)
        
        # Vérifier que l'historique d'entraînement a été restauré
        self.assertEqual(self.trainer.training_history, self.original_training_history)
    
    def test_load_model_empty_training_history(self):
        """Tester le chargement avec un historique d'entraînement vide."""
        # Créer un checkpoint avec un historique vide
        empty_history_checkpoint = {
            'model_state_dict': self.test_model.state_dict(),
            'training_history': {},
            'model_config': {
                'input_channels': 10,
                'num_classes': 2,
                'grid_size': (32, 32),
                'volume_size': (32, 32, 32),
                'input_features': 10
            }
        }
        
        # Sauvegarder le checkpoint
        save_path = os.path.join(self.temp_dir, "test_empty_history.pth")
        torch.save(empty_history_checkpoint, save_path)
        
        # Créer un nouveau modèle
        new_model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        # Charger le modèle
        self.trainer.load_model(new_model, save_path)
        
        # Vérifier que l'historique vide a été restauré
        self.assertEqual(self.trainer.training_history, {})


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(verbosity=2)
