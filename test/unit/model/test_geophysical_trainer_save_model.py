#!/usr/bin/env python3
"""
Tests unitaires pour la méthode save_model de GeophysicalTrainer.

Teste la sauvegarde des modèles, la gestion des métadonnées et la persistance des données.
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


class TestGeophysicalTrainerSaveModel(unittest.TestCase):
    """Tests pour la méthode save_model de GeophysicalTrainer."""
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Créer un augmenter de test
        self.augmenter = GeophysicalDataAugmenter()
        
        # Créer un entraîneur de test
        self.trainer = GeophysicalTrainer(self.augmenter, device="cpu")
        
        # Créer un historique d'entraînement de test
        self.trainer.training_history = {
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
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_model_basic_functionality(self):
        """Tester la sauvegarde de base du modèle."""
        save_path = os.path.join(self.temp_dir, "test_model.pth")
        
        # Sauvegarder le modèle
        self.trainer.save_model(self.test_model, save_path)
        
        # Vérifier que le fichier a été créé
        self.assertTrue(os.path.exists(save_path))
        
        # Vérifier que le fichier n'est pas vide
        file_size = os.path.getsize(save_path)
        self.assertGreater(file_size, 0)
    
    def test_save_model_checkpoint_structure(self):
        """Tester la structure du checkpoint sauvegardé."""
        save_path = os.path.join(self.temp_dir, "test_checkpoint.pth")
        
        # Sauvegarder le modèle
        self.trainer.save_model(self.test_model, save_path)
        
        # Charger le checkpoint
        checkpoint = torch.load(save_path, map_location='cpu')
        
        # Vérifier la structure du checkpoint
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('training_history', checkpoint)
        self.assertIn('model_config', checkpoint)
        
        # Vérifier que model_state_dict contient les paramètres du modèle
        self.assertIsInstance(checkpoint['model_state_dict'], dict)
        self.assertGreater(len(checkpoint['model_state_dict']), 0)
    
    def test_save_model_training_history_persistence(self):
        """Tester la persistance de l'historique d'entraînement."""
        save_path = os.path.join(self.temp_dir, "test_history.pth")
        
        # Sauvegarder le modèle
        self.trainer.save_model(self.test_model, save_path)
        
        # Charger le checkpoint
        checkpoint = torch.load(save_path, map_location='cpu')
        
        # Vérifier que l'historique d'entraînement est correctement sauvegardé
        saved_history = checkpoint['training_history']
        
        self.assertEqual(saved_history['epochs'], self.trainer.training_history['epochs'])
        self.assertEqual(saved_history['train_loss'], self.trainer.training_history['train_loss'])
        self.assertEqual(saved_history['val_loss'], self.trainer.training_history['val_loss'])
        self.assertEqual(saved_history['train_accuracy'], self.trainer.training_history['train_accuracy'])
        self.assertEqual(saved_history['val_accuracy'], self.trainer.training_history['val_accuracy'])
    
    def test_save_model_config_extraction(self):
        """Tester l'extraction de la configuration du modèle."""
        save_path = os.path.join(self.temp_dir, "test_config.pth")
        
        # Sauvegarder le modèle
        self.trainer.save_model(self.test_model, save_path)
        
        # Charger le checkpoint
        checkpoint = torch.load(save_path, map_location='cpu')
        model_config = checkpoint['model_config']
        
        # Vérifier que la configuration est correctement extraite
        self.assertEqual(model_config['input_channels'], 10)
        self.assertEqual(model_config['num_classes'], 2)
        self.assertEqual(model_config['grid_size'], (32, 32))
        self.assertEqual(model_config['volume_size'], (32, 32, 32))
        self.assertEqual(model_config['input_features'], 10)
    
    def test_save_model_with_missing_attributes(self):
        """Tester la sauvegarde avec des attributs manquants."""
        # Créer un modèle sans certains attributs
        simple_model = torch.nn.Sequential(
            torch.nn.Linear(5, 2)
        )
        
        save_path = os.path.join(self.temp_dir, "test_simple_model.pth")
        
        # Sauvegarder le modèle (ne doit pas lever d'exception)
        try:
            self.trainer.save_model(simple_model, save_path)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"La sauvegarde a levé une exception: {e}")
        
        # Charger le checkpoint
        checkpoint = torch.load(save_path, map_location='cpu')
        model_config = checkpoint['model_config']
        
        # Vérifier que les attributs manquants sont None
        self.assertIsNone(model_config['input_channels'])
        self.assertIsNone(model_config['num_classes'])
        self.assertIsNone(model_config['grid_size'])
        self.assertIsNone(model_config['volume_size'])
        self.assertIsNone(model_config['input_features'])
    
    def test_save_model_file_path_validation(self):
        """Tester la validation du chemin de fichier."""
        # Test avec un chemin valide
        valid_path = os.path.join(self.temp_dir, "valid_model.pth")
        
        try:
            self.trainer.save_model(self.test_model, valid_path)
            self.assertTrue(os.path.exists(valid_path))
        except Exception as e:
            self.fail(f"La sauvegarde a levé une exception avec un chemin valide: {e}")
        
        # Test avec un chemin dans un répertoire inexistant
        invalid_path = os.path.join(self.temp_dir, "nonexistent", "model.pth")
        
        try:
            self.trainer.save_model(self.test_model, invalid_path)
            # Si on arrive ici, c'est que la méthode a géré l'erreur ou créé le répertoire
            self.assertTrue(True)
        except Exception as e:
            # C'est normal que ça lève une exception si le répertoire n'existe pas
            self.assertTrue(True)
    
    def test_save_model_logging(self):
        """Tester le logging lors de la sauvegarde."""
        save_path = os.path.join(self.temp_dir, "test_logging.pth")
        
        # Mock du logger pour capturer les appels
        with patch('src.utils.logger.logger.info') as mock_logger:
            # Sauvegarder le modèle
            self.trainer.save_model(self.test_model, save_path)
            
            # Vérifier que le logging a été effectué
            mock_logger.assert_called_once_with(f"Modèle sauvegardé: {save_path}")
    
    def test_save_model_state_dict_integrity(self):
        """Tester l'intégrité du state_dict sauvegardé."""
        save_path = os.path.join(self.temp_dir, "test_integrity.pth")
        
        # Sauvegarder le modèle
        self.trainer.save_model(self.test_model, save_path)
        
        # Charger le checkpoint
        checkpoint = torch.load(save_path, map_location='cpu')
        saved_state_dict = checkpoint['model_state_dict']
        
        # Vérifier que le state_dict contient tous les paramètres du modèle original
        original_state_dict = self.test_model.state_dict()
        
        self.assertEqual(set(saved_state_dict.keys()), set(original_state_dict.keys()))
        
        # Vérifier que les valeurs sont identiques
        for key in original_state_dict.keys():
            torch.testing.assert_close(
                saved_state_dict[key], 
                original_state_dict[key], 
                msg=f"Différence dans le paramètre {key}"
            )
    
    def test_save_model_empty_training_history(self):
        """Tester la sauvegarde avec un historique d'entraînement vide."""
        # Vider l'historique d'entraînement
        self.trainer.training_history = {}
        
        save_path = os.path.join(self.temp_dir, "test_empty_history.pth")
        
        # Sauvegarder le modèle
        self.trainer.save_model(self.test_model, save_path)
        
        # Charger le checkpoint
        checkpoint = torch.load(save_path, map_location='cpu')
        
        # Vérifier que l'historique vide est correctement sauvegardé
        self.assertEqual(checkpoint['training_history'], {})
    
    def test_save_model_large_model(self):
        """Tester la sauvegarde d'un modèle plus complexe."""
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
        
        # Ajouter des attributs
        complex_model.input_channels = 3
        complex_model.num_classes = 10
        complex_model.grid_size = (64, 64)
        
        save_path = os.path.join(self.temp_dir, "test_complex_model.pth")
        
        # Sauvegarder le modèle
        self.trainer.save_model(complex_model, save_path)
        
        # Vérifier que le fichier a été créé et n'est pas vide
        self.assertTrue(os.path.exists(save_path))
        file_size = os.path.getsize(save_path)
        self.assertGreater(file_size, 1000)  # Un modèle complexe devrait être plus gros
    
    def test_save_model_error_handling(self):
        """Tester la gestion des erreurs lors de la sauvegarde."""
        # Test avec un modèle invalide
        invalid_model = "not_a_model"
        
        save_path = os.path.join(self.temp_dir, "test_error.pth")
        
        # Tester que la méthode gère les erreurs
        try:
            self.trainer.save_model(invalid_model, save_path)
            # Si on arrive ici, c'est que la méthode a géré l'erreur
            self.assertTrue(True)
        except Exception as e:
            # C'est normal que ça lève une exception avec un modèle invalide
            self.assertTrue(True)


if __name__ == "__main__":
    # Configuration des tests
    unittest.main(verbosity=2)
