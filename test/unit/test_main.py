#!/usr/bin/env python3
"""
Tests unitaires pour le fichier main.py du projet AI-MAP.
Teste toutes les fonctions principales du pipeline d'entraînement.
"""

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
import pandas as pd
import torch
import json
import argparse

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Imports du module à tester
from main import (
    phase1_data_cleaning,
    phase2_data_processing,
    phase3_data_preparation,
    phase4_model_training,
    phase5_evaluation_and_results,
    train_cnn_2d,
    train_cnn_3d,
    train_hybrid_model,
    train_dataframe_model,
    main,
    main_with_args,
    parse_arguments
)


class TestMainPipeline(unittest.TestCase):
    """Tests pour les fonctions principales du pipeline main.py"""
    
    def setUp(self):
        """Configuration initiale pour chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_argv = sys.argv.copy()
        
        # Mock de la configuration
        self.mock_config = Mock()
        self.mock_config.paths.artifacts_dir = Path(self.temp_dir) / "artifacts"
        self.mock_config.paths.raw_data_dir = Path(self.temp_dir) / "raw_data"
        
    def tearDown(self):
        """Nettoyage après chaque test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        sys.argv = self.original_argv.copy()
    
    @patch('main.CONFIG', new_callable=Mock)
    @patch('main.logger')
    @patch('src.preprocessor.data_cleaner.GeophysicalDataCleaner')
    def test_phase1_data_cleaning_success(self, mock_cleaner_class, mock_logger, mock_config):
        """Test de la phase 1 de nettoyage des données avec succès"""
        # Configuration des mocks
        mock_config.paths.artifacts_dir = Path(self.temp_dir) / "artifacts"
        mock_cleaner = Mock()
        mock_cleaner_class.return_value = mock_cleaner
        
        # Données de test
        cleaning_results = {
            'device1': ('path1.csv', {'cleaned_count': 100, 'original_count': 120}),
            'device2': ('path2.csv', {'cleaned_count': 80, 'original_count': 90})
        }
        mock_cleaner.clean_all_devices.return_value = cleaning_results
        
        # Exécution
        result = phase1_data_cleaning()
        
        # Vérifications
        self.assertEqual(result, cleaning_results)
        mock_cleaner.clean_all_devices.assert_called_once()
        mock_logger.info.assert_called()
        self.assertIn("📋 Phase 1: Nettoyage et prétraitement des données", 
                     [call[0][0] for call in mock_logger.info.call_args_list])
    
    @patch('main.CONFIG', new_callable=Mock)
    @patch('main.logger')
    @patch('src.data.data_processor.GeophysicalDataProcessor')
    def test_phase2_data_processing_success(self, mock_processor_class, mock_logger, mock_config):
        """Test de la phase 2 de traitement des données avec succès"""
        # Configuration des mocks
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.load_and_validate.return_value = {'device1': 'data1'}
        
        # Exécution
        processor, multi_device_tensor, volume_3d = phase2_data_processing()
        
        # Vérifications
        self.assertEqual(processor, mock_processor)
        self.assertIsNone(multi_device_tensor)
        self.assertIsNone(volume_3d)
        mock_processor.load_and_validate.assert_called_once()
        mock_logger.info.assert_called()
    
    @patch('main.CONFIG', new_callable=Mock)
    @patch('main.logger')
    @patch('src.data.data_processor.GeophysicalDataProcessor')
    def test_phase2_data_processing_no_data(self, mock_processor_class, mock_logger, mock_config):
        """Test de la phase 2 avec aucune donnée valide"""
        # Configuration des mocks
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.load_and_validate.return_value = None
        
        # Exécution
        processor, multi_device_tensor, volume_3d = phase2_data_processing()
        
        # Vérifications
        self.assertEqual(processor, mock_processor)
        self.assertIsNone(multi_device_tensor)
        self.assertIsNone(volume_3d)
        mock_logger.warning.assert_called()
    
    @patch('main.logger')
    def test_phase3_data_preparation_with_data(self, mock_logger):
        """Test de la phase 3 avec des données"""
        # Données de test
        processor = Mock()
        processor.split_data.return_value = (np.array([1, 2, 3]), np.array([4, 5]))
        multi_device_tensor = np.random.rand(100, 4, 64, 64)
        
        # Exécution
        x_train, x_test = phase3_data_preparation(processor, multi_device_tensor)
        
        # Vérifications
        self.assertEqual(len(x_train), 3)
        self.assertEqual(len(x_test), 2)
        mock_logger.info.assert_called()
    
    @patch('main.logger')
    def test_phase3_data_preparation_no_data(self, mock_logger):
        """Test de la phase 3 sans données"""
        # Exécution
        x_train, x_test = phase3_data_preparation(None, None)
        
        # Vérifications
        self.assertEqual(len(x_train), 0)
        self.assertEqual(len(x_test), 0)
        mock_logger.warning.assert_called()
    
    @patch('main.CONFIG', new_callable=Mock)
    @patch('main.logger')
    @patch('src.preprocessor.data_augmenter.GeophysicalDataAugmenter')
    @patch('src.model.geophysical_trainer.GeophysicalTrainer')
    def test_phase4_model_training_cnn_2d(self, mock_trainer_class, mock_augmenter_class, 
                                        mock_logger, mock_config):
        """Test de la phase 4 d'entraînement avec modèle CNN 2D"""
        # Configuration des mocks
        mock_config.paths.artifacts_dir = Path(self.temp_dir) / "artifacts"
        mock_augmenter = Mock()
        mock_augmenter_class.return_value = mock_augmenter
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        # Données de test
        x_train = np.random.rand(50, 4, 64, 64)
        x_test = np.random.rand(20, 4, 64, 64)
        training_config = {
            "epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 32,
            "patience": 5,
            "device": "cpu"
        }
        
        # Mock des résultats d'entraînement
        mock_trainer.prepare_data_2d.return_value = (Mock(), Mock())
        mock_trainer.train_model.return_value = {
            "epochs": [1, 2, 3],
            "train_loss": [0.5, 0.4, 0.3],
            "val_loss": [0.6, 0.5, 0.4],
            "train_accuracy": [60, 70, 80],
            "val_accuracy": [55, 65, 75]
        }
        
        # Exécution
        with patch('main.train_cnn_2d') as mock_train_cnn_2d:
            mock_train_cnn_2d.return_value = {
                "model_type": "CNN_2D",
                "model": Mock(),
                "history": {},
                "model_path": "test_path.pth"
            }
            
            result = phase4_model_training(
                model_type="cnn_2d",
                x_train=x_train,
                x_test=x_test,
                training_config=training_config
            )
        
        # Vérifications
        self.assertEqual(result["model_type"], "CNN_2D")
        mock_trainer_class.assert_called_once()
        mock_logger.info.assert_called()
    
    @patch('main.CONFIG', new_callable=Mock)
    @patch('main.logger')
    @patch('src.preprocessor.data_augmenter.GeophysicalDataAugmenter')
    @patch('src.model.geophysical_trainer.GeophysicalTrainer')
    def test_phase4_model_training_invalid_model(self, mock_trainer_class, mock_augmenter_class, 
                                                mock_logger, mock_config):
        """Test de la phase 4 avec un type de modèle invalide"""
        # Configuration des mocks
        mock_augmenter = Mock()
        mock_augmenter_class.return_value = mock_augmenter
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        training_config = {"epochs": 10, "learning_rate": 0.001, "batch_size": 32, 
                          "patience": 5, "device": "cpu"}
        
        # Exécution et vérification
        with self.assertRaises(ValueError):
            phase4_model_training(
                model_type="invalid_model",
                training_config=training_config
            )
    
    @patch('main.CONFIG', new_callable=Mock)
    @patch('main.logger')
    def test_phase5_evaluation_and_results(self, mock_logger, mock_config):
        """Test de la phase 5 d'évaluation et résultats"""
        # Configuration des mocks
        mock_config.paths.artifacts_dir = Path(self.temp_dir) / "artifacts"
        
        # Données de test
        training_results = {
            "model_type": "CNN_2D",
            "model_path": "test_model.pth",
            "history": {
                "epochs": [1, 2, 3],
                "train_loss": [0.5, 0.4, 0.3],
                "val_loss": [0.6, 0.5, 0.4],
                "train_accuracy": [60, 70, 80],
                "val_accuracy": [55, 65, 75]
            }
        }
        
        processor = Mock()
        processor.get_data_summary.return_value = {"total_samples": 100}
        
        # Exécution
        result = phase5_evaluation_and_results(training_results, processor)
        
        # Vérifications
        self.assertIn("training_results", result)
        self.assertIn("evaluation_metrics", result)
        self.assertIn("model_summary", result)
        self.assertEqual(result["training_results"], training_results)
        self.assertEqual(result["model_summary"]["model_type"], "CNN_2D")
        mock_logger.info.assert_called()
    
    @patch('main.CONFIG', new_callable=Mock)
    @patch('main.logger')
    @patch('src.preprocessor.data_augmenter.GeophysicalDataAugmenter')
    @patch('src.model.geophysical_trainer.GeophysicalTrainer')
    def test_train_cnn_2d(self, mock_trainer_class, mock_augmenter_class, 
                         mock_logger, mock_config):
        """Test de la fonction train_cnn_2d"""
        # Configuration des mocks
        mock_config.paths.artifacts_dir = Path(self.temp_dir) / "artifacts"
        mock_augmenter = Mock()
        mock_augmenter_class.return_value = mock_augmenter
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        # Données de test
        x_train = np.random.rand(50, 4, 64, 64)
        x_test = np.random.rand(20, 4, 64, 64)
        config = {
            "epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 32,
            "patience": 5,
            "device": "cpu"
        }
        
        # Mock des résultats
        mock_trainer.prepare_data_2d.return_value = (Mock(), Mock())
        mock_trainer.train_model.return_value = {
            "epochs": [1, 2, 3],
            "train_loss": [0.5, 0.4, 0.3],
            "val_loss": [0.6, 0.5, 0.4],
            "train_accuracy": [60, 70, 80],
            "val_accuracy": [55, 65, 75]
        }
        
        # Exécution
        with patch('src.model.geophysical_trainer.GeophysicalCNN2D') as mock_cnn_class:
            mock_model = Mock()
            mock_cnn_class.return_value = mock_model
            
            result = train_cnn_2d(mock_trainer, x_train, x_test, config)
        
        # Vérifications
        self.assertEqual(result["model_type"], "CNN_2D")
        self.assertIn("model", result)
        self.assertIn("history", result)
        self.assertIn("model_path", result)
        mock_trainer.prepare_data_2d.assert_called_once()
        mock_trainer.train_model.assert_called_once()
        mock_trainer.save_model.assert_called_once()
    
    @patch('main.CONFIG', new_callable=Mock)
    @patch('main.logger')
    @patch('src.preprocessor.data_augmenter.GeophysicalDataAugmenter')
    @patch('src.model.geophysical_trainer.GeophysicalTrainer')
    def test_train_cnn_3d(self, mock_trainer_class, mock_augmenter_class, 
                         mock_logger, mock_config):
        """Test de la fonction train_cnn_3d"""
        # Configuration des mocks
        mock_config.paths.artifacts_dir = Path(self.temp_dir) / "artifacts"
        mock_augmenter = Mock()
        mock_augmenter_class.return_value = mock_augmenter
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        # Données de test
        volume_3d = np.random.rand(10, 4, 32, 32, 32)
        config = {
            "epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 32,
            "patience": 5,
            "device": "cpu"
        }
        
        # Mock des résultats
        mock_trainer.prepare_data_3d.return_value = (Mock(), Mock())
        mock_trainer.train_model.return_value = {
            "epochs": [1, 2, 3],
            "train_loss": [0.5, 0.4, 0.3],
            "val_loss": [0.6, 0.5, 0.4],
            "train_accuracy": [60, 70, 80],
            "val_accuracy": [55, 65, 75]
        }
        
        # Exécution
        with patch('src.model.geophysical_trainer.GeophysicalCNN3D') as mock_cnn_class:
            mock_model = Mock()
            mock_cnn_class.return_value = mock_model
            
            result = train_cnn_3d(mock_trainer, volume_3d, config)
        
        # Vérifications
        self.assertEqual(result["model_type"], "CNN_3D")
        self.assertIn("model", result)
        self.assertIn("history", result)
        self.assertIn("model_path", result)
        mock_trainer.prepare_data_3d.assert_called_once()
        mock_trainer.train_model.assert_called_once()
        mock_trainer.save_model.assert_called_once()
    
    @patch('main.CONFIG', new_callable=Mock)
    @patch('main.logger')
    @patch('src.preprocessor.data_augmenter.GeophysicalDataAugmenter')
    @patch('src.model.geophysical_trainer.GeophysicalTrainer')
    def test_train_cnn_3d_no_volume(self, mock_trainer_class, mock_augmenter_class, 
                                   mock_logger, mock_config):
        """Test de la fonction train_cnn_3d sans volume 3D"""
        # Configuration des mocks
        mock_config.paths.artifacts_dir = Path(self.temp_dir) / "artifacts"
        mock_augmenter = Mock()
        mock_augmenter_class.return_value = mock_augmenter
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        config = {
            "epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 32,
            "patience": 5,
            "device": "cpu"
        }
        
        # Mock des résultats
        mock_trainer.prepare_data_3d.return_value = (Mock(), Mock())
        mock_trainer.train_model.return_value = {}
        
        # Exécution
        with patch('src.model.geophysical_trainer.GeophysicalCNN3D') as mock_cnn_class:
            mock_model = Mock()
            mock_cnn_class.return_value = mock_model
            
            result = train_cnn_3d(mock_trainer, None, config)
        
        # Vérifications
        self.assertEqual(result["model_type"], "CNN_3D")
        mock_logger.warning.assert_called()
    
    @patch('main.CONFIG', new_callable=Mock)
    @patch('main.logger')
    @patch('src.preprocessor.data_augmenter.GeophysicalDataAugmenter')
    @patch('src.model.geophysical_trainer.GeophysicalTrainer')
    def test_train_hybrid_model(self, mock_trainer_class, mock_augmenter_class, 
                               mock_logger, mock_config):
        """Test de la fonction train_hybrid_model"""
        # Configuration des mocks
        mock_config.paths.artifacts_dir = Path(self.temp_dir) / "artifacts"
        mock_config.paths.raw_data_dir = Path(self.temp_dir) / "raw_data"
        mock_augmenter = Mock()
        mock_augmenter_class.return_value = mock_augmenter
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        config = {
            "epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 32,
            "patience": 5,
            "device": "cpu"
        }
        
        # Mock des résultats
        mock_hybrid_trainer = Mock()
        mock_hybrid_trainer.prepare_hybrid_data.return_value = (Mock(), Mock())
        mock_hybrid_trainer.train_hybrid_model.return_value = {
            "epochs": [1, 2, 3],
            "train_loss": [0.5, 0.4, 0.3],
            "val_loss": [0.6, 0.5, 0.4],
            "train_accuracy": [60, 70, 80],
            "val_accuracy": [55, 65, 75]
        }
        
        # Exécution
        with patch('src.model.geophysical_image_trainer.GeophysicalImageTrainer') as mock_image_trainer_class, \
             patch('src.model.geophysical_hybrid_net.GeophysicalHybridNet') as mock_hybrid_class, \
             patch('main.torch.save') as mock_torch_save:
            
            mock_image_trainer_class.return_value = mock_hybrid_trainer
            mock_model = Mock()
            mock_hybrid_class.return_value = mock_model
            
            result = train_hybrid_model(mock_trainer, config)
        
        # Vérifications
        self.assertEqual(result["model_type"], "HYBRID")
        self.assertIn("model", result)
        self.assertIn("history", result)
        self.assertIn("model_path", result)
        mock_hybrid_trainer.prepare_hybrid_data.assert_called_once()
        mock_hybrid_trainer.train_hybrid_model.assert_called_once()
        mock_torch_save.assert_called_once()
    
    @patch('main.CONFIG', new_callable=Mock)
    @patch('main.logger')
    @patch('src.preprocessor.data_augmenter.GeophysicalDataAugmenter')
    @patch('src.model.geophysical_trainer.GeophysicalTrainer')
    def test_train_dataframe_model(self, mock_trainer_class, mock_augmenter_class, 
                                  mock_logger, mock_config):
        """Test de la fonction train_dataframe_model"""
        # Configuration des mocks
        mock_config.paths.artifacts_dir = Path(self.temp_dir) / "artifacts"
        mock_augmenter = Mock()
        mock_augmenter_class.return_value = mock_augmenter
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        config = {
            "epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 32,
            "patience": 5,
            "device": "cpu"
        }
        
        # Mock des résultats
        mock_trainer.prepare_data_dataframe.return_value = (Mock(), Mock())
        mock_trainer.train_model.return_value = {
            "epochs": [1, 2, 3],
            "train_loss": [0.5, 0.4, 0.3],
            "val_loss": [0.6, 0.5, 0.4],
            "train_accuracy": [60, 70, 80],
            "val_accuracy": [55, 65, 75]
        }
        
        # Exécution
        with patch('src.model.geophysical_trainer.GeophysicalDataFrameNet') as mock_dataframe_class:
            mock_model = Mock()
            mock_dataframe_class.return_value = mock_model
            
            result = train_dataframe_model(mock_trainer, config)
        
        # Vérifications
        self.assertEqual(result["model_type"], "DATAFRAME")
        self.assertIn("model", result)
        self.assertIn("history", result)
        self.assertIn("model_path", result)
        mock_trainer.prepare_data_dataframe.assert_called_once()
        mock_trainer.train_model.assert_called_once()
        mock_trainer.save_model.assert_called_once()
    
    def test_parse_arguments_default(self):
        """Test du parsing des arguments avec valeurs par défaut"""
        # Test avec arguments par défaut
        sys.argv = ['main.py']
        args = parse_arguments()
        
        self.assertEqual(args.model, 'cnn_2d')
        self.assertEqual(args.epochs, 50)
        self.assertEqual(args.learning_rate, 0.001)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.patience, 10)
        self.assertEqual(args.device, 'auto')
        self.assertFalse(args.skip_cleaning)
        self.assertFalse(args.skip_processing)
        self.assertFalse(args.skip_training)
        self.assertIsNone(args.output_dir)
        self.assertFalse(args.verbose)
    
    def test_parse_arguments_custom(self):
        """Test du parsing des arguments avec valeurs personnalisées"""
        # Test avec arguments personnalisés
        sys.argv = [
            'main.py',
            '--model', 'hybrid',
            '--epochs', '100',
            '--learning-rate', '0.0001',
            '--batch-size', '16',
            '--patience', '15',
            '--device', 'cuda',
            '--skip-cleaning',
            '--skip-processing',
            '--output-dir', '/tmp/test',
            '--verbose'
        ]
        args = parse_arguments()
        
        self.assertEqual(args.model, 'hybrid')
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.learning_rate, 0.0001)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.patience, 15)
        self.assertEqual(args.device, 'cuda')
        self.assertTrue(args.skip_cleaning)
        self.assertTrue(args.skip_processing)
        self.assertFalse(args.skip_training)
        self.assertEqual(args.output_dir, '/tmp/test')
        self.assertTrue(args.verbose)
    
    @patch('main.phase1_data_cleaning')
    @patch('main.phase2_data_processing')
    @patch('main.phase3_data_preparation')
    @patch('main.phase4_model_training')
    @patch('main.phase5_evaluation_and_results')
    @patch('main.logger')
    def test_main_success(self, mock_logger, mock_phase5, mock_phase4, 
                         mock_phase3, mock_phase2, mock_phase1):
        """Test de la fonction main() avec succès"""
        # Configuration des mocks
        mock_phase1.return_value = {'device1': ('path1', {})}
        mock_phase2.return_value = (Mock(), np.array([]), np.array([]))
        mock_phase3.return_value = (np.array([]), np.array([]))
        mock_phase4.return_value = {'model_type': 'CNN_2D', 'model_path': 'test.pth'}
        mock_phase5.return_value = {'evaluation_metrics': {}}
        
        # Exécution
        result = main()
        
        # Vérifications
        self.assertTrue(result)
        mock_phase1.assert_called_once()
        mock_phase2.assert_called_once()
        mock_phase3.assert_called_once()
        mock_phase4.assert_called_once()
        mock_phase5.assert_called_once()
        mock_logger.info.assert_called()
    
    @patch('main.phase1_data_cleaning')
    @patch('main.logger')
    def test_main_failure(self, mock_logger, mock_phase1):
        """Test de la fonction main() avec échec"""
        # Configuration des mocks pour simuler une erreur
        mock_phase1.side_effect = Exception("Test error")
        
        # Exécution et vérification
        with self.assertRaises(Exception):
            main()
        
        mock_logger.error.assert_called()
    
    @patch('main.phase1_data_cleaning')
    @patch('main.phase2_data_processing')
    @patch('main.phase3_data_preparation')
    @patch('main.phase4_model_training')
    @patch('main.phase5_evaluation_and_results')
    @patch('main.logger')
    @patch('main.CONFIG', new_callable=Mock)
    def test_main_with_args_success(self, mock_config, mock_logger, mock_phase5, 
                                   mock_phase4, mock_phase3, mock_phase2, mock_phase1):
        """Test de la fonction main_with_args() avec succès"""
        # Configuration des mocks
        mock_config.paths.artifacts_dir = Path(self.temp_dir) / "artifacts"
        mock_phase1.return_value = {'device1': ('path1', {})}
        mock_phase2.return_value = (Mock(), np.array([]), np.array([]))
        mock_phase3.return_value = (np.array([]), np.array([]))
        mock_phase4.return_value = {'model_type': 'CNN_2D', 'model_path': 'test.pth'}
        mock_phase5.return_value = {'evaluation_metrics': {}}
        
        # Configuration des arguments
        sys.argv = ['main.py', '--model', 'cnn_2d', '--epochs', '10']
        
        # Exécution
        result = main_with_args()
        
        # Vérifications
        self.assertTrue(result)
        mock_phase1.assert_called_once()
        mock_phase2.assert_called_once()
        mock_phase3.assert_called_once()
        mock_phase4.assert_called_once()
        mock_phase5.assert_called_once()
        mock_logger.info.assert_called()
    
    @patch('main.phase1_data_cleaning')
    @patch('main.phase2_data_processing')
    @patch('main.phase3_data_preparation')
    @patch('main.phase4_model_training')
    @patch('main.phase5_evaluation_and_results')
    @patch('main.logger')
    @patch('main.CONFIG', new_callable=Mock)
    def test_main_with_args_skip_phases(self, mock_config, mock_logger, mock_phase5, 
                                       mock_phase4, mock_phase3, mock_phase2, mock_phase1):
        """Test de la fonction main_with_args() avec phases ignorées"""
        # Configuration des mocks
        mock_config.paths.artifacts_dir = Path(self.temp_dir) / "artifacts"
        mock_phase4.return_value = {'model_type': 'CNN_2D', 'model_path': 'test.pth'}
        mock_phase5.return_value = {'evaluation_metrics': {}}
        
        # Configuration des arguments pour ignorer certaines phases
        sys.argv = [
            'main.py', 
            '--skip-cleaning', 
            '--skip-processing', 
            '--skip-training'
        ]
        
        # Exécution
        result = main_with_args()
        
        # Vérifications
        self.assertTrue(result)
        mock_phase1.assert_not_called()  # Phase ignorée
        mock_phase2.assert_not_called()  # Phase ignorée
        mock_phase3.assert_not_called()  # Phase ignorée
        mock_phase4.assert_not_called()  # Phase ignorée
        mock_phase5.assert_called_once()
        mock_logger.info.assert_called()


class TestMainIntegration(unittest.TestCase):
    """Tests d'intégration pour le fichier main.py"""
    
    def setUp(self):
        """Configuration initiale pour chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_argv = sys.argv.copy()
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        sys.argv = self.original_argv.copy()
    
    @patch('main.CONFIG', new_callable=Mock)
    def test_main_module_import(self, mock_config):
        """Test que le module main peut être importé correctement"""
        # Vérifier que les fonctions principales sont importables
        from main import (
            phase1_data_cleaning,
            phase2_data_processing,
            phase3_data_preparation,
            phase4_model_training,
            phase5_evaluation_and_results,
            main,
            main_with_args,
            parse_arguments
        )
        
        # Vérifier que les fonctions existent
        self.assertTrue(callable(phase1_data_cleaning))
        self.assertTrue(callable(phase2_data_processing))
        self.assertTrue(callable(phase3_data_preparation))
        self.assertTrue(callable(phase4_model_training))
        self.assertTrue(callable(phase5_evaluation_and_results))
        self.assertTrue(callable(main))
        self.assertTrue(callable(main_with_args))
        self.assertTrue(callable(parse_arguments))
    
    def test_parse_arguments_help(self):
        """Test que l'aide des arguments fonctionne"""
        # Test avec --help
        sys.argv = ['main.py', '--help']
        
        with self.assertRaises(SystemExit):
            parse_arguments()
    
    def test_parse_arguments_invalid_model(self):
        """Test avec un modèle invalide"""
        sys.argv = ['main.py', '--model', 'invalid_model']
        
        # Le parser devrait accepter seulement les choix valides
        # Donc cette commande devrait échouer
        with self.assertRaises(SystemExit):
            parse_arguments()


if __name__ == '__main__':
    # Configuration pour les tests
    unittest.main(verbosity=2)
