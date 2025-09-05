#!/usr/bin/env python3
"""
Tests d'intégration pour le fichier main.py du projet AI-MAP.
Teste le pipeline complet avec des données réelles simulées.
"""

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
import torch
import json

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Imports du module à tester
from main import main, main_with_args, parse_arguments


class TestMainIntegration(unittest.TestCase):
    """Tests d'intégration pour le pipeline principal"""
    
    def setUp(self):
        """Configuration initiale pour chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_argv = sys.argv.copy()
        
        # Créer la structure de répertoires nécessaire
        self.artifacts_dir = Path(self.temp_dir) / "artifacts" / "models"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_data_dir = Path(self.temp_dir) / "raw_data"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer des fichiers de données factices
        self.create_fake_data()
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        sys.argv = self.original_argv.copy()
    
    def create_fake_data(self):
        """Créer des données factices pour les tests"""
        # Créer des fichiers CSV factices
        csv_dir = self.raw_data_dir / "csv" / "profiles"
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(1, 6):
            df = pd.DataFrame({
                'x': np.random.rand(50),
                'y': np.random.rand(50),
                'z': np.random.rand(50),
                'resistivity': np.random.uniform(1e-8, 1e9, 50),
                'chargeability': np.random.uniform(0, 200, 50)
            })
            df.to_csv(csv_dir / f"profil {i}.csv", index=False)
        
        # Créer des images factices
        image_dirs = ['resistivity', 'chargeability', 'profiles']
        for img_dir in image_dirs:
            img_path = self.raw_data_dir / "images" / img_dir
            img_path.mkdir(parents=True, exist_ok=True)
            
            # Créer des fichiers d'images factices (juste des fichiers vides)
            for i in range(1, 4):
                if img_dir == 'chargeability':
                    ext = 'PNG'
                else:
                    ext = 'JPG'
                (img_path / f"img{i}.{ext}").touch()
    
    @patch('main.CONFIG', new_callable=Mock)
    @patch('main.logger')
    def test_main_pipeline_complete_simulation(self, mock_logger, mock_config):
        """Test du pipeline complet avec simulation des dépendances"""
        # Configuration des mocks
        mock_config.paths.artifacts_dir = self.artifacts_dir.parent
        mock_config.paths.raw_data_dir = self.raw_data_dir
        
        # Mock de toutes les dépendances
        with patch('src.preprocessor.data_cleaner.GeophysicalDataCleaner') as mock_cleaner_class, \
             patch('src.data.data_processor.GeophysicalDataProcessor') as mock_processor_class, \
             patch('src.preprocessor.data_augmenter.GeophysicalDataAugmenter') as mock_augmenter_class, \
             patch('src.model.geophysical_trainer.GeophysicalTrainer') as mock_trainer_class, \
             patch('src.model.geophysical_trainer.GeophysicalCNN2D') as mock_cnn_class:
            
            # Configuration des mocks
            mock_cleaner = Mock()
            mock_cleaner_class.return_value = mock_cleaner
            mock_cleaner.clean_all_devices.return_value = {
                'device1': ('path1.csv', {'cleaned_count': 100, 'original_count': 120}),
                'device2': ('path2.csv', {'cleaned_count': 80, 'original_count': 90})
            }
            
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.load_and_validate.return_value = {'device1': 'data1'}
            mock_processor.get_data_summary.return_value = {'total_samples': 100}
            
            mock_augmenter = Mock()
            mock_augmenter_class.return_value = mock_augmenter
            
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            mock_trainer.prepare_data_2d.return_value = (Mock(), Mock())
            mock_trainer.train_model.return_value = {
                'epochs': [1, 2, 3],
                'train_loss': [0.5, 0.4, 0.3],
                'val_loss': [0.6, 0.5, 0.4],
                'train_accuracy': [60, 70, 80],
                'val_accuracy': [55, 65, 75]
            }
            
            mock_model = Mock()
            mock_cnn_class.return_value = mock_model
            
            # Exécution du pipeline
            result = main()
            
            # Vérifications
            self.assertTrue(result)
            mock_cleaner.clean_all_devices.assert_called_once()
            mock_processor.load_and_validate.assert_called_once()
            mock_trainer.prepare_data_2d.assert_called_once()
            mock_trainer.train_model.assert_called_once()
            mock_trainer.save_model.assert_called_once()
            
            # Vérifier que les logs ont été appelés
            self.assertTrue(mock_logger.info.called)
            self.assertTrue(mock_logger.error.call_count == 0)
    
    @patch('main.CONFIG', new_callable=Mock)
    @patch('main.logger')
    def test_main_with_args_cnn_2d(self, mock_logger, mock_config):
        """Test du pipeline avec arguments pour CNN 2D"""
        # Configuration des mocks
        mock_config.paths.artifacts_dir = self.artifacts_dir.parent
        mock_config.paths.raw_data_dir = self.raw_data_dir
        
        # Configuration des arguments
        sys.argv = [
            'main.py',
            '--model', 'cnn_2d',
            '--epochs', '5',
            '--learning-rate', '0.001',
            '--batch-size', '16',
            '--patience', '3',
            '--device', 'cpu',
            '--verbose'
        ]
        
        # Mock de toutes les dépendances
        with patch('src.preprocessor.data_cleaner.GeophysicalDataCleaner') as mock_cleaner_class, \
             patch('src.data.data_processor.GeophysicalDataProcessor') as mock_processor_class, \
             patch('src.preprocessor.data_augmenter.GeophysicalDataAugmenter') as mock_augmenter_class, \
             patch('src.model.geophysical_trainer.GeophysicalTrainer') as mock_trainer_class, \
             patch('src.model.geophysical_trainer.GeophysicalCNN2D') as mock_cnn_class:
            
            # Configuration des mocks
            mock_cleaner = Mock()
            mock_cleaner_class.return_value = mock_cleaner
            mock_cleaner.clean_all_devices.return_value = {
                'device1': ('path1.csv', {'cleaned_count': 100, 'original_count': 120})
            }
            
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.load_and_validate.return_value = {'device1': 'data1'}
            mock_processor.get_data_summary.return_value = {'total_samples': 100}
            
            mock_augmenter = Mock()
            mock_augmenter_class.return_value = mock_augmenter
            
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            mock_trainer.prepare_data_2d.return_value = (Mock(), Mock())
            mock_trainer.train_model.return_value = {
                'epochs': [1, 2, 3],
                'train_loss': [0.5, 0.4, 0.3],
                'val_loss': [0.6, 0.5, 0.4],
                'train_accuracy': [60, 70, 80],
                'val_accuracy': [55, 65, 75]
            }
            
            mock_model = Mock()
            mock_cnn_class.return_value = mock_model
            
            # Exécution du pipeline
            result = main_with_args()
            
            # Vérifications
            self.assertTrue(result)
            mock_cleaner.clean_all_devices.assert_called_once()
            mock_processor.load_and_validate.assert_called_once()
            mock_trainer.prepare_data_2d.assert_called_once()
            mock_trainer.train_model.assert_called_once()
            mock_trainer.save_model.assert_called_once()
    
    @patch('main.CONFIG', new_callable=Mock)
    @patch('main.logger')
    def test_main_with_args_skip_phases(self, mock_logger, mock_config):
        """Test du pipeline avec phases ignorées"""
        # Configuration des mocks
        mock_config.paths.artifacts_dir = self.artifacts_dir.parent
        mock_config.paths.raw_data_dir = self.raw_data_dir
        
        # Configuration des arguments pour ignorer certaines phases
        sys.argv = [
            'main.py',
            '--model', 'cnn_2d',
            '--skip-cleaning',
            '--skip-processing',
            '--skip-training',
            '--epochs', '5'
        ]
        
        # Mock de toutes les dépendances
        with patch('src.preprocessor.data_cleaner.GeophysicalDataCleaner') as mock_cleaner_class, \
             patch('src.data.data_processor.GeophysicalDataProcessor') as mock_processor_class, \
             patch('src.preprocessor.data_augmenter.GeophysicalDataAugmenter') as mock_augmenter_class, \
             patch('src.model.geophysical_trainer.GeophysicalTrainer') as mock_trainer_class:
            
            # Configuration des mocks
            mock_cleaner = Mock()
            mock_cleaner_class.return_value = mock_cleaner
            
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            
            mock_augmenter = Mock()
            mock_augmenter_class.return_value = mock_augmenter
            
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            
            # Exécution du pipeline
            result = main_with_args()
            
            # Vérifications
            self.assertTrue(result)
            
            # Vérifier que les phases ignorées n'ont pas été appelées
            mock_cleaner.clean_all_devices.assert_not_called()
            mock_processor.load_and_validate.assert_not_called()
            mock_trainer.prepare_data_2d.assert_not_called()
            mock_trainer.train_model.assert_not_called()
    
    @patch('main.CONFIG', new_callable=Mock)
    @patch('main.logger')
    def test_main_with_args_hybrid_model(self, mock_logger, mock_config):
        """Test du pipeline avec modèle hybride"""
        # Configuration des mocks
        mock_config.paths.artifacts_dir = self.artifacts_dir.parent
        mock_config.paths.raw_data_dir = self.raw_data_dir
        
        # Configuration des arguments
        sys.argv = [
            'main.py',
            '--model', 'hybrid',
            '--epochs', '3',
            '--learning-rate', '0.0001',
            '--batch-size', '8',
            '--patience', '2',
            '--device', 'cpu'
        ]
        
        # Mock de toutes les dépendances
        with patch('src.preprocessor.data_cleaner.GeophysicalDataCleaner') as mock_cleaner_class, \
             patch('src.data.data_processor.GeophysicalDataProcessor') as mock_processor_class, \
             patch('src.preprocessor.data_augmenter.GeophysicalDataAugmenter') as mock_augmenter_class, \
             patch('src.model.geophysical_trainer.GeophysicalTrainer') as mock_trainer_class, \
             patch('src.model.geophysical_image_trainer.GeophysicalImageTrainer') as mock_image_trainer_class, \
             patch('src.model.geophysical_hybrid_net.GeophysicalHybridNet') as mock_hybrid_class, \
             patch('main.torch.save') as mock_torch_save:
            
            # Configuration des mocks
            mock_cleaner = Mock()
            mock_cleaner_class.return_value = mock_cleaner
            mock_cleaner.clean_all_devices.return_value = {
                'device1': ('path1.csv', {'cleaned_count': 100, 'original_count': 120})
            }
            
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.load_and_validate.return_value = {'device1': 'data1'}
            mock_processor.get_data_summary.return_value = {'total_samples': 100}
            
            mock_augmenter = Mock()
            mock_augmenter_class.return_value = mock_augmenter
            
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            
            mock_hybrid_trainer = Mock()
            mock_image_trainer_class.return_value = mock_hybrid_trainer
            mock_hybrid_trainer.prepare_hybrid_data.return_value = (Mock(), Mock())
            mock_hybrid_trainer.train_hybrid_model.return_value = {
                'epochs': [1, 2, 3],
                'train_loss': [0.5, 0.4, 0.3],
                'val_loss': [0.6, 0.5, 0.4],
                'train_accuracy': [60, 70, 80],
                'val_accuracy': [55, 65, 75]
            }
            
            mock_model = Mock()
            mock_hybrid_class.return_value = mock_model
            
            # Exécution du pipeline
            result = main_with_args()
            
            # Vérifications
            self.assertTrue(result)
            mock_cleaner.clean_all_devices.assert_called_once()
            mock_processor.load_and_validate.assert_called_once()
            mock_hybrid_trainer.prepare_hybrid_data.assert_called_once()
            mock_hybrid_trainer.train_hybrid_model.assert_called_once()
            mock_torch_save.assert_called_once()
    
    @patch('main.CONFIG', new_callable=Mock)
    @patch('main.logger')
    def test_main_with_args_dataframe_model(self, mock_logger, mock_config):
        """Test du pipeline avec modèle DataFrame"""
        # Configuration des mocks
        mock_config.paths.artifacts_dir = self.artifacts_dir.parent
        mock_config.paths.raw_data_dir = self.raw_data_dir
        
        # Configuration des arguments
        sys.argv = [
            'main.py',
            '--model', 'dataframe',
            '--epochs', '3',
            '--learning-rate', '0.001',
            '--batch-size', '16',
            '--patience', '2',
            '--device', 'cpu'
        ]
        
        # Mock de toutes les dépendances
        with patch('src.preprocessor.data_cleaner.GeophysicalDataCleaner') as mock_cleaner_class, \
             patch('src.data.data_processor.GeophysicalDataProcessor') as mock_processor_class, \
             patch('src.preprocessor.data_augmenter.GeophysicalDataAugmenter') as mock_augmenter_class, \
             patch('src.model.geophysical_trainer.GeophysicalTrainer') as mock_trainer_class, \
             patch('src.model.geophysical_trainer.GeophysicalDataFrameNet') as mock_dataframe_class:
            
            # Configuration des mocks
            mock_cleaner = Mock()
            mock_cleaner_class.return_value = mock_cleaner
            mock_cleaner.clean_all_devices.return_value = {
                'device1': ('path1.csv', {'cleaned_count': 100, 'original_count': 120})
            }
            
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.load_and_validate.return_value = {'device1': 'data1'}
            mock_processor.get_data_summary.return_value = {'total_samples': 100}
            
            mock_augmenter = Mock()
            mock_augmenter_class.return_value = mock_augmenter
            
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            mock_trainer.prepare_data_dataframe.return_value = (Mock(), Mock())
            mock_trainer.train_model.return_value = {
                'epochs': [1, 2, 3],
                'train_loss': [0.5, 0.4, 0.3],
                'val_loss': [0.6, 0.5, 0.4],
                'train_accuracy': [60, 70, 80],
                'val_accuracy': [55, 65, 75]
            }
            
            mock_model = Mock()
            mock_dataframe_class.return_value = mock_model
            
            # Exécution du pipeline
            result = main_with_args()
            
            # Vérifications
            self.assertTrue(result)
            mock_cleaner.clean_all_devices.assert_called_once()
            mock_processor.load_and_validate.assert_called_once()
            mock_trainer.prepare_data_dataframe.assert_called_once()
            mock_trainer.train_model.assert_called_once()
            mock_trainer.save_model.assert_called_once()
    
    @patch('main.CONFIG', new_callable=Mock)
    @patch('main.logger')
    def test_main_with_args_cnn_3d(self, mock_logger, mock_config):
        """Test du pipeline avec modèle CNN 3D"""
        # Configuration des mocks
        mock_config.paths.artifacts_dir = self.artifacts_dir.parent
        mock_config.paths.raw_data_dir = self.raw_data_dir
        
        # Configuration des arguments
        sys.argv = [
            'main.py',
            '--model', 'cnn_3d',
            '--epochs', '3',
            '--learning-rate', '0.001',
            '--batch-size', '8',
            '--patience', '2',
            '--device', 'cpu'
        ]
        
        # Mock de toutes les dépendances
        with patch('src.preprocessor.data_cleaner.GeophysicalDataCleaner') as mock_cleaner_class, \
             patch('src.data.data_processor.GeophysicalDataProcessor') as mock_processor_class, \
             patch('src.preprocessor.data_augmenter.GeophysicalDataAugmenter') as mock_augmenter_class, \
             patch('src.model.geophysical_trainer.GeophysicalTrainer') as mock_trainer_class, \
             patch('src.model.geophysical_trainer.GeophysicalCNN3D') as mock_cnn_class:
            
            # Configuration des mocks
            mock_cleaner = Mock()
            mock_cleaner_class.return_value = mock_cleaner
            mock_cleaner.clean_all_devices.return_value = {
                'device1': ('path1.csv', {'cleaned_count': 100, 'original_count': 120})
            }
            
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.load_and_validate.return_value = {'device1': 'data1'}
            mock_processor.get_data_summary.return_value = {'total_samples': 100}
            
            mock_augmenter = Mock()
            mock_augmenter_class.return_value = mock_augmenter
            
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            mock_trainer.prepare_data_3d.return_value = (Mock(), Mock())
            mock_trainer.train_model.return_value = {
                'epochs': [1, 2, 3],
                'train_loss': [0.5, 0.4, 0.3],
                'val_loss': [0.6, 0.5, 0.4],
                'train_accuracy': [60, 70, 80],
                'val_accuracy': [55, 65, 75]
            }
            
            mock_model = Mock()
            mock_cnn_class.return_value = mock_model
            
            # Exécution du pipeline
            result = main_with_args()
            
            # Vérifications
            self.assertTrue(result)
            mock_cleaner.clean_all_devices.assert_called_once()
            mock_processor.load_and_validate.assert_called_once()
            mock_trainer.prepare_data_3d.assert_called_once()
            mock_trainer.train_model.assert_called_once()
            mock_trainer.save_model.assert_called_once()
    
    def test_parse_arguments_all_models(self):
        """Test du parsing des arguments pour tous les modèles"""
        models = ['cnn_2d', 'cnn_3d', 'hybrid', 'dataframe']
        
        for model in models:
            with self.subTest(model=model):
                sys.argv = ['main.py', '--model', model]
                args = parse_arguments()
                self.assertEqual(args.model, model)
    
    def test_parse_arguments_all_devices(self):
        """Test du parsing des arguments pour tous les devices"""
        devices = ['auto', 'cpu', 'cuda']
        
        for device in devices:
            with self.subTest(device=device):
                sys.argv = ['main.py', '--device', device]
                args = parse_arguments()
                self.assertEqual(args.device, device)
    
    def test_parse_arguments_numeric_values(self):
        """Test du parsing des arguments numériques"""
        sys.argv = [
            'main.py',
            '--epochs', '100',
            '--learning-rate', '0.0001',
            '--batch-size', '64',
            '--patience', '20'
        ]
        args = parse_arguments()
        
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.learning_rate, 0.0001)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.patience, 20)
    
    def test_parse_arguments_boolean_flags(self):
        """Test du parsing des arguments booléens"""
        sys.argv = [
            'main.py',
            '--skip-cleaning',
            '--skip-processing',
            '--skip-training',
            '--verbose'
        ]
        args = parse_arguments()
        
        self.assertTrue(args.skip_cleaning)
        self.assertTrue(args.skip_processing)
        self.assertTrue(args.skip_training)
        self.assertTrue(args.verbose)
    
    def test_parse_arguments_output_dir(self):
        """Test du parsing de l'argument output-dir"""
        output_dir = "/tmp/test_output"
        sys.argv = ['main.py', '--output-dir', output_dir]
        args = parse_arguments()
        
        self.assertEqual(args.output_dir, output_dir)


if __name__ == '__main__':
    # Configuration pour les tests
    unittest.main(verbosity=2)
