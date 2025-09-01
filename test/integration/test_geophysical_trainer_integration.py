#!/usr/bin/env python3
"""
Test d'intégration complet pour GeophysicalTrainer avec données réelles.

Ce test utilise les données réelles PD.csv et S.csv pour tester le pipeline complet.
"""

import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
import shutil
from pathlib import Path
import sys
import os
from unittest.mock import patch
import warnings

# Ajouter le répertoire racine au path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.model.geophysical_trainer import GeophysicalTrainer, GeophysicalDataFrameNet
from src.preprocessor.data_augmenter import GeophysicalDataAugmenter
from src.utils.logger import logger

# Ignorer les warnings pour les tests
warnings.filterwarnings("ignore")


class TestGeophysicalTrainerIntegration(unittest.TestCase):
    """Test d'intégration complet pour GeophysicalTrainer avec données réelles."""
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Créer un augmenter de test
        self.augmenter = GeophysicalDataAugmenter()
        
        # Créer un entraîneur de test
        self.trainer = GeophysicalTrainer(self.augmenter, device="cpu")
        
        # Chemins vers les données réelles
        self.pd_csv_path = project_root / "data" / "raw" / "PD.csv"
        self.s_csv_path = project_root / "data" / "raw" / "S.csv"
        
        # Vérifier que les fichiers existent
        if not self.pd_csv_path.exists():
            raise FileNotFoundError(f"Fichier PD.csv non trouvé: {self.pd_csv_path}")
        if not self.s_csv_path.exists():
            raise FileNotFoundError(f"Fichier S.csv non trouvé: {self.s_csv_path}")
        
        # Charger les données réelles
        self.pd_data = self.load_pd_data()
        self.s_data = self.load_s_data()
        
        # Créer des labels factices pour la classification
        self.pd_labels = self.create_geological_labels(self.pd_data)
        self.s_labels = self.create_geological_labels(self.s_data)
        
        logger.info(f"Données chargées: PD={len(self.pd_data)} lignes, S={len(self.s_data)} lignes")
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        shutil.rmtree(self.temp_dir)
    
    def load_pd_data(self) -> pd.DataFrame:
        """Charger et nettoyer les données PD.csv."""
        try:
            df = pd.read_csv(self.pd_csv_path, sep=';', decimal=',')
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=['Rho(ohm.m)', 'M (mV/V)', 'SP (mV)', 'VMN (mV)', 'IAB (mA)'])
            
            numeric_columns = ['Rho(ohm.m)', 'M (mV/V)', 'SP (mV)', 'VMN (mV)', 'IAB (mA)']
            df_clean = df[numeric_columns].copy()
            
            # Forcer la conversion en float pour toutes les colonnes
            for col in numeric_columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Supprimer les lignes avec des valeurs manquantes après conversion
            df_clean = df_clean.dropna()
            
            # Normaliser les données
            for col in numeric_columns:
                mean_val = df_clean[col].mean()
                std_val = df_clean[col].std()
                if std_val > 0:
                    df_clean[col] = (df_clean[col] - mean_val) / std_val
            
            logger.info(f"Données PD chargées: {df_clean.shape}, types: {df_clean.dtypes.to_dict()}")
            return df_clean
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données PD: {e}")
            raise
    
    def load_s_data(self) -> pd.DataFrame:
        """Charger et nettoyer les données S.csv."""
        try:
            df = pd.read_csv(self.s_csv_path, sep=';', decimal=',')
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=['Rho (Ohm.m)', 'M (mV/V)', 'SP (mV) ', 'VMN (mV)', 'IAB (mA)'])
            
            numeric_columns = ['Rho (Ohm.m)', 'M (mV/V)', 'SP (mV) ', 'VMN (mV)', 'IAB (mA)']
            df_clean = df[numeric_columns].copy()
            
            # Normaliser les données
            for col in numeric_columns:
                if df_clean[col].dtype in ['float64', 'int64']:
                    mean_val = df_clean[col].mean()
                    std_val = df_clean[col].std()
                    if std_val > 0:
                        df_clean[col] = (df_clean[col] - mean_val) / std_val
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données S: {e}")
            raise
    
    def create_geological_labels(self, df: pd.DataFrame) -> list:
        """Créer des labels factices basés sur les caractéristiques géophysiques."""
        labels = []
        
        for _, row in df.iterrows():
            if 'Rho' in df.columns:
                rho_col = 'Rho' if 'Rho' in df.columns else 'Rho(ohm.m)'
                rho_val = row[rho_col]
                
                if rho_val < -1.0:
                    label = 0  # Anomalie conductrice
                elif rho_val > 1.0:
                    label = 1  # Anomalie résistive
                else:
                    label = 2  # Sol normal
            else:
                label = np.random.randint(0, 3)
            
            labels.append(label)
        
        return labels
    
    def test_data_loading_and_preparation(self):
        """Tester le chargement et la préparation des données réelles."""
        self.assertIsInstance(self.pd_data, pd.DataFrame)
        self.assertIsInstance(self.s_data, pd.DataFrame)
        self.assertGreater(len(self.pd_data), 0)
        self.assertGreater(len(self.s_data), 0)
        self.assertEqual(len(self.pd_labels), len(self.pd_data))
        self.assertEqual(len(self.s_labels), len(self.s_data))
        self.assertTrue(all(0 <= label <= 2 for label in self.pd_labels))
        self.assertTrue(all(0 <= label <= 2 for label in self.s_labels))
        
        logger.info("✅ Chargement et préparation des données réelles réussi")
    
    def test_prepare_data_dataframe_with_real_data(self):
        """Tester la préparation des données avec augmentation sur données réelles."""
        # Test avec les données PD
        # Pour DataFrameNet, nous devons passer un label par DataFrame, pas par ligne
        train_loader, val_loader = self.trainer.prepare_data_dataframe(
            dataframes=[self.pd_data],
            labels=[0],  # Un seul label pour tout le DataFrame
            augmentations=["gaussian_noise", "value_variation"],
            num_augmentations=3,
            test_size=0.2,
            random_state=42
        )
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        self.assertEqual(len(train_batch), 2)
        self.assertEqual(len(val_batch), 2)
        
        X_train, y_train = train_batch
        X_val, y_val = val_batch
        
        self.assertIsInstance(X_train, torch.Tensor)
        self.assertIsInstance(y_train, torch.Tensor)
        self.assertIsInstance(X_val, torch.Tensor)
        self.assertIsInstance(y_val, torch.Tensor)
        
        # Debug: afficher les formes des tenseurs
        logger.info(f"Formes des tenseurs - X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"Formes des tenseurs - X_val: {X_val.shape}, y_val: {y_val.shape}")
        
        # Vérifier que les dimensions sont correctes pour DataFrameNet
        # X_train devrait être [batch_size, num_features]
        # y_train devrait être [batch_size]
        self.assertEqual(len(X_train.shape), 2)  # 2D: [batch_size, num_features]
        self.assertEqual(len(y_train.shape), 1)  # 1D: [batch_size]
        self.assertEqual(X_train.shape[1], self.pd_data.shape[1])  # num_features
        
        logger.info(f"✅ Préparation des données PD réussie: {X_train.shape}, {y_train.shape}")
    
    def test_model_creation_and_training_with_real_data(self):
        """Tester la création et l'entraînement du modèle avec données réelles."""
        # Préparer les données
        train_loader, val_loader = self.trainer.prepare_data_dataframe(
            dataframes=[self.pd_data],
            labels=[0],  # Un seul label pour tout le DataFrame
            augmentations=["gaussian_noise"],
            num_augmentations=2,
            test_size=0.2,
            random_state=42
        )
        
        # Créer le modèle
        input_features = self.pd_data.shape[1]
        num_classes = len(set(self.pd_labels))
        
        model = GeophysicalDataFrameNet(
            input_features=input_features,
            num_classes=num_classes,
            hidden_layers=[64, 32],
            dropout_rate=0.2
        )
        
        self.assertIsInstance(model, GeophysicalDataFrameNet)
        self.assertEqual(model.input_features, input_features)
        self.assertEqual(model.num_classes, num_classes)
        
        logger.info(f"✅ Modèle créé: {input_features} features → {num_classes} classes")
        
        # Entraîner le modèle
        training_history = self.trainer.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=5,
            learning_rate=0.001,
            weight_decay=1e-5,
            patience=3
        )
        
        self.assertIsInstance(training_history, dict)
        self.assertIn('train_loss', training_history)
        self.assertIn('val_loss', training_history)
        self.assertIn('train_accuracy', training_history)
        self.assertIn('val_accuracy', training_history)
        self.assertIn('epochs', training_history)
        
        self.assertEqual(len(training_history['epochs']), 5)
        self.assertEqual(len(training_history['train_loss']), 5)
        self.assertEqual(len(training_history['val_loss']), 5)
        
        final_train_loss = training_history['train_loss'][-1]
        initial_train_loss = training_history['train_loss'][0]
        self.assertLessEqual(final_train_loss, initial_train_loss * 1.5)
        
        logger.info(f"✅ Entraînement réussi: Loss finale = {final_train_loss:.4f}")
    
    def test_model_evaluation_with_real_data(self):
        """Tester l'évaluation du modèle sur données réelles."""
        # Préparer les données
        train_loader, val_loader = self.trainer.prepare_data_dataframe(
            dataframes=[self.pd_data],
            labels=[0],  # Un seul label pour tout le DataFrame
            augmentations=["gaussian_noise"],
            num_augmentations=1,
            test_size=0.3,
            random_state=42
        )
        
        # Créer et entraîner un modèle simple
        input_features = self.pd_data.shape[1]
        num_classes = len(set(self.pd_labels))
        
        model = GeophysicalDataFrameNet(
            input_features=input_features,
            num_classes=num_classes,
            hidden_layers=[32],
            dropout_rate=0.1
        )
        
        # Entraînement rapide
        self.trainer.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=3,
            learning_rate=0.01,
            patience=2
        )
        
        # Évaluer le modèle
        metrics = self.trainer.evaluate_model(model, val_loader)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('test_loss', metrics)
        self.assertIn('test_accuracy', metrics)
        self.assertIn('classification_report', metrics)
        self.assertIn('confusion_matrix', metrics)
        
        self.assertGreaterEqual(metrics['test_accuracy'], 0.0)
        self.assertLessEqual(metrics['test_accuracy'], 100.0)
        self.assertGreaterEqual(metrics['test_loss'], 0.0)
        
        logger.info(f"✅ Évaluation réussie: Accuracy = {metrics['test_accuracy']:.2f}%, Loss = {metrics['test_loss']:.4f}")
    
    def test_end_to_end_pipeline_with_both_datasets(self):
        """Test complet du pipeline avec les deux datasets."""
        logger.info("🚀 Démarrage du test end-to-end complet...")
        
        # Test 1: Pipeline complet avec PD.csv
        logger.info("📊 Test du pipeline avec données PD (Pole-Dipole)...")
        
        pd_train_loader, pd_val_loader = self.trainer.prepare_data_dataframe(
            dataframes=[self.pd_data],
            labels=[0],  # Un seul label pour tout le DataFrame
            augmentations=["gaussian_noise", "value_variation"],
            num_augmentations=2,
            test_size=0.2,
            random_state=42
        )
        
        pd_model = GeophysicalDataFrameNet(
            input_features=self.pd_data.shape[1],
            num_classes=len(set(self.pd_labels)),
            hidden_layers=[64, 32],
            dropout_rate=0.2
        )
        
        # Entraînement PD
        pd_history = self.trainer.train_model(
            model=pd_model,
            train_loader=pd_train_loader,
            val_loader=pd_val_loader,
            num_epochs=3,
            learning_rate=0.001,
            patience=2
        )
        
        # Évaluation PD
        pd_metrics = self.trainer.evaluate_model(pd_model, pd_val_loader)
        
        logger.info(f"📊 PD - Accuracy finale: {pd_metrics['test_accuracy']:.2f}%")
        
        # Test 2: Pipeline complet avec S.csv
        logger.info("📊 Test du pipeline avec données S (Schlumberger)...")
        
        s_train_loader, s_val_loader = self.trainer.prepare_data_dataframe(
            dataframes=[self.s_data],
            labels=[1],  # Un seul label pour tout le DataFrame
            augmentations=["gaussian_noise", "value_variation"],
            num_augmentations=2,
            test_size=0.2,
            random_state=42
        )
        
        s_model = GeophysicalDataFrameNet(
            input_features=self.s_data.shape[1],
            num_classes=len(set(self.s_labels)),
            hidden_layers=[64, 32],
            dropout_rate=0.2
        )
        
        # Entraînement S
        s_history = self.trainer.train_model(
            model=s_model,
            train_loader=s_train_loader,
            val_loader=s_val_loader,
            num_epochs=3,
            learning_rate=0.001,
            patience=2
        )
        
        # Évaluation S
        s_metrics = self.trainer.evaluate_model(s_model, s_val_loader)
        
        logger.info(f"📊 S - Accuracy finale: {s_metrics['test_accuracy']:.2f}%")
        
        # Vérifications finales
        self.assertGreater(pd_metrics['test_accuracy'], 0.0)
        self.assertGreater(s_metrics['test_accuracy'], 0.0)
        
        logger.info("🎉 Test end-to-end complet réussi !")


if __name__ == "__main__":
    unittest.main(verbosity=2)
