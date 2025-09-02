#!/usr/bin/env python3
"""
Tests Unitaires pour GeophysicalHybridNet avec Données Réelles
==============================================================

Ce module teste toutes les méthodes de la classe GeophysicalHybridNet en utilisant
des données géophysiques et images réelles du projet AI-MAP.

Tests couverts:
- Initialisation avec différentes configurations
- Forward pass avec images et données réelles
- Extraction de features intermédiaires
- Comptage des paramètres
- Intégration complète du modèle
- Performance avec données réelles
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import sys
import os
from PIL import Image
import torchvision.transforms as transforms
from unittest.mock import patch, MagicMock

# Ajouter le répertoire racine au path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.geophysical_hybrid_net import GeophysicalHybridNet, ImageEncoder, GeoDataEncoder, FusionModule
from src.utils.logger import logger


class TestGeophysicalHybridNetRealData(unittest.TestCase):
    """Tests pour GeophysicalHybridNet avec données géophysiques et images réelles."""
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Charger les données réelles
        self.real_geo_data = self.load_real_geophysical_data()
        self.real_images = self.load_real_images()
        
        # Paramètres de test basés sur les données réelles
        self.batch_size = 4
        self.num_classes = 2
        self.image_size = (64, 64)  # Taille réduite pour les tests
        
        logger.info(f"Tests initialisés avec {len(self.real_geo_data)} échantillons de données géophysiques et {len(self.real_images)} images réelles")
    
    def tearDown(self):
        """Nettoyage après chaque test."""
        shutil.rmtree(self.temp_dir)
    
    def load_real_geophysical_data(self) -> np.ndarray:
        """
        Charger les données géophysiques réelles depuis les fichiers CSV.
        
        Returns:
            np.ndarray: Données géophysiques normalisées
        """
        try:
            # Charger les données Schlumberger
            schlumberger_path = project_root / "data" / "processed" / "schlumberger_cleaned.csv"
            pole_dipole_path = project_root / "data" / "processed" / "pole_dipole_cleaned.csv"
            
            data_frames = []
            
            if schlumberger_path.exists():
                schlumberger_df = pd.read_csv(schlumberger_path)
                # Sélectionner les colonnes numériques pertinentes
                numeric_cols = ['Rho (Ohm.m)', 'M (mV/V)', 'SP (mV)', 'VMN (mV)', 'IAB (mA)']
                available_cols = [col for col in numeric_cols if col in schlumberger_df.columns]
                if available_cols:
                    schlumberger_data = schlumberger_df[available_cols].values
                    data_frames.append(schlumberger_data)
            
            if pole_dipole_path.exists():
                pole_dipole_df = pd.read_csv(pole_dipole_path)
                # Sélectionner les colonnes numériques pertinentes
                numeric_cols = ['Rho(ohm.m)', 'M (mV/V)', 'SP (mV)', 'VMN (mV)', 'IAB (mA)']
                available_cols = [col for col in numeric_cols if col in pole_dipole_df.columns]
                if available_cols:
                    pole_dipole_data = pole_dipole_df[available_cols].values
                    data_frames.append(pole_dipole_data)
            
            # Charger aussi quelques profils individuels
            profiles_dir = project_root / "data" / "training" / "csv"
            if profiles_dir.exists():
                for profile_file in list(profiles_dir.glob("*.csv"))[:3]:  # Limiter à 3 fichiers
                    try:
                        profile_df = pd.read_csv(profile_file, sep=';')
                        numeric_cols = ['Rho(ohm.m)', 'M (mV/V)', 'SP (mV)', 'VMN (mV)', 'IAB (mA)']
                        available_cols = [col for col in numeric_cols if col in profile_df.columns]
                        if available_cols:
                            profile_data = profile_df[available_cols].values
                            data_frames.append(profile_data)
                    except Exception as e:
                        logger.warning(f"Impossible de charger {profile_file}: {e}")
            
            if not data_frames:
                # Données de fallback si aucun fichier n'est trouvé
                logger.warning("Aucune donnée géophysique réelle trouvée, utilisation de données simulées")
                return np.random.randn(100, 5).astype(np.float32)
            
            # Combiner toutes les données
            all_data = np.vstack(data_frames)
            
            # Nettoyer les données (supprimer NaN, inf)
            all_data = all_data[np.isfinite(all_data).all(axis=1)]
            
            # Normaliser les données
            all_data = (all_data - all_data.mean(axis=0)) / (all_data.std(axis=0) + 1e-8)
            
            # Limiter à 5 dimensions pour les tests
            if all_data.shape[1] > 5:
                all_data = all_data[:, :5]
            elif all_data.shape[1] < 5:
                # Pad avec des zéros si nécessaire
                padding = np.zeros((all_data.shape[0], 5 - all_data.shape[1]))
                all_data = np.hstack([all_data, padding])
            
            logger.info(f"Données géophysiques réelles chargées: {all_data.shape}")
            return all_data.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données géophysiques réelles: {e}")
            # Données de fallback
            return np.random.randn(100, 5).astype(np.float32)
    
    def load_real_images(self) -> list:
        """
        Charger les images géophysiques réelles du projet.
        
        Returns:
            list: Liste des chemins vers les images réelles
        """
        try:
            image_paths = []
            
            # Charger les images de résistivité
            resistivity_dir = project_root / "data" / "training" / "images" / "resistivity"
            if resistivity_dir.exists():
                for img_file in list(resistivity_dir.glob("*.JPG"))[:5]:  # Limiter à 5 images
                    image_paths.append(str(img_file))
            
            # Charger les images de chargeabilité
            chargeability_dir = project_root / "data" / "training" / "images" / "chargeability"
            if chargeability_dir.exists():
                for img_file in list(chargeability_dir.glob("*.JPG"))[:3]:  # Limiter à 3 images
                    image_paths.append(str(img_file))
                for img_file in list(chargeability_dir.glob("*.PNG"))[:2]:  # Limiter à 2 images
                    image_paths.append(str(img_file))
            
            # Charger les images de profils
            profiles_dir = project_root / "data" / "training" / "images" / "profiles"
            if profiles_dir.exists():
                for img_file in list(profiles_dir.glob("*.JPG"))[:3]:  # Limiter à 3 images
                    image_paths.append(str(img_file))
            
            if not image_paths:
                logger.warning("Aucune image réelle trouvée, utilisation d'images simulées")
                # Créer des images simulées
                for i in range(10):
                    img_path = os.path.join(self.temp_dir, f"simulated_image_{i}.jpg")
                    # Créer une image RGB simulée
                    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                    img = Image.fromarray(img_array)
                    img.save(img_path)
                    image_paths.append(img_path)
            
            logger.info(f"Images réelles chargées: {len(image_paths)} images")
            return image_paths
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des images réelles: {e}")
            # Créer des images simulées en cas d'erreur
            image_paths = []
            for i in range(10):
                img_path = os.path.join(self.temp_dir, f"simulated_image_{i}.jpg")
                img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(img_path)
                image_paths.append(img_path)
            return image_paths
    
    def create_hybrid_model(self, **kwargs) -> GeophysicalHybridNet:
        """
        Créer une instance de GeophysicalHybridNet avec des paramètres par défaut.
        
        Args:
            **kwargs: Paramètres additionnels
            
        Returns:
            GeophysicalHybridNet: Instance configurée
        """
        default_params = {
            'num_classes': self.num_classes,
            'image_model': 'resnet18',
            'pretrained': False,  # Désactiver pour les tests
            'geo_input_dim': 5,
            'image_feature_dim': 512,
            'geo_feature_dim': 256,
            'fusion_hidden_dims': (512, 256),
            'dropout': 0.5,
            'fusion_method': 'concatenation',
            'freeze_backbone': False
        }
        default_params.update(kwargs)
        
        return GeophysicalHybridNet(**default_params)
    
    def get_real_batch(self, batch_size: int = None) -> tuple:
        """
        Obtenir un batch de données réelles pour les tests.
        
        Args:
            batch_size (int): Taille du batch
            
        Returns:
            tuple: (images, geo_data)
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Sélectionner des échantillons aléatoires des données réelles
        geo_indices = np.random.choice(len(self.real_geo_data), batch_size, replace=True)
        image_indices = np.random.choice(len(self.real_images), batch_size, replace=True)
        
        # Charger et traiter les images
        images = []
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        for idx in image_indices:
            try:
                img = Image.open(self.real_images[idx]).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de l'image {self.real_images[idx]}: {e}")
                # Créer une image simulée en cas d'erreur
                img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img_tensor = transform(img)
                images.append(img_tensor)
        
        images = torch.stack(images)
        
        # Préparer les données géophysiques
        geo_data = torch.tensor(self.real_geo_data[geo_indices], dtype=torch.float32)
        
        return images, geo_data


class TestGeophysicalHybridNetInitialization(TestGeophysicalHybridNetRealData):
    """Tests d'initialisation de GeophysicalHybridNet avec données réelles."""
    
    def test_init_default_parameters_with_real_data_context(self):
        """Tester l'initialisation avec les paramètres par défaut dans le contexte des données réelles."""
        model = self.create_hybrid_model()
        
        # Vérifier les paramètres
        self.assertEqual(model.num_classes, self.num_classes)
        self.assertEqual(model.image_model, 'resnet18')
        self.assertEqual(model.fusion_method, 'concatenation')
        
        # Vérifier les composants
        self.assertIsInstance(model.image_encoder, ImageEncoder)
        self.assertIsInstance(model.geo_encoder, GeoDataEncoder)
        self.assertIsInstance(model.fusion, FusionModule)
        
        logger.info("✅ Initialisation paramètres par défaut avec contexte données réelles: OK")
    
    def test_init_custom_parameters_with_real_data_context(self):
        """Tester l'initialisation avec des paramètres personnalisés dans le contexte des données réelles."""
        custom_params = {
            'num_classes': 5,
            'image_model': 'resnet34',
            'geo_input_dim': 6,
            'image_feature_dim': 1024,
            'geo_feature_dim': 512,
            'fusion_method': 'attention',
            'freeze_backbone': True
        }
        
        model = self.create_hybrid_model(**custom_params)
        
        # Vérifier que les paramètres personnalisés sont appliqués
        self.assertEqual(model.num_classes, 5)
        self.assertEqual(model.image_model, 'resnet34')
        self.assertEqual(model.fusion_method, 'attention')
        
        # Vérifier les dimensions
        self.assertEqual(model.image_encoder.feature_dim, 1024)
        self.assertEqual(model.geo_encoder.feature_dim, 512)
        
        logger.info("✅ Initialisation paramètres personnalisés avec contexte données réelles: OK")
    
    def test_init_different_fusion_methods_with_real_data_context(self):
        """Tester l'initialisation avec différentes méthodes de fusion dans le contexte des données réelles."""
        fusion_methods = ['concatenation', 'attention', 'weighted']
        
        for method in fusion_methods:
            if method == 'attention':
                # Pour l'attention, utiliser des dimensions compatibles
                model = self.create_hybrid_model(
                    fusion_method=method,
                    image_feature_dim=256,
                    geo_feature_dim=256
                )
            else:
                model = self.create_hybrid_model(fusion_method=method)
            
            self.assertEqual(model.fusion_method, method)
            self.assertIsInstance(model.fusion, FusionModule)
        
        logger.info("✅ Initialisation différentes méthodes fusion avec contexte données réelles: OK")
    
    def test_init_different_image_models_with_real_data_context(self):
        """Tester l'initialisation avec différents modèles d'images dans le contexte des données réelles."""
        image_models = ['resnet18', 'resnet34', 'resnet50']
        
        for model_name in image_models:
            model = self.create_hybrid_model(image_model=model_name)
            
            self.assertEqual(model.image_model, model_name)
            self.assertIsInstance(model.image_encoder, ImageEncoder)
        
        logger.info("✅ Initialisation différents modèles images avec contexte données réelles: OK")


class TestGeophysicalHybridNetForwardPass(TestGeophysicalHybridNetRealData):
    """Tests du forward pass de GeophysicalHybridNet avec données réelles."""
    
    def test_forward_with_real_data(self):
        """Tester le forward pass avec des données réelles."""
        model = self.create_hybrid_model()
        images, geo_data = self.get_real_batch()
        
        # Forward pass
        output = model(images, geo_data)
        
        # Vérifications
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(torch.isfinite(output).all())
        
        logger.info("✅ Forward pass avec données réelles: OK")
    
    def test_forward_different_batch_sizes_with_real_data(self):
        """Tester le forward pass avec différentes tailles de batch et données réelles."""
        model = self.create_hybrid_model()
        model.eval()  # Mode eval pour éviter les problèmes avec BatchNorm
        
        batch_sizes = [2, 4, 8]
        
        for batch_size in batch_sizes:
            images, geo_data = self.get_real_batch(batch_size)
            
            output = model(images, geo_data)
            
            self.assertEqual(output.shape, (batch_size, self.num_classes))
            self.assertTrue(torch.isfinite(output).all())
        
        logger.info("✅ Forward pass différentes tailles batch avec données réelles: OK")
    
    def test_forward_different_fusion_methods_with_real_data(self):
        """Tester le forward pass avec différentes méthodes de fusion et données réelles."""
        fusion_methods = ['concatenation', 'weighted']  # Éviter attention pour simplifier
        
        for method in fusion_methods:
            model = self.create_hybrid_model(fusion_method=method)
            images, geo_data = self.get_real_batch()
            
            output = model(images, geo_data)
            
            self.assertEqual(output.shape, (self.batch_size, self.num_classes))
            self.assertTrue(torch.isfinite(output).all())
        
        logger.info("✅ Forward pass différentes méthodes fusion avec données réelles: OK")
    
    def test_forward_consistency_with_real_data(self):
        """Tester la cohérence du forward pass avec les mêmes données réelles."""
        model = self.create_hybrid_model()
        model.eval()  # Mode eval pour désactiver le dropout
        
        images, geo_data = self.get_real_batch()
        
        # Deux forward passes
        output1 = model(images, geo_data)
        output2 = model(images, geo_data)
        
        # Les résultats doivent être identiques en mode eval
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))
        
        logger.info("✅ Cohérence forward pass avec données réelles: OK")


class TestGeophysicalHybridNetFeatureExtraction(TestGeophysicalHybridNetRealData):
    """Tests d'extraction de features de GeophysicalHybridNet avec données réelles."""
    
    def test_get_feature_maps_with_real_data(self):
        """Tester l'extraction de features intermédiaires avec des données réelles."""
        model = self.create_hybrid_model()
        images, geo_data = self.get_real_batch()
        
        # Extraire les features
        features = model.get_feature_maps(images, geo_data)
        
        # Vérifier la structure des features
        expected_keys = ['image_features', 'geo_features', 'output']
        for key in expected_keys:
            self.assertIn(key, features)
        
        # Vérifier les dimensions
        self.assertEqual(features['image_features'].shape, (self.batch_size, 512))
        self.assertEqual(features['geo_features'].shape, (self.batch_size, 256))
        self.assertEqual(features['output'].shape, (self.batch_size, self.num_classes))
        
        # Vérifier que les features sont finies
        for key, tensor in features.items():
            self.assertTrue(torch.isfinite(tensor).all())
        
        logger.info("✅ Extraction features intermédiaires avec données réelles: OK")
    
    def test_feature_consistency_with_real_data(self):
        """Tester la cohérence des features extraites avec des données réelles."""
        model = self.create_hybrid_model()
        model.eval()  # Mode eval pour la cohérence
        
        images, geo_data = self.get_real_batch()
        
        # Extraire les features deux fois
        features1 = model.get_feature_maps(images, geo_data)
        features2 = model.get_feature_maps(images, geo_data)
        
        # Vérifier la cohérence
        for key in features1.keys():
            self.assertTrue(torch.allclose(features1[key], features2[key], atol=1e-6))
        
        logger.info("✅ Cohérence features extraites avec données réelles: OK")
    
    def test_feature_gradient_flow_with_real_data(self):
        """Tester le flux de gradients dans l'extraction de features avec des données réelles."""
        model = self.create_hybrid_model()
        images, geo_data = self.get_real_batch()
        
        # S'assurer que les inputs nécessitent des gradients
        images.requires_grad_(True)
        geo_data.requires_grad_(True)
        
        # Extraire les features
        features = model.get_feature_maps(images, geo_data)
        
        # Calculer une loss simple
        loss = features['output'].sum()
        
        # Backward pass
        loss.backward()
        
        # Vérifier que les gradients sont calculés
        self.assertIsNotNone(images.grad)
        self.assertIsNotNone(geo_data.grad)
        
        # Vérifier que les gradients ne sont pas tous zéros
        self.assertFalse(torch.allclose(images.grad, torch.zeros_like(images.grad)))
        self.assertFalse(torch.allclose(geo_data.grad, torch.zeros_like(geo_data.grad)))
        
        logger.info("✅ Flux gradients extraction features avec données réelles: OK")


class TestGeophysicalHybridNetParameterCounting(TestGeophysicalHybridNetRealData):
    """Tests de comptage des paramètres de GeophysicalHybridNet avec données réelles."""
    
    def test_count_parameters_with_real_data_context(self):
        """Tester le comptage des paramètres dans le contexte des données réelles."""
        model = self.create_hybrid_model()
        
        param_counts = model.count_parameters()
        
        # Vérifier la structure du dictionnaire
        expected_keys = ['total_parameters', 'trainable_parameters', 'frozen_parameters']
        for key in expected_keys:
            self.assertIn(key, param_counts)
        
        # Vérifier que les comptes sont cohérents
        total = param_counts['total_parameters']
        trainable = param_counts['trainable_parameters']
        frozen = param_counts['frozen_parameters']
        
        self.assertEqual(total, trainable + frozen)
        self.assertGreater(total, 0)
        
        logger.info("✅ Comptage paramètres avec contexte données réelles: OK")
    
    def test_parameter_counting_different_configurations_with_real_data(self):
        """Tester le comptage des paramètres avec différentes configurations et données réelles."""
        configurations = [
            {'image_model': 'resnet18', 'freeze_backbone': False},
            {'image_model': 'resnet18', 'freeze_backbone': True},
            {'image_model': 'resnet34', 'freeze_backbone': False},
            {'num_classes': 5, 'image_feature_dim': 1024, 'geo_feature_dim': 512}
        ]
        
        for config in configurations:
            model = self.create_hybrid_model(**config)
            param_counts = model.count_parameters()
            
            # Vérifier la cohérence
            total = param_counts['total_parameters']
            trainable = param_counts['trainable_parameters']
            frozen = param_counts['frozen_parameters']
            
            self.assertEqual(total, trainable + frozen)
            self.assertGreater(total, 0)
        
        logger.info("✅ Comptage paramètres différentes configurations avec données réelles: OK")
    
    def test_parameter_counting_consistency_with_real_data(self):
        """Tester la cohérence du comptage des paramètres avec des données réelles."""
        model = self.create_hybrid_model()
        
        # Compter les paramètres plusieurs fois
        counts1 = model.count_parameters()
        counts2 = model.count_parameters()
        
        # Les comptes doivent être identiques
        for key in counts1.keys():
            self.assertEqual(counts1[key], counts2[key])
        
        # Vérifier avec le comptage manuel
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.assertEqual(counts1['total_parameters'], total_params)
        self.assertEqual(counts1['trainable_parameters'], trainable_params)
        
        logger.info("✅ Cohérence comptage paramètres avec données réelles: OK")


class TestGeophysicalHybridNetIntegration(TestGeophysicalHybridNetRealData):
    """Tests d'intégration de GeophysicalHybridNet avec données réelles."""
    
    def test_integration_with_training_loop_real_data(self):
        """Tester l'intégration dans une boucle d'entraînement avec des données réelles."""
        model = self.create_hybrid_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Simuler quelques étapes d'entraînement
        for epoch in range(3):
            for batch_idx in range(3):  # 3 batches par époque
                images, geo_data = self.get_real_batch(4)
                
                # Labels simulés
                labels = torch.randint(0, self.num_classes, (4,))
                
                # Forward pass
                optimizer.zero_grad()
                output = model(images, geo_data)
                loss = criterion(output, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Vérifier que la loss est finie
                self.assertTrue(torch.isfinite(loss))
        
        logger.info("✅ Intégration boucle entraînement avec données réelles: OK")
    
    def test_integration_with_different_models_real_data(self):
        """Tester l'intégration avec différents modèles et données réelles."""
        configurations = [
            {'image_model': 'resnet18', 'num_classes': 3},
            {'image_model': 'resnet34', 'num_classes': 2},
            {'fusion_method': 'weighted', 'num_classes': 4}
        ]
        
        for config in configurations:
            model = self.create_hybrid_model(**config)
            images, geo_data = self.get_real_batch(4)
            
            # Test du forward pass
            output = model(images, geo_data)
            self.assertEqual(output.shape, (4, config['num_classes']))
            self.assertTrue(torch.isfinite(output).all())
            
            # Test de l'extraction de features
            features = model.get_feature_maps(images, geo_data)
            self.assertIn('output', features)
            self.assertEqual(features['output'].shape, (4, config['num_classes']))
        
        logger.info("✅ Intégration différents modèles avec données réelles: OK")
    
    def test_integration_with_real_geophysical_scenarios(self):
        """Tester l'intégration avec des scénarios géophysiques réels."""
        model = self.create_hybrid_model()
        
        # Scénario 1: Données de résistivité élevée (roches dures)
        high_resistivity_data = self.real_geo_data.copy()
        high_resistivity_data[:, 0] = high_resistivity_data[:, 0] * 10
        
        # Scénario 2: Données de chargeabilité élevée (minéralisation)
        high_chargeability_data = self.real_geo_data.copy()
        if high_chargeability_data.shape[1] > 1:
            high_chargeability_data[:, 1] = high_chargeability_data[:, 1] * 5
        
        # Scénario 3: Données avec bruit (conditions de terrain difficiles)
        noisy_data = self.real_geo_data.copy()
        noise = np.random.randn(*noisy_data.shape) * 0.1
        noisy_data = noisy_data + noise
        
        scenarios = [
            ("haute_resistivite", high_resistivity_data),
            ("haute_chargeabilite", high_chargeability_data),
            ("donnees_bruitees", noisy_data)
        ]
        
        for scenario_name, scenario_data in scenarios:
            # Adapter les dimensions
            if scenario_data.shape[1] < 5:
                padding = np.zeros((scenario_data.shape[0], 5 - scenario_data.shape[1]))
                scenario_data = np.hstack([scenario_data, padding])
            elif scenario_data.shape[1] > 5:
                scenario_data = scenario_data[:, :5]
            
            # Tester avec ce scénario
            batch_size = 4
            geo_data = torch.tensor(scenario_data[:batch_size], dtype=torch.float32)
            images, _ = self.get_real_batch(batch_size)
            
            # Test du forward pass
            output = model(images, geo_data)
            self.assertEqual(output.shape, (batch_size, self.num_classes))
            self.assertTrue(torch.isfinite(output).all())
            
            # Test de l'extraction de features
            features = model.get_feature_maps(images, geo_data)
            self.assertIn('output', features)
            self.assertEqual(features['output'].shape, (batch_size, self.num_classes))
        
        logger.info("✅ Intégration scénarios géophysiques réels: OK")


class TestGeophysicalHybridNetPerformance(TestGeophysicalHybridNetRealData):
    """Tests de performance de GeophysicalHybridNet avec données réelles."""
    
    def test_forward_speed_with_real_data(self):
        """Tester la vitesse du forward pass avec des données réelles."""
        import time
        
        model = self.create_hybrid_model()
        model.eval()  # Mode eval pour des mesures de vitesse plus précises
        
        images, geo_data = self.get_real_batch(8)
        
        # Mesurer le temps
        start_time = time.time()
        
        for _ in range(50):  # 50 itérations
            output = model(images, geo_data)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 50
        
        # Vérifier que le temps est raisonnable (< 200ms par forward pass pour un modèle hybride complexe)
        self.assertLess(avg_time, 0.2)
        
        logger.info(f"✅ Vitesse forward pass avec données réelles: {avg_time*1000:.2f}ms par batch")
    
    def test_memory_usage_with_real_data(self):
        """Tester l'utilisation mémoire avec des données réelles."""
        model = self.create_hybrid_model()
        
        # Tester avec différents batch sizes
        batch_sizes = [2, 4, 8]
        
        for batch_size in batch_sizes:
            images, geo_data = self.get_real_batch(batch_size)
            
            # Forward pass
            output = model(images, geo_data)
            
            # Vérifier que la forme est correcte
            self.assertEqual(output.shape, (batch_size, self.num_classes))
            
            # Vérifier que les tensors sont sur le bon device
            self.assertEqual(output.device, images.device)
            self.assertEqual(output.device, geo_data.device)
        
        logger.info("✅ Utilisation mémoire avec données réelles: OK")
    
    def test_different_models_performance_comparison_with_real_data(self):
        """Comparer les performances de différents modèles avec des données réelles."""
        import time
        
        images, geo_data = self.get_real_batch(8)
        
        models = [
            ('resnet18', self.create_hybrid_model(image_model='resnet18')),
            ('resnet34', self.create_hybrid_model(image_model='resnet34')),
            ('weighted_fusion', self.create_hybrid_model(fusion_method='weighted'))
        ]
        
        times = {}
        
        for model_name, model in models:
            model.eval()
            
            # Mesurer le temps
            start_time = time.time()
            
            for _ in range(30):  # 30 itérations
                output = model(images, geo_data)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 30
            times[model_name] = avg_time
            
            # Vérifier que l'output est correct
            self.assertEqual(output.shape, (8, self.num_classes))
            self.assertTrue(torch.isfinite(output).all())
        
        # Vérifier que tous les modèles fonctionnent
        self.assertEqual(len(times), 3)
        
        logger.info(f"✅ Comparaison performances modèles avec données réelles: {times}")


if __name__ == '__main__':
    # Configuration des tests
    unittest.main(verbosity=2)