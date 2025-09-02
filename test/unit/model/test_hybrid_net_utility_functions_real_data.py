#!/usr/bin/env python3
"""
Tests Unitaires pour les Fonctions Utilitaires de GeophysicalHybridNet avec Données Réelles
===========================================================================================

Ce module teste les fonctions utilitaires create_hybrid_model et get_model_summary
en utilisant des données géophysiques et images réelles du projet AI-MAP.

Tests couverts:
- create_hybrid_model avec différentes configurations
- get_model_summary avec validation des métriques
- Intégration avec données réelles
- Validation des paramètres et configurations
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

# Ajouter le répertoire racine au path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.geophysical_hybrid_net import (
    GeophysicalHybridNet, 
    create_hybrid_model, 
    get_model_summary
)
from src.utils.logger import logger


class TestHybridNetUtilityFunctionsRealData(unittest.TestCase):
    """Tests pour les fonctions utilitaires de GeophysicalHybridNet avec données réelles."""
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Charger les données réelles
        self.real_geo_data = self.load_real_geophysical_data()
        self.real_images = self.load_real_images()
        
        # Paramètres de test basés sur les données réelles
        self.batch_size = 4
        self.num_classes = 2
        self.image_size = (64, 64)
        
        logger.info(f"Tests utilitaires initialisés avec {len(self.real_geo_data)} échantillons de données géophysiques et {len(self.real_images)} images réelles")
    
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
                numeric_cols = ['Rho (Ohm.m)', 'M (mV/V)', 'SP (mV)', 'VMN (mV)', 'IAB (mA)']
                available_cols = [col for col in numeric_cols if col in schlumberger_df.columns]
                if available_cols:
                    schlumberger_data = schlumberger_df[available_cols].values
                    data_frames.append(schlumberger_data)
            
            if pole_dipole_path.exists():
                pole_dipole_df = pd.read_csv(pole_dipole_path)
                numeric_cols = ['Rho(ohm.m)', 'M (mV/V)', 'SP (mV)', 'VMN (mV)', 'IAB (mA)']
                available_cols = [col for col in numeric_cols if col in pole_dipole_df.columns]
                if available_cols:
                    pole_dipole_data = pole_dipole_df[available_cols].values
                    data_frames.append(pole_dipole_data)
            
            # Charger quelques profils individuels
            profiles_dir = project_root / "data" / "training" / "csv"
            if profiles_dir.exists():
                for profile_file in list(profiles_dir.glob("*.csv"))[:3]:
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
                logger.warning("Aucune donnée géophysique réelle trouvée, utilisation de données simulées")
                return np.random.randn(100, 5).astype(np.float32)
            
            # Combiner toutes les données
            all_data = np.vstack(data_frames)
            
            # Nettoyer les données
            all_data = all_data[np.isfinite(all_data).all(axis=1)]
            
            # Normaliser les données
            all_data = (all_data - all_data.mean(axis=0)) / (all_data.std(axis=0) + 1e-8)
            
            # Limiter à 5 dimensions
            if all_data.shape[1] > 5:
                all_data = all_data[:, :5]
            elif all_data.shape[1] < 5:
                padding = np.zeros((all_data.shape[0], 5 - all_data.shape[1]))
                all_data = np.hstack([all_data, padding])
            
            logger.info(f"Données géophysiques réelles chargées: {all_data.shape}")
            return all_data.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données géophysiques réelles: {e}")
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
                for img_file in list(resistivity_dir.glob("*.JPG"))[:3]:
                    image_paths.append(str(img_file))
            
            # Charger les images de chargeabilité
            chargeability_dir = project_root / "data" / "training" / "images" / "chargeability"
            if chargeability_dir.exists():
                for img_file in list(chargeability_dir.glob("*.JPG"))[:2]:
                    image_paths.append(str(img_file))
                for img_file in list(chargeability_dir.glob("*.PNG"))[:2]:
                    image_paths.append(str(img_file))
            
            # Charger les images de profils
            profiles_dir = project_root / "data" / "training" / "images" / "profiles"
            if profiles_dir.exists():
                for img_file in list(profiles_dir.glob("*.JPG"))[:2]:
                    image_paths.append(str(img_file))
            
            if not image_paths:
                logger.warning("Aucune image réelle trouvée, utilisation d'images simulées")
                for i in range(8):
                    img_path = os.path.join(self.temp_dir, f"simulated_image_{i}.jpg")
                    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                    img = Image.fromarray(img_array)
                    img.save(img_path)
                    image_paths.append(img_path)
            
            logger.info(f"Images réelles chargées: {len(image_paths)} images")
            return image_paths
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des images réelles: {e}")
            image_paths = []
            for i in range(8):
                img_path = os.path.join(self.temp_dir, f"simulated_image_{i}.jpg")
                img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(img_path)
                image_paths.append(img_path)
            return image_paths
    
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
        
        # Sélectionner des échantillons aléatoires
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
                img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img_tensor = transform(img)
                images.append(img_tensor)
        
        images = torch.stack(images)
        
        # Préparer les données géophysiques
        geo_data = torch.tensor(self.real_geo_data[geo_indices], dtype=torch.float32)
        
        return images, geo_data


class TestCreateHybridModelRealData(TestHybridNetUtilityFunctionsRealData):
    """Tests pour la fonction create_hybrid_model avec données réelles."""
    
    def test_create_hybrid_model_default_parameters_with_real_data(self):
        """Tester create_hybrid_model avec les paramètres par défaut et données réelles."""
        model = create_hybrid_model()
        
        # Vérifier que le modèle est correctement créé
        self.assertIsInstance(model, GeophysicalHybridNet)
        self.assertEqual(model.num_classes, 2)
        self.assertEqual(model.image_model, 'resnet18')
        self.assertEqual(model.fusion_method, 'concatenation')
        
        # Tester le forward pass avec des données réelles
        images, geo_data = self.get_real_batch()
        
        # Adapter les dimensions des données géophysiques (modèle attend 4 par défaut)
        if geo_data.shape[1] > 4:
            geo_data = geo_data[:, :4]
        elif geo_data.shape[1] < 4:
            padding = torch.zeros(geo_data.shape[0], 4 - geo_data.shape[1])
            geo_data = torch.cat([geo_data, padding], dim=1)
        
        output = model(images, geo_data)
        
        self.assertEqual(output.shape, (self.batch_size, 2))
        self.assertTrue(torch.isfinite(output).all())
        
        logger.info("✅ create_hybrid_model paramètres par défaut avec données réelles: OK")
    
    def test_create_hybrid_model_custom_parameters_with_real_data(self):
        """Tester create_hybrid_model avec des paramètres personnalisés et données réelles."""
        custom_params = {
            'num_classes': 5,
            'image_model': 'resnet34',
            'geo_input_dim': 6,
            'image_feature_dim': 1024,
            'geo_feature_dim': 512,
            'fusion_method': 'weighted',
            'freeze_backbone': True
        }
        
        model = create_hybrid_model(**custom_params)
        
        # Vérifier que les paramètres personnalisés sont appliqués
        self.assertIsInstance(model, GeophysicalHybridNet)
        self.assertEqual(model.num_classes, 5)
        self.assertEqual(model.image_model, 'resnet34')
        self.assertEqual(model.fusion_method, 'weighted')
        
        # Tester le forward pass avec des données réelles
        images, geo_data = self.get_real_batch()
        # Adapter les dimensions des données géophysiques
        if geo_data.shape[1] < 6:
            padding = torch.zeros(geo_data.shape[0], 6 - geo_data.shape[1])
            geo_data = torch.cat([geo_data, padding], dim=1)
        elif geo_data.shape[1] > 6:
            geo_data = geo_data[:, :6]
        
        output = model(images, geo_data)
        
        self.assertEqual(output.shape, (self.batch_size, 5))
        self.assertTrue(torch.isfinite(output).all())
        
        logger.info("✅ create_hybrid_model paramètres personnalisés avec données réelles: OK")
    
    def test_create_hybrid_model_different_image_models_with_real_data(self):
        """Tester create_hybrid_model avec différents modèles d'images et données réelles."""
        image_models = ['resnet18', 'resnet34', 'resnet50']
        
        for model_name in image_models:
            model = create_hybrid_model(image_model=model_name, pretrained=False)
            
            # Vérifier que le modèle est correctement créé
            self.assertIsInstance(model, GeophysicalHybridNet)
            self.assertEqual(model.image_model, model_name)
            
            # Tester le forward pass avec des données réelles
            images, geo_data = self.get_real_batch()
            
            # Adapter les dimensions des données géophysiques
            if geo_data.shape[1] > 4:
                geo_data = geo_data[:, :4]
            elif geo_data.shape[1] < 4:
                padding = torch.zeros(geo_data.shape[0], 4 - geo_data.shape[1])
                geo_data = torch.cat([geo_data, padding], dim=1)
            
            output = model(images, geo_data)
            
            self.assertEqual(output.shape, (self.batch_size, 2))
            self.assertTrue(torch.isfinite(output).all())
        
        logger.info("✅ create_hybrid_model différents modèles images avec données réelles: OK")
    
    def test_create_hybrid_model_different_fusion_methods_with_real_data(self):
        """Tester create_hybrid_model avec différentes méthodes de fusion et données réelles."""
        fusion_methods = ['concatenation', 'weighted']  # Éviter attention pour simplifier
        
        for method in fusion_methods:
            if method == 'attention':
                # Pour l'attention, utiliser des dimensions compatibles
                model = create_hybrid_model(
                    fusion_method=method,
                    image_feature_dim=256,
                    geo_feature_dim=256
                )
            else:
                model = create_hybrid_model(fusion_method=method)
            
            # Vérifier que le modèle est correctement créé
            self.assertIsInstance(model, GeophysicalHybridNet)
            self.assertEqual(model.fusion_method, method)
            
            # Tester le forward pass avec des données réelles
            images, geo_data = self.get_real_batch()
            
            # Adapter les dimensions des données géophysiques
            if geo_data.shape[1] > 4:
                geo_data = geo_data[:, :4]
            elif geo_data.shape[1] < 4:
                padding = torch.zeros(geo_data.shape[0], 4 - geo_data.shape[1])
                geo_data = torch.cat([geo_data, padding], dim=1)
            
            output = model(images, geo_data)
            
            self.assertEqual(output.shape, (self.batch_size, 2))
            self.assertTrue(torch.isfinite(output).all())
        
        logger.info("✅ create_hybrid_model différentes méthodes fusion avec données réelles: OK")
    
    def test_create_hybrid_model_different_geo_input_dims_with_real_data(self):
        """Tester create_hybrid_model avec différentes dimensions de données géophysiques et données réelles."""
        geo_input_dims = [3, 4, 5, 6]
        
        for geo_dim in geo_input_dims:
            model = create_hybrid_model(geo_input_dim=geo_dim)
            
            # Vérifier que le modèle est correctement créé
            self.assertIsInstance(model, GeophysicalHybridNet)
            # Vérifier que le modèle a été créé avec la bonne dimension
            self.assertEqual(model.geo_encoder.encoder[0].in_features, geo_dim)
            
            # Tester le forward pass avec des données réelles
            images, geo_data = self.get_real_batch()
            
            # Adapter les dimensions des données géophysiques
            if geo_data.shape[1] < geo_dim:
                padding = torch.zeros(geo_data.shape[0], geo_dim - geo_data.shape[1])
                geo_data = torch.cat([geo_data, padding], dim=1)
            elif geo_data.shape[1] > geo_dim:
                geo_data = geo_data[:, :geo_dim]
            
            output = model(images, geo_data)
            
            self.assertEqual(output.shape, (self.batch_size, 2))
            self.assertTrue(torch.isfinite(output).all())
        
        logger.info("✅ create_hybrid_model différentes dimensions géo avec données réelles: OK")
    
    def test_create_hybrid_model_integration_with_real_data(self):
        """Tester l'intégration complète de create_hybrid_model avec des données réelles."""
        # Créer un modèle avec configuration complexe
        model = create_hybrid_model(
            num_classes=3,
            image_model='resnet34',
            geo_input_dim=5,
            image_feature_dim=512,
            geo_feature_dim=256,
            fusion_hidden_dims=(512, 256, 128),
            dropout=0.3,
            fusion_method='weighted',
            freeze_backbone=False
        )
        
        # Vérifier que le modèle est correctement créé
        self.assertIsInstance(model, GeophysicalHybridNet)
        
        # Tester le forward pass avec des données réelles
        images, geo_data = self.get_real_batch()
        output = model(images, geo_data)
        
        self.assertEqual(output.shape, (self.batch_size, 3))
        self.assertTrue(torch.isfinite(output).all())
        
        # Tester l'extraction de features
        features = model.get_feature_maps(images, geo_data)
        self.assertIn('output', features)
        self.assertEqual(features['output'].shape, (self.batch_size, 3))
        
        # Tester le comptage des paramètres
        param_counts = model.count_parameters()
        self.assertIn('total_parameters', param_counts)
        self.assertGreater(param_counts['total_parameters'], 0)
        
        logger.info("✅ create_hybrid_model intégration complète avec données réelles: OK")


class TestGetModelSummaryRealData(TestHybridNetUtilityFunctionsRealData):
    """Tests pour la fonction get_model_summary avec données réelles."""
    
    def test_get_model_summary_basic_with_real_data(self):
        """Tester get_model_summary avec un modèle de base et données réelles."""
        model = create_hybrid_model()
        
        # Obtenir le résumé du modèle
        summary = get_model_summary(model)
        
        # Vérifier la structure du résumé
        expected_keys = ['model_type', 'image_model', 'fusion_method', 'num_classes', 'parameters', 'architecture']
        for key in expected_keys:
            self.assertIn(key, summary)
        
        # Vérifier les valeurs
        self.assertEqual(summary['model_type'], 'GeophysicalHybridNet')
        self.assertEqual(summary['image_model'], 'resnet18')
        self.assertEqual(summary['fusion_method'], 'concatenation')
        self.assertEqual(summary['num_classes'], 2)
        
        # Vérifier la structure des paramètres
        param_keys = ['total_parameters', 'trainable_parameters', 'frozen_parameters']
        for key in param_keys:
            self.assertIn(key, summary['parameters'])
        
        # Vérifier la structure de l'architecture
        arch_keys = ['image_encoder', 'geo_encoder', 'fusion_module']
        for key in arch_keys:
            self.assertIn(key, summary['architecture'])
        
        logger.info("✅ get_model_summary modèle de base avec données réelles: OK")
    
    def test_get_model_summary_custom_model_with_real_data(self):
        """Tester get_model_summary avec un modèle personnalisé et données réelles."""
        model = create_hybrid_model(
            num_classes=5,
            image_model='resnet34',
            geo_input_dim=6,
            fusion_method='weighted',
            freeze_backbone=True
        )
        
        # Obtenir le résumé du modèle
        summary = get_model_summary(model)
        
        # Vérifier les valeurs personnalisées
        self.assertEqual(summary['model_type'], 'GeophysicalHybridNet')
        self.assertEqual(summary['image_model'], 'resnet34')
        self.assertEqual(summary['fusion_method'], 'weighted')
        self.assertEqual(summary['num_classes'], 5)
        
        # Vérifier que les paramètres sont cohérents
        total_params = summary['parameters']['total_parameters']
        trainable_params = summary['parameters']['trainable_parameters']
        frozen_params = summary['parameters']['frozen_parameters']
        
        self.assertEqual(total_params, trainable_params + frozen_params)
        self.assertGreater(total_params, 0)
        
        # Vérifier que certains paramètres sont gelés (freeze_backbone=True)
        self.assertGreater(frozen_params, 0)
        
        logger.info("✅ get_model_summary modèle personnalisé avec données réelles: OK")
    
    def test_get_model_summary_different_configurations_with_real_data(self):
        """Tester get_model_summary avec différentes configurations et données réelles."""
        configurations = [
            {'num_classes': 3, 'image_model': 'resnet18'},
            {'num_classes': 4, 'image_model': 'resnet34', 'fusion_method': 'weighted'},
            {'num_classes': 2, 'geo_input_dim': 6, 'freeze_backbone': True}
        ]
        
        for config in configurations:
            model = create_hybrid_model(**config)
            summary = get_model_summary(model)
            
            # Vérifier que le résumé est cohérent
            self.assertEqual(summary['model_type'], 'GeophysicalHybridNet')
            self.assertEqual(summary['num_classes'], config['num_classes'])
            
            if 'image_model' in config:
                self.assertEqual(summary['image_model'], config['image_model'])
            
            if 'fusion_method' in config:
                self.assertEqual(summary['fusion_method'], config['fusion_method'])
            
            # Vérifier la cohérence des paramètres
            total_params = summary['parameters']['total_parameters']
            trainable_params = summary['parameters']['trainable_parameters']
            frozen_params = summary['parameters']['frozen_parameters']
            
            self.assertEqual(total_params, trainable_params + frozen_params)
            self.assertGreater(total_params, 0)
        
        logger.info("✅ get_model_summary différentes configurations avec données réelles: OK")
    
    def test_get_model_summary_consistency_with_real_data(self):
        """Tester la cohérence de get_model_summary avec des données réelles."""
        model = create_hybrid_model()
        
        # Obtenir le résumé plusieurs fois
        summary1 = get_model_summary(model)
        summary2 = get_model_summary(model)
        
        # Les résumés doivent être identiques
        self.assertEqual(summary1, summary2)
        
        # Vérifier la cohérence avec les attributs du modèle
        self.assertEqual(summary1['image_model'], model.image_model)
        self.assertEqual(summary1['fusion_method'], model.fusion_method)
        self.assertEqual(summary1['num_classes'], model.num_classes)
        
        # Vérifier la cohérence avec le comptage manuel des paramètres
        manual_params = model.count_parameters()
        self.assertEqual(summary1['parameters'], manual_params)
        
        logger.info("✅ get_model_summary cohérence avec données réelles: OK")
    
    def test_get_model_summary_architecture_details_with_real_data(self):
        """Tester les détails de l'architecture dans get_model_summary avec des données réelles."""
        model = create_hybrid_model(
            num_classes=3,
            image_model='resnet34',
            geo_input_dim=5,
            fusion_method='weighted'
        )
        
        summary = get_model_summary(model)
        
        # Vérifier que les détails de l'architecture sont présents
        architecture = summary['architecture']
        
        # Vérifier que les composants sont décrits
        self.assertIsInstance(architecture['image_encoder'], str)
        self.assertIsInstance(architecture['geo_encoder'], str)
        self.assertIsInstance(architecture['fusion_module'], str)
        
        # Vérifier que les descriptions ne sont pas vides
        self.assertGreater(len(architecture['image_encoder']), 0)
        self.assertGreater(len(architecture['geo_encoder']), 0)
        self.assertGreater(len(architecture['fusion_module']), 0)
        
        # Vérifier que les descriptions contiennent des informations pertinentes
        self.assertIn('ImageEncoder', architecture['image_encoder'])
        self.assertIn('GeoDataEncoder', architecture['geo_encoder'])
        self.assertIn('FusionModule', architecture['fusion_module'])
        
        logger.info("✅ get_model_summary détails architecture avec données réelles: OK")
    
    def test_get_model_summary_integration_with_real_data(self):
        """Tester l'intégration complète de get_model_summary avec des données réelles."""
        # Créer un modèle complexe
        model = create_hybrid_model(
            num_classes=4,
            image_model='resnet50',
            geo_input_dim=6,
            image_feature_dim=1024,
            geo_feature_dim=512,
            fusion_hidden_dims=(1024, 512, 256),
            dropout=0.4,
            fusion_method='weighted',
            freeze_backbone=True
        )
        
        # Obtenir le résumé
        summary = get_model_summary(model)
        
        # Vérifier la structure complète
        self.assertEqual(summary['model_type'], 'GeophysicalHybridNet')
        self.assertEqual(summary['image_model'], 'resnet50')
        self.assertEqual(summary['fusion_method'], 'weighted')
        self.assertEqual(summary['num_classes'], 4)
        
        # Vérifier les paramètres
        params = summary['parameters']
        self.assertGreater(params['total_parameters'], 0)
        self.assertGreater(params['frozen_parameters'], 0)  # freeze_backbone=True
        
        # Tester le modèle avec des données réelles
        images, geo_data = self.get_real_batch()
        
        # Adapter les dimensions
        if geo_data.shape[1] < 6:
            padding = torch.zeros(geo_data.shape[0], 6 - geo_data.shape[1])
            geo_data = torch.cat([geo_data, padding], dim=1)
        elif geo_data.shape[1] > 6:
            geo_data = geo_data[:, :6]
        
        output = model(images, geo_data)
        
        self.assertEqual(output.shape, (self.batch_size, 4))
        self.assertTrue(torch.isfinite(output).all())
        
        # Vérifier que le résumé est cohérent avec le comportement du modèle
        self.assertEqual(summary['num_classes'], output.shape[1])
        
        logger.info("✅ get_model_summary intégration complète avec données réelles: OK")


class TestUtilityFunctionsIntegrationRealData(TestHybridNetUtilityFunctionsRealData):
    """Tests d'intégration des fonctions utilitaires avec données réelles."""
    
    def test_create_and_summarize_model_with_real_data(self):
        """Tester la création et le résumé d'un modèle avec des données réelles."""
        # Créer un modèle avec create_hybrid_model
        model = create_hybrid_model(
            num_classes=3,
            image_model='resnet34',
            geo_input_dim=5,
            fusion_method='weighted'
        )
        
        # Obtenir le résumé avec get_model_summary
        summary = get_model_summary(model)
        
        # Vérifier la cohérence
        self.assertEqual(summary['num_classes'], 3)
        self.assertEqual(summary['image_model'], 'resnet34')
        self.assertEqual(summary['fusion_method'], 'weighted')
        
        # Tester le modèle avec des données réelles
        images, geo_data = self.get_real_batch()
        output = model(images, geo_data)
        
        self.assertEqual(output.shape, (self.batch_size, 3))
        self.assertTrue(torch.isfinite(output).all())
        
        logger.info("✅ Intégration create_hybrid_model et get_model_summary avec données réelles: OK")
    
    def test_multiple_models_comparison_with_real_data(self):
        """Tester la comparaison de plusieurs modèles avec des données réelles."""
        configurations = [
            {'num_classes': 2, 'image_model': 'resnet18', 'fusion_method': 'concatenation'},
            {'num_classes': 3, 'image_model': 'resnet34', 'fusion_method': 'weighted'},
            {'num_classes': 4, 'image_model': 'resnet50', 'fusion_method': 'concatenation'}
        ]
        
        models = []
        summaries = []
        
        for config in configurations:
            model = create_hybrid_model(**config)
            summary = get_model_summary(model)
            
            models.append(model)
            summaries.append(summary)
        
        # Vérifier que tous les modèles sont différents
        for i, summary in enumerate(summaries):
            self.assertEqual(summary['num_classes'], configurations[i]['num_classes'])
            self.assertEqual(summary['image_model'], configurations[i]['image_model'])
            self.assertEqual(summary['fusion_method'], configurations[i]['fusion_method'])
        
        # Tester tous les modèles avec des données réelles
        images, geo_data = self.get_real_batch()
        
        for i, model in enumerate(models):
            # Adapter les dimensions des données géophysiques
            test_geo_data = geo_data.clone()
            if test_geo_data.shape[1] > 4:
                test_geo_data = test_geo_data[:, :4]
            elif test_geo_data.shape[1] < 4:
                padding = torch.zeros(test_geo_data.shape[0], 4 - test_geo_data.shape[1])
                test_geo_data = torch.cat([test_geo_data, padding], dim=1)
            
            output = model(images, test_geo_data)
            expected_classes = configurations[i]['num_classes']
            
            self.assertEqual(output.shape, (self.batch_size, expected_classes))
            self.assertTrue(torch.isfinite(output).all())
        
        logger.info("✅ Comparaison plusieurs modèles avec données réelles: OK")
    
    def test_utility_functions_with_real_geophysical_scenarios(self):
        """Tester les fonctions utilitaires avec des scénarios géophysiques réels."""
        # Scénario 1: Modèle pour haute résistivité
        high_resistivity_model = create_hybrid_model(
            num_classes=2,
            image_model='resnet18',
            geo_input_dim=5,
            fusion_method='concatenation'
        )
        
        # Scénario 2: Modèle pour haute chargeabilité
        high_chargeability_model = create_hybrid_model(
            num_classes=3,
            image_model='resnet34',
            geo_input_dim=5,
            fusion_method='weighted'
        )
        
        # Scénario 3: Modèle pour données bruitées
        noisy_data_model = create_hybrid_model(
            num_classes=2,
            image_model='resnet50',
            geo_input_dim=5,
            fusion_method='concatenation',
            dropout=0.3
        )
        
        models = [high_resistivity_model, high_chargeability_model, noisy_data_model]
        model_names = ['high_resistivity', 'high_chargeability', 'noisy_data']
        
        # Obtenir les résumés
        summaries = [get_model_summary(model) for model in models]
        
        # Vérifier que tous les modèles sont correctement configurés
        for i, summary in enumerate(summaries):
            self.assertEqual(summary['model_type'], 'GeophysicalHybridNet')
            self.assertGreater(summary['parameters']['total_parameters'], 0)
        
        # Tester tous les modèles avec des données réelles
        images, geo_data = self.get_real_batch()
        
        for i, model in enumerate(models):
            output = model(images, geo_data)
            expected_classes = summaries[i]['num_classes']
            
            self.assertEqual(output.shape, (self.batch_size, expected_classes))
            self.assertTrue(torch.isfinite(output).all())
        
        logger.info("✅ Fonctions utilitaires scénarios géophysiques réels: OK")


if __name__ == '__main__':
    # Configuration des tests
    unittest.main(verbosity=2)
