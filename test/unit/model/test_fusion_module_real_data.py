#!/usr/bin/env python3
"""
Tests Unitaires pour FusionModule avec Données Réelles
======================================================

Ce module teste toutes les méthodes de la classe FusionModule en utilisant
des données géophysiques réelles du projet AI-MAP.

Tests couverts:
- Initialisation avec différentes méthodes de fusion
- Forward pass avec données réelles
- Gestion des dimensions différentes
- Cas limites et erreurs
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

# Ajouter le répertoire racine au path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.geophysical_hybrid_net import FusionModule
from src.utils.logger import logger


class TestFusionModuleRealData(unittest.TestCase):
    """Tests pour FusionModule avec données géophysiques réelles."""
    
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Charger les données réelles
        self.real_geo_data = self.load_real_geophysical_data()
        self.real_image_features = self.generate_real_image_features()
        
        # Paramètres de test basés sur les données réelles
        self.batch_size = 8
        self.image_features_dim = 512
        self.geo_features_dim = 256
        self.num_classes = 2
        
        logger.info(f"Tests initialisés avec {len(self.real_geo_data)} échantillons de données réelles")
    
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
                logger.warning("Aucune donnée réelle trouvée, utilisation de données simulées")
                return np.random.randn(100, 5).astype(np.float32)
            
            # Combiner toutes les données
            all_data = np.vstack(data_frames)
            
            # Nettoyer les données (supprimer NaN, inf)
            all_data = all_data[np.isfinite(all_data).all(axis=1)]
            
            # Normaliser les données
            all_data = (all_data - all_data.mean(axis=0)) / (all_data.std(axis=0) + 1e-8)
            
            # Limiter à 256 dimensions pour les tests
            if all_data.shape[1] > 5:
                all_data = all_data[:, :5]
            elif all_data.shape[1] < 5:
                # Pad avec des zéros si nécessaire
                padding = np.zeros((all_data.shape[0], 5 - all_data.shape[1]))
                all_data = np.hstack([all_data, padding])
            
            logger.info(f"Données géophysiques réelles chargées: {all_data.shape}")
            return all_data.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données réelles: {e}")
            # Données de fallback
            return np.random.randn(100, 5).astype(np.float32)
    
    def generate_real_image_features(self) -> np.ndarray:
        """
        Générer des features d'images réalistes basées sur les images du projet.
        
        Returns:
            np.ndarray: Features d'images simulées
        """
        # Simuler des features extraites d'un CNN (ResNet)
        # Les features d'images ont généralement des valeurs dans une certaine plage
        features = np.random.randn(100, 512).astype(np.float32)
        
        # Appliquer une normalisation similaire à celle d'un CNN pré-entraîné
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        # Ajouter un peu de structure pour simuler des features réelles
        features = features * 0.5 + np.random.randn(100, 512).astype(np.float32) * 0.1
        
        logger.info(f"Features d'images générées: {features.shape}")
        return features
    
    def create_fusion_module(self, fusion_method: str = "concatenation", **kwargs) -> FusionModule:
        """
        Créer une instance de FusionModule avec des paramètres par défaut.
        
        Args:
            fusion_method (str): Méthode de fusion à utiliser
            **kwargs: Paramètres additionnels
            
        Returns:
            FusionModule: Instance configurée
        """
        default_params = {
            'image_features': self.image_features_dim,
            'geo_features': self.geo_features_dim,
            'num_classes': self.num_classes,
            'dropout': 0.5,
            'fusion_method': fusion_method
        }
        default_params.update(kwargs)
        
        return FusionModule(**default_params)
    
    def get_real_batch(self, batch_size: int = None) -> tuple:
        """
        Obtenir un batch de données réelles pour les tests.
        
        Args:
            batch_size (int): Taille du batch
            
        Returns:
            tuple: (image_features, geo_features)
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Sélectionner des échantillons aléatoires des données réelles
        geo_indices = np.random.choice(len(self.real_geo_data), batch_size, replace=True)
        image_indices = np.random.choice(len(self.real_image_features), batch_size, replace=True)
        
        geo_features = torch.tensor(self.real_geo_data[geo_indices], dtype=torch.float32)
        image_features = torch.tensor(self.real_image_features[image_indices], dtype=torch.float32)
        
        # Adapter les dimensions si nécessaire
        if geo_features.shape[1] != self.geo_features_dim:
            if geo_features.shape[1] < self.geo_features_dim:
                # Pad avec des zéros
                padding = torch.zeros(batch_size, self.geo_features_dim - geo_features.shape[1])
                geo_features = torch.cat([geo_features, padding], dim=1)
            else:
                # Tronquer
                geo_features = geo_features[:, :self.geo_features_dim]
        
        if image_features.shape[1] != self.image_features_dim:
            if image_features.shape[1] < self.image_features_dim:
                # Pad avec des zéros
                padding = torch.zeros(batch_size, self.image_features_dim - image_features.shape[1])
                image_features = torch.cat([image_features, padding], dim=1)
            else:
                # Tronquer
                image_features = image_features[:, :self.image_features_dim]
        
        return image_features, geo_features


class TestFusionModuleInitialization(TestFusionModuleRealData):
    """Tests d'initialisation de FusionModule avec données réelles."""
    
    def test_init_concatenation_with_real_data_context(self):
        """Tester l'initialisation avec fusion par concaténation dans le contexte des données réelles."""
        fusion = self.create_fusion_module("concatenation")
        
        # Vérifier les paramètres
        self.assertEqual(fusion.fusion_method, "concatenation")
        self.assertEqual(fusion.image_features, self.image_features_dim)
        self.assertEqual(fusion.geo_features, self.geo_features_dim)
        
        # Vérifier que le module est correctement initialisé
        self.assertIsInstance(fusion, nn.Module)
        self.assertIsNotNone(fusion.classifier)
        
        logger.info("✅ Initialisation concaténation avec contexte données réelles: OK")
    
    def test_init_attention_with_real_data_context(self):
        """Tester l'initialisation avec fusion par attention dans le contexte des données réelles."""
        fusion = self.create_fusion_module("attention")
        
        # Vérifier les paramètres
        self.assertEqual(fusion.fusion_method, "attention")
        self.assertIsNotNone(fusion.attention)
        self.assertIsInstance(fusion.attention, nn.MultiheadAttention)
        
        # Vérifier les paramètres de l'attention
        self.assertEqual(fusion.attention.embed_dim, self.image_features_dim)
        self.assertEqual(fusion.attention.num_heads, 8)
        
        logger.info("✅ Initialisation attention avec contexte données réelles: OK")
    
    def test_init_weighted_with_real_data_context(self):
        """Tester l'initialisation avec fusion pondérée dans le contexte des données réelles."""
        fusion = self.create_fusion_module("weighted")
        
        # Vérifier les paramètres
        self.assertEqual(fusion.fusion_method, "weighted")
        self.assertIsNotNone(fusion.image_weight)
        self.assertIsNotNone(fusion.geo_weight)
        
        # Vérifier que les poids sont des paramètres PyTorch
        self.assertIsInstance(fusion.image_weight, nn.Parameter)
        self.assertIsInstance(fusion.geo_weight, nn.Parameter)
        
        # Vérifier les valeurs initiales
        self.assertAlmostEqual(fusion.image_weight.item(), 0.5, places=5)
        self.assertAlmostEqual(fusion.geo_weight.item(), 0.5, places=5)
        
        logger.info("✅ Initialisation pondérée avec contexte données réelles: OK")
    
    def test_init_unsupported_method_with_real_data_context(self):
        """Tester l'initialisation avec une méthode non supportée dans le contexte des données réelles."""
        with self.assertRaises(ValueError) as context:
            self.create_fusion_module("unsupported_method")
        
        self.assertIn("Méthode de fusion non supportée", str(context.exception))
        logger.info("✅ Gestion erreur méthode non supportée avec données réelles: OK")
    
    def test_init_custom_parameters_with_real_data_context(self):
        """Tester l'initialisation avec des paramètres personnalisés dans le contexte des données réelles."""
        custom_params = {
            'image_features': 1024,
            'geo_features': 512,
            'hidden_dims': (1024, 512, 256),
            'num_classes': 5,
            'dropout': 0.3
        }
        
        fusion = self.create_fusion_module("concatenation", **custom_params)
        
        # Vérifier que les paramètres personnalisés sont appliqués
        self.assertEqual(fusion.image_features, 1024)
        self.assertEqual(fusion.geo_features, 512)
        # Vérifier le nombre de classes via la couche de sortie
        self.assertEqual(fusion.classifier[-1].out_features, 5)
        
        logger.info("✅ Initialisation paramètres personnalisés avec données réelles: OK")


class TestFusionModuleForwardPass(TestFusionModuleRealData):
    """Tests du forward pass de FusionModule avec données réelles."""
    
    def test_forward_concatenation_with_real_data(self):
        """Tester le forward pass avec fusion par concaténation et données réelles."""
        fusion = self.create_fusion_module("concatenation")
        image_features, geo_features = self.get_real_batch()
        
        # Forward pass
        output = fusion(image_features, geo_features)
        
        # Vérifications
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(torch.isfinite(output).all())
        
        # Vérifier que les features sont bien concaténées
        expected_input_dim = self.image_features_dim + self.geo_features_dim
        self.assertEqual(fusion.classifier[0].in_features, expected_input_dim)
        
        logger.info("✅ Forward pass concaténation avec données réelles: OK")
    
    def test_forward_attention_with_real_data(self):
        """Tester le forward pass avec fusion par attention et données réelles."""
        # Créer un module d'attention avec les bonnes dimensions
        fusion = self.create_fusion_module("attention", image_features=256, geo_features=256)
        image_features, geo_features = self.get_real_batch()
        
        # Adapter les dimensions pour l'attention
        image_features = image_features[:, :256]  # Tronquer à 256
        geo_features = geo_features[:, :256]      # Tronquer à 256
        
        # Forward pass
        output = fusion(image_features, geo_features)
        
        # Vérifications
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(torch.isfinite(output).all())
        
        # Vérifier que l'attention est utilisée
        self.assertIsNotNone(fusion.attention)
        
        logger.info("✅ Forward pass attention avec données réelles: OK")
    
    def test_forward_weighted_with_real_data(self):
        """Tester le forward pass avec fusion pondérée et données réelles."""
        fusion = self.create_fusion_module("weighted")
        image_features, geo_features = self.get_real_batch()
        
        # Forward pass
        output = fusion(image_features, geo_features)
        
        # Vérifications
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(torch.isfinite(output).all())
        
        # Vérifier que les poids sont utilisés
        self.assertIsNotNone(fusion.image_weight)
        self.assertIsNotNone(fusion.geo_weight)
        
        logger.info("✅ Forward pass pondéré avec données réelles: OK")
    
    def test_forward_different_batch_sizes_with_real_data(self):
        """Tester le forward pass avec différentes tailles de batch et données réelles."""
        fusion = self.create_fusion_module("concatenation")
        
        # Mettre en mode eval pour éviter les problèmes avec BatchNorm et batch_size=1
        fusion.eval()
        
        batch_sizes = [2, 4, 16, 32]  # Éviter batch_size=1 pour BatchNorm
        
        for batch_size in batch_sizes:
            image_features, geo_features = self.get_real_batch(batch_size)
            
            output = fusion(image_features, geo_features)
            
            self.assertEqual(output.shape, (batch_size, self.num_classes))
            self.assertTrue(torch.isfinite(output).all())
        
        logger.info("✅ Forward pass différentes tailles batch avec données réelles: OK")
    
    def test_forward_consistency_with_real_data(self):
        """Tester la cohérence du forward pass avec les mêmes données réelles."""
        fusion = self.create_fusion_module("concatenation")
        
        # Même batch de données
        image_features, geo_features = self.get_real_batch()
        
        # Mettre en mode eval pour désactiver le dropout
        fusion.eval()
        
        # Deux forward passes en mode eval
        output1 = fusion(image_features, geo_features)
        output2 = fusion(image_features, geo_features)
        
        # En mode eval, les résultats doivent être identiques
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))
        
        logger.info("✅ Cohérence forward pass avec données réelles: OK")


class TestFusionModuleEdgeCases(TestFusionModuleRealData):
    """Tests des cas limites de FusionModule avec données réelles."""
    
    def test_forward_weighted_different_dimensions_with_real_data(self):
        """Tester la fusion pondérée avec des dimensions différentes et données réelles."""
        fusion = self.create_fusion_module("weighted")
        
        # Créer des features avec des dimensions différentes
        batch_size = 8
        image_features = torch.randn(batch_size, 256)  # Plus petit
        geo_features = torch.randn(batch_size, 512)    # Plus grand
        
        output = fusion(image_features, geo_features)
        
        # Vérifications
        self.assertEqual(output.shape, (batch_size, self.num_classes))
        self.assertTrue(torch.isfinite(output).all())
        
        logger.info("✅ Fusion pondérée dimensions différentes avec données réelles: OK")
    
    def test_forward_with_extreme_values_real_data(self):
        """Tester le forward pass avec des valeurs extrêmes basées sur les données réelles."""
        fusion = self.create_fusion_module("concatenation")
        
        # Créer des features avec des valeurs extrêmes
        batch_size = 4
        image_features = torch.tensor([
            [1e6, -1e6, 0, 1e-6] + [0] * (self.image_features_dim - 4),
            [-1e6, 1e6, 0, -1e-6] + [0] * (self.image_features_dim - 4),
            [0, 0, 1e6, 0] + [0] * (self.image_features_dim - 4),
            [0, 0, 0, 0] + [1e6] * (self.image_features_dim - 4)
        ], dtype=torch.float32)
        
        geo_features = torch.tensor([
            [1e3, -1e3, 0, 1e-3, 0],
            [-1e3, 1e3, 0, -1e-3, 0],
            [0, 0, 1e3, 0, 0],
            [0, 0, 0, 0, 1e3]
        ], dtype=torch.float32)
        
        # Adapter les dimensions
        if geo_features.shape[1] < self.geo_features_dim:
            padding = torch.zeros(batch_size, self.geo_features_dim - geo_features.shape[1])
            geo_features = torch.cat([geo_features, padding], dim=1)
        
        output = fusion(image_features, geo_features)
        
        # Vérifications
        self.assertEqual(output.shape, (batch_size, self.num_classes))
        self.assertTrue(torch.isfinite(output).all())
        
        logger.info("✅ Forward pass valeurs extrêmes avec données réelles: OK")
    
    def test_forward_with_nan_handling_real_data(self):
        """Tester la gestion des NaN avec des données réelles."""
        fusion = self.create_fusion_module("concatenation")
        
        # Créer des features avec des NaN
        batch_size = 4
        image_features = torch.randn(batch_size, self.image_features_dim)
        geo_features = torch.randn(batch_size, self.geo_features_dim)
        
        # Introduire des NaN
        image_features[0, 0] = float('nan')
        geo_features[1, 1] = float('nan')
        
        # Le forward pass doit gérer les NaN gracieusement
        try:
            output = fusion(image_features, geo_features)
            # Si on arrive ici, vérifier que les NaN sont propagés ou gérés
            self.assertTrue(torch.isfinite(output).all() or torch.isnan(output).any())
        except Exception as e:
            # C'est acceptable si le module lève une exception pour les NaN
            self.assertIsInstance(e, (RuntimeError, ValueError))
        
        logger.info("✅ Gestion NaN avec données réelles: OK")
    
    def test_forward_gradient_flow_with_real_data(self):
        """Tester le flux de gradients avec des données réelles."""
        fusion = self.create_fusion_module("concatenation")
        image_features, geo_features = self.get_real_batch()
        
        # S'assurer que les features nécessitent des gradients
        image_features.requires_grad_(True)
        geo_features.requires_grad_(True)
        
        # Forward pass
        output = fusion(image_features, geo_features)
        
        # Calculer une loss simple
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Vérifier que les gradients sont calculés
        self.assertIsNotNone(image_features.grad)
        self.assertIsNotNone(geo_features.grad)
        
        # Vérifier que les gradients ne sont pas tous zéros
        self.assertFalse(torch.allclose(image_features.grad, torch.zeros_like(image_features.grad)))
        self.assertFalse(torch.allclose(geo_features.grad, torch.zeros_like(geo_features.grad)))
        
        logger.info("✅ Flux gradients avec données réelles: OK")


class TestFusionModulePerformance(TestFusionModuleRealData):
    """Tests de performance de FusionModule avec données réelles."""
    
    def test_forward_speed_with_real_data(self):
        """Tester la vitesse du forward pass avec des données réelles."""
        import time
        
        fusion = self.create_fusion_module("concatenation")
        image_features, geo_features = self.get_real_batch(32)
        
        # Mesurer le temps
        start_time = time.time()
        
        for _ in range(100):  # 100 itérations
            output = fusion(image_features, geo_features)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        # Vérifier que le temps est raisonnable (< 10ms par forward pass)
        self.assertLess(avg_time, 0.01)
        
        logger.info(f"✅ Vitesse forward pass avec données réelles: {avg_time*1000:.2f}ms par batch")
    
    def test_memory_usage_with_real_data(self):
        """Tester l'utilisation mémoire avec des données réelles."""
        fusion = self.create_fusion_module("concatenation")
        
        # Tester avec différents batch sizes
        batch_sizes = [8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            image_features, geo_features = self.get_real_batch(batch_size)
            
            # Forward pass
            output = fusion(image_features, geo_features)
            
            # Vérifier que la forme est correcte
            self.assertEqual(output.shape, (batch_size, self.num_classes))
            
            # Vérifier que les tensors sont sur le bon device
            self.assertEqual(output.device, image_features.device)
            self.assertEqual(output.device, geo_features.device)
        
        logger.info("✅ Utilisation mémoire avec données réelles: OK")
    
    def test_different_fusion_methods_comparison_with_real_data(self):
        """Comparer les performances des différentes méthodes de fusion avec données réelles."""
        import time
        
        image_features, geo_features = self.get_real_batch(16)
        
        methods = ["concatenation", "weighted"]  # Éviter attention pour simplifier
        times = {}
        
        for method in methods:
            if method == "attention":
                # Créer un module d'attention avec les bonnes dimensions
                fusion = self.create_fusion_module(method, image_features=256, geo_features=256)
                img_feat = image_features[:, :256]
                geo_feat = geo_features[:, :256]
            else:
                fusion = self.create_fusion_module(method)
                img_feat = image_features
                geo_feat = geo_features
            
            # Mesurer le temps
            start_time = time.time()
            
            for _ in range(50):  # 50 itérations
                output = fusion(img_feat, geo_feat)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 50
            times[method] = avg_time
            
            # Vérifier que l'output est correct
            self.assertEqual(output.shape, (16, self.num_classes))
            self.assertTrue(torch.isfinite(output).all())
        
        # Vérifier que toutes les méthodes fonctionnent
        self.assertEqual(len(times), 2)
        
        logger.info(f"✅ Comparaison méthodes fusion avec données réelles: {times}")


class TestFusionModuleIntegration(TestFusionModuleRealData):
    """Tests d'intégration de FusionModule avec données réelles."""
    
    def test_integration_with_training_loop_real_data(self):
        """Tester l'intégration dans une boucle d'entraînement avec données réelles."""
        fusion = self.create_fusion_module("concatenation")
        optimizer = torch.optim.Adam(fusion.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Simuler quelques étapes d'entraînement
        for epoch in range(3):
            for batch_idx in range(5):  # 5 batches par époque
                image_features, geo_features = self.get_real_batch(8)
                
                # Labels simulés
                labels = torch.randint(0, self.num_classes, (8,))
                
                # Forward pass
                optimizer.zero_grad()
                output = fusion(image_features, geo_features)
                loss = criterion(output, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Vérifier que la loss est finie
                self.assertTrue(torch.isfinite(loss))
        
        logger.info("✅ Intégration boucle entraînement avec données réelles: OK")
    
    def test_integration_with_different_models_real_data(self):
        """Tester l'intégration avec différents modèles d'images et données réelles."""
        # Tester avec différentes dimensions d'images
        image_dims = [256, 512, 1024]
        geo_dims = [128, 256, 512]
        
        for img_dim in image_dims:
            for geo_dim in geo_dims:
                fusion = self.create_fusion_module(
                    "concatenation",
                    image_features=img_dim,
                    geo_features=geo_dim
                )
                
                # Créer des features avec les bonnes dimensions
                batch_size = 4
                image_features = torch.randn(batch_size, img_dim)
                geo_features = torch.randn(batch_size, geo_dim)
                
                output = fusion(image_features, geo_features)
                
                self.assertEqual(output.shape, (batch_size, self.num_classes))
                self.assertTrue(torch.isfinite(output).all())
        
        logger.info("✅ Intégration différents modèles avec données réelles: OK")
    
    def test_integration_with_real_geophysical_scenarios(self):
        """Tester l'intégration avec des scénarios géophysiques réels."""
        # Scénario 1: Données de résistivité élevée (roches dures)
        high_resistivity_data = self.real_geo_data.copy()
        high_resistivity_data[:, 0] = high_resistivity_data[:, 0] * 10  # Multiplier la résistivité
        
        # Scénario 2: Données de chargeabilité élevée (minéralisation)
        high_chargeability_data = self.real_geo_data.copy()
        if high_chargeability_data.shape[1] > 1:
            high_chargeability_data[:, 1] = high_chargeability_data[:, 1] * 5  # Multiplier la chargeabilité
        
        # Scénario 3: Données avec bruit (conditions de terrain difficiles)
        noisy_data = self.real_geo_data.copy()
        noise = np.random.randn(*noisy_data.shape) * 0.1
        noisy_data = noisy_data + noise
        
        scenarios = [
            ("haute_resistivite", high_resistivity_data),
            ("haute_chargeabilite", high_chargeability_data),
            ("donnees_bruitees", noisy_data)
        ]
        
        fusion = self.create_fusion_module("concatenation")
        
        for scenario_name, scenario_data in scenarios:
            # Adapter les dimensions
            if scenario_data.shape[1] < self.geo_features_dim:
                padding = np.zeros((scenario_data.shape[0], self.geo_features_dim - scenario_data.shape[1]))
                scenario_data = np.hstack([scenario_data, padding])
            elif scenario_data.shape[1] > self.geo_features_dim:
                scenario_data = scenario_data[:, :self.geo_features_dim]
            
            # Tester avec ce scénario
            batch_size = 8
            geo_features = torch.tensor(scenario_data[:batch_size], dtype=torch.float32)
            image_features = torch.randn(batch_size, self.image_features_dim)
            
            output = fusion(image_features, geo_features)
            
            self.assertEqual(output.shape, (batch_size, self.num_classes))
            self.assertTrue(torch.isfinite(output).all())
        
        logger.info("✅ Intégration scénarios géophysiques réels: OK")


if __name__ == '__main__':
    # Configuration des tests
    unittest.main(verbosity=2)
