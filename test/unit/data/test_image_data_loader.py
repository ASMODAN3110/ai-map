#!/usr/bin/env python3
"""
Tests unitaires pour ImageDataLoader avec données réelles
========================================================

Ce module teste toutes les méthodes de la classe ImageDataLoader
en utilisant les vraies images géophysiques du projet AI-MAP.
"""

import unittest
import numpy as np
import torch
from PIL import Image
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ajouter le chemin du projet
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.data.image_processor import GeophysicalImageProcessor, ImageAugmenter
from src.model.geophysical_image_trainer import ImageDataLoader


class TestImageDataLoaderRealData(unittest.TestCase):
    """Tests pour la classe ImageDataLoader avec données réelles."""
    
    def setUp(self):
        """Initialiser les tests avec des données réelles."""
        # Créer un processeur d'images
        self.image_processor = GeophysicalImageProcessor(
            target_size=(64, 64),
            channels=3,
            normalize=True
        )
        
        # Créer un augmenteur d'images
        self.augmenter = ImageAugmenter(random_seed=42)
        
        # Créer le gestionnaire de données
        self.data_loader = ImageDataLoader(
            image_processor=self.image_processor,
            augmenter=self.augmenter
        )
        
        # Chemins vers les données réelles du projet
        self.data_root = Path("data")
        self.training_images = self.data_root / "training" / "images"
        self.test_images = self.data_root / "test" / "images"
        
        # Images réelles de résistivité
        self.resistivity_images = list((self.training_images / "resistivity").glob("*.JPG"))
        self.chargeability_images = list((self.training_images / "chargeability").glob("*.JPG"))
        self.chargeability_images.extend(list((self.training_images / "chargeability").glob("*.PNG")))
        
        # Vérifier que nous avons des données réelles
        self.assertGreater(len(self.resistivity_images), 0, "Aucune image de résistivité trouvée")
        self.assertGreater(len(self.chargeability_images), 0, "Aucune image de chargeabilité trouvée")
        
        # Utiliser les 3 premières images pour les tests
        self.test_image_paths = [str(img) for img in self.resistivity_images[:3]]
        self.test_labels = [0, 1, 0]  # Labels simulés pour les tests
        
        # Créer un dossier temporaire
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Nettoyer après les tests."""
        # Supprimer le dossier temporaire
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init_with_real_components(self):
        """Tester l'initialisation avec des composants réels."""
        # Vérifier que les composants sont correctement assignés
        self.assertIs(self.data_loader.image_processor, self.image_processor)
        self.assertIs(self.data_loader.augmenter, self.augmenter)
        
        # Vérifier que les attributs sont initialisés
        self.assertIsInstance(self.data_loader.processed_images, dict)
        self.assertIsInstance(self.data_loader.processed_features, dict)
        
        # Vérifier que l'augmenteur par défaut est créé si aucun n'est fourni
        data_loader_default = ImageDataLoader(image_processor=self.image_processor)
        self.assertIsInstance(data_loader_default.augmenter, ImageAugmenter)
    
    def test_load_and_process_images_basic(self):
        """Tester le chargement et traitement d'images sans augmentation."""
        # Traiter les images sans augmentation
        batch_tensor = self.data_loader.load_and_process_images(
            self.test_image_paths,
            augmentations=None,
            num_augmentations=0
        )
        
        # Vérifications de base
        self.assertIsInstance(batch_tensor, torch.Tensor)
        self.assertEqual(batch_tensor.dim(), 4)  # batch, channels, height, width
        self.assertEqual(batch_tensor.shape[0], 3)  # 3 images
        self.assertEqual(batch_tensor.shape[1], 3)  # 3 canaux RGB
        self.assertEqual(batch_tensor.shape[2:], (64, 64))  # Taille cible
        
        # Vérifier que les images contiennent des données valides
        self.assertGreater(torch.mean(batch_tensor).item(), -1.0)
        self.assertLess(torch.mean(batch_tensor).item(), 2.0)
    
    def test_load_and_process_images_with_augmentation(self):
        """Tester le chargement et traitement d'images avec augmentation."""
        # Techniques d'augmentation simples
        augmentations = ["rotation", "flip_horizontal"]
        num_augmentations = 2
        
        # Traiter les images avec augmentation
        batch_tensor = self.data_loader.load_and_process_images(
            self.test_image_paths,
            augmentations=augmentations,
            num_augmentations=num_augmentations
        )
        
        # Vérifications avec augmentation
        self.assertIsInstance(batch_tensor, torch.Tensor)
        self.assertEqual(batch_tensor.dim(), 4)
        
        # Calculer le nombre total d'images attendu
        expected_images = len(self.test_image_paths) * (1 + num_augmentations)
        self.assertEqual(batch_tensor.shape[0], expected_images)
        
        # Vérifier que les dimensions sont cohérentes
        self.assertEqual(batch_tensor.shape[1], 3)  # 3 canaux RGB
        self.assertEqual(batch_tensor.shape[2:], (64, 64))  # Taille cible
    
    def test_load_and_process_images_with_geophysical_augmentation(self):
        """Tester l'augmentation géophysique sur des données réelles."""
        # Techniques d'augmentation géophysiques
        geophysical_augmentations = [
            "geological_stratification", "fracture_patterns", "mineral_inclusions"
        ]
        
        # Traiter avec augmentation géophysique
        batch_tensor = self.data_loader.load_and_process_images(
            self.test_image_paths[:2],  # Utiliser seulement 2 images pour le test
            augmentations=geophysical_augmentations,
            num_augmentations=1
        )
        
        # Vérifications
        self.assertIsInstance(batch_tensor, torch.Tensor)
        self.assertEqual(batch_tensor.dim(), 4)
        
        # Vérifier que l'augmentation a eu un effet
        expected_images = 2 * (1 + 1)  # 2 images originales + 2 augmentées
        self.assertEqual(batch_tensor.shape[0], expected_images)
    
    def test_load_and_process_images_empty_list(self):
        """Tester le traitement d'une liste d'images vide."""
        with self.assertRaises(ValueError):
            self.data_loader.load_and_process_images([], augmentations=None, num_augmentations=0)
    
    def test_load_and_process_images_invalid_path(self):
        """Tester le traitement avec des chemins d'images invalides."""
        invalid_paths = ["nonexistent1.jpg", "nonexistent2.jpg"]
        
        # Cela devrait lever une exception ou gérer l'erreur gracieusement
        try:
            batch_tensor = self.data_loader.load_and_process_images(
                invalid_paths, augmentations=None, num_augmentations=0
            )
            # Si aucune exception n'est levée, vérifier que le résultat est vide
            self.assertEqual(batch_tensor.shape[0], 0)
        except ValueError as e:
            # C'est le comportement attendu
            self.assertIn("Aucune image n'a pu être traitée", str(e))
    
    def test_extract_features_batch(self):
        """Tester l'extraction de features d'un lot d'images."""
        # Extraire les features
        features_batch = self.data_loader.extract_features_batch(self.test_image_paths)
        
        # Vérifier la structure des features
        self.assertIsInstance(features_batch, dict)
        self.assertIn('mean_intensities', features_batch)
        self.assertIn('gradient_magnitudes', features_batch)
        self.assertIn('histograms', features_batch)
        self.assertIn('image_sizes', features_batch)
        
        # Vérifier que les features sont des arrays numpy
        for key in features_batch:
            self.assertIsInstance(features_batch[key], np.ndarray)
        
        # Vérifier les dimensions
        self.assertEqual(len(features_batch['mean_intensities']), 3)
        self.assertEqual(len(features_batch['gradient_magnitudes']), 3)
        self.assertEqual(len(features_batch['histograms']), 3)
        self.assertEqual(len(features_batch['image_sizes']), 3)
        
        # Vérifier que les features ont des valeurs cohérentes
        self.assertTrue(np.all(features_batch['mean_intensities'] > 0))
        self.assertTrue(np.all(features_batch['mean_intensities'] < 255))
        self.assertTrue(np.all(features_batch['gradient_magnitudes'] >= 0))
        
        # Vérifier la forme des histogrammes
        for hist in features_batch['histograms']:
            self.assertEqual(hist.shape, (256,))
    
    def test_extract_features_batch_empty_list(self):
        """Tester l'extraction de features d'une liste vide."""
        features_batch = self.data_loader.extract_features_batch([])
        
        # Vérifier que les listes sont vides
        for key in features_batch:
            self.assertEqual(len(features_batch[key]), 0)
    
    def test_extract_features_batch_partial_failure(self):
        """Tester l'extraction de features avec certaines images qui échouent."""
        # Mélanger des chemins valides et invalides
        mixed_paths = self.test_image_paths[:2] + ["nonexistent.jpg"]
        
        features_batch = self.data_loader.extract_features_batch(mixed_paths)
        
        # Vérifier que seules les images valides ont été traitées
        self.assertEqual(len(features_batch['mean_intensities']), 2)
        self.assertEqual(len(features_batch['gradient_magnitudes']), 2)
    
    def test_create_image_dataset_basic(self):
        """Tester la création d'un dataset sans augmentation."""
        # Créer un dataset de base
        dataset = self.data_loader.create_image_dataset(
            self.test_image_paths,
            self.test_labels,
            augmentations=None,
            num_augmentations=0
        )
        
        # Vérifications de base
        self.assertIsInstance(dataset, torch.utils.data.TensorDataset)
        self.assertEqual(len(dataset), 3)  # 3 images, 3 labels
        
        # Vérifier les données et labels
        images, labels = dataset[0]
        self.assertIsInstance(images, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(images.shape, (3, 64, 64))  # 3 canaux, 64x64
        self.assertEqual(labels.item(), 0)
    
    def test_create_image_dataset_with_augmentation(self):
        """Tester la création d'un dataset avec augmentation."""
        # Créer un dataset avec augmentation
        augmentations = ["rotation", "flip_horizontal"]
        num_augmentations = 2
        
        dataset = self.data_loader.create_image_dataset(
            self.test_image_paths,
            self.test_labels,
            augmentations=augmentations,
            num_augmentations=num_augmentations
        )
        
        # Vérifications avec augmentation
        self.assertIsInstance(dataset, torch.utils.data.TensorDataset)
        
        # Calculer le nombre total d'échantillons attendu
        expected_samples = len(self.test_image_paths) * (1 + num_augmentations)
        self.assertEqual(len(dataset), expected_samples)
        
        # Vérifier que les labels sont correctement étendus
        for i in range(len(dataset)):
            _, label = dataset[i]
            self.assertIsInstance(label, torch.Tensor)
            self.assertIn(label.item(), self.test_labels)
    
    def test_create_image_dataset_with_geophysical_augmentation(self):
        """Tester la création d'un dataset avec augmentation géophysique."""
        # Techniques d'augmentation géophysiques
        geophysical_augmentations = [
            "geological_stratification", "fracture_patterns"
        ]
        
        dataset = self.data_loader.create_image_dataset(
            self.test_image_paths[:2],  # Utiliser seulement 2 images
            self.test_labels[:2],
            augmentations=geophysical_augmentations,
            num_augmentations=1
        )
        
        # Vérifications
        self.assertIsInstance(dataset, torch.utils.data.TensorDataset)
        expected_samples = 2 * (1 + 1)  # 2 images originales + 2 augmentées
        self.assertEqual(len(dataset), expected_samples)
    
    def test_create_image_dataset_mismatched_lengths(self):
        """Tester la création d'un dataset avec des longueurs de chemins et labels différentes."""
        # Plus de chemins que de labels - cela devrait échouer lors de la création du TensorDataset
        try:
            dataset = self.data_loader.create_image_dataset(
                self.test_image_paths,  # 3 chemins
                self.test_labels[:2],   # 2 labels
                augmentations=None,
                num_augmentations=0
            )
            # Si aucune exception n'est levée, vérifier que le dataset a la bonne taille
            self.assertEqual(len(dataset), 2)  # Seulement 2 échantillons (labels)
        except AssertionError as e:
            # C'est le comportement attendu de PyTorch TensorDataset
            self.assertIn("Size mismatch between tensors", str(e))
    
    def test_integration_with_real_data(self):
        """Test d'intégration complet avec des données réelles."""
        # Test complet du pipeline
        augmentations = ["rotation", "flip_horizontal"]
        num_augmentations = 1
        
        # 1. Charger et traiter les images
        batch_tensor = self.data_loader.load_and_process_images(
            self.test_image_paths,
            augmentations=augmentations,
            num_augmentations=num_augmentations
        )
        
        # 2. Extraire les features
        features_batch = self.data_loader.extract_features_batch(self.test_image_paths)
        
        # 3. Créer le dataset
        dataset = self.data_loader.create_image_dataset(
            self.test_image_paths,
            self.test_labels,
            augmentations=augmentations,
            num_augmentations=num_augmentations
        )
        
        # Vérifications d'intégration
        self.assertIsInstance(batch_tensor, torch.Tensor)
        self.assertIsInstance(features_batch, dict)
        self.assertIsInstance(dataset, torch.utils.data.TensorDataset)
        
        # Vérifier la cohérence des dimensions
        expected_images = len(self.test_image_paths) * (1 + num_augmentations)
        self.assertEqual(batch_tensor.shape[0], expected_images)
        self.assertEqual(len(dataset), expected_images)
        
        # Vérifier que les features correspondent aux images
        self.assertEqual(len(features_batch['mean_intensities']), len(self.test_image_paths))
    
    def test_error_handling_and_logging(self):
        """Tester la gestion d'erreurs et le logging."""
        # Créer un gestionnaire avec un processeur qui peut échouer
        failing_processor = MagicMock()
        failing_processor.process_image.side_effect = Exception("Erreur de traitement")
        
        failing_loader = ImageDataLoader(
            image_processor=failing_processor,
            augmenter=self.augmenter
        )
        
        # Tester que les erreurs sont gérées gracieusement
        try:
            batch_tensor = failing_loader.load_and_process_images(
                self.test_image_paths[:1],
                augmentations=None,
                num_augmentations=0
            )
            # Si aucune exception n'est levée, vérifier que le résultat est vide
            self.assertEqual(batch_tensor.shape[0], 0)
        except ValueError as e:
            # C'est le comportement attendu
            self.assertIn("Aucune image n'a pu être traitée", str(e))


if __name__ == '__main__':
    # Créer la suite de tests
    test_suite = unittest.TestSuite()
    
    # Ajouter tous les tests
    test_suite.addTest(unittest.makeSuite(TestImageDataLoaderRealData))
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Afficher le résumé
    print(f"\n{'='*60}")
    print(f"RÉSULTATS DES TESTS ImageDataLoader")
    print(f"{'='*60}")
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    print(f"Succès: {result.testsRun - len(result.failures) - len(result.errors)}")
    
    if result.failures:
        print(f"\nÉCHECS:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\nERREURS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    print(f"\n{'='*60}")
    print(f"TESTS ImageDataLoader TERMINÉS")
    print(f"{'='*60}")

