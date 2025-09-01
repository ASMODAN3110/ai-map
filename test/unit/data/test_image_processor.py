#!/usr/bin/env python3
"""
Tests unitaires pour GeophysicalImageProcessor avec données réelles
================================================================

Ce module teste toutes les méthodes de la classe GeophysicalImageProcessor
en utilisant les vraies images géophysiques et données du projet AI-MAP.
"""

import unittest
import numpy as np
import torch
from PIL import Image
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

# Ajouter le chemin du projet
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.data.image_processor import GeophysicalImageProcessor, ImageAugmenter


class TestGeophysicalImageProcessorRealData(unittest.TestCase):
    """Tests pour la classe GeophysicalImageProcessor avec données réelles."""
    
    def setUp(self):
        """Initialiser les tests avec des données réelles."""
        self.processor = GeophysicalImageProcessor(
            target_size=(64, 64),
            channels=3,
            normalize=True
        )
        
        # Chemins vers les données réelles du projet
        self.data_root = Path("data")
        self.training_images = self.data_root / "training" / "images"
        self.test_images = self.data_root / "test" / "images"
        self.training_csv = self.data_root / "training" / "csv"
        
        # Images réelles de résistivité
        self.resistivity_images = list((self.training_images / "resistivity").glob("*.JPG"))
        self.chargeability_images = list((self.training_images / "chargeability").glob("*.JPG"))
        self.chargeability_images.extend(list((self.training_images / "chargeability").glob("*.PNG")))
        
        # Fichiers CSV réels
        self.csv_files = list(self.training_csv.glob("*.csv"))
        
        # Vérifier que nous avons des données réelles
        self.assertGreater(len(self.resistivity_images), 0, "Aucune image de résistivité trouvée")
        self.assertGreater(len(self.chargeability_images), 0, "Aucune image de chargeabilité trouvée")
        self.assertGreater(len(self.csv_files), 0, "Aucun fichier CSV trouvé")
        
        # Utiliser la première image de résistivité comme image de test principale
        self.test_image_path = str(self.resistivity_images[0])
        self.test_image = Image.open(self.test_image_path)
        self.test_image_array = np.array(self.test_image)
        
        # Créer un dossier temporaire pour les tests de sauvegarde
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Nettoyer après les tests."""
        # Supprimer le dossier temporaire
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init_with_real_data_context(self):
        """Tester l'initialisation avec le contexte des données réelles."""
        self.assertEqual(self.processor.target_size, (64, 64))
        self.assertEqual(self.processor.channels, 3)
        self.assertTrue(self.processor.normalize)
        self.assertIn('.jpg', self.processor.supported_formats)
        self.assertIn('.png', self.processor.supported_formats)
        
        # Vérifier que nous pouvons traiter les formats réels du projet
        self.assertIn('.JPG', [f.upper() for f in self.processor.supported_formats])
        self.assertIn('.PNG', [f.upper() for f in self.processor.supported_formats])
    
    def test_create_transforms_for_real_images(self):
        """Tester la création des transformations pour les vraies images."""
        transforms = self.processor._create_transforms()
        self.assertIsNotNone(transforms)
        
        # Tester avec grayscale pour les images qui pourraient être en niveaux de gris
        processor_gray = GeophysicalImageProcessor(channels=1, normalize=True)
        transforms_gray = processor_gray._create_transforms()
        self.assertIsNotNone(transforms_gray)
    
    def test_load_real_resistivity_image(self):
        """Tester le chargement d'une vraie image de résistivité."""
        image = self.processor.load_image(self.test_image_path)
        self.assertIsInstance(image, Image.Image)
        self.assertGreater(image.size[0], 0)
        self.assertGreater(image.size[1], 0)
        
        # Vérifier que l'image a du contenu
        img_array = np.array(image)
        self.assertGreater(np.mean(img_array), 0)
        self.assertLess(np.mean(img_array), 255)
    
    def test_load_real_chargeability_image(self):
        """Tester le chargement d'une vraie image de chargeabilité."""
        if self.chargeability_images:
            chargeability_path = str(self.chargeability_images[0])
            image = self.processor.load_image(chargeability_path)
            self.assertIsInstance(image, Image.Image)
            self.assertGreater(image.size[0], 0)
            self.assertGreater(image.size[1], 0)
    
    def test_load_image_file_not_found(self):
        """Tester le chargement d'une image inexistante."""
        with self.assertRaises(FileNotFoundError):
            self.processor.load_image("nonexistent.jpg")
    
    def test_load_image_unsupported_format(self):
        """Tester le chargement d'un format non supporté."""
        # Créer un fichier avec une extension non supportée
        unsupported_path = os.path.join(self.temp_dir, "test.txt")
        with open(unsupported_path, 'w') as f:
            f.write("test")
        
        with self.assertRaises(ValueError):
            self.processor.load_image(unsupported_path)
    
    def test_process_real_resistivity_image(self):
        """Tester le traitement d'une vraie image de résistivité."""
        tensor = self.processor.process_image(self.test_image_path)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dim(), 4)  # batch, channels, height, width
        self.assertEqual(tensor.shape[1], 3)  # 3 canaux RGB
        self.assertEqual(tensor.shape[2:], (64, 64))  # Taille cible
        
        # Vérifier que le tensor contient des données valides
        self.assertGreater(torch.mean(tensor).item(), -1.0)  # Normalisation ImageNet peut donner des valeurs négatives
        self.assertLess(torch.mean(tensor).item(), 2.0)  # Normalisation ImageNet peut donner des valeurs > 1
    
    def test_process_real_chargeability_image(self):
        """Tester le traitement d'une vraie image de chargeabilité."""
        if self.chargeability_images:
            chargeability_path = str(self.chargeability_images[0])
            tensor = self.processor.process_image(chargeability_path)
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertEqual(tensor.dim(), 4)
            self.assertEqual(tensor.shape[1], 3)
            self.assertEqual(tensor.shape[2:], (64, 64))
    
    def test_process_image_batch_with_real_data(self):
        """Tester le traitement d'un lot d'images réelles."""
        # Utiliser les 3 premières images de résistivité
        image_paths = [str(img) for img in self.resistivity_images[:3]]
        batch_tensor = self.processor.process_image_batch(image_paths)
        self.assertIsInstance(batch_tensor, torch.Tensor)
        self.assertEqual(batch_tensor.shape[0], 3)  # 3 images
        self.assertEqual(batch_tensor.shape[1], 3)  # 3 canaux RGB
        self.assertEqual(batch_tensor.shape[2:], (64, 64))  # Taille cible
    
    def test_process_image_batch_empty(self):
        """Tester le traitement d'un lot vide."""
        with self.assertRaises(ValueError):
            self.processor.process_image_batch([])
    
    def test_noise_reduction_on_real_resistivity_image(self):
        """Tester la réduction de bruit sur une vraie image de résistivité."""
        # Utiliser l'image réelle sans ajouter de bruit artificiel
        real_image = Image.open(self.test_image_path)
        
        # Tester différentes méthodes de réduction de bruit
        methods = ["gaussian", "median", "bilateral"]
        for method in methods:
            with self.subTest(method=method):
                cleaned = self.processor.apply_noise_reduction(real_image, method=method)
                self.assertIsInstance(cleaned, Image.Image)
                self.assertEqual(cleaned.size, real_image.size)
                
                # Vérifier que l'image nettoyée a des dimensions cohérentes
                cleaned_array = np.array(cleaned)
                self.assertEqual(cleaned_array.shape, real_image.size[::-1] + (3,))
    
    def test_noise_reduction_on_real_chargeability_image(self):
        """Tester la réduction de bruit sur une vraie image de chargeabilité."""
        if self.chargeability_images:
            chargeability_path = str(self.chargeability_images[0])
            real_image = Image.open(chargeability_path)
            
            cleaned = self.processor.apply_noise_reduction(real_image, method="gaussian", sigma=1.0)
            self.assertIsInstance(cleaned, Image.Image)
            self.assertEqual(cleaned.size, real_image.size)
    
    def test_noise_reduction_invalid_method(self):
        """Tester une méthode de réduction de bruit invalide."""
        with self.assertRaises(ValueError):
            self.processor.apply_noise_reduction(self.test_image, method="invalid_method")
    
    def test_correct_artifacts_on_real_data(self):
        """Tester la correction d'artefacts sur des données réelles."""
        real_image = Image.open(self.test_image_path)
        
        # Tester la correction des lignes de balayage
        corrected = self.processor.correct_artifacts(real_image, "scan_lines")
        self.assertIsInstance(corrected, Image.Image)
        self.assertEqual(corrected.size, real_image.size)
        
        # Tester la correction du bruit sel-et-poivre
        corrected = self.processor.correct_artifacts(real_image, "salt_pepper")
        self.assertIsInstance(corrected, Image.Image)
        self.assertEqual(corrected.size, real_image.size)
    
    def test_correct_artifacts_invalid_type(self):
        """Tester un type d'artefact invalide."""
        with self.assertRaises(ValueError):
            self.processor.correct_artifacts(self.test_image, "invalid_artifact")
    
    def test_enhance_contrast_on_real_resistivity_image(self):
        """Tester l'amélioration du contraste sur une vraie image de résistivité."""
        real_image = Image.open(self.test_image_path)
        
        # Tester différentes méthodes d'amélioration du contraste
        methods = ["histogram_equalization", "adaptive_histogram", "clahe", "gamma_correction"]
        for method in methods:
            with self.subTest(method=method):
                if method == "gamma_correction":
                    enhanced = self.processor.enhance_contrast(real_image, method=method, gamma=1.2)
                else:
                    enhanced = self.processor.enhance_contrast(real_image, method=method)
                
                self.assertIsInstance(enhanced, Image.Image)
                self.assertEqual(enhanced.size, real_image.size)
                
                # Vérifier que l'amélioration a eu un effet
                enhanced_array = np.array(enhanced)
                real_array = np.array(real_image)
                # Les images ne devraient pas être identiques après amélioration
                self.assertFalse(np.array_equal(enhanced_array, real_array))
    
    def test_enhance_contrast_on_real_chargeability_image(self):
        """Tester l'amélioration du contraste sur une vraie image de chargeabilité."""
        if self.chargeability_images:
            chargeability_path = str(self.chargeability_images[0])
            real_image = Image.open(chargeability_path)
            
            enhanced = self.processor.enhance_contrast(real_image, method="clahe")
            self.assertIsInstance(enhanced, Image.Image)
            self.assertEqual(enhanced.size, real_image.size)
    
    def test_enhance_contrast_invalid_method(self):
        """Tester une méthode d'amélioration du contraste invalide."""
        with self.assertRaises(ValueError):
            self.processor.enhance_contrast(self.test_image, method="invalid_method")
    
    def test_apply_geophysical_specific_cleaning_on_real_data(self):
        """Tester le nettoyage géophysique sur des données réelles."""
        real_image = Image.open(self.test_image_path)
        
        # Pipeline de nettoyage complet
        cleaning_steps = ["noise_reduction", "scan_lines_removal", "contrast_enhancement"]
        cleaned = self.processor.apply_geophysical_specific_cleaning(real_image, cleaning_steps)
        self.assertIsInstance(cleaned, Image.Image)
        self.assertEqual(cleaned.size, real_image.size)
        
        # Vérifier que le nettoyage a eu un effet
        cleaned_array = np.array(cleaned)
        real_array = np.array(real_image)
        self.assertFalse(np.array_equal(cleaned_array, real_array))
    
    def test_get_cleaning_summary_on_real_data(self):
        """Tester l'obtention du résumé de nettoyage sur des données réelles."""
        real_image = Image.open(self.test_image_path)
        cleaning_methods = ["noise_reduction", "contrast_enhancement"]
        
        summary = self.processor.get_cleaning_summary(real_image, cleaning_methods)
        self.assertIsInstance(summary, dict)
        self.assertIn('noise_reduction', summary)
        self.assertIn('contrast_improvement', summary)
        self.assertIn('cleaning_methods_applied', summary)
        self.assertIn('original_std', summary)
        self.assertIn('cleaned_std', summary)
        
        # Vérifier que les valeurs sont cohérentes
        self.assertIsInstance(summary['original_std'], float)
        self.assertIsInstance(summary['cleaned_std'], float)
    
    def test_extract_geophysical_features_from_real_image(self):
        """Tester l'extraction de features depuis une vraie image géophysique."""
        real_image = Image.open(self.test_image_path)
        features = self.processor.extract_geophysical_features_from_image(real_image)
        
        self.assertIsInstance(features, dict)
        self.assertIn('mean_intensity', features)
        self.assertIn('std_intensity', features)
        self.assertIn('gradient_magnitude', features)
        self.assertIn('histogram', features)
        self.assertIn('image_size', features)
        
        # Vérifier que les features sont cohérentes avec l'image réelle
        self.assertIsInstance(features['histogram'], np.ndarray)
        self.assertEqual(features['histogram'].shape, (256,))
        self.assertEqual(features['image_size'], real_image.size[::-1] + (3,))
        
        # Vérifier que les valeurs d'intensité sont dans des plages raisonnables
        self.assertGreater(features['mean_intensity'], 0)
        self.assertLess(features['mean_intensity'], 255)
        self.assertGreater(features['std_intensity'], 0)
    
    def test_extract_geophysical_features_from_real_file(self):
        """Tester l'extraction de features depuis un fichier d'image réel."""
        features = self.processor.extract_geophysical_features(self.test_image_path)
        self.assertIsInstance(features, dict)
        self.assertIn('mean_intensity', features)
        self.assertIn('std_intensity', features)
        self.assertIn('gradient_magnitude', features)
    
    def test_extract_features_from_multiple_real_images(self):
        """Tester l'extraction de features depuis plusieurs images réelles."""
        # Tester sur les 3 premières images de résistivité
        for i, img_path in enumerate(self.resistivity_images[:3]):
            with self.subTest(image=i, path=str(img_path)):
                features = self.processor.extract_geophysical_features(str(img_path))
                self.assertIsInstance(features, dict)
                self.assertIn('mean_intensity', features)
                self.assertIn('std_intensity', features)
                
                # Vérifier que les features sont cohérentes
                self.assertGreater(features['mean_intensity'], 0)
                self.assertLess(features['mean_intensity'], 255)
    
    def test_save_processed_real_image(self):
        """Tester la sauvegarde d'une image réelle prétraitée."""
        output_path = os.path.join(self.temp_dir, "processed_real_image.jpg")
        
        saved_path = self.processor.save_processed_image(self.test_image_path, output_path)
        self.assertEqual(saved_path, output_path)
        self.assertTrue(os.path.exists(output_path))
        
        # Vérifier que l'image a été redimensionnée
        saved_image = Image.open(output_path)
        self.assertEqual(saved_image.size, (64, 64))
        
        # Vérifier que l'image contient des données
        saved_array = np.array(saved_image)
        self.assertGreater(np.mean(saved_array), 0)
    
    def test_process_different_image_formats(self):
        """Tester le traitement de différents formats d'images réelles."""
        # Tester les formats JPG et PNG
        test_images = []
        if self.resistivity_images:
            test_images.append(str(self.resistivity_images[0]))  # JPG
        if self.chargeability_images:
            test_images.append(str(self.chargeability_images[0]))  # JPG ou PNG
        
        for img_path in test_images:
            with self.subTest(image_path=img_path):
                # Charger l'image
                image = self.processor.load_image(img_path)
                self.assertIsInstance(image, Image.Image)
                
                # Traiter l'image
                tensor = self.processor.process_image(img_path)
                self.assertIsInstance(tensor, torch.Tensor)
                self.assertEqual(tensor.dim(), 4)
                self.assertEqual(tensor.shape[1], 3)
                self.assertEqual(tensor.shape[2:], (64, 64))
    
    def test_create_image_processor_function(self):
        """Tester la fonction factory create_image_processor."""
        from src.data.image_processor import create_image_processor
        
        # Test avec paramètres par défaut
        processor_default = create_image_processor()
        self.assertIsInstance(processor_default, GeophysicalImageProcessor)
        self.assertEqual(processor_default.target_size, (64, 64))
        self.assertEqual(processor_default.channels, 3)
        self.assertTrue(processor_default.normalize)
        
        # Test avec paramètres personnalisés
        processor_custom = create_image_processor(target_size=(128, 128), channels=1)
        self.assertIsInstance(processor_custom, GeophysicalImageProcessor)
        self.assertEqual(processor_custom.target_size, (128, 128))
        self.assertEqual(processor_custom.channels, 1)
        self.assertTrue(processor_custom.normalize)
        
        # Test avec différentes tailles
        sizes_to_test = [(32, 32), (256, 256), (512, 512)]
        for size in sizes_to_test:
            with self.subTest(size=size):
                processor = create_image_processor(target_size=size)
                self.assertEqual(processor.target_size, size)
                self.assertEqual(processor.channels, 3)
        
        # Test avec différents nombres de canaux
        channels_to_test = [1, 3, 4]
        for channels in channels_to_test:
            with self.subTest(channels=channels):
                processor = create_image_processor(channels=channels)
                self.assertEqual(processor.target_size, (64, 64))
                self.assertEqual(processor.channels, channels)


class TestImageProcessorUtilityFunctions(unittest.TestCase):
    """Tests pour les fonctions utilitaires du module image_processor."""
    
    def test_create_image_processor_function(self):
        """Tester la fonction factory create_image_processor."""
        from src.data.image_processor import create_image_processor
        
        # Test avec paramètres par défaut
        processor_default = create_image_processor()
        self.assertIsInstance(processor_default, GeophysicalImageProcessor)
        self.assertEqual(processor_default.target_size, (64, 64))
        self.assertEqual(processor_default.channels, 3)
        self.assertTrue(processor_default.normalize)
        
        # Test avec paramètres personnalisés
        processor_custom = create_image_processor(target_size=(128, 128), channels=1)
        self.assertIsInstance(processor_custom, GeophysicalImageProcessor)
        self.assertEqual(processor_custom.target_size, (128, 128))
        self.assertEqual(processor_custom.channels, 1)
        self.assertTrue(processor_custom.normalize)
        
        # Test avec différentes tailles
        sizes_to_test = [(32, 32), (256, 256), (512, 512)]
        for size in sizes_to_test:
            with self.subTest(size=size):
                processor = create_image_processor(target_size=size)
                self.assertEqual(processor.target_size, size)
                self.assertEqual(processor.channels, 3)
        
        # Test avec différents nombres de canaux
        channels_to_test = [1, 3, 4]
        for channels in channels_to_test:
            with self.subTest(channels=channels):
                processor = create_image_processor(channels=channels)
                self.assertEqual(processor.target_size, (64, 64))
                self.assertEqual(processor.channels, channels)
    
    def test_create_image_processor_edge_cases(self):
        """Tester les cas limites de create_image_processor."""
        from src.data.image_processor import create_image_processor
        
        # Test avec taille minimale
        processor_min = create_image_processor(target_size=(1, 1))
        self.assertEqual(processor_min.target_size, (1, 1))
        
        # Test avec taille maximale raisonnable
        processor_max = create_image_processor(target_size=(1024, 1024))
        self.assertEqual(processor_max.target_size, (1024, 1024))
        
        # Test avec canaux extrêmes
        processor_1ch = create_image_processor(channels=1)
        self.assertEqual(processor_1ch.channels, 1)
        
        processor_4ch = create_image_processor(channels=4)
        self.assertEqual(processor_4ch.channels, 4)


class TestImageAugmenterRealData(unittest.TestCase):
    """Tests pour la classe ImageAugmenter avec données réelles."""
    
    def setUp(self):
        """Initialiser les tests avec des données réelles."""
        self.augmenter = ImageAugmenter(random_seed=42)
        
        # Utiliser une vraie image de résistivité
        self.data_root = Path("data")
        self.training_images = self.data_root / "training" / "images"
        self.resistivity_images = list((self.training_images / "resistivity").glob("*.JPG"))
        
        self.assertGreater(len(self.resistivity_images), 0, "Aucune image de résistivité trouvée")
        
        # Charger une vraie image
        self.test_image_path = str(self.resistivity_images[0])
        self.test_image = Image.open(self.test_image_path)
        
        # Créer un dossier temporaire
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Nettoyer après les tests."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init_with_real_data_context(self):
        """Tester l'initialisation avec le contexte des données réelles."""
        self.assertIsNotNone(self.augmenter.augmentation_history)
        self.assertIsNotNone(self.augmenter.geophysical_patterns)
    
    def test_augment_real_resistivity_image(self):
        """Tester l'augmentation d'une vraie image de résistivité."""
        augmentations = ["rotation", "flip_horizontal"]
        augmented_images = self.augmenter.augment_image(
            self.test_image, augmentations, num_augmentations=2
        )
        
        self.assertIsInstance(augmented_images, list)
        self.assertEqual(len(augmented_images), 3)  # Original + 2 augmentations
        self.assertIsInstance(augmented_images[0], Image.Image)  # Original
        
        # Vérifier que les images augmentées ont des dimensions cohérentes
        for i, img in enumerate(augmented_images):
            self.assertEqual(img.size, self.test_image.size)
            self.assertEqual(img.mode, self.test_image.mode)
    
    def test_geophysical_augmentation_on_real_data(self):
        """Tester l'augmentation géophysique sur des données réelles."""
        geophysical_augmentations = [
            "geological_stratification", "fracture_patterns", "mineral_inclusions"
        ]
        
        augmented_images = self.augmenter.augment_image(
            self.test_image, geophysical_augmentations, num_augmentations=1
        )
        
        self.assertIsInstance(augmented_images, list)
        self.assertGreater(len(augmented_images), 1)
        
        # Vérifier que les augmentations géophysiques ont eu un effet
        for i, img in enumerate(augmented_images[1:], 1):  # Ignorer l'original
            self.assertEqual(img.size, self.test_image.size)
            self.assertEqual(img.mode, self.test_image.mode)
    
    def test_augmentation_summary_with_real_data(self):
        """Tester l'obtention du résumé des augmentations avec des données réelles."""
        # Effectuer quelques augmentations
        self.augmenter.augment_image(
            self.test_image, ["rotation"], num_augmentations=1
        )
        
        summary = self.augmenter.get_augmentation_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('total_augmentations', summary)
        self.assertIn('available_techniques', summary)
        
        # Vérifier que le résumé est cohérent
        self.assertGreater(summary['total_augmentations'], 0)
        self.assertIsInstance(summary['available_techniques'], list)
        self.assertGreater(len(summary['available_techniques']), 0)


if __name__ == '__main__':
    # Créer la suite de tests
    test_suite = unittest.TestSuite()
    
    # Ajouter tous les tests
    test_suite.addTest(unittest.makeSuite(TestGeophysicalImageProcessorRealData))
    test_suite.addTest(unittest.makeSuite(TestImageProcessorUtilityFunctions))
    test_suite.addTest(unittest.makeSuite(TestImageAugmenterRealData))
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Afficher le résumé
    print(f"\n{'='*60}")
    print(f"RÉSULTATS COMPLETS DES TESTS")
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
    print(f"TOUS LES TESTS TERMINÉS")
    print(f"{'='*60}")
