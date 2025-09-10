#!/usr/bin/env python3
"""
Test unitaire pour la méthode augment_3d_volume de GeophysicalDataAugmenter

Ce test vérifie le bon fonctionnement de la méthode augment_3d_volume
avec des données géophysiques réelles (PD.csv et S.csv) et des scénarios de test variés.
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.preprocessor.data_augmenter import GeophysicalDataAugmenter
from backend.preprocessor.data_cleaner import GeophysicalDataCleaner


class TestDataAugmenterAugment3dVolume(unittest.TestCase):
    """Tests pour la méthode augment_3d_volume de GeophysicalDataAugmenter"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        # Créer un répertoire temporaire pour les tests
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Créer la structure des dossiers de test
        self.test_raw_dir = self.test_dir / "raw"
        self.test_processed_dir = self.test_dir / "processed"
        self.test_raw_dir.mkdir(parents=True, exist_ok=True)
        self.test_processed_dir.mkdir(exist_ok=True)
        
        # Copier les fichiers de test depuis fixtures
        fixtures_dir = Path(__file__).parent.parent.parent / "fixtures" / "raw"
        self.pd_test_file = fixtures_dir / "PD.csv"
        self.s_test_file = fixtures_dir / "S.csv"
        
        # Copier les fichiers vers le dossier de test
        if self.pd_test_file.exists():
            shutil.copy2(self.pd_test_file, self.test_raw_dir / "PD.csv")
        if self.s_test_file.exists():
            shutil.copy2(self.s_test_file, self.test_raw_dir / "S.csv")
        
        # Créer une instance du cleaner avec des chemins de test
        with patch('backend.preprocessor.data_cleaner.CONFIG') as mock_config:
            mock_config.paths.raw_data_dir = str(self.test_raw_dir)
            mock_config.paths.processed_data_dir = str(self.test_processed_dir)
            mock_config.geophysical_data.coordinate_systems = {
                'wgs84': "EPSG:4326",
                'utm_proj': "EPSG:32630"
            }
            self.cleaner = GeophysicalDataCleaner()
        
        # Initialiser l'augmenteur
        self.augmenter = GeophysicalDataAugmenter(random_seed=42)
        
        # Créer des volumes de test basés sur les vraies données
        self._create_test_volumes()
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Supprimer le répertoire temporaire
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _create_test_volumes(self):
        """Créer des volumes de test basés sur les vraies données"""
        # Charger les données PD.csv
        pd_df = pd.read_csv(self.test_raw_dir / "PD.csv", sep=';')
        
        # Créer un volume 3D basé sur PD.csv (8x8x8x4)
        volume_size = 8
        self.volume_3d_pd = np.zeros((volume_size, volume_size, volume_size, 4))
        
        # Remplir le volume avec les données PD.csv
        for i in range(min(len(pd_df), volume_size * volume_size * volume_size)):
            d = i // (volume_size * volume_size)
            h = (i % (volume_size * volume_size)) // volume_size
            w = i % volume_size
            if d < volume_size and h < volume_size and w < volume_size:
                self.volume_3d_pd[d, h, w, 0] = pd_df.iloc[i]['Rho(ohm.m)'] if i < len(pd_df) else 0
                self.volume_3d_pd[d, h, w, 1] = pd_df.iloc[i]['M (mV/V)'] if i < len(pd_df) else 0
                self.volume_3d_pd[d, h, w, 2] = pd_df.iloc[i]['x'] if i < len(pd_df) else 0
                self.volume_3d_pd[d, h, w, 3] = pd_df.iloc[i]['y'] if i < len(pd_df) else 0
        
        # Charger les données S.csv
        s_df = pd.read_csv(self.test_raw_dir / "S.csv", sep=';')
        
        # Créer un volume 3D basé sur S.csv (16x16x16x4)
        volume_size_s = 16
        self.volume_3d_s = np.zeros((volume_size_s, volume_size_s, volume_size_s, 4))
        
        # Remplir le volume avec les données S.csv
        for i in range(min(len(s_df), volume_size_s * volume_size_s * volume_size_s)):
            d = i // (volume_size_s * volume_size_s)
            h = (i % (volume_size_s * volume_size_s)) // volume_size_s
            w = i % volume_size_s
            if d < volume_size_s and h < volume_size_s and w < volume_size_s:
                self.volume_3d_s[d, h, w, 0] = s_df.iloc[i]['Rho (Ohm.m)'] if i < len(s_df) else 0
                self.volume_3d_s[d, h, w, 1] = s_df.iloc[i]['M (mV/V)'] if i < len(s_df) else 0
                self.volume_3d_s[d, h, w, 2] = s_df.iloc[i]['LAT'] if i < len(s_df) else 0
                self.volume_3d_s[d, h, w, 3] = s_df.iloc[i]['LON'] if i < len(s_df) else 0
        
        # Créer un volume de test simple pour les tests de base
        self.volume_3d_simple = np.random.rand(6, 6, 6, 4)
        
        print(f"📊 Volumes de test créés:")
        print(f"   - PD.csv: {self.volume_3d_pd.shape}")
        print(f"   - S.csv: {self.volume_3d_s.shape}")
        print(f"   - Simple: {self.volume_3d_simple.shape}")
    
    def test_augment_3d_volume_basic_functionality(self):
        """Test de la fonctionnalité de base de augment_3d_volume"""
        # Test avec un volume simple
        augmentations = ["flip_horizontal"]
        result = self.augmenter.augment_3d_volume(self.volume_3d_simple, augmentations, 1)
        
        # Vérifications de base
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, self.volume_3d_simple.shape)
        
        # Vérifier que l'augmentation a été appliquée
        self.assertFalse(np.array_equal(result[0], self.volume_3d_simple))
        
        print("✅ Fonctionnalité de base de augment_3d_volume validée")
    
    def test_augment_3d_volume_with_pd_csv_data(self):
        """Test de augment_3d_volume avec les données PD.csv"""
        augmentations = ["rotation", "flip_horizontal", "gaussian_noise"]
        num_augmentations = 3
        
        result = self.augmenter.augment_3d_volume(
            self.volume_3d_pd, 
            augmentations, 
            num_augmentations
        )
        
        # Vérifications
        self.assertEqual(len(result), num_augmentations)
        for augmented_volume in result:
            self.assertEqual(augmented_volume.shape, self.volume_3d_pd.shape)
            self.assertFalse(np.array_equal(augmented_volume, self.volume_3d_pd))
        
        # Vérifier l'historique
        self.assertEqual(len(self.augmenter.augmentation_history), num_augmentations)
        
        print(f"✅ augment_3d_volume avec PD.csv validé: {num_augmentations} augmentations générées")
    
    def test_augment_3d_volume_with_s_csv_data(self):
        """Test de augment_3d_volume avec les données S.csv"""
        augmentations = ["flip_vertical", "gaussian_noise", "value_variation"]
        num_augmentations = 2
        
        result = self.augmenter.augment_3d_volume(
            self.volume_3d_s, 
            augmentations, 
            num_augmentations
        )
        
        # Vérifications
        self.assertEqual(len(result), num_augmentations)
        for augmented_volume in result:
            self.assertEqual(augmented_volume.shape, self.volume_3d_s.shape)
            self.assertFalse(np.array_equal(augmented_volume, self.volume_3d_s))
        
        print(f"✅ augment_3d_volume avec S.csv validé: {num_augmentations} augmentations générées")
    
    def test_augment_3d_volume_rotation_technique(self):
        """Test spécifique de la technique de rotation 3D"""
        augmentations = ["rotation"]
        result = self.augmenter.augment_3d_volume(self.volume_3d_simple, augmentations, 1)
        
        # Vérifier que la rotation a été appliquée
        rotated_volume = result[0]
        self.assertFalse(np.array_equal(rotated_volume, self.volume_3d_simple))
        
        # Vérifier que les dimensions sont préservées
        self.assertEqual(rotated_volume.shape, self.volume_3d_simple.shape)
        
        print("✅ Technique de rotation 3D validée")
    
    def test_augment_3d_volume_flip_horizontal_technique(self):
        """Test spécifique de la technique de retournement horizontal 3D"""
        augmentations = ["flip_horizontal"]
        result = self.augmenter.augment_3d_volume(self.volume_3d_simple, augmentations, 1)
        
        # Vérifier que le retournement horizontal a été appliqué
        flipped_volume = result[0]
        self.assertFalse(np.array_equal(flipped_volume, self.volume_3d_simple))
        
        # Vérifier que les dimensions sont préservées
        self.assertEqual(flipped_volume.shape, self.volume_3d_simple.shape)
        
        print("✅ Technique de retournement horizontal 3D validée")
    
    def test_augment_3d_volume_flip_vertical_technique(self):
        """Test spécifique de la technique de retournement vertical 3D"""
        augmentations = ["flip_vertical"]
        result = self.augmenter.augment_3d_volume(self.volume_3d_simple, augmentations, 1)
        
        # Vérifier que le retournement vertical a été appliqué
        flipped_volume = result[0]
        self.assertFalse(np.array_equal(flipped_volume, self.volume_3d_simple))
        
        # Vérifier que les dimensions sont préservées
        self.assertEqual(flipped_volume.shape, self.volume_3d_simple.shape)
        
        print("✅ Technique de retournement vertical 3D validée")
    
    def test_augment_3d_volume_gaussian_noise_technique(self):
        """Test spécifique de la technique de bruit gaussien 3D"""
        augmentations = ["gaussian_noise"]
        result = self.augmenter.augment_3d_volume(self.volume_3d_simple, augmentations, 1)
        
        # Vérifier que le bruit gaussien a été appliqué
        noisy_volume = result[0]
        self.assertFalse(np.array_equal(noisy_volume, self.volume_3d_simple))
        
        # Vérifier que les dimensions sont préservées
        self.assertEqual(noisy_volume.shape, self.volume_3d_simple.shape)
        
        print("✅ Technique de bruit gaussien 3D validée")
    
    def test_augment_3d_volume_value_variation_technique(self):
        """Test spécifique de la technique de variation des valeurs 3D"""
        augmentations = ["value_variation"]
        result = self.augmenter.augment_3d_volume(self.volume_3d_simple, augmentations, 1)
        
        # Vérifier que la variation des valeurs a été appliquée
        varied_volume = result[0]
        self.assertFalse(np.array_equal(varied_volume, self.volume_3d_simple))
        
        # Vérifier que les dimensions sont préservées
        self.assertEqual(varied_volume.shape, self.volume_3d_simple.shape)
        
        print("✅ Technique de variation des valeurs 3D validée")
    
    def test_augment_3d_volume_multiple_techniques(self):
        """Test de augment_3d_volume avec plusieurs techniques combinées"""
        augmentations = ["rotation", "flip_horizontal", "gaussian_noise", "value_variation"]
        num_augmentations = 2
        
        result = self.augmenter.augment_3d_volume(
            self.volume_3d_pd, 
            augmentations, 
            num_augmentations
        )
        
        # Vérifications
        self.assertEqual(len(result), num_augmentations)
        for augmented_volume in result:
            self.assertEqual(augmented_volume.shape, self.volume_3d_pd.shape)
            self.assertFalse(np.array_equal(augmented_volume, self.volume_3d_pd))
        
        print(f"✅ Combinaison de techniques 3D validée: {len(augmentations)} techniques appliquées")
    
    def test_augment_3d_volume_reproducibility(self):
        """Test de reproductibilité avec la même graine"""
        # Créer deux augmenteurs avec la même graine
        augmenter1 = GeophysicalDataAugmenter(random_seed=42)
        augmenter2 = GeophysicalDataAugmenter(random_seed=42)
        
        # Appliquer une seule augmentation déterministe (pas de mélange aléatoire)
        augmentations = ["flip_horizontal"]  # Technique déterministe
        result1 = augmenter1.augment_3d_volume(self.volume_3d_simple, augmentations, 1)
        result2 = augmenter2.augment_3d_volume(self.volume_3d_simple, augmentations, 1)
        
        # Les résultats devraient être identiques avec la même graine pour une technique déterministe
        np.testing.assert_array_equal(result1[0], result2[0])
        
        print("✅ Reproductibilité 3D avec la même graine validée")
    
    def test_augment_3d_volume_error_handling(self):
        """Test de gestion des erreurs"""
        # Test avec un volume de mauvaise dimension
        invalid_volume = np.random.rand(6, 6, 6)  # 3D au lieu de 4D
        
        with self.assertRaises(ValueError):
            self.augmenter.augment_3d_volume(invalid_volume, ["flip_horizontal"], 1)
        
        # Test avec un type invalide
        with self.assertRaises(ValueError):
            self.augmenter.augment_3d_volume("invalid", ["flip_horizontal"], 1)
        
        print("✅ Gestion des erreurs 3D validée")
    
    def test_augment_3d_volume_with_cleaned_data(self):
        """Test de augment_3d_volume avec des données nettoyées"""
        # Nettoyer les données PD.csv
        cleaning_results = self.cleaner.clean_all_devices()
        
        if 'pole_dipole' in cleaning_results:
            clean_path, report = cleaning_results['pole_dipole']
            if clean_path and Path(clean_path).exists():
                # Charger les données nettoyées
                cleaned_df = pd.read_csv(clean_path)
                
                # Créer un volume à partir des données nettoyées
                volume_size = min(8, int(np.cbrt(len(cleaned_df))))
                cleaned_volume = np.zeros((volume_size, volume_size, volume_size, 4))
                
                # Remplir le volume
                for i in range(min(len(cleaned_df), volume_size * volume_size * volume_size)):
                    d = i // (volume_size * volume_size)
                    h = (i % (volume_size * volume_size)) // volume_size
                    w = i % volume_size
                    if d < volume_size and h < volume_size and w < volume_size:
                        cleaned_volume[d, h, w, 0] = cleaned_df.iloc[i]['resistivity'] if 'resistivity' in cleaned_df.columns else 0
                        cleaned_volume[d, h, w, 1] = cleaned_df.iloc[i]['chargeability'] if 'chargeability' in cleaned_df.columns else 0
                        cleaned_volume[d, h, w, 2] = cleaned_df.iloc[i]['x'] if 'x' in cleaned_df.columns else 0
                        cleaned_volume[d, h, w, 3] = cleaned_df.iloc[i]['y'] if 'y' in cleaned_df.columns else 0
                
                # Tester l'augmentation
                augmentations = ["rotation", "flip_horizontal"]
                result = self.augmenter.augment_3d_volume(cleaned_volume, augmentations, 1)
                
                # Vérifications
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].shape, cleaned_volume.shape)
                
                print(f"✅ augment_3d_volume avec données nettoyées validé: {cleaned_volume.shape}")
            else:
                print("⚠️ Données PD nettoyées non disponibles")
        else:
            print("⚠️ Nettoyage des données PD non effectué")
    
    def test_augment_3d_volume_performance(self):
        """Test de performance de augment_3d_volume"""
        import time
        
        # Mesurer le temps d'exécution
        start_time = time.time()
        
        augmentations = ["rotation", "flip_horizontal", "gaussian_noise", "value_variation"]
        result = self.augmenter.augment_3d_volume(
            self.volume_3d_s,  # Volume plus grand (16x16x16x4)
            augmentations, 
            3  # 3 augmentations
        )
        
        execution_time = time.time() - start_time
        
        # Vérifier que l'exécution est raisonnable
        self.assertLess(execution_time, 10.0, "L'augmentation 3D devrait être rapide (< 10 secondes)")
        
        print(f"✅ Performance 3D validée: 3 augmentations en {execution_time:.3f} secondes")
    
    def test_augment_3d_volume_history_tracking(self):
        """Test du suivi de l'historique des augmentations 3D"""
        # Réinitialiser l'historique
        self.augmenter.reset_history()
        self.assertEqual(len(self.augmenter.augmentation_history), 0)
        
        # Effectuer des augmentations
        augmentations = ["rotation", "flip_horizontal"]
        self.augmenter.augment_3d_volume(self.volume_3d_simple, augmentations, 3)
        
        # Vérifier l'historique
        self.assertEqual(len(self.augmenter.augmentation_history), 3)
        
        # Vérifier la structure de l'historique
        for entry in self.augmenter.augmentation_history:
            self.assertIn('volume_shape', entry)
            self.assertIn('augmentations_applied', entry)
            self.assertIn('augmentation_index', entry)
            self.assertEqual(entry['volume_shape'], self.volume_3d_simple.shape)
            self.assertEqual(entry['augmentations_applied'], augmentations)
        
        print("✅ Suivi de l'historique 3D validé")
    
    def test_augment_3d_volume_edge_cases(self):
        """Test des cas limites de augment_3d_volume"""
        # Test avec une seule augmentation
        result = self.augmenter.augment_3d_volume(
            self.volume_3d_simple, 
            ["flip_horizontal"], 
            1
        )
        self.assertEqual(len(result), 1)
        
        # Test avec zéro augmentation
        result = self.augmenter.augment_3d_volume(
            self.volume_3d_simple, 
            ["flip_horizontal"], 
            0
        )
        self.assertEqual(len(result), 0)
        
        # Test avec une liste d'augmentations vide
        result = self.augmenter.augment_3d_volume(
            self.volume_3d_simple, 
            [], 
            1
        )
        self.assertEqual(len(result), 1)
        # Le volume devrait être identique car aucune augmentation n'a été appliquée
        np.testing.assert_array_equal(result[0], self.volume_3d_simple)
        
        print("✅ Cas limites 3D validés")
    
    def test_augment_3d_volume_integration_with_real_data(self):
        """Test d'intégration avec les vraies données géophysiques 3D"""
        # Test avec PD.csv
        pd_result = self.augmenter.augment_3d_volume(
            self.volume_3d_pd, 
            ["rotation", "gaussian_noise"], 
            2
        )
        
        # Test avec S.csv
        s_result = self.augmenter.augment_3d_volume(
            self.volume_3d_s, 
            ["flip_horizontal", "value_variation"], 
            2
        )
        
        # Vérifications
        self.assertEqual(len(pd_result), 2)
        self.assertEqual(len(s_result), 2)
        
        # Vérifier que les augmentations ont modifié les données
        for pd_volume in pd_result:
            self.assertFalse(np.array_equal(pd_volume, self.volume_3d_pd))
        
        for s_volume in s_result:
            self.assertFalse(np.array_equal(s_volume, self.volume_3d_s))
        
        print("✅ Intégration 3D avec vraies données géophysiques validée")
    
    def test_augment_3d_volume_data_integrity(self):
        """Test de l'intégrité des données après augmentation 3D"""
        # Sauvegarder les valeurs originales
        original_min = np.min(self.volume_3d_pd)
        original_max = np.max(self.volume_3d_pd)
        original_mean = np.mean(self.volume_3d_pd)
        
        # Appliquer des augmentations
        augmentations = ["gaussian_noise", "value_variation"]
        result = self.augmenter.augment_3d_volume(self.volume_3d_pd, augmentations, 1)
        
        augmented_volume = result[0]
        
        # Vérifier que les données ont été modifiées mais restent dans des limites raisonnables
        self.assertFalse(np.array_equal(augmented_volume, self.volume_3d_pd))
        
        # Vérifier que les dimensions sont préservées
        self.assertEqual(augmented_volume.shape, self.volume_3d_pd.shape)
        
        # Vérifier que les valeurs restent dans des limites raisonnables
        augmented_min = np.min(augmented_volume)
        augmented_max = np.max(augmented_volume)
        
        # Les valeurs ne devraient pas être extrêmement différentes
        self.assertLess(abs(augmented_min - original_min), 1000)
        self.assertLess(abs(augmented_max - original_max), 1000)
        
        print("✅ Intégrité des données 3D après augmentation validée")
    
    def test_augment_3d_volume_different_sizes(self):
        """Test de augment_3d_volume avec différentes tailles de volume"""
        # Test avec un petit volume
        small_volume = np.random.rand(4, 4, 4, 4)
        result_small = self.augmenter.augment_3d_volume(small_volume, ["flip_horizontal"], 1)
        self.assertEqual(result_small[0].shape, small_volume.shape)
        
        # Test avec un volume moyen
        medium_volume = np.random.rand(12, 12, 12, 4)
        result_medium = self.augmenter.augment_3d_volume(medium_volume, ["rotation"], 1)
        self.assertEqual(result_medium[0].shape, medium_volume.shape)
        
        # Test avec un grand volume
        large_volume = np.random.rand(24, 24, 24, 4)
        result_large = self.augmenter.augment_3d_volume(large_volume, ["gaussian_noise"], 1)
        self.assertEqual(result_large[0].shape, large_volume.shape)
        
        print("✅ Tests avec différentes tailles de volume 3D validés")
    
    def test_augment_3d_volume_channel_preservation(self):
        """Test de préservation des canaux dans augment_3d_volume"""
        # Créer un volume avec des canaux distincts
        test_volume = np.zeros((6, 6, 6, 4))
        test_volume[:, :, :, 0] = 1.0  # Canal 0: résistivité
        test_volume[:, :, :, 1] = 2.0  # Canal 1: chargeabilité
        test_volume[:, :, :, 2] = 3.0  # Canal 2: coordonnée X
        test_volume[:, :, :, 3] = 4.0  # Canal 3: coordonnée Y
        
        # Appliquer des augmentations
        augmentations = ["flip_horizontal", "gaussian_noise"]
        result = self.augmenter.augment_3d_volume(test_volume, augmentations, 1)
        
        augmented_volume = result[0]
        
        # Vérifier que les dimensions sont préservées
        self.assertEqual(augmented_volume.shape, test_volume.shape)
        
        # Vérifier que les canaux sont toujours présents
        self.assertEqual(augmented_volume.shape[3], 4)
        
        print("✅ Préservation des canaux 3D validée")
    
    def test_augment_3d_volume_technique_combinations(self):
        """Test de différentes combinaisons de techniques 3D"""
        # Test avec techniques géométriques
        geometric_augmentations = ["rotation", "flip_horizontal", "flip_vertical"]
        result_geometric = self.augmenter.augment_3d_volume(
            self.volume_3d_simple, 
            geometric_augmentations, 
            1
        )
        self.assertEqual(result_geometric[0].shape, self.volume_3d_simple.shape)
        
        # Test avec techniques de bruit
        noise_augmentations = ["gaussian_noise", "value_variation"]
        result_noise = self.augmenter.augment_3d_volume(
            self.volume_3d_simple, 
            noise_augmentations, 
            1
        )
        self.assertEqual(result_noise[0].shape, self.volume_3d_simple.shape)
        
        # Test avec techniques mixtes
        mixed_augmentations = ["rotation", "gaussian_noise", "flip_vertical"]
        result_mixed = self.augmenter.augment_3d_volume(
            self.volume_3d_simple, 
            mixed_augmentations, 
            1
        )
        self.assertEqual(result_mixed[0].shape, self.volume_3d_simple.shape)
        
        print("✅ Combinaisons de techniques 3D validées")
    
    def test_augment_3d_volume_memory_efficiency(self):
        """Test de l'efficacité mémoire de augment_3d_volume"""
        import psutil
        import os
        
        # Obtenir l'utilisation mémoire avant
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Effectuer des augmentations
        augmentations = ["rotation", "flip_horizontal", "gaussian_noise"]
        result = self.augmenter.augment_3d_volume(
            self.volume_3d_s,  # Volume de taille moyenne
            augmentations, 
            5  # 5 augmentations
        )
        
        # Obtenir l'utilisation mémoire après
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Vérifier que l'utilisation mémoire est raisonnable
        memory_increase = memory_after - memory_before
        self.assertLess(memory_increase, 1000, "L'augmentation mémoire devrait être raisonnable (< 1GB)")
        
        print(f"✅ Efficacité mémoire 3D validée: augmentation de {memory_increase:.1f} MB")
    
    def test_augment_3d_volume_validation_parameters(self):
        """Test de validation des paramètres de augment_3d_volume"""
        # Test avec des techniques valides
        valid_augmentations = ["rotation", "flip_horizontal", "flip_vertical", "gaussian_noise", "value_variation"]
        is_valid = self.augmenter.validate_augmentation_parameters(valid_augmentations, "3d_volume")
        self.assertTrue(is_valid)
        
        # Test avec des techniques invalides
        invalid_augmentations = ["spatial_shift", "elastic_deformation"]  # Techniques 2D uniquement
        is_valid = self.augmenter.validate_augmentation_parameters(invalid_augmentations, "3d_volume")
        self.assertFalse(is_valid)
        
        print("✅ Validation des paramètres 3D validée")


if __name__ == "__main__":
    unittest.main(verbosity=2)
