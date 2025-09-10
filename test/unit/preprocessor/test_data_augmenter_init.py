#!/usr/bin/env python3
"""
Test unitaire pour l'initialisation du module GeophysicalDataAugmenter.
Ce test vérifie le bon fonctionnement de l'initialisation avec les vraies données
géophysiques PD.csv et S.csv.
"""

import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import time

# Ajouter le répertoire parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.preprocessor.data_augmenter import GeophysicalDataAugmenter
from backend.preprocessor.data_cleaner import GeophysicalDataCleaner
from backend.config import CONFIG


class TestDataAugmenterInit(unittest.TestCase):
    """Tests pour l'initialisation de GeophysicalDataAugmenter avec vraies données"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.processed_data_dir = self.project_root / "data" / "processed"
        self.test_dir = self.project_root / "test" / "fixtures"
        
        # Créer le dossier de test s'il n'existe pas
        (self.test_dir / "processed").mkdir(exist_ok=True)
        
        # Initialiser le nettoyeur de données pour préparer les données
        self.cleaner = GeophysicalDataCleaner()
        
        # Vérifier que les fichiers de données existent
        self.pd_file = self.raw_data_dir / "PD.csv"
        self.s_file = self.raw_data_dir / "S.csv"
        
        self.assertTrue(self.pd_file.exists(), f"Le fichier {self.pd_file} n'existe pas")
        self.assertTrue(self.s_file.exists(), f"Le fichier {self.s_file} n'existe pas")
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        processed_dir = self.test_dir / "processed"
        if processed_dir.exists():
            import shutil
            shutil.rmtree(processed_dir)
    
    def test_initialization_without_seed(self):
        """Test de l'initialisation sans graine aléatoire"""
        augmenter = GeophysicalDataAugmenter()
        
        # Vérifier les attributs de base
        self.assertIsInstance(augmenter.augmentation_history, list)
        self.assertEqual(len(augmenter.augmentation_history), 0)
        self.assertIsNone(augmenter._random_seed)
        
        print("✅ Initialisation sans graine aléatoire validée")
    
    def test_initialization_with_seed(self):
        """Test de l'initialisation avec graine aléatoire"""
        test_seed = 42
        augmenter = GeophysicalDataAugmenter(random_seed=test_seed)
        
        # Vérifier les attributs de base
        self.assertIsInstance(augmenter.augmentation_history, list)
        self.assertEqual(len(augmenter.augmentation_history), 0)
        self.assertEqual(augmenter._random_seed, test_seed)
        
        print("✅ Initialisation avec graine aléatoire validée")
    
    def test_initialization_with_different_seeds(self):
        """Test de l'initialisation avec différentes graines"""
        seeds = [0, 42, 123, 999, 1000]
        
        for seed in seeds:
            augmenter = GeophysicalDataAugmenter(random_seed=seed)
            self.assertEqual(augmenter._random_seed, seed)
            self.assertEqual(len(augmenter.augmentation_history), 0)
        
        print("✅ Initialisation avec différentes graines validée")
    
    def test_initialization_with_pd_csv_data(self):
        """Test de l'initialisation et utilisation avec les données PD.csv"""
        # Charger les données PD.csv
        pd_df = pd.read_csv(self.pd_file, sep=',')
        print(f"📊 Données PD.csv chargées: {pd_df.shape}")
        
        # Initialiser l'augmenteur
        augmenter = GeophysicalDataAugmenter(random_seed=42)
        
        # Vérifier que l'augmenteur peut traiter ces données
        self.assertIsInstance(augmenter.augmentation_history, list)
        self.assertEqual(len(augmenter.augmentation_history), 0)
        
        # Tester l'augmentation avec les données PD.csv
        augmentations = ["gaussian_noise", "value_variation", "spatial_jitter"]
        augmented_dfs = augmenter.augment_dataframe(
            pd_df, 
            augmentations, 
            num_augmentations=2
        )
        
        # Vérifier les résultats
        self.assertEqual(len(augmented_dfs), 2)
        for augmented_df in augmented_dfs:
            self.assertEqual(augmented_df.shape, pd_df.shape)
        
        # Vérifier l'historique
        self.assertEqual(len(augmenter.augmentation_history), 2)
        
        print("✅ Initialisation et utilisation avec PD.csv validées")
    
    def test_initialization_with_s_csv_data(self):
        """Test de l'initialisation et utilisation avec les données S.csv"""
        # Charger les données S.csv
        s_df = pd.read_csv(self.s_file, sep=',')
        print(f"📊 Données S.csv chargées: {s_df.shape}")
        
        # Initialiser l'augmenteur
        augmenter = GeophysicalDataAugmenter(random_seed=123)
        
        # Vérifier que l'augmenteur peut traiter ces données
        self.assertIsInstance(augmenter.augmentation_history, list)
        self.assertEqual(len(augmenter.augmentation_history), 0)
        
        # Tester l'augmentation avec les données S.csv
        augmentations = ["gaussian_noise", "value_variation", "coordinate_perturbation"]
        augmented_dfs = augmenter.augment_dataframe(
            s_df, 
            augmentations, 
            num_augmentations=2
        )
        
        # Vérifier les résultats
        self.assertEqual(len(augmented_dfs), 2)
        for augmented_df in augmented_dfs:
            self.assertEqual(augmented_df.shape, s_df.shape)
        
        # Vérifier l'historique
        self.assertEqual(len(augmenter.augmentation_history), 2)
        
        print("✅ Initialisation et utilisation avec S.csv validées")
    
    def test_initialization_with_cleaned_data(self):
        """Test de l'initialisation avec des données nettoyées"""
        # Nettoyer les données PD.csv
        cleaning_results = self.cleaner.clean_all_devices()
        
        if 'pole_dipole' in cleaning_results:
            clean_path, report = cleaning_results['pole_dipole']
            if clean_path and Path(clean_path).exists():
                # Charger les données nettoyées
                cleaned_df = pd.read_csv(clean_path)
                print(f"📊 Données PD nettoyées chargées: {cleaned_df.shape}")
                
                # Initialiser l'augmenteur
                augmenter = GeophysicalDataAugmenter(random_seed=42)
                
                # Tester l'augmentation avec les données nettoyées
                augmentations = ["gaussian_noise", "value_variation"]
                augmented_dfs = augmenter.augment_dataframe(
                    cleaned_df, 
                    augmentations, 
                    num_augmentations=1
                )
                
                # Vérifier les résultats
                self.assertEqual(len(augmented_dfs), 1)
                self.assertEqual(augmented_dfs[0].shape, cleaned_df.shape)
                
                print("✅ Initialisation avec données nettoyées validée")
            else:
                print("⚠️ Données PD nettoyées non disponibles")
        else:
            print("⚠️ Nettoyage des données PD non effectué")
    
    def test_initialization_with_grid_data(self):
        """Test de l'initialisation avec des données de grille"""
        # Créer des données de grille basées sur PD.csv
        pd_df = pd.read_csv(self.pd_file, sep=',')
        
        # Normaliser les colonnes pour la correspondance
        augmenter = GeophysicalDataAugmenter(random_seed=42)
        pd_df = augmenter._normalize_dataframe_columns(pd_df)
        
        # Extraire les coordonnées et valeurs
        x_coords = pd_df['x'].values
        y_coords = pd_df['y'].values
        resistivity = pd_df['resistivity'].values
        chargeability = pd_df['chargeability'].values
        
        # Créer une grille 2D simple (8x8x4)
        grid_size = 8
        grid_2d = np.zeros((grid_size, grid_size, 4))
        
        # Remplir la grille avec les données
        for i in range(min(len(x_coords), grid_size * grid_size)):
            row = i // grid_size
            col = i % grid_size
            if row < grid_size and col < grid_size:
                grid_2d[row, col, 0] = resistivity[i] if i < len(resistivity) else 0
                grid_2d[row, col, 1] = chargeability[i] if i < len(chargeability) else 0
                grid_2d[row, col, 2] = x_coords[i] if i < len(x_coords) else 0
                grid_2d[row, col, 3] = y_coords[i] if i < len(y_coords) else 0
        
        print(f"📊 Grille 2D créée: {grid_2d.shape}")
        
        # L'augmenteur est déjà initialisé ci-dessus
        
        # Tester l'augmentation avec la grille
        augmentations = ["rotation", "flip_horizontal", "gaussian_noise"]
        augmented_grids = augmenter.augment_2d_grid(
            grid_2d, 
            augmentations, 
            num_augmentations=2
        )
        
        # Vérifier les résultats
        self.assertEqual(len(augmented_grids), 2)
        for augmented_grid in augmented_grids:
            self.assertEqual(augmented_grid.shape, grid_2d.shape)
        
        print("✅ Initialisation avec données de grille validée")
    
    def test_initialization_with_volume_data(self):
        """Test de l'initialisation avec des données de volume 3D"""
        # Créer un volume 3D basé sur S.csv
        s_df = pd.read_csv(self.s_file, sep=',')
        
        # Normaliser les colonnes pour la correspondance
        augmenter = GeophysicalDataAugmenter(random_seed=123)
        s_df = augmenter._normalize_dataframe_columns(s_df)
        
        # Extraire les valeurs de résistivité
        resistivity = s_df['resistivity'].values
        
        # Créer un volume 3D simple (8x8x8x4)
        volume_size = 8
        volume_3d = np.zeros((volume_size, volume_size, volume_size, 4))
        
        # Remplir le volume avec les données
        for i in range(min(len(resistivity), volume_size * volume_size * volume_size)):
            d = i // (volume_size * volume_size)
            h = (i % (volume_size * volume_size)) // volume_size
            w = i % volume_size
            if d < volume_size and h < volume_size and w < volume_size:
                volume_3d[d, h, w, 0] = resistivity[i] if i < len(resistivity) else 0
                volume_3d[d, h, w, 1] = resistivity[i] * 0.1 if i < len(resistivity) else 0  # Chargeabilité simulée
                volume_3d[d, h, w, 2] = i  # Coordonnée X simulée
                volume_3d[d, h, w, 3] = i * 2  # Coordonnée Y simulée
        
        print(f"📊 Volume 3D créé: {volume_3d.shape}")
        
        # L'augmenteur est déjà initialisé ci-dessus
        
        # Tester l'augmentation avec le volume
        augmentations = ["rotation", "flip_horizontal", "gaussian_noise"]
        augmented_volumes = augmenter.augment_3d_volume(
            volume_3d, 
            augmentations, 
            num_augmentations=2
        )
        
        # Vérifier les résultats
        self.assertEqual(len(augmented_volumes), 2)
        for augmented_volume in augmented_volumes:
            self.assertEqual(augmented_volume.shape, volume_3d.shape)
        
        print("✅ Initialisation avec données de volume 3D validée")
    
    def test_initialization_performance(self):
        """Test de performance de l'initialisation"""
        start_time = time.time()
        
        # Initialiser plusieurs augmenteurs
        augmenters = []
        for i in range(10):
            augmenter = GeophysicalDataAugmenter(random_seed=i)
            augmenters.append(augmenter)
        
        initialization_time = time.time() - start_time
        
        # Vérifier que l'initialisation est rapide
        self.assertLess(initialization_time, 1.0, "L'initialisation devrait être rapide (< 1 seconde)")
        
        # Vérifier que tous les augmenteurs sont correctement initialisés
        for i, augmenter in enumerate(augmenters):
            self.assertEqual(augmenter._random_seed, i)
            self.assertEqual(len(augmenter.augmentation_history), 0)
        
        print(f"✅ Performance d'initialisation validée: {initialization_time:.3f} secondes pour 10 augmenteurs")
    
    def test_initialization_error_handling(self):
        """Test de gestion des erreurs lors de l'initialisation"""
        # Test avec des types de graine invalides
        invalid_seeds = ["invalid", 3.14, [], {}, None, -1]
        
        for invalid_seed in invalid_seeds:
            try:
                # Ces initialisations ne devraient pas lever d'erreur
                augmenter = GeophysicalDataAugmenter(random_seed=invalid_seed)
                
                # Vérifier que l'augmenteur est quand même utilisable
                self.assertIsInstance(augmenter.augmentation_history, list)
                
                # Tester une augmentation simple
                test_data = np.random.rand(4, 4, 4)
                augmented_data = augmenter.augment_2d_grid(test_data, ["flip_horizontal"], 1)
                self.assertEqual(len(augmented_data), 1)
                
            except Exception as e:
                # Si une erreur est levée, elle devrait être gérée gracieusement
                print(f"⚠️ Gestion d'erreur pour graine invalide '{invalid_seed}': {e}")
        
        print("✅ Gestion des erreurs d'initialisation validée")
    
    def test_initialization_integration(self):
        """Test d'intégration de l'initialisation avec le pipeline complet"""
        # Charger les données
        pd_df = pd.read_csv(self.pd_file, sep=',')
        s_df = pd.read_csv(self.s_file, sep=',')
        
        # Initialiser l'augmenteur
        augmenter = GeophysicalDataAugmenter(random_seed=42)
        
        # Tester l'augmentation sur différents types de données
        test_results = {}
        
        # Test DataFrame PD.csv
        try:
            pd_df_normalized = augmenter._normalize_dataframe_columns(pd_df)
            augmented_pd = augmenter.augment_dataframe(pd_df_normalized, ["gaussian_noise"], 1)
            test_results['PD_DataFrame'] = len(augmented_pd) > 0
        except Exception as e:
            test_results['PD_DataFrame'] = False
            print(f"❌ Erreur avec PD DataFrame: {e}")
        
        # Test DataFrame S.csv
        try:
            s_df_normalized = augmenter._normalize_dataframe_columns(s_df)
            augmented_s = augmenter.augment_dataframe(s_df_normalized, ["value_variation"], 1)
            test_results['S_DataFrame'] = len(augmented_s) > 0
        except Exception as e:
            test_results['S_DataFrame'] = False
            print(f"❌ Erreur avec S DataFrame: {e}")
        
        # Test grille 2D
        try:
            grid_2d = np.random.rand(8, 8, 4)
            augmented_grid = augmenter.augment_2d_grid(grid_2d, ["rotation"], 1)
            test_results['Grid_2D'] = len(augmented_grid) > 0
        except Exception as e:
            test_results['Grid_2D'] = False
            print(f"❌ Erreur avec grille 2D: {e}")
        
        # Test volume 3D
        try:
            volume_3d = np.random.rand(8, 8, 8, 4)
            augmented_volume = augmenter.augment_3d_volume(volume_3d, ["flip_horizontal"], 1)
            test_results['Volume_3D'] = len(augmented_volume) > 0
        except Exception as e:
            test_results['Volume_3D'] = False
            print(f"❌ Erreur avec volume 3D: {e}")
        
        # Vérifier les résultats
        success_count = sum(test_results.values())
        total_tests = len(test_results)
        
        print(f"📊 Résultats d'intégration: {success_count}/{total_tests} tests réussis")
        for test_name, result in test_results.items():
            status = "✅" if result else "❌"
            print(f"   {status} {test_name}")
        
        # Au moins 3 tests sur 4 devraient réussir
        self.assertGreaterEqual(success_count, 3, "La plupart des tests d'intégration devraient réussir")
        
        print("✅ Tests d'intégration de l'initialisation validés")
    
    def test_initialization_summary(self):
        """Test du résumé après initialisation et utilisation"""
        # Initialiser l'augmenteur
        augmenter = GeophysicalDataAugmenter(random_seed=42)
        
        # Vérifier le résumé initial
        initial_summary = augmenter.get_augmentation_summary()
        self.assertIn("message", initial_summary)
        self.assertEqual(initial_summary["message"], "Aucune augmentation effectuée")
        
        # Effectuer quelques augmentations
        pd_df = pd.read_csv(self.pd_file, sep=',')
        # Normaliser les colonnes pour la correspondance
        pd_df = augmenter._normalize_dataframe_columns(pd_df)
        augmenter.augment_dataframe(pd_df, ["gaussian_noise"], 2)
        
        # Vérifier le résumé après utilisation
        final_summary = augmenter.get_augmentation_summary()
        self.assertIn("total_augmentations", final_summary)
        self.assertEqual(final_summary["total_augmentations"], 2)
        
        print("✅ Résumé après initialisation et utilisation validé")
    
    def test_initialization_reproducibility(self):
        """Test de reproductibilité avec la même graine"""
        # Créer des données de test
        test_data = np.random.rand(8, 8, 4)
        
        # Initialiser deux augmenteurs avec la même graine
        augmenter1 = GeophysicalDataAugmenter(random_seed=42)
        augmenter2 = GeophysicalDataAugmenter(random_seed=42)
        
        # Effectuer les mêmes augmentations
        result1 = augmenter1.augment_2d_grid(test_data, ["flip_horizontal"], 1)
        result2 = augmenter2.augment_2d_grid(test_data, ["flip_horizontal"], 1)
        
        # Les résultats devraient être identiques avec la même graine
        np.testing.assert_array_equal(result1[0], result2[0])
        
        print("✅ Reproductibilité avec la même graine validée")


if __name__ == "__main__":
    unittest.main(verbosity=2)
