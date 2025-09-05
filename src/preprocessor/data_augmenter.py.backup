#!/usr/bin/env python3
"""
Module d'augmentation de données géophysiques pour améliorer l'entraînement des modèles CNN.
Fournit des techniques d'augmentation spécifiquement adaptées aux données de résistivité et chargeabilité.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import random
from scipy import ndimage
from scipy.stats import norm

from src.utils.logger import logger


class GeophysicalDataAugmenter:
    """
    Augmenteur de données géophysiques avec des techniques spécifiques au domaine.
    
    Techniques disponibles :
    - Augmentations géométriques (rotation, retournement, décalage)
    - Augmentations de bruit (gaussien, poivre et sel)
    - Variations de valeurs (résistivité, chargeabilité)
    - Perturbations spatiales
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialiser l'augmenteur de données.
        
        Args:
            random_seed: Graine aléatoire pour la reproductibilité
        """
        self._random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        self.augmentation_history = []
        logger.info("GeophysicalDataAugmenter initialisé")
    
    def augment_2d_grid(self, grid: np.ndarray, augmentations: List[str], 
                        num_augmentations: int = 1) -> List[np.ndarray]:
        """
        Augmenter une grille 2D avec les techniques spécifiées.
        
        Args:
            grid: Grille 2D de forme (height, width, channels)
            augmentations: Liste des techniques d'augmentation à appliquer
            num_augmentations: Nombre d'augmentations à générer
            
        Returns:
            Liste des grilles augmentées
        """
        if not isinstance(grid, np.ndarray) or grid.ndim != 3:
            raise ValueError("grid doit être un tableau numpy 3D (height, width, channels)")
        
        augmented_grids = []
        
        for i in range(num_augmentations):
            augmented_grid = grid.copy()
            
            # Appliquer les augmentations dans un ordre aléatoire
            # Mais préserver l'ordre pour la reproductibilité si une graine est définie
            if hasattr(self, '_random_seed') and self._random_seed is not None:
                # Utiliser un ordre déterministe basé sur l'index
                augmentation_order = augmentations.copy()
                random.Random(self._random_seed + i).shuffle(augmentation_order)
            else:
                augmentation_order = augmentations.copy()
                random.shuffle(augmentation_order)
            
            for aug_type in augmentation_order:
                if aug_type == "rotation":
                    augmented_grid = self._rotate_2d_grid(augmented_grid)
                elif aug_type == "flip_horizontal":
                    augmented_grid = self._flip_horizontal_2d_grid(augmented_grid)
                elif aug_type == "flip_vertical":
                    augmented_grid = self._flip_vertical_2d_grid(augmented_grid)
                elif aug_type == "spatial_shift":
                    augmented_grid = self._spatial_shift_2d_grid(augmented_grid)
                elif aug_type == "gaussian_noise":
                    augmented_grid = self._add_gaussian_noise_2d_grid(augmented_grid)
                elif aug_type == "salt_pepper_noise":
                    augmented_grid = self._add_salt_pepper_noise_2d_grid(augmented_grid)
                elif aug_type == "value_variation":
                    augmented_grid = self._vary_values_2d_grid(augmented_grid)
                elif aug_type == "elastic_deformation":
                    augmented_grid = self._elastic_deformation_2d_grid(augmented_grid)
            
            augmented_grids.append(augmented_grid)
            
            # Enregistrer l'historique
            self.augmentation_history.append({
                'grid_shape': grid.shape,
                'augmentations_applied': augmentations.copy(),
                'augmentation_index': i
            })
        
        logger.info(f"Généré {len(augmented_grids)} grilles augmentées")
        return augmented_grids
    
    def augment_3d_volume(self, volume: np.ndarray, augmentations: List[str], 
                          num_augmentations: int = 1) -> List[np.ndarray]:
        """
        Augmenter un volume 3D avec les techniques spécifiées.
        
        Args:
            volume: Volume 3D de forme (depth, height, width, channels)
            augmentations: Liste des techniques d'augmentation à appliquer
            num_augmentations: Nombre d'augmentations à générer
            
        Returns:
            Liste des volumes augmentés
        """
        if not isinstance(volume, np.ndarray) or volume.ndim != 4:
            raise ValueError("volume doit être un tableau numpy 4D (depth, height, width, channels)")
        
        augmented_volumes = []
        
        for i in range(num_augmentations):
            augmented_volume = volume.copy()
            
            # Appliquer les augmentations dans un ordre aléatoire
            # Mais préserver l'ordre pour la reproductibilité si une graine est définie
            if hasattr(self, '_random_seed') and self._random_seed is not None:
                # Utiliser un ordre déterministe basé sur l'index
                augmentation_order = augmentations.copy()
                random.Random(self._random_seed + i).shuffle(augmentation_order)
            else:
                augmentation_order = augmentations.copy()
                random.shuffle(augmentation_order)
            
            for aug_type in augmentation_order:
                if aug_type == "rotation":
                    augmented_volume = self._rotate_3d_volume(augmented_volume)
                elif aug_type == "flip_horizontal":
                    augmented_volume = self._flip_horizontal_3d_volume(augmented_volume)
                elif aug_type == "flip_vertical":
                    augmented_volume = self._flip_vertical_3d_volume(augmented_volume)
                elif aug_type == "gaussian_noise":
                    augmented_volume = self._add_gaussian_noise_3d_volume(augmented_volume)
                elif aug_type == "value_variation":
                    augmented_volume = self._vary_values_3d_volume(augmented_volume)
            
            augmented_volumes.append(augmented_volume)
            
            # Enregistrer l'historique
            self.augmentation_history.append({
                'volume_shape': volume.shape,
                'augmentations_applied': augmentations.copy(),
                'augmentation_index': i
            })
        
        logger.info(f"Généré {len(augmented_volumes)} volumes augmentés")
        return augmented_volumes
    
    def augment_dataframe(self, df: pd.DataFrame, augmentations: List[str], 
                         num_augmentations: int = 1) -> List[pd.DataFrame]:
        """
        Augmenter un DataFrame avec des techniques appropriées.
        
        Args:
            df: DataFrame contenant les données géophysiques
            augmentations: Liste des techniques d'augmentation à appliquer
            num_augmentations: Nombre d'augmentations à générer
            
        Returns:
            Liste des DataFrames augmentés
        """
        augmented_dfs = []
        
        for i in range(num_augmentations):
            augmented_df = df.copy()
            
            # Appliquer les augmentations dans un ordre aléatoire
            # Mais préserver l'ordre pour la reproductibilité si une graine est définie
            if hasattr(self, '_random_seed') and self._random_seed is not None:
                # Utiliser un ordre déterministe basé sur l'index
                augmentation_order = augmentations.copy()
                random.Random(self._random_seed + i).shuffle(augmentation_order)
            else:
                augmentation_order = augmentations.copy()
                random.shuffle(augmentation_order)
            
            for aug_type in augmentation_order:
                if aug_type == "gaussian_noise":
                    augmented_df = self._add_gaussian_noise_dataframe(augmented_df)
                elif aug_type == "value_variation":
                    augmented_df = self._vary_values_dataframe(augmented_df)
                elif aug_type == "spatial_jitter":
                    augmented_df = self._spatial_jitter_dataframe(augmented_df)
                elif aug_type == "coordinate_perturbation":
                    augmented_df = self._perturb_coordinates_dataframe(augmented_df)
            
            augmented_dfs.append(augmented_df)
            
            # Enregistrer l'historique
            self.augmentation_history.append({
                'dataframe_shape': df.shape,
                'augmentations_applied': augmentations.copy(),
                'augmentation_index': i
            })
        
        logger.info(f"Généré {len(augmented_dfs)} DataFrames augmentés")
        return augmented_dfs
    
    # ==================== MÉTHODES PRIVÉES 2D ====================
    
    def _rotate_2d_grid(self, grid: np.ndarray) -> np.ndarray:
        """Rotation aléatoire de 90°, 180° ou 270°."""
        angle = random.choice([90, 180, 270])
        return ndimage.rotate(grid, angle, reshape=False, order=1)
    
    def _flip_horizontal_2d_grid(self, grid: np.ndarray) -> np.ndarray:
        """Retournement horizontal de la grille."""
        return np.flip(grid, axis=1)
    
    def _flip_vertical_2d_grid(self, grid: np.ndarray) -> np.ndarray:
        """Retournement vertical de la grille."""
        return np.flip(grid, axis=0)
    
    def _spatial_shift_2d_grid(self, grid: np.ndarray, max_shift: int = 2) -> np.ndarray:
        """Décalage spatial aléatoire avec remplissage par zéros."""
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        
        shifted = np.roll(grid, shift_x, axis=1)
        shifted = np.roll(shifted, shift_y, axis=0)
        
        # Remplir les bords avec des zéros si nécessaire
        if shift_x > 0:
            shifted[:, :shift_x] = 0
        elif shift_x < 0:
            shifted[:, shift_x:] = 0
        
        if shift_y > 0:
            shifted[:shift_y, :] = 0
        elif shift_y < 0:
            shifted[shift_y:, :] = 0
        
        return shifted
    
    def _add_gaussian_noise_2d_grid(self, grid: np.ndarray, 
                                   noise_std: float = 0.01) -> np.ndarray:
        """Ajouter du bruit gaussien à la grille."""
        noise = np.random.normal(0, noise_std, grid.shape)
        # Appliquer le bruit seulement aux canaux de données (pas aux coordonnées)
        noisy_grid = grid.copy()
        noisy_grid[:, :, :2] += noise[:, :, :2]  # Résistivité et chargeabilité
        return noisy_grid
    
    def _add_salt_pepper_noise_2d_grid(self, grid: np.ndarray, 
                                       noise_prob: float = 0.01) -> np.ndarray:
        """Ajouter du bruit poivre et sel à la grille."""
        noisy_grid = grid.copy()
        
        # Générer des masques pour le bruit
        salt_mask = np.random.random(grid.shape) < noise_prob / 2
        pepper_mask = np.random.random(grid.shape) < noise_prob / 2
        
        # Appliquer le bruit seulement aux canaux de données
        noisy_grid[:, :, :2][salt_mask[:, :, :2]] = 1.0  # Valeur maximale
        noisy_grid[:, :, :2][pepper_mask[:, :, :2]] = 0.0  # Valeur minimale
        
        return noisy_grid
    
    def _vary_values_2d_grid(self, grid: np.ndarray, 
                             variation_factor: float = 0.1) -> np.ndarray:
        """Varier légèrement les valeurs de résistivité et chargeabilité."""
        varied_grid = grid.copy()
        
        # Appliquer des variations aléatoires aux canaux de données
        variation = np.random.uniform(1 - variation_factor, 1 + variation_factor, grid.shape[:2])
        variation = np.expand_dims(variation, axis=2)
        
        varied_grid[:, :, :2] *= variation
        
        return varied_grid
    
    def _elastic_deformation_2d_grid(self, grid: np.ndarray, 
                                    alpha: float = 1.0, sigma: float = 50.0) -> np.ndarray:
        """Déformation élastique de la grille."""
        shape = grid.shape[:2]
        
        # Générer des champs de déformation aléatoires
        dx = np.random.randn(*shape) * alpha
        dy = np.random.randn(*shape) * alpha
        
        # Lisser avec un filtre gaussien
        dx = ndimage.gaussian_filter(dx, sigma=sigma)
        dy = ndimage.gaussian_filter(dy, sigma=sigma)
        
        # Normaliser
        dx = dx * alpha / np.max(np.abs(dx))
        dy = dy * alpha / np.max(np.abs(dy))
        
        # Appliquer la déformation
        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        x_deformed = x + dx
        y_deformed = y + dy
        
        # Interpolation
        deformed_grid = np.zeros_like(grid)
        for c in range(grid.shape[2]):
            deformed_grid[:, :, c] = ndimage.map_coordinates(
                grid[:, :, c], [y_deformed, x_deformed], order=1
            )
        
        return deformed_grid
    
    # ==================== MÉTHODES PRIVÉES 3D ====================
    
    def _rotate_3d_volume(self, volume: np.ndarray) -> np.ndarray:
        """Rotation aléatoire du volume 3D."""
        angle = random.choice([90, 180, 270])
        axis = random.choice([0, 1, 2])  # Axe de rotation
        
        if axis == 0:
            return np.rot90(volume, k=angle//90, axes=(1, 2))
        elif axis == 1:
            return np.rot90(volume, k=angle//90, axes=(0, 2))
        else:
            return np.rot90(volume, k=angle//90, axes=(0, 1))
    
    def _flip_horizontal_3d_volume(self, volume: np.ndarray) -> np.ndarray:
        """Retournement horizontal du volume 3D."""
        return np.flip(volume, axis=2)
    
    def _flip_vertical_3d_volume(self, volume: np.ndarray) -> np.ndarray:
        """Retournement vertical du volume 3D."""
        return np.flip(volume, axis=1)
    
    def _add_gaussian_noise_3d_volume(self, volume: np.ndarray, 
                                     noise_std: float = 0.01) -> np.ndarray:
        """Ajouter du bruit gaussien au volume 3D."""
        noise = np.random.normal(0, noise_std, volume.shape)
        noisy_volume = volume.copy()
        noisy_volume[:, :, :, :2] += noise[:, :, :, :2]  # Résistivité et chargeabilité
        return noisy_volume
    
    def _vary_values_3d_volume(self, volume: np.ndarray, 
                               variation_factor: float = 0.1) -> np.ndarray:
        """Varier légèrement les valeurs du volume 3D."""
        varied_volume = volume.copy()
        
        # Appliquer des variations aléatoires aux canaux de données
        variation = np.random.uniform(1 - variation_factor, 1 + variation_factor, volume.shape[:3])
        variation = np.expand_dims(variation, axis=3)
        
        varied_volume[:, :, :, :2] *= variation
        
        return varied_volume
    
    # ==================== MÉTHODES PRIVÉES DATAFRAME ====================
    
    def _add_gaussian_noise_dataframe(self, df: pd.DataFrame, 
                                     noise_std: float = 0.01) -> pd.DataFrame:
        """Ajouter du bruit gaussien aux colonnes numériques."""
        noisy_df = df.copy()
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in ['resistivity', 'chargeability']:
                noise = np.random.normal(0, noise_std * df[col].std(), len(df))
                noisy_df[col] = df[col] + noise
        
        return noisy_df
    
    def _vary_values_dataframe(self, df: pd.DataFrame, 
                              variation_factor: float = 0.1) -> pd.DataFrame:
        """Varier légèrement les valeurs des colonnes numériques."""
        varied_df = df.copy()
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in ['resistivity', 'chargeability']:
                variation = np.random.uniform(1 - variation_factor, 1 + variation_factor, len(df))
                varied_df[col] = df[col] * variation
        
        return varied_df
    
    def _spatial_jitter_dataframe(self, df: pd.DataFrame, 
                                 jitter_std: float = 0.5) -> pd.DataFrame:
        """Ajouter du jitter spatial aux coordonnées."""
        jittered_df = df.copy()
        
        if 'x' in df.columns:
            jitter_x = np.random.normal(0, jitter_std, len(df))
            jittered_df['x'] = df['x'] + jitter_x
        
        if 'y' in df.columns:
            jitter_y = np.random.normal(0, jitter_std, len(df))
            jittered_df['y'] = df['y'] + jitter_y
        
        return jittered_df
    
    def _perturb_coordinates_dataframe(self, df: pd.DataFrame, 
                                     perturbation_factor: float = 0.01) -> pd.DataFrame:
        """Perturber légèrement les coordonnées."""
        perturbed_df = df.copy()
        
        if 'x' in df.columns:
            x_range = df['x'].max() - df['x'].min()
            perturbation = np.random.uniform(-perturbation_factor * x_range, 
                                          perturbation_factor * x_range, len(df))
            perturbed_df['x'] = df['x'] + perturbation
        
        if 'y' in df.columns:
            y_range = df['y'].max() - df['y'].min()
            perturbation = np.random.uniform(-perturbation_factor * y_range, 
                                          perturbation_factor * y_range, len(df))
            perturbed_df['y'] = df['y'] + perturbation
        
        return perturbed_df
    
    # ==================== MÉTHODES UTILITAIRES ====================
    
    def get_augmentation_summary(self) -> Dict:
        """Obtenir un résumé des augmentations effectuées."""
        if not self.augmentation_history:
            return {"message": "Aucune augmentation effectuée"}
        
        summary = {
            "total_augmentations": len(self.augmentation_history),
            "augmentation_types": {},
            "shape_distribution": {},
            "recent_augmentations": self.augmentation_history[-5:]  # 5 dernières
        }
        
        # Compter les types d'augmentation
        for aug in self.augmentation_history:
            for aug_type in aug.get('augmentations_applied', []):
                summary['augmentation_types'][aug_type] = summary['augmentation_types'].get(aug_type, 0) + 1
        
        # Compter les formes
        for aug in self.augmentation_history:
            shape_key = str(aug.get('grid_shape', aug.get('volume_shape', aug.get('dataframe_shape', 'unknown'))))
            summary['shape_distribution'][shape_key] = summary['shape_distribution'].get(shape_key, 0) + 1
        
        return summary
    
    def reset_history(self):
        """Réinitialiser l'historique des augmentations."""
        self.augmentation_history = []
        logger.info("Historique des augmentations réinitialisé")
    
    def get_recommended_augmentations(self, data_type: str = "2d_grid") -> List[str]:
        """
        Obtenir des recommandations d'augmentation selon le type de données.
        
        Args:
            data_type: Type de données ("2d_grid", "3d_volume", "dataframe")
            
        Returns:
            Liste des techniques d'augmentation recommandées
        """
        recommendations = {
            "2d_grid": [
                "rotation", "flip_horizontal", "flip_vertical", 
                "gaussian_noise", "spatial_shift", "value_variation"
            ],
            "3d_volume": [
                "rotation", "flip_horizontal", "flip_vertical",
                "gaussian_noise", "value_variation"
            ],
            "dataframe": [
                "gaussian_noise", "value_variation", 
                "spatial_jitter", "coordinate_perturbation"
            ]
        }
        
        return recommendations.get(data_type, [])
    
    def validate_augmentation_parameters(self, augmentations: List[str], 
                                       data_type: str = "2d_grid") -> bool:
        """
        Valider que les techniques d'augmentation sont appropriées pour le type de données.
        
        Args:
            augmentations: Liste des techniques d'augmentation
            data_type: Type de données
            
        Returns:
            True si les paramètres sont valides, False sinon
        """
        valid_augmentations = {
            "2d_grid": [
                "rotation", "flip_horizontal", "flip_vertical", 
                "spatial_shift", "gaussian_noise", "salt_pepper_noise",
                "value_variation", "elastic_deformation"
            ],
            "3d_volume": [
                "rotation", "flip_horizontal", "flip_vertical",
                "gaussian_noise", "value_variation"
            ],
            "dataframe": [
                "gaussian_noise", "value_variation", 
                "spatial_jitter", "coordinate_perturbation"
            ]
        }
        
        valid_for_type = valid_augmentations.get(data_type, [])
        invalid_augmentations = [aug for aug in augmentations if aug not in valid_for_type]
        
        if invalid_augmentations:
            logger.warning(f"Techniques d'augmentation invalides pour {data_type}: {invalid_augmentations}")
            return False
        
        return True
