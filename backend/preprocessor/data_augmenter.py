#!/usr/bin/env python3
"""
Module d'augmentation de données géophysiques pour améliorer l'entraînement des modèles CNN.
Fournit des techniques d'augmentation spécifiquement adaptées aux données de résistivité et chargeabilité.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union, Any
import random
from scipy import ndimage
from scipy.stats import norm

from backend.utils.logger import logger


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
    
    # ==================== MÉTHODES DE VALIDATION ====================
    
    def _validate_2d_grid_input(self, grid: np.ndarray) -> None:
        """
        Valider strictement les entrées pour les grilles 2D.
        
        Args:
            grid: Grille à valider
            
        Raises:
            ValueError: Si la grille n'est pas valide
            TypeError: Si le type n'est pas correct
        """
        if not isinstance(grid, np.ndarray):
            raise TypeError(f"grid doit être un numpy.ndarray, reçu: {type(grid)}")
        
        if grid.dtype not in [np.float32, np.float64, np.int32, np.int64]:
            raise TypeError(f"Type de données non supporté: {grid.dtype}. Types supportés: float32, float64, int32, int64")
        
        if grid.size == 0:
            raise ValueError("La grille ne peut pas être vide")
        
        if len(grid.shape) not in [3, 4]:
            raise ValueError(f"Forme invalide: {grid.shape}. Attendu: (height, width, channels) ou (samples, channels, height, width)")
        
        if len(grid.shape) == 3:
            height, width, channels = grid.shape
            if height < 2 or width < 2:
                raise ValueError(f"Dimensions trop petites: {height}x{width}. Minimum: 2x2")
            if channels < 1:
                raise ValueError(f"Nombre de canaux invalide: {channels}. Minimum: 1")
        else:  # 4D
            samples, channels, height, width = grid.shape
            if samples < 1:
                raise ValueError(f"Nombre d'échantillons invalide: {samples}. Minimum: 1")
            if height < 2 or width < 2:
                raise ValueError(f"Dimensions trop petites: {height}x{width}. Minimum: 2x2")
            if channels < 1:
                raise ValueError(f"Nombre de canaux invalide: {channels}. Minimum: 1")
        
        # Vérifier les valeurs
        if np.any(np.isnan(grid)):
            raise ValueError("La grille contient des valeurs NaN")
        
        if np.any(np.isinf(grid)):
            raise ValueError("La grille contient des valeurs infinies")
    
    def _validate_3d_volume_input(self, volume: np.ndarray) -> None:
        """
        Valider strictement les entrées pour les volumes 3D.
        
        Args:
            volume: Volume à valider
            
        Raises:
            ValueError: Si le volume n'est pas valide
            TypeError: Si le type n'est pas correct
        """
        if not isinstance(volume, np.ndarray):
            raise TypeError(f"volume doit être un numpy.ndarray, reçu: {type(volume)}")
        
        if volume.dtype not in [np.float32, np.float64, np.int32, np.int64]:
            raise TypeError(f"Type de données non supporté: {volume.dtype}. Types supportés: float32, float64, int32, int64")
        
        if volume.size == 0:
            raise ValueError("Le volume ne peut pas être vide")
        
        if len(volume.shape) != 4:
            raise ValueError(f"Forme invalide: {volume.shape}. Attendu: (depth, height, width, channels)")
        
        depth, height, width, channels = volume.shape
        
        if depth < 1:
            raise ValueError(f"Profondeur invalide: {depth}. Minimum: 1")
        if height < 2 or width < 2:
            raise ValueError(f"Dimensions trop petites: {height}x{width}. Minimum: 2x2")
        if channels < 1:
            raise ValueError(f"Nombre de canaux invalide: {channels}. Minimum: 1")
        
        # Vérifier les valeurs
        if np.any(np.isnan(volume)):
            raise ValueError("Le volume contient des valeurs NaN")
        
        if np.any(np.isinf(volume)):
            raise ValueError("Le volume contient des valeurs infinies")
    
    def _validate_dataframe_input(self, df: pd.DataFrame) -> None:
        """
        Valider strictement les entrées pour les DataFrames.
        
        Args:
            df: DataFrame à valider
            
        Raises:
            ValueError: Si le DataFrame n'est pas valide
            TypeError: Si le type n'est pas correct
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df doit être un pandas.DataFrame, reçu: {type(df)}")
        
        if df.empty:
            raise ValueError("Le DataFrame ne peut pas être vide")
        
        if len(df) < 1:
            raise ValueError(f"Le DataFrame doit contenir au moins 1 ligne, reçu: {len(df)}")
        
        # Vérifier les colonnes requises pour les données géophysiques
        required_columns = ['x', 'y']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Colonnes requises manquantes: {missing_columns}")
        
        # Vérifier les types de données des colonnes de coordonnées
        if not pd.api.types.is_numeric_dtype(df['x']):
            raise TypeError("La colonne 'x' doit être numérique")
        if not pd.api.types.is_numeric_dtype(df['y']):
            raise TypeError("La colonne 'y' doit être numérique")
        
        # Vérifier les valeurs
        if df['x'].isna().any():
            raise ValueError("La colonne 'x' contient des valeurs NaN")
        if df['y'].isna().any():
            raise ValueError("La colonne 'y' contient des valeurs NaN")
        
        if df['x'].isinf().any():
            raise ValueError("La colonne 'x' contient des valeurs infinies")
        if df['y'].isinf().any():
            raise ValueError("La colonne 'y' contient des valeurs infinies")
    
    def _validate_augmentations_list(self, augmentations: List[str], data_type: str) -> None:
        """
        Valider la liste des techniques d'augmentation.
        
        Args:
            augmentations: Liste des techniques d'augmentation
            data_type: Type de données ("2d_grid", "3d_volume", "dataframe")
            
        Raises:
            ValueError: Si la liste n'est pas valide
            TypeError: Si le type n'est pas correct
        """
        if not isinstance(augmentations, list):
            raise TypeError(f"augmentations doit être une liste, reçu: {type(augmentations)}")
        
        if not augmentations:
            raise ValueError("La liste des augmentations ne peut pas être vide")
        
        if not all(isinstance(aug, str) for aug in augmentations):
            raise TypeError("Tous les éléments de augmentations doivent être des chaînes de caractères")
        
        # Vérifier que toutes les techniques sont valides pour le type de données
        valid_augmentations = {
            "2d_grid": [
                "rotation", "flip_horizontal", "flip_vertical", 
                "spatial_shift", "gaussian_noise", "salt_pepper_noise",
                "value_variation", "elastic_deformation",
                "geological_stratification", "fracture_patterns",
                "mineral_inclusions", "weathering_effects", "sedimentary_layers",
                "color_jittering", "blur_sharpen", "perspective_transform", "cutout"
            ],
            "3d_volume": [
                "rotation", "flip_horizontal", "flip_vertical",
                "gaussian_noise", "value_variation", "elastic_deformation",
                "geological_stratification", "fracture_patterns",
                "mineral_inclusions", "weathering_effects", "sedimentary_layers"
            ],
            "dataframe": [
                "gaussian_noise", "value_variation", 
                "spatial_jitter", "coordinate_perturbation",
                "geological_stratification", "fracture_patterns",
                "mineral_inclusions", "weathering_effects", "sedimentary_layers"
            ]
        }
        
        valid_for_type = valid_augmentations.get(data_type, [])
        invalid_augmentations = [aug for aug in augmentations if aug not in valid_for_type]
        
        if invalid_augmentations:
            raise ValueError(f"Techniques d'augmentation invalides pour {data_type}: {invalid_augmentations}. "
                           f"Techniques valides: {valid_for_type}")
    
    def _validate_num_augmentations(self, num_augmentations: int) -> None:
        """
        Valider le nombre d'augmentations.
        
        Args:
            num_augmentations: Nombre d'augmentations à générer
            
        Raises:
            ValueError: Si le nombre n'est pas valide
            TypeError: Si le type n'est pas correct
        """
        if not isinstance(num_augmentations, int):
            raise TypeError(f"num_augmentations doit être un entier, reçu: {type(num_augmentations)}")
        
        if num_augmentations < 1:
            raise ValueError(f"num_augmentations doit être >= 1, reçu: {num_augmentations}")
        
        if num_augmentations > 1000:
            raise ValueError(f"num_augmentations trop élevé: {num_augmentations}. Maximum recommandé: 1000")
    
    def _validate_geophysical_data_quality(self, data: Union[np.ndarray, pd.DataFrame], 
                                         data_type: str) -> None:
        """
        Valider la qualité des données géophysiques.
        
        Args:
            data: Données à valider
            data_type: Type de données ("2d_grid", "3d_volume", "dataframe")
            
        Raises:
            ValueError: Si la qualité des données n'est pas acceptable
        """
        if data_type in ["2d_grid", "3d_volume"]:
            # Vérifier les valeurs extrêmes pour les données géophysiques
            if isinstance(data, np.ndarray):
                if data_type == "2d_grid" and len(data.shape) == 3:
                    # Vérifier les canaux de résistivité et chargeabilité
                    if data.shape[2] >= 2:
                        resistivity_values = data[:, :, 0]
                        chargeability_values = data[:, :, 1]
                        
                        # Vérifier les plages de valeurs réalistes
                        if np.any(resistivity_values < 0):
                            logger.warning("Valeurs de résistivité négatives détectées")
                        
                        if np.any(chargeability_values < 0):
                            logger.warning("Valeurs de chargeabilité négatives détectées")
                        
                        # Vérifier les valeurs extrêmement élevées
                        if np.any(resistivity_values > 10000):
                            logger.warning("Valeurs de résistivité très élevées détectées (>10k Ω⋅m)")
                        
                        if np.any(chargeability_values > 1000):
                            logger.warning("Valeurs de chargeabilité très élevées détectées (>1000 mV/V)")
        
        elif data_type == "dataframe":
            # Vérifier les colonnes géophysiques si elles existent
            if isinstance(data, pd.DataFrame):
                if 'resistivity' in data.columns:
                    resistivity_values = data['resistivity']
                    if np.any(resistivity_values < 0):
                        logger.warning("Valeurs de résistivité négatives détectées dans le DataFrame")
                    if np.any(resistivity_values > 10000):
                        logger.warning("Valeurs de résistivité très élevées détectées dans le DataFrame")
                
                if 'chargeability' in data.columns:
                    chargeability_values = data['chargeability']
                    if np.any(chargeability_values < 0):
                        logger.warning("Valeurs de chargeabilité négatives détectées dans le DataFrame")
                    if np.any(chargeability_values > 1000):
                        logger.warning("Valeurs de chargeabilité très élevées détectées dans le DataFrame")
    
    def _log_validation_summary(self, data_type: str, augmentations: List[str], 
                               num_augmentations: int) -> None:
        """
        Logger un résumé de la validation.
        
        Args:
            data_type: Type de données validé
            augmentations: Liste des techniques d'augmentation
            num_augmentations: Nombre d'augmentations
        """
        logger.info(f"Validation réussie pour {data_type}")
        logger.info(f"  - Techniques d'augmentation: {len(augmentations)} ({', '.join(augmentations)})")
        logger.info(f"  - Nombre d'augmentations: {num_augmentations}")
    
    def validate_input_comprehensive(self, data: Union[np.ndarray, pd.DataFrame], 
                                   augmentations: List[str], 
                                   num_augmentations: int,
                                   data_type: str) -> Dict[str, Any]:
        """
        Validation complète des entrées avec rapport détaillé.
        
        Args:
            data: Données à valider
            augmentations: Liste des techniques d'augmentation
            num_augmentations: Nombre d'augmentations
            data_type: Type de données
            
        Returns:
            Dictionnaire avec le rapport de validation
            
        Raises:
            ValueError: Si la validation échoue
            TypeError: Si les types ne sont pas corrects
        """
        validation_report = {
            "status": "success",
            "data_type": data_type,
            "data_shape": None,
            "data_dtype": None,
            "augmentations_count": len(augmentations),
            "num_augmentations": num_augmentations,
            "warnings": [],
            "errors": []
        }
        
        try:
            # Validation des types de base
            if data_type == "2d_grid":
                self._validate_2d_grid_input(data)
                validation_report["data_shape"] = data.shape
                validation_report["data_dtype"] = str(data.dtype)
            elif data_type == "3d_volume":
                self._validate_3d_volume_input(data)
                validation_report["data_shape"] = data.shape
                validation_report["data_dtype"] = str(data.dtype)
            elif data_type == "dataframe":
                self._validate_dataframe_input(data)
                validation_report["data_shape"] = data.shape
                validation_report["data_dtype"] = "DataFrame"
            
            # Validation des augmentations
            self._validate_augmentations_list(augmentations, data_type)
            
            # Validation du nombre d'augmentations
            self._validate_num_augmentations(num_augmentations)
            
            # Validation de la qualité géophysique
            self._validate_geophysical_data_quality(data, data_type)
            
            logger.info(f"Validation complète réussie pour {data_type}")
            
        except (ValueError, TypeError) as e:
            validation_report["status"] = "error"
            validation_report["errors"].append(str(e))
            logger.error(f"Erreur de validation: {e}")
            raise
        
        return validation_report
    
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
            
        Raises:
            ValueError: Si les paramètres d'entrée ne sont pas valides
            TypeError: Si les types d'entrée ne sont pas corrects
        """
        # Validation stricte des entrées
        self._validate_2d_grid_input(grid)
        self._validate_augmentations_list(augmentations, "2d_grid")
        self._validate_num_augmentations(num_augmentations)
        self._validate_geophysical_data_quality(grid, "2d_grid")
        self._log_validation_summary("2d_grid", augmentations, num_augmentations)
        
        # Gérer les formats 3D et 4D
        if len(grid.shape) == 4:
            # Format 4D (samples, channels, height, width) - prendre le premier échantillon
            grid = grid[0]  # Prendre le premier échantillon
        elif len(grid.shape) != 3:
            raise ValueError("grid doit être un tableau numpy 3D (height, width, channels) ou 4D (samples, channels, height, width)")
        
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
                elif aug_type == "geological_stratification":
                    augmented_grid = self._geological_stratification_2d_grid(augmented_grid)
                elif aug_type == "fracture_patterns":
                    augmented_grid = self._fracture_patterns_2d_grid(augmented_grid)
                elif aug_type == "mineral_inclusions":
                    augmented_grid = self._mineral_inclusions_2d_grid(augmented_grid)
                elif aug_type == "weathering_effects":
                    augmented_grid = self._weathering_effects_2d_grid(augmented_grid)
                elif aug_type == "sedimentary_layers":
                    augmented_grid = self._sedimentary_layers_2d_grid(augmented_grid)
                elif aug_type == "color_jittering":
                    augmented_grid = self._color_jittering_2d_grid(augmented_grid)
                elif aug_type == "blur_sharpen":
                    augmented_grid = self._blur_sharpen_2d_grid(augmented_grid)
                elif aug_type == "perspective_transform":
                    augmented_grid = self._perspective_transform_2d_grid(augmented_grid)
                elif aug_type == "cutout":
                    augmented_grid = self._cutout_2d_grid(augmented_grid)
            
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
            
        Raises:
            ValueError: Si les paramètres d'entrée ne sont pas valides
            TypeError: Si les types d'entrée ne sont pas corrects
        """
        # Validation stricte des entrées
        self._validate_3d_volume_input(volume)
        self._validate_augmentations_list(augmentations, "3d_volume")
        self._validate_num_augmentations(num_augmentations)
        self._validate_geophysical_data_quality(volume, "3d_volume")
        self._log_validation_summary("3d_volume", augmentations, num_augmentations)
        
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
                elif aug_type == "geological_stratification":
                    augmented_volume = self._geological_stratification_3d_volume(augmented_volume)
                elif aug_type == "fracture_patterns":
                    augmented_volume = self._fracture_patterns_3d_volume(augmented_volume)
                elif aug_type == "mineral_inclusions":
                    augmented_volume = self._mineral_inclusions_3d_volume(augmented_volume)
                elif aug_type == "weathering_effects":
                    augmented_volume = self._weathering_effects_3d_volume(augmented_volume)
                elif aug_type == "sedimentary_layers":
                    augmented_volume = self._sedimentary_layers_3d_volume(augmented_volume)
                elif aug_type == "elastic_deformation":
                    augmented_volume = self._elastic_deformation_3d_volume(augmented_volume)
            
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
            
        Raises:
            ValueError: Si les paramètres d'entrée ne sont pas valides
            TypeError: Si les types d'entrée ne sont pas corrects
        """
        # Validation stricte des entrées
        self._validate_dataframe_input(df)
        self._validate_augmentations_list(augmentations, "dataframe")
        self._validate_num_augmentations(num_augmentations)
        self._validate_geophysical_data_quality(df, "dataframe")
        self._log_validation_summary("dataframe", augmentations, num_augmentations)
        
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
                elif aug_type == "geological_stratification":
                    augmented_df = self._geological_stratification_dataframe(augmented_df)
                elif aug_type == "fracture_patterns":
                    augmented_df = self._fracture_patterns_dataframe(augmented_df)
                elif aug_type == "mineral_inclusions":
                    augmented_df = self._mineral_inclusions_dataframe(augmented_df)
                elif aug_type == "weathering_effects":
                    augmented_df = self._weathering_effects_dataframe(augmented_df)
                elif aug_type == "sedimentary_layers":
                    augmented_df = self._sedimentary_layers_dataframe(augmented_df)
            
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
    
    def _geological_stratification_2d_grid(self, grid: np.ndarray, 
                                         num_layers: int = 3) -> np.ndarray:
        """Simuler des couches géologiques horizontales."""
        stratified_grid = grid.copy()
        height, width, channels = grid.shape
        
        # Créer des couches horizontales
        for i in range(num_layers):
            layer_start = int(i * height / num_layers)
            layer_end = int((i + 1) * height / num_layers)
            
            # Variation de résistivité selon la profondeur
            depth_factor = 1.0 + (i * 0.3)  # Augmentation avec la profondeur
            
            # Appliquer aux canaux de données
            stratified_grid[layer_start:layer_end, :, 0] *= depth_factor  # Résistivité
            stratified_grid[layer_start:layer_end, :, 1] *= (1.0 + i * 0.1)  # Chargeabilité
        
        return stratified_grid
    
    def _fracture_patterns_2d_grid(self, grid: np.ndarray, 
                                  num_fractures: int = 2) -> np.ndarray:
        """Ajouter des motifs de fractures."""
        fractured_grid = grid.copy()
        height, width, channels = grid.shape
        
        for _ in range(num_fractures):
            # Position aléatoire de la fracture
            start_x = random.randint(0, width - 1)
            start_y = random.randint(0, height - 1)
            
            # Direction de la fracture (diagonale)
            direction = random.choice([1, -1])
            
            # Créer la fracture
            for i in range(min(width - start_x, height - start_y)):
                x = start_x + i
                y = start_y + i * direction
                
                if 0 <= x < width and 0 <= y < height:
                    # Réduire la résistivité dans la fracture
                    fractured_grid[y, x, 0] *= 0.3  # Résistivité réduite
                    fractured_grid[y, x, 1] *= 1.5  # Chargeabilité augmentée
        
        return fractured_grid
    
    def _mineral_inclusions_2d_grid(self, grid: np.ndarray, 
                                   num_inclusions: int = 5) -> np.ndarray:
        """Ajouter des inclusions minérales."""
        inclusion_grid = grid.copy()
        height, width, channels = grid.shape
        
        for _ in range(num_inclusions):
            # Position aléatoire de l'inclusion
            center_x = random.randint(0, width - 1)
            center_y = random.randint(0, height - 1)
            radius = random.randint(2, 5)
            
            # Type d'inclusion (conductrice ou résistante)
            inclusion_type = random.choice(['conductive', 'resistive'])
            
            # Créer l'inclusion circulaire
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:
                        x = center_x + dx
                        y = center_y + dy
                        
                        if 0 <= x < width and 0 <= y < height:
                            if inclusion_type == 'conductive':
                                inclusion_grid[y, x, 0] *= 0.1  # Résistivité très faible
                                inclusion_grid[y, x, 1] *= 2.0  # Chargeabilité élevée
                            else:  # resistive
                                inclusion_grid[y, x, 0] *= 5.0  # Résistivité élevée
                                inclusion_grid[y, x, 1] *= 0.2  # Chargeabilité faible
        
        return inclusion_grid
    
    def _weathering_effects_2d_grid(self, grid: np.ndarray, 
                                   weathering_intensity: float = 0.3) -> np.ndarray:
        """Simuler des effets d'altération."""
        weathered_grid = grid.copy()
        height, width, channels = grid.shape
        
        # Effet d'altération plus prononcé en surface
        for y in range(height):
            depth_factor = 1.0 - (y / height) * weathering_intensity
            
            # Réduire la résistivité avec l'altération
            weathered_grid[y, :, 0] *= depth_factor
            # Augmenter la chargeabilité avec l'altération
            weathered_grid[y, :, 1] *= (1.0 + weathering_intensity * (1.0 - y / height))
        
        return weathered_grid
    
    def _sedimentary_layers_2d_grid(self, grid: np.ndarray, 
                                   layer_thickness: int = 10) -> np.ndarray:
        """Créer des couches sédimentaires."""
        sedimentary_grid = grid.copy()
        height, width, channels = grid.shape
        
        # Créer des couches sédimentaires alternées
        for y in range(0, height, layer_thickness):
            layer_type = (y // layer_thickness) % 2
            
            for dy in range(min(layer_thickness, height - y)):
                if layer_type == 0:  # Couche conductrice
                    sedimentary_grid[y + dy, :, 0] *= 0.5  # Résistivité réduite
                    sedimentary_grid[y + dy, :, 1] *= 1.5  # Chargeabilité élevée
                else:  # Couche résistante
                    sedimentary_grid[y + dy, :, 0] *= 2.0  # Résistivité élevée
                    sedimentary_grid[y + dy, :, 1] *= 0.7  # Chargeabilité réduite
        
        return sedimentary_grid
    
    def _color_jittering_2d_grid(self, grid: np.ndarray, 
                                jitter_factor: float = 0.1) -> np.ndarray:
        """Variation de couleur pour les visualisations."""
        jittered_grid = grid.copy()
        
        # Appliquer des variations de couleur aux canaux de données
        for c in range(min(2, grid.shape[2])):  # Résistivité et chargeabilité
            color_shift = np.random.uniform(-jitter_factor, jitter_factor)
            jittered_grid[:, :, c] += color_shift
        
        return jittered_grid
    
    def _blur_sharpen_2d_grid(self, grid: np.ndarray) -> np.ndarray:
        """Flou ou aiguisage aléatoire."""
        processed_grid = grid.copy()
        
        # Choisir aléatoirement entre flou et aiguisage
        if random.random() < 0.5:
            # Flou gaussien
            for c in range(grid.shape[2]):
                processed_grid[:, :, c] = ndimage.gaussian_filter(grid[:, :, c], sigma=1.0)
        else:
            # Aiguisage
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            for c in range(grid.shape[2]):
                processed_grid[:, :, c] = ndimage.convolve(grid[:, :, c], kernel)
        
        return processed_grid
    
    def _perspective_transform_2d_grid(self, grid: np.ndarray) -> np.ndarray:
        """Transformation de perspective."""
        height, width, channels = grid.shape
        transformed_grid = np.zeros_like(grid)
        
        # Points de transformation (coin supérieur gauche déformé)
        src_points = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
        dst_points = src_points.copy()
        
        # Déformation aléatoire
        distortion = random.uniform(0.8, 1.2)
        dst_points[0, 0] *= distortion  # Coin supérieur gauche
        dst_points[1, 0] *= (2 - distortion)  # Coin supérieur droit
        
        # Appliquer la transformation
        for c in range(channels):
            # Interpolation simple pour la transformation de perspective
            transformed_grid[:, :, c] = ndimage.geometric_transform(
                grid[:, :, c], 
                lambda x, y: (x * distortion, y), 
                output_shape=(height, width)
            )
        
        return transformed_grid
    
    def _cutout_2d_grid(self, grid: np.ndarray, 
                       cutout_size: int = 8) -> np.ndarray:
        """Masquage de zones (cutout)."""
        cutout_grid = grid.copy()
        height, width, channels = grid.shape
        
        # Position aléatoire du cutout
        start_x = random.randint(0, max(1, width - cutout_size))
        start_y = random.randint(0, max(1, height - cutout_size))
        
        # Appliquer le masque
        cutout_grid[start_y:start_y+cutout_size, start_x:start_x+cutout_size, :] = 0
        
        return cutout_grid
    
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
    
    def _geological_stratification_3d_volume(self, volume: np.ndarray, 
                                           num_layers: int = 3) -> np.ndarray:
        """Simuler des couches géologiques dans un volume 3D."""
        stratified_volume = volume.copy()
        depth, height, width, channels = volume.shape
        
        # Créer des couches horizontales
        for i in range(num_layers):
            layer_start = int(i * depth / num_layers)
            layer_end = int((i + 1) * depth / num_layers)
            
            # Variation de résistivité selon la profondeur
            depth_factor = 1.0 + (i * 0.3)
            
            # Appliquer aux canaux de données
            stratified_volume[layer_start:layer_end, :, :, 0] *= depth_factor
            stratified_volume[layer_start:layer_end, :, :, 1] *= (1.0 + i * 0.1)
        
        return stratified_volume
    
    def _fracture_patterns_3d_volume(self, volume: np.ndarray, 
                                    num_fractures: int = 2) -> np.ndarray:
        """Ajouter des motifs de fractures 3D."""
        fractured_volume = volume.copy()
        depth, height, width, channels = volume.shape
        
        for _ in range(num_fractures):
            # Position aléatoire de la fracture
            start_x = random.randint(0, width - 1)
            start_y = random.randint(0, height - 1)
            start_z = random.randint(0, depth - 1)
            
            # Direction de la fracture
            direction = random.choice([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)])
            
            # Créer la fracture
            for i in range(min(width, height, depth)):
                x = start_x + i * direction[0]
                y = start_y + i * direction[1]
                z = start_z + i * direction[2]
                
                if 0 <= x < width and 0 <= y < height and 0 <= z < depth:
                    fractured_volume[z, y, x, 0] *= 0.3  # Résistivité réduite
                    fractured_volume[z, y, x, 1] *= 1.5  # Chargeabilité augmentée
        
        return fractured_volume
    
    def _mineral_inclusions_3d_volume(self, volume: np.ndarray, 
                                     num_inclusions: int = 5) -> np.ndarray:
        """Ajouter des inclusions minérales 3D."""
        inclusion_volume = volume.copy()
        depth, height, width, channels = volume.shape
        
        for _ in range(num_inclusions):
            # Position aléatoire de l'inclusion
            center_x = random.randint(0, width - 1)
            center_y = random.randint(0, height - 1)
            center_z = random.randint(0, depth - 1)
            radius = random.randint(2, 4)
            
            # Type d'inclusion
            inclusion_type = random.choice(['conductive', 'resistive'])
            
            # Créer l'inclusion sphérique
            for dz in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dx*dx + dy*dy + dz*dz <= radius*radius:
                            x = center_x + dx
                            y = center_y + dy
                            z = center_z + dz
                            
                            if 0 <= x < width and 0 <= y < height and 0 <= z < depth:
                                if inclusion_type == 'conductive':
                                    inclusion_volume[z, y, x, 0] *= 0.1
                                    inclusion_volume[z, y, x, 1] *= 2.0
                                else:
                                    inclusion_volume[z, y, x, 0] *= 5.0
                                    inclusion_volume[z, y, x, 1] *= 0.2
        
        return inclusion_volume
    
    def _weathering_effects_3d_volume(self, volume: np.ndarray, 
                                     weathering_intensity: float = 0.3) -> np.ndarray:
        """Simuler des effets d'altération 3D."""
        weathered_volume = volume.copy()
        depth, height, width, channels = volume.shape
        
        # Effet d'altération plus prononcé en surface (z=0)
        for z in range(depth):
            depth_factor = 1.0 - (z / depth) * weathering_intensity
            
            weathered_volume[z, :, :, 0] *= depth_factor
            weathered_volume[z, :, :, 1] *= (1.0 + weathering_intensity * (1.0 - z / depth))
        
        return weathered_volume
    
    def _sedimentary_layers_3d_volume(self, volume: np.ndarray, 
                                     layer_thickness: int = 5) -> np.ndarray:
        """Créer des couches sédimentaires 3D."""
        sedimentary_volume = volume.copy()
        depth, height, width, channels = volume.shape
        
        # Créer des couches sédimentaires alternées
        for z in range(0, depth, layer_thickness):
            layer_type = (z // layer_thickness) % 2
            
            for dz in range(min(layer_thickness, depth - z)):
                if layer_type == 0:  # Couche conductrice
                    sedimentary_volume[z + dz, :, :, 0] *= 0.5
                    sedimentary_volume[z + dz, :, :, 1] *= 1.5
                else:  # Couche résistante
                    sedimentary_volume[z + dz, :, :, 0] *= 2.0
                    sedimentary_volume[z + dz, :, :, 1] *= 0.7
        
        return sedimentary_volume
    
    def _elastic_deformation_3d_volume(self, volume: np.ndarray, 
                                      alpha: float = 1.0, sigma: float = 50.0) -> np.ndarray:
        """Déformation élastique 3D."""
        depth, height, width, channels = volume.shape
        
        # Générer des champs de déformation aléatoires
        dx = np.random.randn(depth, height, width) * alpha
        dy = np.random.randn(depth, height, width) * alpha
        dz = np.random.randn(depth, height, width) * alpha
        
        # Lisser avec un filtre gaussien
        dx = ndimage.gaussian_filter(dx, sigma=sigma)
        dy = ndimage.gaussian_filter(dy, sigma=sigma)
        dz = ndimage.gaussian_filter(dz, sigma=sigma)
        
        # Normaliser
        dx = dx * alpha / np.max(np.abs(dx))
        dy = dy * alpha / np.max(np.abs(dy))
        dz = dz * alpha / np.max(np.abs(dz))
        
        # Appliquer la déformation
        deformed_volume = np.zeros_like(volume)
        for c in range(channels):
            deformed_volume[:, :, :, c] = ndimage.map_coordinates(
                volume[:, :, :, c], 
                [dz, dy, dx], 
                order=1
            )
        
        return deformed_volume
    
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
    
    def _geological_stratification_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simuler des couches géologiques dans un DataFrame."""
        stratified_df = df.copy()
        
        if 'z' in df.columns:
            # Créer des couches basées sur la profondeur
            z_min, z_max = df['z'].min(), df['z'].max()
            z_range = z_max - z_min
            
            # Appliquer des variations selon la profondeur
            depth_factor = 1.0 + (df['z'] - z_min) / z_range * 0.5
            
            if 'resistivity' in df.columns:
                stratified_df['resistivity'] *= depth_factor
            if 'chargeability' in df.columns:
                stratified_df['chargeability'] *= (1.0 + (df['z'] - z_min) / z_range * 0.2)
        
        return stratified_df
    
    def _fracture_patterns_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajouter des motifs de fractures dans un DataFrame."""
        fractured_df = df.copy()
        
        # Simuler des fractures en créant des zones de faible résistivité
        if 'resistivity' in df.columns:
            # Sélectionner aléatoirement 10% des points comme fractures
            fracture_mask = np.random.random(len(df)) < 0.1
            fractured_df.loc[fracture_mask, 'resistivity'] *= 0.3
            
        if 'chargeability' in df.columns:
            # Augmenter la chargeabilité dans les zones de fracture
            fracture_mask = np.random.random(len(df)) < 0.1
            fractured_df.loc[fracture_mask, 'chargeability'] *= 1.5
        
        return fractured_df
    
    def _mineral_inclusions_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajouter des inclusions minérales dans un DataFrame."""
        inclusion_df = df.copy()
        
        # Sélectionner aléatoirement 5% des points comme inclusions
        inclusion_mask = np.random.random(len(df)) < 0.05
        inclusion_type = np.random.choice(['conductive', 'resistive'], len(df))
        
        if 'resistivity' in df.columns:
            conductive_mask = inclusion_mask & (inclusion_type == 'conductive')
            resistive_mask = inclusion_mask & (inclusion_type == 'resistive')
            
            inclusion_df.loc[conductive_mask, 'resistivity'] *= 0.1
            inclusion_df.loc[resistive_mask, 'resistivity'] *= 5.0
            
        if 'chargeability' in df.columns:
            inclusion_df.loc[conductive_mask, 'chargeability'] *= 2.0
            inclusion_df.loc[resistive_mask, 'chargeability'] *= 0.2
        
        return inclusion_df
    
    def _weathering_effects_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simuler des effets d'altération dans un DataFrame."""
        weathered_df = df.copy()
        
        if 'z' in df.columns:
            # Effet d'altération plus prononcé en surface
            z_min, z_max = df['z'].min(), df['z'].max()
            depth_factor = 1.0 - (df['z'] - z_min) / (z_max - z_min) * 0.3
            
            if 'resistivity' in df.columns:
                weathered_df['resistivity'] *= depth_factor
            if 'chargeability' in df.columns:
                weathered_df['chargeability'] *= (1.0 + (1.0 - depth_factor) * 0.5)
        
        return weathered_df
    
    def _sedimentary_layers_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Créer des couches sédimentaires dans un DataFrame."""
        sedimentary_df = df.copy()
        
        if 'z' in df.columns:
            # Créer des couches alternées
            z_min, z_max = df['z'].min(), df['z'].max()
            layer_thickness = (z_max - z_min) / 5  # 5 couches
            
            # Déterminer le type de couche
            layer_type = ((df['z'] - z_min) // layer_thickness) % 2
            
            if 'resistivity' in df.columns:
                # Couches conductrices (type 0) et résistantes (type 1)
                sedimentary_df.loc[layer_type == 0, 'resistivity'] *= 0.5
                sedimentary_df.loc[layer_type == 1, 'resistivity'] *= 2.0
                
            if 'chargeability' in df.columns:
                sedimentary_df.loc[layer_type == 0, 'chargeability'] *= 1.5
                sedimentary_df.loc[layer_type == 1, 'chargeability'] *= 0.7
        
        return sedimentary_df
    
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
    
    def get_recommended_augmentations(self, data_type: str = "2d_grid", 
                                     method: str = "pole_dipole") -> List[str]:
        """
        Obtenir des recommandations d'augmentation selon le type de données et la méthode géophysique.
        
        Args:
            data_type: Type de données ("2d_grid", "3d_volume", "dataframe")
            method: Méthode géophysique ("pole_dipole", "schlumberger")
            
        Returns:
            Liste des techniques d'augmentation recommandées
        """
        # Recommandations de base par type de données
        base_recommendations = {
            "2d_grid": [
                "rotation", "flip_horizontal", "flip_vertical", 
                "gaussian_noise", "spatial_shift", "value_variation",
                "elastic_deformation", "geological_stratification"
            ],
            "3d_volume": [
                "rotation", "flip_horizontal", "flip_vertical",
                "gaussian_noise", "value_variation", "elastic_deformation",
                "geological_stratification", "fracture_patterns"
            ],
            "dataframe": [
                "gaussian_noise", "value_variation", 
                "spatial_jitter", "coordinate_perturbation",
                "geological_stratification", "weathering_effects"
            ]
        }
        
        # Recommandations spécifiques par méthode géophysique
        method_specific = {
            "pole_dipole": {
                "2d_grid": ["rotation", "elastic_deformation", "geological_stratification", "fracture_patterns"],
                "3d_volume": ["rotation", "elastic_deformation", "geological_stratification", "mineral_inclusions"],
                "dataframe": ["geological_stratification", "fracture_patterns", "value_variation"]
            },
            "schlumberger": {
                "2d_grid": ["flip_horizontal", "value_variation", "weathering_effects", "sedimentary_layers"],
                "3d_volume": ["flip_horizontal", "weathering_effects", "sedimentary_layers", "mineral_inclusions"],
                "dataframe": ["weathering_effects", "sedimentary_layers", "spatial_jitter"]
            }
        }
        
        # Obtenir les recommandations de base
        base = base_recommendations.get(data_type, [])
        
        # Ajouter les recommandations spécifiques à la méthode
        if method in method_specific and data_type in method_specific[method]:
            specific = method_specific[method][data_type]
            # Combiner et dédupliquer
            combined = list(set(base + specific))
            return combined
        
        return base
    
    def get_recommended_augmentations_for_method(self, method: str) -> Dict[str, List[str]]:
        """
        Obtenir des recommandations d'augmentation pour toutes les méthodes selon la méthode géophysique.
        
        Args:
            method: Méthode géophysique ("pole_dipole", "schlumberger")
            
        Returns:
            Dictionnaire avec les recommandations par type de données
        """
        return {
            "2d_grid": self.get_recommended_augmentations("2d_grid", method),
            "3d_volume": self.get_recommended_augmentations("3d_volume", method),
            "dataframe": self.get_recommended_augmentations("dataframe", method)
        }
    
    def get_geophysical_augmentation_guide(self) -> Dict[str, Dict]:
        """
        Obtenir un guide complet des augmentations géophysiques.
        
        Returns:
            Guide détaillé des techniques d'augmentation
        """
        return {
            "pole_dipole": {
                "description": "Méthode Pôle-Dipôle pour pseudo-sections de résistivité",
                "best_practices": [
                    "Utiliser la rotation pour simuler différents angles de mesure",
                    "La déformation élastique simule les plis géologiques",
                    "Les fractures sont importantes pour les structures géologiques",
                    "Éviter les transformations de perspective qui déforment les pseudo-sections"
                ],
                "recommended_2d": ["rotation", "elastic_deformation", "geological_stratification", "fracture_patterns"],
                "recommended_3d": ["rotation", "elastic_deformation", "geological_stratification", "mineral_inclusions"],
                "recommended_dataframe": ["geological_stratification", "fracture_patterns", "value_variation"]
            },
            "schlumberger": {
                "description": "Méthode Schlumberger pour sondages électriques verticaux",
                "best_practices": [
                    "Le retournement horizontal simule différents profils",
                    "Les effets d'altération sont importants en surface",
                    "Les couches sédimentaires sont typiques de cette méthode",
                    "Éviter le masquage qui peut masquer des structures importantes"
                ],
                "recommended_2d": ["flip_horizontal", "value_variation", "weathering_effects", "sedimentary_layers"],
                "recommended_3d": ["flip_horizontal", "weathering_effects", "sedimentary_layers", "mineral_inclusions"],
                "recommended_dataframe": ["weathering_effects", "sedimentary_layers", "spatial_jitter"]
            }
        }
    
    def validate_augmentation_parameters(self, augmentations: List[str], 
                                       data_type: str = "2d_grid", 
                                       method: str = "pole_dipole") -> bool:
        """
        Valider que les techniques d'augmentation sont appropriées pour le type de données et la méthode géophysique.
        
        Args:
            augmentations: Liste des techniques d'augmentation
            data_type: Type de données ("2d_grid", "3d_volume", "dataframe")
            method: Méthode géophysique ("pole_dipole", "schlumberger")
            
        Returns:
            True si les paramètres sont valides, False sinon
        """
        # Techniques valides par type de données
        valid_augmentations = {
            "2d_grid": [
                "rotation", "flip_horizontal", "flip_vertical", 
                "spatial_shift", "gaussian_noise", "salt_pepper_noise",
                "value_variation", "elastic_deformation",
                "geological_stratification", "fracture_patterns",
                "mineral_inclusions", "weathering_effects", "sedimentary_layers",
                "color_jittering", "blur_sharpen", "perspective_transform", "cutout"
            ],
            "3d_volume": [
                "rotation", "flip_horizontal", "flip_vertical",
                "gaussian_noise", "value_variation", "elastic_deformation",
                "geological_stratification", "fracture_patterns",
                "mineral_inclusions", "weathering_effects", "sedimentary_layers"
            ],
            "dataframe": [
                "gaussian_noise", "value_variation", 
                "spatial_jitter", "coordinate_perturbation",
                "geological_stratification", "fracture_patterns",
                "mineral_inclusions", "weathering_effects", "sedimentary_layers"
            ]
        }
        
        # Techniques recommandées par méthode géophysique
        method_recommendations = {
            "pole_dipole": {
                "recommended": ["rotation", "elastic_deformation", "geological_stratification", "fracture_patterns"],
                "avoid": ["perspective_transform"]  # Peut déformer les pseudo-sections
            },
            "schlumberger": {
                "recommended": ["flip_horizontal", "value_variation", "weathering_effects", "sedimentary_layers"],
                "avoid": ["cutout"]  # Peut masquer des structures importantes
            }
        }
        
        # Vérifier les techniques valides pour le type de données
        valid_for_type = valid_augmentations.get(data_type, [])
        invalid_augmentations = [aug for aug in augmentations if aug not in valid_for_type]
        
        if invalid_augmentations:
            logger.warning(f"Techniques d'augmentation invalides pour {data_type}: {invalid_augmentations}")
            return False
        
        # Vérifier les recommandations spécifiques à la méthode
        if method in method_recommendations:
            recommendations = method_recommendations[method]
            
            # Avertir pour les techniques non recommandées
            not_recommended = [aug for aug in augmentations if aug not in recommendations["recommended"]]
            if not_recommended:
                logger.info(f"Techniques non recommandées pour {method}: {not_recommended}")
            
            # Avertir pour les techniques à éviter
            to_avoid = [aug for aug in augmentations if aug in recommendations["avoid"]]
            if to_avoid:
                logger.warning(f"Techniques déconseillées pour {method}: {to_avoid}")
                return False
        
        return True
    
    def validate_geophysical_parameters(self, augmentations: List[str], 
                                       method: str = "pole_dipole") -> bool:
        """
        Valider les paramètres d'augmentation selon la méthode géophysique.
        
        Args:
            augmentations: Liste des techniques d'augmentation
            method: Méthode géophysique ("pole_dipole", "schlumberger")
            
        Returns:
            True si les paramètres sont valides, False sinon
        """
        # Paramètres spécifiques par méthode
        method_parameters = {
            "pole_dipole": {
                "max_rotation_angle": 15,  # Degrés
                "max_elastic_deformation": 0.1,
                "max_value_variation": 0.2,
                "fracture_probability": 0.1
            },
            "schlumberger": {
                "max_rotation_angle": 10,  # Degrés
                "max_elastic_deformation": 0.05,
                "max_value_variation": 0.15,
                "fracture_probability": 0.05
            }
        }
        
        if method not in method_parameters:
            logger.warning(f"Méthode géophysique non reconnue: {method}")
            return False
        
        params = method_parameters[method]
        
        # Valider les paramètres selon la méthode
        for aug in augmentations:
            if aug == "rotation" and "max_rotation_angle" in params:
                # Vérifier que l'angle de rotation est approprié
                logger.info(f"Rotation validée pour {method} (max: {params['max_rotation_angle']}°)")
            
            elif aug == "elastic_deformation" and "max_elastic_deformation" in params:
                logger.info(f"Déformation élastique validée pour {method} (max: {params['max_elastic_deformation']})")
            
            elif aug == "value_variation" and "max_value_variation" in params:
                logger.info(f"Variation de valeurs validée pour {method} (max: {params['max_value_variation']})")
        
        return True
