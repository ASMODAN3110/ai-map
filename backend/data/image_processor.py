"""
Processeur d'Images Géophysiques pour CNN
=========================================

Ce module fournit des fonctionnalités pour traiter des images géophysiques
et les convertir en tenseurs PyTorch compatibles avec les modèles CNN.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage
from scipy.signal import wiener
import torch
from torchvision import transforms
import os
from typing import List, Dict, Tuple, Union, Optional
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeophysicalImageProcessor:
    """Processeur d'images géophysiques pour CNN avec méthodes de nettoyage avancées."""
    
    def __init__(self, target_size: Tuple[int, int] = (64, 64), 
                 channels: int = 3, normalize: bool = True):
        """
        Initialiser le processeur d'images.
        
        Args:
            target_size (tuple): Taille cible (width, height)
            channels (int): Nombre de canaux (3=RGB, 1=grayscale)
            normalize (bool): Normaliser les images avec ImageNet stats
        """
        self.target_size = target_size
        self.channels = channels
        self.normalize = normalize
        self.transform = self._create_transforms()
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        logger.info(f"Processeur d'images initialisé: {target_size}, {channels} canaux")
    
    def _create_transforms(self) -> transforms.Compose:
        """Créer les transformations d'images PyTorch."""
        transform_list = [
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ]
        
        if self.normalize and self.channels == 3:
            # Normalisation ImageNet
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
        elif self.normalize and self.channels == 1:
            # Normalisation pour grayscale
            transform_list.append(transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            ))
        
        return transforms.Compose(transform_list)
    
    def load_image(self, image_path: str) -> Image.Image:
        """Charger une image depuis un fichier."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image non trouvée: {image_path}")
        
        # Vérifier l'extension
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Format non supporté: {file_ext}")
        
        try:
            # Charger avec PIL
            if self.channels == 3:
                image = Image.open(image_path).convert('RGB')
            else:
                image = Image.open(image_path).convert('L')
            
            logger.debug(f"Image chargée: {image_path}, taille: {image.size}")
            return image
            
        except Exception as e:
            raise ValueError(f"Impossible de charger l'image {image_path}: {e}")
    
    def process_image(self, image_path: str) -> torch.Tensor:
        """Traiter une image pour CNN."""
        # Charger l'image
        image = self.load_image(image_path)
        
        # Appliquer les transformations
        tensor = self.transform(image)
        
        # Ajouter dimension batch si nécessaire
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        logger.debug(f"Image traitée: {image_path}, forme tensor: {tensor.shape}")
        return tensor
    
    def process_image_batch(self, image_paths: List[str]) -> torch.Tensor:
        """Traiter un lot d'images."""
        if not image_paths:
            raise ValueError("Liste d'images vide")
        
        tensors = []
        for path in image_paths:
            try:
                tensor = self.process_image(path)
                tensors.append(tensor)
            except Exception as e:
                logger.warning(f"Erreur lors du traitement de {path}: {e}")
                continue
        
        if not tensors:
            raise ValueError("Aucune image n'a pu être traitée")
        
        # Concaténer en batch
        batch_tensor = torch.cat(tensors, dim=0)
        logger.info(f"Lot d'images traité: {len(tensors)} images, forme: {batch_tensor.shape}")
        
        return batch_tensor
    
    # ============================================================================
    # MÉTHODES DE NETTOYAGE AVANCÉES
    # ============================================================================
    
    def apply_noise_reduction(self, image: Image.Image, method: str = "gaussian", 
                            **kwargs) -> Image.Image:
        """
        Réduire le bruit d'une image avec différentes méthodes.
        
        Args:
            image: Image PIL à nettoyer
            method: Méthode de réduction de bruit
            **kwargs: Paramètres spécifiques à la méthode
        
        Returns:
            Image nettoyée
        """
        if method == "gaussian":
            return self._gaussian_noise_reduction(image, **kwargs)
        elif method == "median":
            return self._median_noise_reduction(image, **kwargs)
        elif method == "bilateral":
            return self._bilateral_noise_reduction(image, **kwargs)
        elif method == "wiener":
            return self._wiener_noise_reduction(image, **kwargs)
        elif method == "non_local_means":
            return self._non_local_means_reduction(image, **kwargs)
        else:
            raise ValueError(f"Méthode de réduction de bruit non supportée: {method}")
    
    def _gaussian_noise_reduction(self, image: Image.Image, kernel_size: int = 5, 
                                sigma: float = 1.0) -> Image.Image:
        """Réduction de bruit par filtre gaussien."""
        # Convertir en array numpy
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:  # RGB
            # Appliquer le filtre sur chaque canal
            filtered = np.zeros_like(img_array)
            for i in range(3):
                filtered[:, :, i] = ndimage.gaussian_filter(
                    img_array[:, :, i], sigma=sigma, mode='reflect'
                )
        else:  # Grayscale
            filtered = ndimage.gaussian_filter(img_array, sigma=sigma, mode='reflect')
        
        return Image.fromarray(filtered.astype(np.uint8))
    
    def _median_noise_reduction(self, image: Image.Image, kernel_size: int = 5) -> Image.Image:
        """Réduction de bruit par filtre médian (excellent pour le bruit impulsionnel)."""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:  # RGB
            filtered = np.zeros_like(img_array)
            for i in range(3):
                filtered[:, :, i] = ndimage.median_filter(
                    img_array[:, :, i], size=kernel_size
                )
        else:  # Grayscale
            filtered = ndimage.median_filter(img_array, size=kernel_size)
        
        return Image.fromarray(filtered.astype(np.uint8))
    
    def _bilateral_noise_reduction(self, image: Image.Image, d: int = 15, 
                                 sigma_color: float = 75, sigma_space: float = 75) -> Image.Image:
        """Réduction de bruit par filtre bilatéral (préserve les contours)."""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:  # RGB
            # Convertir en BGR pour OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            filtered_bgr = cv2.bilateralFilter(img_bgr, d, sigma_color, sigma_space)
            # Reconvertir en RGB
            filtered = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
        else:  # Grayscale
            filtered = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)
        
        return Image.fromarray(filtered.astype(np.uint8))
    
    def _wiener_noise_reduction(self, image: Image.Image, noise_power: float = 0.1) -> Image.Image:
        """Réduction de bruit par filtre de Wiener (optimal pour le bruit gaussien)."""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:  # RGB
            filtered = np.zeros_like(img_array)
            for i in range(3):
                filtered[:, :, i] = wiener(img_array[:, :, i], noise_power)
        else:  # Grayscale
            filtered = wiener(img_array, noise_power)
        
        # Normaliser et convertir en uint8
        filtered = np.clip(filtered, 0, 255).astype(np.uint8)
        return Image.fromarray(filtered)
    
    def _non_local_means_reduction(self, image: Image.Image, h: float = 10, 
                                  template_window_size: int = 7, 
                                  search_window_size: int = 21) -> Image.Image:
        """Réduction de bruit par méthode non-locale (très efficace mais lente)."""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:  # RGB
            # Convertir en BGR pour OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            filtered_bgr = cv2.fastNlMeansDenoisingColored(
                img_bgr, h=h, templateWindowSize=template_window_size, 
                searchWindowSize=search_window_size
            )
            # Reconvertir en RGB
            filtered = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
        else:  # Grayscale
            filtered = cv2.fastNlMeansDenoising(
                img_array, h=h, templateWindowSize=template_window_size, 
                searchWindowSize=search_window_size
            )
        
        return Image.fromarray(filtered.astype(np.uint8))
    
    def correct_artifacts(self, image: Image.Image, artifact_type: str, **kwargs) -> Image.Image:
        """
        Corriger des artefacts spécifiques aux images géophysiques.
        
        Args:
            image: Image PIL à corriger
            artifact_type: Type d'artefact à corriger
            **kwargs: Paramètres spécifiques à la correction
        
        Returns:
            Image corrigée
        """
        if artifact_type == "scan_lines":
            return self._remove_scan_lines(image, **kwargs)
        elif artifact_type == "salt_pepper":
            return self._remove_salt_pepper_noise(image, **kwargs)
        elif artifact_type == "streaking":
            return self._remove_streaking(image, **kwargs)
        elif artifact_type == "banding":
            return self._remove_banding(image, **kwargs)
        else:
            raise ValueError(f"Type d'artefact non supporté: {artifact_type}")
    
    def _remove_scan_lines(self, image: Image.Image, line_thickness: int = 1) -> Image.Image:
        """Supprimer les lignes de balayage horizontales."""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:  # RGB
            # Détecter et corriger les lignes horizontales
            for i in range(3):
                channel = img_array[:, :, i]
                # Filtre pour supprimer les lignes horizontales
                kernel = np.ones((1, line_thickness), np.uint8)
                lines = cv2.morphologyEx(channel, cv2.MORPH_OPEN, kernel)
                # Interpolation pour remplir les lignes
                channel = cv2.inpaint(channel, lines.astype(np.uint8), 3, cv2.INPAINT_TELEA)
                img_array[:, :, i] = channel
        else:  # Grayscale
            kernel = np.ones((1, line_thickness), np.uint8)
            lines = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)
            img_array = cv2.inpaint(img_array, lines.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _remove_salt_pepper_noise(self, image: Image.Image, kernel_size: int = 3) -> Image.Image:
        """Supprimer le bruit sel-et-poivre."""
        return self._median_noise_reduction(image, kernel_size=kernel_size)
    
    def _remove_streaking(self, image: Image.Image, direction: str = "horizontal") -> Image.Image:
        """Supprimer les stries directionnelles."""
        img_array = np.array(image)
        
        if direction == "horizontal":
            kernel = np.ones((1, 5), np.uint8)
        elif direction == "vertical":
            kernel = np.ones((5, 1), np.uint8)
        else:
            raise ValueError("Direction doit être 'horizontal' ou 'vertical'")
        
        if len(img_array.shape) == 3:  # RGB
            for i in range(3):
                channel = img_array[:, :, i]
                # Supprimer les stries
                streaking = cv2.morphologyEx(channel, cv2.MORPH_OPEN, kernel)
                channel = cv2.inpaint(channel, streaking.astype(np.uint8), 3, cv2.INPAINT_TELEA)
                img_array[:, :, i] = channel
        else:  # Grayscale
            streaking = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)
            img_array = cv2.inpaint(img_array, streaking.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _remove_banding(self, image: Image.Image, band_width: int = 10) -> Image.Image:
        """Supprimer les bandes de moiré."""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:  # RGB
            for i in range(3):
                channel = img_array[:, :, i]
                # Filtre passe-bas pour supprimer les bandes
                channel = cv2.GaussianBlur(channel, (band_width, band_width), 0)
                img_array[:, :, i] = channel
        else:  # Grayscale
            img_array = cv2.GaussianBlur(img_array, (band_width, band_width), 0)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def enhance_contrast(self, image: Image.Image, method: str = "histogram_equalization", 
                        **kwargs) -> Image.Image:
        """
        Améliorer le contraste d'une image.
        
        Args:
            image: Image PIL à améliorer
            method: Méthode d'amélioration du contraste
            **kwargs: Paramètres spécifiques à la méthode
        
        Returns:
            Image avec contraste amélioré
        """
        if method == "histogram_equalization":
            return self._histogram_equalization(image, **kwargs)
        elif method == "adaptive_histogram":
            return self._adaptive_histogram_equalization(image, **kwargs)
        elif method == "clahe":
            return self._clahe_enhancement(image, **kwargs)
        elif method == "gamma_correction":
            return self._gamma_correction(image, **kwargs)
        else:
            raise ValueError(f"Méthode d'amélioration du contraste non supportée: {method}")
    
    def _histogram_equalization(self, image: Image.Image) -> Image.Image:
        """Égalisation d'histogramme classique."""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:  # RGB
            # Convertir en YUV pour traiter la luminance
            img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
            # Égaliser le canal Y (luminance)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            # Reconvertir en RGB
            enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        else:  # Grayscale
            enhanced = cv2.equalizeHist(img_array)
        
        return Image.fromarray(enhanced.astype(np.uint8))
    
    def _adaptive_histogram_equalization(self, image: Image.Image, 
                                       clip_limit: float = 2.0, 
                                       tile_grid_size: Tuple[int, int] = (8, 8)) -> Image.Image:
        """Égalisation d'histogramme adaptative (CLAHE)."""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:  # RGB
            # Convertir en LAB pour traiter la luminance
            img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            # Créer le CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            # Appliquer sur le canal L
            img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
            # Reconvertir en RGB
            enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        else:  # Grayscale
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe.apply(img_array)
        
        return Image.fromarray(enhanced.astype(np.uint8))
    
    def _clahe_enhancement(self, image: Image.Image, clip_limit: float = 2.0, 
                          tile_grid_size: Tuple[int, int] = (8, 8)) -> Image.Image:
        """Enhancement CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        return self._adaptive_histogram_equalization(image, clip_limit, tile_grid_size)
    
    def _gamma_correction(self, image: Image.Image, gamma: float = 1.2) -> Image.Image:
        """Correction gamma pour ajuster la luminosité."""
        img_array = np.array(image)
        
        # Table de lookup pour la correction gamma
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        if len(img_array.shape) == 3:  # RGB
            corrected = cv2.LUT(img_array, table)
        else:  # Grayscale
            corrected = cv2.LUT(img_array, table)
        
        return Image.fromarray(corrected.astype(np.uint8))
    
    def apply_geophysical_specific_cleaning(self, image: Image.Image, 
                                          cleaning_steps: List[str]) -> Image.Image:
        """
        Appliquer une séquence de nettoyage spécifique aux images géophysiques.
        
        Args:
            image: Image PIL à nettoyer
            cleaning_steps: Liste des étapes de nettoyage à appliquer
        
        Returns:
            Image nettoyée
        """
        cleaned_image = image.copy()
        
        for step in cleaning_steps:
            if step == "noise_reduction":
                cleaned_image = self.apply_noise_reduction(cleaned_image, method="bilateral")
            elif step == "scan_lines_removal":
                cleaned_image = self.correct_artifacts(cleaned_image, "scan_lines")
            elif step == "contrast_enhancement":
                cleaned_image = self.enhance_contrast(cleaned_image, method="clahe")
            elif step == "salt_pepper_removal":
                cleaned_image = self.correct_artifacts(cleaned_image, "salt_pepper")
            elif step == "streaking_removal":
                cleaned_image = self.correct_artifacts(cleaned_image, "streaking", direction="horizontal")
            else:
                logger.warning(f"Étape de nettoyage non reconnue: {step}")
        
        logger.info(f"Nettoyage géophysique appliqué: {len(cleaning_steps)} étapes")
        return cleaned_image
    
    def get_cleaning_summary(self, image: Image.Image, 
                           cleaning_methods: List[str]) -> Dict[str, Union[float, str]]:
        """
        Obtenir un résumé des améliorations apportées par le nettoyage.
        
        Args:
            image: Image originale
            cleaning_methods: Méthodes de nettoyage à évaluer
        
        Returns:
            Résumé des améliorations
        """
        original_features = self.extract_geophysical_features_from_image(image)
        
        # Appliquer le nettoyage
        cleaned_image = self.apply_geophysical_specific_cleaning(image, cleaning_methods)
        cleaned_features = self.extract_geophysical_features_from_image(cleaned_image)
        
        # Calculer les améliorations
        improvements = {
            'noise_reduction': original_features['std_intensity'] - cleaned_features['std_intensity'],
            'contrast_improvement': cleaned_features['std_intensity'] - original_features['std_intensity'],
            'gradient_enhancement': cleaned_features['gradient_magnitude'] - original_features['gradient_magnitude'],
            'cleaning_methods_applied': cleaning_methods,
            'original_std': original_features['std_intensity'],
            'cleaned_std': cleaned_features['std_intensity']
        }
        
        return improvements
    
    def extract_geophysical_features_from_image(self, image: Image.Image) -> Dict[str, Union[float, np.ndarray]]:
        """Extraire des features géophysiques d'une image PIL."""
        # Convertir en numpy array
        img_array = np.array(image)
        
        # Features basiques d'intensité
        features = {
            'mean_intensity': float(np.mean(img_array)),
            'std_intensity': float(np.std(img_array)),
            'min_intensity': float(np.min(img_array)),
            'max_intensity': float(np.max(img_array)),
            'image_size': img_array.shape
        }
        
        # Histogramme
        if len(img_array.shape) == 3:  # RGB
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:  # Grayscale
            gray = img_array
        
        hist = np.histogram(gray, bins=256, range=(0, 255))[0]
        features['histogram'] = hist.astype(np.float32)
        
        # Features de gradient (contours géologiques)
        features['gradient_magnitude'] = float(self._calculate_gradient(gray))
        
        return features
    
    def extract_geophysical_features(self, image_path: str) -> Dict[str, Union[float, np.ndarray]]:
        """Extraire des features géophysiques d'une image depuis un fichier."""
        # Charger l'image
        image = self.load_image(image_path)
        
        # Extraire les features
        return self.extract_geophysical_features_from_image(image)
    
    def _calculate_gradient(self, gray_image: np.ndarray) -> float:
        """Calculer la magnitude du gradient."""
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return float(np.mean(magnitude))
    
    def save_processed_image(self, image_path: str, output_path: str) -> str:
        """Sauvegarder une image prétraitée."""
        # Charger et traiter l'image
        image = self.load_image(image_path)
        
        # Redimensionner
        image = image.resize(self.target_size, Image.LANCZOS)
        
        # Créer le dossier de sortie si nécessaire
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Sauvegarder
        image.save(output_path, quality=95)
        logger.info(f"Image prétraitée sauvegardée: {output_path}")
        
        return output_path


class ImageAugmenter:
    """Augmenteur d'images géophysiques avec techniques avancées."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialiser l'augmenteur."""
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        self.augmentation_history = []
        self.geophysical_patterns = self._initialize_geophysical_patterns()
        logger.info("Augmenteur d'images géophysiques avancé initialisé")
    
    def _initialize_geophysical_patterns(self) -> Dict[str, np.ndarray]:
        """Initialiser les motifs géophysiques de base."""
        patterns = {}
        
        # Motif de stratification
        strat_pattern = np.zeros((64, 64), dtype=np.uint8)
        for i in range(0, 64, 8):
            strat_pattern[i:i+4, :] = 128
        patterns['stratification'] = strat_pattern
        
        # Motif de fractures
        fracture_pattern = np.zeros((64, 64), dtype=np.uint8)
        for _ in range(5):
            x1, y1 = np.random.randint(0, 64), np.random.randint(0, 64)
            x2, y2 = x1 + np.random.randint(-20, 20), y1 + np.random.randint(-20, 20)
            cv2.line(fracture_pattern, (x1, y1), (x2, y2), 255, 2)
        patterns['fractures'] = fracture_pattern
        
        # Motif d'inclusions minérales
        inclusion_pattern = np.zeros((64, 64), dtype=np.uint8)
        for _ in range(8):
            x, y = np.random.randint(0, 64), np.random.randint(0, 64)
            radius = np.random.randint(2, 6)
            cv2.circle(inclusion_pattern, (x, y), radius, 200, -1)
        patterns['inclusions'] = inclusion_pattern
        
        return patterns
    
    def augment_image(self, image: Image.Image, augmentations: List[str], 
                     num_augmentations: int = 1) -> List[Image.Image]:
        """Augmenter une image avec les techniques spécifiées."""
        augmented_images = [image]  # Image originale incluse
        
        for _ in range(num_augmentations):
            augmented = image.copy()
            
            for aug_type in augmentations:
                try:
                    if aug_type == "rotation":
                        augmented = self._apply_rotation(augmented)
                    elif aug_type == "flip_horizontal":
                        augmented = self._apply_flip_horizontal(augmented)
                    elif aug_type == "flip_vertical":
                        augmented = self._apply_flip_vertical(augmented)
                    elif aug_type == "brightness":
                        augmented = self._apply_brightness(augmented)
                    elif aug_type == "contrast":
                        augmented = self._apply_contrast(augmented)
                    # Nouvelles techniques avancées
                    elif aug_type == "elastic_deformation":
                        augmented = self._apply_elastic_deformation(augmented)
                    elif aug_type == "color_jittering":
                        augmented = self._apply_color_jittering(augmented)
                    elif aug_type == "gaussian_noise":
                        augmented = self._apply_gaussian_noise(augmented)
                    elif aug_type == "blur_sharpen":
                        augmented = self._apply_blur_sharpen(augmented)
                    elif aug_type == "perspective_transform":
                        augmented = self._apply_perspective_transform(augmented)
                    elif aug_type == "cutout":
                        augmented = self._apply_cutout(augmented)
                    elif aug_type == "geological_stratification":
                        augmented = self._apply_geological_stratification(augmented)
                    elif aug_type == "fracture_patterns":
                        augmented = self._apply_fracture_patterns(augmented)
                    elif aug_type == "mineral_inclusions":
                        augmented = self._apply_mineral_inclusions(augmented)
                    elif aug_type == "weathering_effects":
                        augmented = self._apply_weathering_effects(augmented)
                    elif aug_type == "sedimentary_layers":
                        augmented = self._apply_sedimentary_layers(augmented)
                    else:
                        logger.warning(f"Type d'augmentation non reconnu: {aug_type}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Erreur lors de l'augmentation {aug_type}: {e}")
                    continue
            
            augmented_images.append(augmented)
            self.augmentation_history.append({
                'augmentations': augmentations,
                'timestamp': np.datetime64('now')
            })
        
        logger.info(f"Image augmentée: {len(augmented_images)} versions créées")
        return augmented_images
    
    # ============================================================================
    # TECHNIQUES D'AUGMENTATION DE BASE
    # ============================================================================
    
    def _apply_rotation(self, image: Image.Image) -> Image.Image:
        """Appliquer une rotation aléatoire."""
        angle = np.random.uniform(-15, 15)
        return image.rotate(angle, fillcolor=128)
    
    def _apply_flip_horizontal(self, image: Image.Image) -> Image.Image:
        """Appliquer un flip horizontal aléatoire."""
        if np.random.random() > 0.5:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image
    
    def _apply_flip_vertical(self, image: Image.Image) -> Image.Image:
        """Appliquer un flip vertical aléatoire."""
        if np.random.random() > 0.5:
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        return image
    
    def _apply_brightness(self, image: Image.Image) -> Image.Image:
        """Appliquer une variation de luminosité."""
        factor = np.random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def _apply_contrast(self, image: Image.Image) -> Image.Image:
        """Appliquer une variation de contraste."""
        factor = np.random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    # ============================================================================
    # TECHNIQUES D'AUGMENTATION AVANCÉES
    # ============================================================================
    
    def _apply_elastic_deformation(self, image: Image.Image, alpha: float = 1.0, sigma: float = 50.0) -> Image.Image:
        """Appliquer une déformation élastique pour simuler les plis géologiques."""
        img_array = np.array(image)
        
        # Créer une grille de déformation
        h, w = img_array.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Générer des déplacements aléatoires
        dx = np.random.randn(h, w) * alpha
        dy = np.random.randn(h, w) * alpha
        
        # Appliquer un filtre gaussien pour lisser les déplacements
        dx = ndimage.gaussian_filter(dx, sigma)
        dy = ndimage.gaussian_filter(dy, sigma)
        
        # Normaliser les déplacements
        dx = dx * (w / 100)
        dy = dy * (h / 100)
        
        # Appliquer la déformation
        grid_x = np.clip(grid_x + dx, 0, w - 1).astype(np.int32)
        grid_y = np.clip(grid_y + dy, 0, h - 1).astype(np.int32)
        
        # Interpoler l'image
        if len(img_array.shape) == 3:
            deformed = np.zeros_like(img_array)
            for i in range(3):
                deformed[:, :, i] = img_array[grid_y, grid_x, i]
        else:
            deformed = img_array[grid_y, grid_x]
        
        return Image.fromarray(deformed.astype(np.uint8))
    
    def _apply_color_jittering(self, image: Image.Image, 
                              hue_factor: float = 0.1, 
                              saturation_factor: float = 0.2,
                              value_factor: float = 0.2) -> Image.Image:
        """Appliquer des variations de couleur pour différents types de roches."""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            # Convertir en HSV
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Variations aléatoires
            hsv[:, :, 0] *= (1 + np.random.uniform(-hue_factor, hue_factor))  # Teinte
            hsv[:, :, 1] *= (1 + np.random.uniform(-saturation_factor, saturation_factor))  # Saturation
            hsv[:, :, 2] *= (1 + np.random.uniform(-value_factor, value_factor))  # Valeur
            
            # Normaliser
            hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            
            # Reconvertir en RGB
            jittered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            return Image.fromarray(jittered)
        
        return image
    
    def _apply_gaussian_noise(self, image: Image.Image, std: float = 25.0) -> Image.Image:
        """Ajouter du bruit gaussien réaliste."""
        img_array = np.array(image)
        noise = np.random.normal(0, std, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)
    
    def _apply_blur_sharpen(self, image: Image.Image) -> Image.Image:
        """Appliquer un flou ou un aiguisage aléatoire."""
        if np.random.random() > 0.5:
            # Flou gaussien
            radius = np.random.uniform(0.5, 2.0)
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        else:
            # Aiguisage
            factor = np.random.uniform(1.5, 2.5)
            enhancer = ImageEnhance.Sharpness(image)
            return enhancer.enhance(factor)
    
    def _apply_perspective_transform(self, image: Image.Image, max_shift: float = 0.1) -> Image.Image:
        """Appliquer une transformation de perspective."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Points source (coins de l'image)
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Points de destination avec décalage aléatoire
        shift_x = w * max_shift
        shift_y = h * max_shift
        
        dst_points = np.float32([
            [np.random.uniform(-shift_x, shift_x), np.random.uniform(-shift_y, shift_y)],
            [w + np.random.uniform(-shift_x, shift_x), np.random.uniform(-shift_y, shift_y)],
            [w + np.random.uniform(-shift_x, shift_x), h + np.random.uniform(-shift_y, shift_y)],
            [np.random.uniform(-shift_x, shift_x), h + np.random.uniform(-shift_y, shift_y)]
        ])
        
        # Calculer la matrice de transformation
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Appliquer la transformation
        transformed = cv2.warpPerspective(img_array, matrix, (w, h))
        return Image.fromarray(transformed)
    
    def _apply_cutout(self, image: Image.Image, num_holes: int = 3, hole_size: float = 0.1) -> Image.Image:
        """Appliquer un masquage de zones pour améliorer la robustesse."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        for _ in range(num_holes):
            # Taille du trou
            hole_w = int(w * hole_size * np.random.uniform(0.5, 1.5))
            hole_h = int(h * hole_size * np.random.uniform(0.5, 1.5))
            
            # Position du trou
            x = np.random.randint(0, w - hole_w)
            y = np.random.randint(0, h - hole_h)
            
            # Couleur de remplissage (moyenne des pixels voisins)
            if len(img_array.shape) == 3:
                img_array[y:y+hole_h, x:x+hole_w] = [128, 128, 128]
            else:
                img_array[y:y+hole_h, x:x+hole_w] = 128
        
        return Image.fromarray(img_array)
    
    # ============================================================================
    # TECHNIQUES SPÉCIFIQUES AUX IMAGES GÉOPHYSIQUES
    # ============================================================================
    
    def _apply_geological_stratification(self, image: Image.Image, intensity: float = 0.3) -> Image.Image:
        """Ajouter des motifs de stratification géologique."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Créer un motif de stratification
        strat_pattern = np.zeros((h, w), dtype=np.uint8)
        layer_height = max(4, h // 20)
        
        for i in range(0, h, layer_height):
            thickness = np.random.randint(2, layer_height // 2)
            strat_pattern[i:i+thickness, :] = 255
        
        # Appliquer le motif avec transparence
        if len(img_array.shape) == 3:
            for i in range(3):
                img_array[:, :, i] = np.clip(
                    img_array[:, :, i] * (1 - intensity) + 
                    strat_pattern * intensity, 0, 255
                ).astype(np.uint8)
        else:
            img_array = np.clip(
                img_array * (1 - intensity) + strat_pattern * intensity, 0, 255
            ).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def _apply_fracture_patterns(self, image: Image.Image, num_fractures: int = 5) -> Image.Image:
        """Ajouter des motifs de fractures géologiques."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Créer des fractures aléatoires
        for _ in range(num_fractures):
            # Points de départ et d'arrivée
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            x2 = x1 + np.random.randint(-w//4, w//4)
            y2 = y1 + np.random.randint(-h//4, h//4)
            
            # Limiter aux bords de l'image
            x2 = np.clip(x2, 0, w-1)
            y2 = np.clip(y2, 0, h-1)
            
            # Largeur de la fracture
            width = np.random.randint(1, 4)
            
            # Dessiner la fracture
            if len(img_array.shape) == 3:
                cv2.line(img_array, (x1, y1), (x2, y2), [0, 0, 0], width)
            else:
                cv2.line(img_array, (x1, y1), (x2, y2), 0, width)
        
        return Image.fromarray(img_array)
    
    def _apply_mineral_inclusions(self, image: Image.Image, num_inclusions: int = 8) -> Image.Image:
        """Ajouter des inclusions minérales."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Couleurs minérales typiques
        mineral_colors = [
            [255, 215, 0],   # Or
            [192, 192, 192], # Argent
            [184, 115, 51],  # Cuivre
            [255, 69, 0],    # Rouge-orange
            [0, 128, 128],   # Teal
            [128, 0, 128]    # Violet
        ]
        
        for _ in range(num_inclusions):
            # Position et taille
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            radius = np.random.randint(2, 8)
            
            # Couleur aléatoire
            color = mineral_colors[np.random.randint(0, len(mineral_colors))]
            
            # Dessiner l'inclusion
            if len(img_array.shape) == 3:
                cv2.circle(img_array, (x, y), radius, color, -1)
            else:
                cv2.circle(img_array, (x, y), radius, 200, -1)
        
        return Image.fromarray(img_array)
    
    def _apply_weathering_effects(self, image: Image.Image, intensity: float = 0.4) -> Image.Image:
        """Simuler les effets d'altération et d'érosion."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Créer un motif d'altération
        weathering = np.random.rand(h, w) * 255
        
        # Appliquer un filtre gaussien pour lisser
        weathering = ndimage.gaussian_filter(weathering, sigma=3)
        
        # Appliquer avec transparence
        if len(img_array.shape) == 3:
            for i in range(3):
                img_array[:, :, i] = np.clip(
                    img_array[:, :, i] * (1 - intensity) + 
                    weathering * intensity, 0, 255
                ).astype(np.uint8)
        else:
            img_array = np.clip(
                img_array * (1 - intensity) + weathering * intensity, 0, 255
            ).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def _apply_sedimentary_layers(self, image: Image.Image, num_layers: int = 6) -> Image.Image:
        """Simuler des couches sédimentaires avec variations de couleur."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        if len(img_array.shape) == 3:
            # Couleurs sédimentaires
            sedimentary_colors = [
                [139, 69, 19],   # Brun sable
                [160, 82, 45],   # Brun terre
                [210, 180, 140], # Beige
                [245, 245, 220], # Blanc cassé
                [188, 143, 143], # Rosé
                [165, 42, 42]    # Rouge-brun
            ]
            
            layer_height = h // num_layers
            
            for i in range(num_layers):
                y_start = i * layer_height
                y_end = min((i + 1) * layer_height, h)
                
                # Couleur de la couche
                color = sedimentary_colors[i % len(sedimentary_colors)]
                
                # Appliquer la couleur avec transparence
                alpha = 0.3
                for j in range(3):
                    img_array[y_start:y_end, :, j] = np.clip(
                        img_array[y_start:y_end, :, j] * (1 - alpha) + 
                        color[j] * alpha, 0, 255
                    ).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def get_augmentation_summary(self) -> Dict[str, Union[int, list]]:
        """Obtenir un résumé des augmentations effectuées."""
        return {
            'total_augmentations': len(self.augmentation_history),
            'augmentation_types': list(set(
                aug_type for hist in self.augmentation_history 
                for aug_type in hist['augmentations']
            )),
            'available_techniques': [
                # Techniques de base
                'rotation', 'flip_horizontal', 'flip_vertical', 
                'brightness', 'contrast',
                # Techniques avancées
                'elastic_deformation', 'color_jittering', 'gaussian_noise',
                'blur_sharpen', 'perspective_transform', 'cutout',
                # Techniques géophysiques
                'geological_stratification', 'fracture_patterns', 
                'mineral_inclusions', 'weathering_effects', 'sedimentary_layers'
            ]
        }


# Fonction utilitaire pour créer un processeur avec configuration par défaut
def create_image_processor(target_size: Tuple[int, int] = (64, 64), 
                          channels: int = 3) -> GeophysicalImageProcessor:
    """Créer un processeur d'images avec configuration par défaut."""
    return GeophysicalImageProcessor(target_size=target_size, channels=channels)


if __name__ == "__main__":
    # Test du processeur
    processor = GeophysicalImageProcessor()
    augmenter = ImageAugmenter()
    
    print("✅ Processeur d'images géophysiques initialisé avec succès!")
    print(f"Formats supportés: {processor.supported_formats}")
    print(f"Taille cible: {processor.target_size}")
    print(f"Canaux: {processor.channels}")

