#!/usr/bin/env python3
"""
Module de générateurs géophysiques conformes au cahier des charges.

Ce module implémente les architectures U-Net 2D et VoxNet 3D spécifiées dans le cahier des charges
pour la génération de pseudo-sections 2D, cartes d'iso-résistivité/chargeabilité et modèles 3D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
from PIL import Image
import io
import base64
from mpl_toolkits.mplot3d import Axes3D

from backend.utils.logger import logger


class UNet2D(nn.Module):
    """
    Architecture U-Net 2D conforme au cahier des charges.
    
    Spécifications:
    - Entrée: Tenseur 4D (64×64×4) - 4 canaux pour les dispositifs
    - Encodeur: 4 blocs convolutionnels (64→128→256→512→1024 filtres)
    - Décodeur: 4 blocs de déconvolution avec connexions résiduelles
    - Sortie: 2 canaux (résistivité vraie, chargeabilité vraie)
    - Paramètres: ~31M paramètres entraînables
    """
    
    def __init__(self, input_channels: int = 4, output_channels: int = 2, 
                 grid_size: int = 64, dropout_rate: float = 0.2):
        super(UNet2D, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.grid_size = grid_size
        self.dropout_rate = dropout_rate
        
        # Encodeur (downsampling)
        self.enc1 = self._conv_block(input_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        self.enc5 = self._conv_block(512, 1024)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
        # Décodeur (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Couche de sortie
        self.final_conv = nn.Conv2d(64, output_channels, 1)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Initialisation des poids
        self._initialize_weights()
        
        # Calcul du nombre de paramètres
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"U-Net 2D initialisé: {input_channels}→{output_channels} canaux, {grid_size}×{grid_size}, {total_params:,} paramètres")
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Bloc de convolution avec BatchNorm et ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        """Initialiser les poids du modèle."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du modèle U-Net 2D.
        
        Args:
            x: Tenseur d'entrée (batch_size, 4, 64, 64)
            
        Returns:
            Tenseur de sortie (batch_size, 2, 64, 64)
        """
        # Encodeur
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))
        
        # Décodeur avec connexions résiduelles
        dec4 = self.upconv4(enc5)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        dec4 = self.dropout(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        dec3 = self.dropout(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        dec2 = self.dropout(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Sortie finale
        output = self.final_conv(dec1)
        
        return output


class VoxNet3D(nn.Module):
    """
    Architecture VoxNet 3D conforme au cahier des charges.
    
    Spécifications:
    - Entrée: Tenseur 5D (32×32×32×4) - Volume 3D multi-canaux
    - Convolutions 3D: 3 couches (32→64→128 filtres)
    - Déconvolutions 3D: Reconstruction volumétrique
    - Sortie: Volume 3D de chargeabilité
    - Paramètres: ~15M paramètres entraînables
    """
    
    def __init__(self, input_channels: int = 4, output_channels: int = 1, 
                 volume_size: int = 32, dropout_rate: float = 0.2):
        super(VoxNet3D, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.volume_size = volume_size
        self.dropout_rate = dropout_rate
        
        # Encodeur 3D
        self.enc1 = self._conv3d_block(input_channels, 32)
        self.enc2 = self._conv3d_block(32, 64)
        self.enc3 = self._conv3d_block(64, 128)
        
        # Pooling 3D
        self.pool = nn.MaxPool3d(2)
        
        # Décodeur 3D
        self.upconv2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec2 = self._conv3d_block(128, 64)
        
        self.upconv1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = self._conv3d_block(64, 32)
        
        # Couche de sortie
        self.final_conv = nn.Conv3d(32, output_channels, 1)
        
        # Dropout
        self.dropout = nn.Dropout3d(dropout_rate)
        
        # Initialisation des poids
        self._initialize_weights()
        
        # Calcul du nombre de paramètres
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"VoxNet 3D initialisé: {input_channels}→{output_channels} canaux, {volume_size}×{volume_size}×{volume_size}, {total_params:,} paramètres")
    
    def _conv3d_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Bloc de convolution 3D avec BatchNorm et ReLU."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        """Initialiser les poids du modèle."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du modèle VoxNet 3D.
        
        Args:
            x: Tenseur d'entrée (batch_size, 4, 32, 32, 32)
            
        Returns:
            Tenseur de sortie (batch_size, 1, 32, 32, 32)
        """
        # Encodeur 3D
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        
        # Décodeur 3D avec connexions résiduelles
        dec2 = self.upconv2(enc3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        dec2 = self.dropout(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Sortie finale
        output = self.final_conv(dec1)
        
        return output


class GeophysicalDataProcessor:
    """
    Processeur de données géophysiques pour préparer les entrées des modèles.
    """
    
    def __init__(self, grid_size_2d: int = 64, volume_size_3d: int = 32):
        self.grid_size_2d = grid_size_2d
        self.volume_size_3d = volume_size_3d
        
        logger.info(f"Processeur de données initialisé: 2D={grid_size_2d}×{grid_size_2d}, 3D={volume_size_3d}×{volume_size_3d}×{volume_size_3d}")
    
    def process_csv_to_2d_grid(self, csv_data: np.ndarray, method: str = "pole-dipole") -> torch.Tensor:
        """
        Convertir les données CSV en grille 2D pour U-Net.
        
        Args:
            csv_data: Données CSV (n_samples, n_features)
            method: Méthode géophysique
            
        Returns:
            Grille 2D (batch_size, 4, 64, 64)
        """
        batch_size = len(csv_data)
        
        # Créer une grille 2D pour chaque échantillon
        grids = []
        
        for i in range(batch_size):
            # Extraire les caractéristiques
            resistivity = csv_data[i, 0] if csv_data.shape[1] > 0 else 100.0
            chargeability = csv_data[i, 1] if csv_data.shape[1] > 1 else 10.0
            x_coord = csv_data[i, 2] if csv_data.shape[1] > 2 else 50.0
            y_coord = csv_data[i, 3] if csv_data.shape[1] > 3 else 50.0
            
            # Créer une grille 2D avec 4 canaux
            grid = np.zeros((4, self.grid_size_2d, self.grid_size_2d))
            
            # Canal 1: Résistivité (normalisée)
            grid[0] = self._create_resistivity_channel(resistivity, x_coord, y_coord)
            
            # Canal 2: Chargeabilité (normalisée)
            grid[1] = self._create_chargeability_channel(chargeability, x_coord, y_coord)
            
            # Canal 3: Coordonnées X (normalisées)
            grid[2] = self._create_coordinate_channel(x_coord, 'x')
            
            # Canal 4: Coordonnées Y (normalisées)
            grid[3] = self._create_coordinate_channel(y_coord, 'y')
            
            grids.append(grid)
        
        return torch.FloatTensor(np.array(grids))
    
    def process_csv_to_3d_volume(self, csv_data: np.ndarray, method: str = "pole-dipole") -> torch.Tensor:
        """
        Convertir les données CSV en volume 3D pour VoxNet.
        
        Args:
            csv_data: Données CSV (n_samples, n_features)
            method: Méthode géophysique
            
        Returns:
            Volume 3D (batch_size, 4, 32, 32, 32)
        """
        batch_size = len(csv_data)
        
        # Créer un volume 3D pour chaque échantillon
        volumes = []
        
        for i in range(batch_size):
            # Extraire les caractéristiques
            resistivity = csv_data[i, 0] if csv_data.shape[1] > 0 else 100.0
            chargeability = csv_data[i, 1] if csv_data.shape[1] > 1 else 10.0
            x_coord = csv_data[i, 2] if csv_data.shape[1] > 2 else 50.0
            y_coord = csv_data[i, 3] if csv_data.shape[1] > 3 else 50.0
            
            # Créer un volume 3D avec 4 canaux
            volume = np.zeros((4, self.volume_size_3d, self.volume_size_3d, self.volume_size_3d))
            
            # Canal 1: Résistivité 3D
            volume[0] = self._create_resistivity_volume(resistivity, x_coord, y_coord)
            
            # Canal 2: Chargeabilité 3D
            volume[1] = self._create_chargeability_volume(chargeability, x_coord, y_coord)
            
            # Canal 3: Coordonnées X 3D
            volume[2] = self._create_coordinate_volume(x_coord, 'x')
            
            # Canal 4: Coordonnées Y 3D
            volume[3] = self._create_coordinate_volume(y_coord, 'y')
            
            volumes.append(volume)
        
        return torch.FloatTensor(np.array(volumes))
    
    def _create_resistivity_channel(self, resistivity: float, x_coord: float, y_coord: float) -> np.ndarray:
        """Créer un canal de résistivité 2D."""
        # Normaliser la résistivité (log scale)
        log_resistivity = np.log10(max(resistivity, 1.0))
        normalized_resistivity = (log_resistivity - 1.0) / 3.0  # Normaliser entre 0 et 1
        
        # Créer une grille avec variation spatiale
        x = np.linspace(0, 1, self.grid_size_2d)
        y = np.linspace(0, 1, self.grid_size_2d)
        X, Y = np.meshgrid(x, y)
        
        # Créer une distribution gaussienne centrée sur les coordonnées
        center_x = x_coord / 100.0  # Normaliser
        center_y = y_coord / 100.0  # Normaliser
        
        # Distance du centre
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # Créer la grille de résistivité
        grid = normalized_resistivity * np.exp(-distance**2 / 0.1)
        
        return grid
    
    def _create_chargeability_channel(self, chargeability: float, x_coord: float, y_coord: float) -> np.ndarray:
        """Créer un canal de chargeabilité 2D."""
        # Normaliser la chargeabilité
        normalized_chargeability = chargeability / 50.0  # Normaliser entre 0 et 1
        
        # Créer une grille avec variation spatiale
        x = np.linspace(0, 1, self.grid_size_2d)
        y = np.linspace(0, 1, self.grid_size_2d)
        X, Y = np.meshgrid(x, y)
        
        # Créer une distribution gaussienne centrée sur les coordonnées
        center_x = x_coord / 100.0  # Normaliser
        center_y = y_coord / 100.0  # Normaliser
        
        # Distance du centre
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # Créer la grille de chargeabilité
        grid = normalized_chargeability * np.exp(-distance**2 / 0.1)
        
        return grid
    
    def _create_coordinate_channel(self, coord: float, axis: str) -> np.ndarray:
        """Créer un canal de coordonnées 2D."""
        # Normaliser la coordonnée
        normalized_coord = coord / 100.0  # Normaliser entre 0 et 1
        
        # Créer une grille avec gradient
        if axis == 'x':
            grid = np.linspace(0, 1, self.grid_size_2d)
            grid = np.tile(grid, (self.grid_size_2d, 1))
        else:  # y
            grid = np.linspace(0, 1, self.grid_size_2d)
            grid = np.tile(grid.reshape(-1, 1), (1, self.grid_size_2d))
        
        return grid
    
    def _create_resistivity_volume(self, resistivity: float, x_coord: float, y_coord: float) -> np.ndarray:
        """Créer un volume de résistivité 3D."""
        # Normaliser la résistivité (log scale)
        log_resistivity = np.log10(max(resistivity, 1.0))
        normalized_resistivity = (log_resistivity - 1.0) / 3.0  # Normaliser entre 0 et 1
        
        # Créer un volume 3D avec variation spatiale
        x = np.linspace(0, 1, self.volume_size_3d)
        y = np.linspace(0, 1, self.volume_size_3d)
        z = np.linspace(0, 1, self.volume_size_3d)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Créer une distribution gaussienne centrée sur les coordonnées
        center_x = x_coord / 100.0  # Normaliser
        center_y = y_coord / 100.0  # Normaliser
        center_z = 0.5  # Centre en profondeur
        
        # Distance du centre
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2)
        
        # Créer le volume de résistivité
        volume = normalized_resistivity * np.exp(-distance**2 / 0.1)
        
        return volume
    
    def _create_chargeability_volume(self, chargeability: float, x_coord: float, y_coord: float) -> np.ndarray:
        """Créer un volume de chargeabilité 3D."""
        # Normaliser la chargeabilité
        normalized_chargeability = chargeability / 50.0  # Normaliser entre 0 et 1
        
        # Créer un volume 3D avec variation spatiale
        x = np.linspace(0, 1, self.volume_size_3d)
        y = np.linspace(0, 1, self.volume_size_3d)
        z = np.linspace(0, 1, self.volume_size_3d)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Créer une distribution gaussienne centrée sur les coordonnées
        center_x = x_coord / 100.0  # Normaliser
        center_y = y_coord / 100.0  # Normaliser
        center_z = 0.5  # Centre en profondeur
        
        # Distance du centre
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2)
        
        # Créer le volume de chargeabilité
        volume = normalized_chargeability * np.exp(-distance**2 / 0.1)
        
        return volume
    
    def _create_coordinate_volume(self, coord: float, axis: str) -> np.ndarray:
        """Créer un volume de coordonnées 3D."""
        # Normaliser la coordonnée
        normalized_coord = coord / 100.0  # Normaliser entre 0 et 1
        
        # Créer un volume avec gradient
        if axis == 'x':
            grid = np.linspace(0, 1, self.volume_size_3d)
            volume = np.tile(grid, (self.volume_size_3d, self.volume_size_3d, 1))
        else:  # y
            grid = np.linspace(0, 1, self.volume_size_3d)
            volume = np.tile(grid.reshape(-1, 1, 1), (1, self.volume_size_3d, self.volume_size_3d))
        
        return volume


class GeophysicalImageGenerator:
    """
    Générateur d'images géophysiques utilisant U-Net 2D et VoxNet 3D.
    """
    
    def __init__(self, model_path_2d: Optional[str] = None, model_path_3d: Optional[str] = None):
        self.unet_2d = UNet2D()
        self.voxnet_3d = VoxNet3D()
        self.data_processor = GeophysicalDataProcessor()
        
        # Charger les modèles pré-entraînés si disponibles
        if model_path_2d and Path(model_path_2d).exists():
            self.load_model_2d(model_path_2d)
        if model_path_3d and Path(model_path_3d).exists():
            self.load_model_3d(model_path_3d)
        
        logger.info("Générateur d'images géophysiques initialisé avec U-Net 2D et VoxNet 3D")
    
    def load_model_2d(self, model_path: str):
        """Charger un modèle U-Net 2D pré-entraîné."""
        checkpoint = torch.load(model_path, map_location='cpu')
        self.unet_2d.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Modèle U-Net 2D chargé depuis: {model_path}")
    
    def load_model_3d(self, model_path: str):
        """Charger un modèle VoxNet 3D pré-entraîné."""
        checkpoint = torch.load(model_path, map_location='cpu')
        self.voxnet_3d.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Modèle VoxNet 3D chargé depuis: {model_path}")
    
    def generate_pseudo_sections(self, csv_data: np.ndarray, method: str = "pole-dipole") -> List[str]:
        """
        Générer des pseudo-sections 2D à partir de données CSV.
        
        Args:
            csv_data: Données CSV (n_samples, n_features)
            method: Méthode géophysique
            
        Returns:
            Liste d'images de pseudo-sections en base64
        """
        logger.info(f"Génération de pseudo-sections pour {len(csv_data)} échantillons")
        
        # Traiter les données CSV en grilles 2D
        grids_2d = self.data_processor.process_csv_to_2d_grid(csv_data, method)
        
        # Générer les pseudo-sections avec U-Net 2D
        with torch.no_grad():
            outputs = self.unet_2d(grids_2d)
        
        # Convertir en images
        pseudo_sections = []
        for i in range(len(csv_data)):
            # Extraire la résistivité (canal 0) et la chargeabilité (canal 1)
            resistivity = outputs[i, 0].cpu().numpy()
            chargeability = outputs[i, 1].cpu().numpy()
            
            # Créer l'image de pseudo-section
            pseudo_section_img = self._create_pseudo_section_image(resistivity, chargeability, method)
            pseudo_sections.append(pseudo_section_img)
        
        logger.info(f"✅ {len(pseudo_sections)} pseudo-sections générées")
        return pseudo_sections
    
    def generate_3d_models(self, csv_data: np.ndarray, method: str = "pole-dipole") -> List[str]:
        """
        Générer des modèles 3D à partir de données CSV.
        
        Args:
            csv_data: Données CSV (n_samples, n_features)
            method: Méthode géophysique
            
        Returns:
            Liste d'images de modèles 3D en base64
        """
        logger.info(f"Génération de modèles 3D pour {len(csv_data)} échantillons")
        
        # Traiter les données CSV en volumes 3D
        volumes_3d = self.data_processor.process_csv_to_3d_volume(csv_data, method)
        
        # Générer les modèles 3D avec VoxNet 3D
        with torch.no_grad():
            outputs = self.voxnet_3d(volumes_3d)
        
        # Convertir en images
        models_3d = []
        for i in range(len(csv_data)):
            # Extraire le volume de chargeabilité
            chargeability_volume = outputs[i, 0].cpu().numpy()
            
            # Créer l'image du modèle 3D
            model_3d_img = self._create_3d_model_image(chargeability_volume, method)
            models_3d.append(model_3d_img)
        
        logger.info(f"✅ {len(models_3d)} modèles 3D générés")
        return models_3d
    
    def _create_pseudo_section_image(self, resistivity: np.ndarray, chargeability: np.ndarray, method: str) -> str:
        """Créer une image de pseudo-section combinée."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pseudo-section de résistivité
        im1 = ax1.imshow(resistivity, cmap='viridis', aspect='auto', origin='lower')
        ax1.set_title(f'Pseudo-section de Résistivité - {method.upper()}')
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Profondeur (m)')
        plt.colorbar(im1, ax=ax1, label='Résistivité (Ω⋅m)')
        
        # Pseudo-section de chargeabilité
        im2 = ax2.imshow(chargeability, cmap='plasma', aspect='auto', origin='lower')
        ax2.set_title(f'Pseudo-section de Chargeabilité - {method.upper()}')
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Profondeur (m)')
        plt.colorbar(im2, ax=ax2, label='Chargeabilité (mV/V)')
        
        plt.tight_layout()
        
        # Convertir en base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
    
    def _create_3d_model_image(self, chargeability_volume: np.ndarray, method: str) -> str:
        """Créer une image de modèle 3D."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Créer une grille 3D
        x = np.linspace(0, 100, 32)
        y = np.linspace(0, 100, 32)
        z = np.linspace(0, 50, 32)
        
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Créer la visualisation 3D
        scatter = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), 
                           c=chargeability_volume.flatten(), cmap='plasma', alpha=0.6)
        
        ax.set_xlabel('Distance X (m)')
        ax.set_ylabel('Distance Y (m)')
        ax.set_zlabel('Profondeur (m)')
        ax.set_title(f'Modèle 3D de Chargeabilité - {method.upper()}')
        
        plt.colorbar(scatter, label='Chargeabilité (mV/V)')
        
        # Convertir en base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"


def create_sample_csv_data(n_samples: int = 5, n_features: int = 4) -> np.ndarray:
    """
    Créer des données CSV d'exemple pour tester les générateurs.
    
    Args:
        n_samples: Nombre d'échantillons
        n_features: Nombre de caractéristiques (résistivité, chargeabilité, x, y)
        
    Returns:
        Données CSV simulées
    """
    # Données géophysiques simulées
    resistivity = np.random.uniform(10, 1000, n_samples)  # Ω⋅m
    chargeability = np.random.uniform(0, 50, n_samples)   # mV/V
    x_coord = np.random.uniform(0, 100, n_samples)        # m
    y_coord = np.random.uniform(0, 100, n_samples)        # m
    
    csv_data = np.column_stack([resistivity, chargeability, x_coord, y_coord])
    
    logger.info(f"Données CSV d'exemple créées: {csv_data.shape}")
    return csv_data.astype(np.float32)


def main():
    """Test des générateurs géophysiques."""
    logger.info("🚀 TEST DES GÉNÉRATEURS GÉOPHYSIQUES")
    logger.info("=" * 60)
    
    try:
        # Créer le générateur
        generator = GeophysicalImageGenerator()
        
        # Créer des données d'exemple
        csv_data = create_sample_csv_data(n_samples=3)
        
        # Générer les pseudo-sections 2D
        logger.info("🖼️ Génération des pseudo-sections 2D...")
        pseudo_sections = generator.generate_pseudo_sections(csv_data, method="pole-dipole")
        
        # Générer les modèles 3D
        logger.info("🌍 Génération des modèles 3D...")
        models_3d = generator.generate_3d_models(csv_data, method="pole-dipole")
        
        # Afficher les résultats
        logger.info(f"✅ Génération terminée:")
        logger.info(f"   - Pseudo-sections 2D: {len(pseudo_sections)}")
        logger.info(f"   - Modèles 3D: {len(models_3d)}")
        
        logger.info("🎉 TEST TERMINÉ AVEC SUCCÈS!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
