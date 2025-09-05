#!/usr/bin/env python3
"""
Module d'entraînement pour les modèles géophysiques

Ce module fournit des fonctionnalités d'entraînement spécialisées pour les données géophysiques,
en utilisant l'augmenteur de données pour améliorer la robustesse des modèles.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

from src.preprocessor.data_augmenter import GeophysicalDataAugmenter
from src.utils.logger import logger


class GeophysicalCNN2D(nn.Module):
    """
    Réseau de neurones convolutif 2D spécialisé pour les données géophysiques.
    
    Architecture optimisée pour les grilles de résistivité et chargeabilité.
    """
    
    def __init__(self, input_channels: int = 4, num_classes: int = 2, 
                 grid_size: int = 64, dropout_rate: float = 0.3):
        """
        Initialiser le modèle CNN 2D.
        
        Args:
            input_channels: Nombre de canaux d'entrée (résistivité, chargeabilité, x, y)
            num_classes: Nombre de classes de sortie
            grid_size: Taille de la grille d'entrée (supposée carrée)
            dropout_rate: Taux de dropout pour la régularisation
        """
        super(GeophysicalCNN2D, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.dropout_rate = dropout_rate
        
        # Calcul de la taille de sortie après convolutions
        self.feature_size = self._calculate_feature_size()
        
        # Couches de convolution
        self.conv_layers = nn.Sequential(
            # Première couche de convolution
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Deuxième couche de convolution
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Troisième couche de convolution
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Quatrième couche de convolution
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Couches fully connected
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * self.feature_size * self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialisation des poids
        self._initialize_weights()
        
        logger.info(f"CNN 2D géophysique initialisé: {input_channels} canaux, {grid_size}x{grid_size}, {num_classes} classes")
    
    def _calculate_feature_size(self) -> int:
        """Calculer la taille des features après les couches de convolution."""
        size = self.grid_size
        # Après 4 couches de MaxPool2d(2)
        for _ in range(4):
            size = size // 2
        return size
    
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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du modèle.
        
        Args:
            x: Tenseur d'entrée de forme (batch_size, channels, height, width)
            
        Returns:
            Tenseur de sortie de forme (batch_size, num_classes)
        """
        # Vérification de la forme d'entrée
        if x.shape[1:] != (self.input_channels, self.grid_size, self.grid_size):
            raise ValueError(f"Forme d'entrée attendue: (batch_size, {self.input_channels}, {self.grid_size}, {self.grid_size})")
        
        # Couches de convolution
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Couches fully connected
        x = self.fc_layers(x)
        
        return x


class GeophysicalCNN3D(nn.Module):
    """
    Réseau de neurones convolutif 3D pour les volumes géophysiques.
    
    Architecture optimisée pour les volumes de données 3D.
    """
    
    def __init__(self, input_channels: int = 4, num_classes: int = 2, 
                 volume_size: int = 32, dropout_rate: float = 0.3):
        """
        Initialiser le modèle CNN 3D.
        
        Args:
            input_channels: Nombre de canaux d'entrée
            num_classes: Nombre de classes de sortie
            volume_size: Taille du volume d'entrée (supposé cubique)
            dropout_rate: Taux de dropout pour la régularisation
        """
        super(GeophysicalCNN3D, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.volume_size = volume_size
        self.dropout_rate = dropout_rate
        
        # Calcul de la taille de sortie après convolutions
        self.feature_size = self._calculate_feature_size()
        
        # Couches de convolution 3D
        self.conv_layers = nn.Sequential(
            # Première couche de convolution 3D
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(dropout_rate),
            
            # Deuxième couche de convolution 3D
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(dropout_rate),
            
            # Troisième couche de convolution 3D
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(dropout_rate)
        )
        
        # Couches fully connected
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * self.feature_size * self.feature_size * self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialisation des poids
        self._initialize_weights()
        
        logger.info(f"CNN 3D géophysique initialisé: {input_channels} canaux, {volume_size}x{volume_size}x{volume_size}, {num_classes} classes")
    
    def _calculate_feature_size(self) -> int:
        """Calculer la taille des features après les couches de convolution."""
        size = self.volume_size
        # Après 3 couches de MaxPool3d(2)
        for _ in range(3):
            size = size // 2
        return size
    
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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du modèle.
        
        Args:
            x: Tenseur d'entrée de forme (batch_size, channels, depth, height, width)
            
        Returns:
            Tenseur de sortie de forme (batch_size, num_classes)
        """
        # Vérification de la forme d'entrée
        if x.shape[1:] != (self.input_channels, self.volume_size, self.volume_size, self.volume_size):
            raise ValueError(f"Forme d'entrée attendue: (batch_size, {self.input_channels}, {self.volume_size}, {self.volume_size}, {self.volume_size})")
        
        # Couches de convolution 3D
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Couches fully connected
        x = self.fc_layers(x)
        
        return x


class GeophysicalDataFrameNet(nn.Module):
    """
    Réseau de neurones pour les DataFrames géophysiques.
    
    Architecture optimisée pour les données tabulaires géophysiques.
    """
    
    def __init__(self, input_features: int, num_classes: int = 2, 
                 hidden_layers: List[int] = [256, 128, 64], dropout_rate: float = 0.3):
        """
        Initialiser le modèle pour DataFrames.
        
        Args:
            input_features: Nombre de features d'entrée
            num_classes: Nombre de classes de sortie
            hidden_layers: Liste des tailles des couches cachées
            dropout_rate: Taux de dropout pour la régularisation
        """
        super(GeophysicalDataFrameNet, self).__init__()
        
        self.input_features = input_features
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        
        # Construction des couches
        layers = []
        prev_size = input_features
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Couche de sortie
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialisation des poids
        self._initialize_weights()
        
        logger.info(f"DataFrameNet géophysique initialisé: {input_features} features, {hidden_layers}, {num_classes} classes")
    
    def _initialize_weights(self):
        """Initialiser les poids du modèle."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du modèle.
        
        Args:
            x: Tenseur d'entrée de forme (batch_size, input_features)
            
        Returns:
            Tenseur de sortie de forme (batch_size, num_classes)
        """
        return self.network(x)


class GeophysicalTrainer:
    """
    Entraîneur spécialisé pour les modèles géophysiques.
    
    Utilise l'augmenteur de données pour améliorer la robustesse des modèles.
    """
    
    def __init__(self, augmenter: GeophysicalDataAugmenter, device: str = "auto"):
        """
        Initialiser l'entraîneur.
        
        Args:
            augmenter: Instance de l'augmenteur de données
            device: Device pour l'entraînement ("auto", "cpu", "cuda")
        """
        self.augmenter = augmenter
        
        # Configuration du device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Historique d'entraînement
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "epochs": []
        }
        
        logger.info(f"GeophysicalTrainer initialisé sur device: {self.device}")
    
    def prepare_data_2d(self, grids: List[np.ndarray], labels: List[int], 
                        augmentations: List[str] = None, num_augmentations: int = 5,
                        test_size: float = 0.2, random_state: int = 42) -> Tuple[DataLoader, DataLoader]:
        """
        Préparer les données 2D avec augmentation.
        
        Args:
            grids: Liste des grilles 2D
            labels: Liste des labels correspondants
            augmentations: Techniques d'augmentation à appliquer
            num_augmentations: Nombre d'augmentations par grille
            test_size: Proportion des données de test
            random_state: Seed pour la reproductibilité
            
        Returns:
            Tuple de DataLoaders (train, validation)
        """
        # Validation des entrées
        if not grids or not labels:
            raise ValueError("Les listes grids et labels ne peuvent pas être vides")
        if len(grids) != len(labels):
            raise ValueError("Le nombre de grilles doit correspondre au nombre de labels")
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size doit être entre 0 et 1")
        if num_augmentations < 0:
            raise ValueError("num_augmentations doit être >= 0")
        
        if augmentations is None:
            augmentations = ["rotation", "flip_horizontal", "gaussian_noise"]
        
        # Valider les augmentations
        self.validate_augmentations_for_data_type(augmentations, "2d")
        
        # Augmenter les données
        augmented_grids = []
        augmented_labels = []
        
        for grid, label in zip(grids, labels):
            # Ajouter la grille originale
            augmented_grids.append(grid)
            augmented_labels.append(label)
            
            # Générer des augmentations
            if num_augmentations > 0:
                augmented = self.augmenter.augment_2d_grid(
                    grid, augmentations, num_augmentations
                )
                augmented_grids.extend(augmented)
                augmented_labels.extend([label] * len(augmented))
        
        # Convertir en tenseurs PyTorch
        X = torch.FloatTensor(np.array(augmented_grids))
        y = torch.LongTensor(augmented_labels)
        
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Créer les datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # Créer les DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        logger.info(f"Données 2D préparées: {len(X_train)} train, {len(X_val)} validation")
        return train_loader, val_loader
    
    def prepare_data_3d(self, volumes: List[np.ndarray], labels: List[int],
                        augmentations: List[str] = None, num_augmentations: int = 3,
                        test_size: float = 0.2, random_state: int = 42) -> Tuple[DataLoader, DataLoader]:
        """
        Préparer les données 3D avec augmentation.
        
        Args:
            volumes: Liste des volumes 3D
            labels: Liste des labels correspondants
            augmentations: Techniques d'augmentation à appliquer
            num_augmentations: Nombre d'augmentations par volume
            test_size: Proportion des données de test
            random_state: Seed pour la reproductibilité
            
        Returns:
            Tuple de DataLoaders (train, validation)
        """
        # Validation des entrées
        if not volumes or not labels:
            raise ValueError("Les listes volumes et labels ne peuvent pas être vides")
        if len(volumes) != len(labels):
            raise ValueError("Le nombre de volumes doit correspondre au nombre de labels")
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size doit être entre 0 et 1")
        if num_augmentations < 0:
            raise ValueError("num_augmentations doit être >= 0")
        
        if augmentations is None:
            augmentations = ["rotation", "gaussian_noise"]
        
        # Valider les augmentations
        self.validate_augmentations_for_data_type(augmentations, "3d")
        
        # Augmenter les données
        augmented_volumes = []
        augmented_labels = []
        
        for volume, label in zip(volumes, labels):
            # Ajouter le volume original
            augmented_volumes.append(volume)
            augmented_labels.append(label)
            
            # Générer des augmentations
            if num_augmentations > 0:
                augmented = self.augmenter.augment_3d_volume(
                    volume, augmentations, num_augmentations
                )
                augmented_volumes.extend(augmented)
                augmented_labels.extend([label] * len(augmented))
        
        # Convertir en tenseurs PyTorch
        X = torch.FloatTensor(np.array(augmented_volumes))
        y = torch.LongTensor(augmented_labels)
        
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Créer les datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # Créer les DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Batch size plus petit pour 3D
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        logger.info(f"Données 3D préparées: {len(X_train)} train, {len(X_val)} validation")
        return train_loader, val_loader
    
    def prepare_data_dataframe(self, dataframes: List[pd.DataFrame], labels: List[int],
                              augmentations: List[str] = None, num_augmentations: int = 5,
                              test_size: float = 0.2, random_state: int = 42) -> Tuple[DataLoader, DataLoader]:
        """
        Préparer les données DataFrame avec augmentation.
        
        Args:
            dataframes: Liste des DataFrames
            labels: Liste des labels correspondants
            augmentations: Techniques d'augmentation à appliquer
            num_augmentations: Nombre d'augmentations par DataFrame
            test_size: Proportion des données de test
            random_state: Seed pour la reproductibilité
            
        Returns:
            Tuple de DataLoaders (train, validation)
        """
        # Validation des entrées
        if not dataframes or not labels:
            raise ValueError("Les listes dataframes et labels ne peuvent pas être vides")
        if len(dataframes) != len(labels):
            raise ValueError("Le nombre de dataframes doit correspondre au nombre de labels")
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size doit être entre 0 et 1")
        if num_augmentations < 0:
            raise ValueError("num_augmentations doit être >= 0")
        
        if augmentations is None:
            augmentations = ["gaussian_noise", "value_variation"]
        
        # Valider les augmentations
        self.validate_augmentations_for_data_type(augmentations, "dataframe")
        
        # Augmenter les données
        augmented_dfs = []
        augmented_labels = []
        
        for df, label in zip(dataframes, labels):
            # Ajouter le DataFrame original
            augmented_dfs.append(df)
            augmented_labels.append(label)
            
            # Générer des augmentations
            if num_augmentations > 0:
                augmented = self.augmenter.augment_dataframe(
                    df, augmentations, num_augmentations
                )
                augmented_dfs.extend(augmented)
                augmented_labels.extend([label] * len(augmented))
        
        # Convertir en tenseurs PyTorch
        try:
            # S'assurer que tous les DataFrames ont les mêmes colonnes numériques
            all_samples = []
            all_labels = []
            
            for df, label in zip(augmented_dfs, augmented_labels):
                # Debug: afficher les types de toutes les colonnes
                logger.info(f"Types des colonnes: {df.dtypes.to_dict()}")
                
                # Sélectionner seulement les colonnes numériques
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.empty:
                    # Fallback: essayer de convertir toutes les colonnes en numérique
                    logger.warning("Aucune colonne numérique trouvée, tentative de conversion...")
                    try:
                        numeric_df = df.astype(float)
                    except Exception as e:
                        logger.error(f"Impossible de convertir en float: {e}")
                        raise ValueError(f"Impossible de convertir les colonnes en numérique: {e}")
                
                # Debug: afficher les colonnes sélectionnées
                logger.info(f"Colonnes numériques sélectionnées: {numeric_df.columns.tolist()}")
                logger.info(f"Forme du DataFrame numérique: {numeric_df.shape}")
                
                # Convertir le DataFrame en array numpy et traiter chaque ligne
                sample_values = numeric_df.values.astype(np.float32)
                for i in range(len(sample_values)):
                    all_samples.append(sample_values[i])  # Chaque ligne devient un sample
                    all_labels.append(label)
            
            # Debug: afficher les informations avant conversion
            logger.info(f"Nombre de samples: {len(all_samples)}")
            logger.info(f"Nombre de labels: {len(all_labels)}")
            logger.info(f"Forme du premier sample: {all_samples[0].shape if all_samples else 'N/A'}")
            logger.info(f"Premiers labels: {all_labels[:10]}")
            logger.info(f"Derniers labels: {all_labels[-10:]}")
            logger.info(f"Types des labels: {type(all_labels[0]) if all_labels else 'N/A'}")
            
            # Convertir en tenseurs avec la forme [num_samples, num_features]
            X = torch.FloatTensor(np.array(all_samples))
            y = torch.LongTensor(all_labels)
            
            # Vérification finale des formes
            if len(X.shape) != 2:
                raise ValueError(f"X doit avoir 2 dimensions, mais a {len(X.shape)} dimensions: {X.shape}")
            if len(y.shape) != 1:
                raise ValueError(f"y doit avoir 1 dimension, mais a {len(y.shape)} dimensions: {y.shape}")
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"X et y doivent avoir le même nombre de samples: X={X.shape[0]}, y={y.shape[0]}")
            
            # Debug: afficher les formes après conversion
            logger.info(f"Forme de X après conversion: {X.shape}")
            logger.info(f"Forme de y après conversion: {y.shape}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la conversion des DataFrames: {e}")
            raise ValueError(f"Impossible de convertir les DataFrames en tenseurs: {e}")
        
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Créer les datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # Créer les DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        logger.info(f"Données DataFrame préparées: {len(X_train)} train, {len(X_val)} validation")
        return train_loader, val_loader
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                   num_epochs: int = 100, learning_rate: float = 0.001, 
                   weight_decay: float = 1e-5, patience: int = 10) -> Dict[str, List[float]]:
        """
        Entraîner le modèle.
        
        Args:
            model: Modèle PyTorch à entraîner
            train_loader: DataLoader pour l'entraînement
            val_loader: DataLoader pour la validation
            num_epochs: Nombre d'époques d'entraînement
            learning_rate: Taux d'apprentissage
            weight_decay: Régularisation L2
            patience: Nombre d'époques sans amélioration avant early stopping
            
        Returns:
            Historique d'entraînement
        """
        # Validation des paramètres
        if num_epochs < 0:
            raise ValueError("num_epochs doit être >= 0")
        if learning_rate <= 0:
            raise ValueError("learning_rate doit être > 0")
        if patience < 0:
            raise ValueError("patience doit être >= 0")
        
        model = model.to(self.device)
        
        # Critère et optimiseur
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Début de l'entraînement sur {num_epochs} époques")
        
        for epoch in range(num_epochs):
            # Mode entraînement
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
            
            # Mode évaluation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            # Calcul des métriques
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_accuracy = 100. * train_correct / train_total
            val_accuracy = 100. * val_correct / val_total
            
            # Mise à jour du scheduler
            scheduler.step(avg_val_loss)
            
            # Sauvegarde de l'historique
            self.training_history["train_loss"].append(avg_train_loss)
            self.training_history["val_loss"].append(avg_val_loss)
            self.training_history["train_accuracy"].append(train_accuracy)
            self.training_history["val_accuracy"].append(val_accuracy)
            self.training_history["epochs"].append(epoch)
            
            # Log des métriques
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                          f"Val Acc: {val_accuracy:.2f}%")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                torch.save(model.state_dict(), "best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping à l'époque {epoch}")
                    break
        
        logger.info("Entraînement terminé")
        return self.training_history
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """
        Évaluer le modèle sur des données de test.
        
        Args:
            model: Modèle entraîné
            test_loader: DataLoader pour les données de test
            
        Returns:
            Dictionnaire des métriques d'évaluation
        """
        model = model.to(self.device)
        model.eval()
        
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_targets = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calcul des métriques
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100. * test_correct / test_total
        
        # Métriques supplémentaires
        from sklearn.metrics import classification_report, confusion_matrix
        
        metrics = {
            "test_loss": avg_test_loss,
            "test_accuracy": test_accuracy,
            "classification_report": classification_report(all_targets, all_predictions),
            "confusion_matrix": confusion_matrix(all_targets, all_predictions).tolist()
        }
        
        logger.info(f"Évaluation terminée - Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
        return metrics
    
    def plot_training_history(self, save_path: str = None):
        """
        Tracer l'historique d'entraînement.
        
        Args:
            save_path: Chemin pour sauvegarder le graphique
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.training_history["epochs"]
        
        # Loss
        ax1.plot(epochs, self.training_history["train_loss"], label="Train Loss")
        ax1.plot(epochs, self.training_history["val_loss"], label="Validation Loss")
        ax1.set_title("Évolution de la Loss")
        ax1.set_xlabel("Époque")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(epochs, self.training_history["train_accuracy"], label="Train Accuracy")
        ax2.plot(epochs, self.training_history["val_accuracy"], label="Validation Accuracy")
        ax2.set_title("Évolution de l'Accuracy")
        ax2.set_xlabel("Époque")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate (si disponible)
        if "learning_rate" in self.training_history:
            ax3.plot(epochs, self.training_history["learning_rate"])
            ax3.set_title("Évolution du Learning Rate")
            ax3.set_xlabel("Époque")
            ax3.set_ylabel("Learning Rate")
            ax3.grid(True)
        
        # Métriques supplémentaires
        ax4.plot(epochs, self.training_history["train_loss"], label="Train Loss", alpha=0.7)
        ax4.plot(epochs, self.training_history["val_loss"], label="Val Loss", alpha=0.7)
        ax4.set_title("Vue d'ensemble")
        ax4.set_xlabel("Époque")
        ax4.set_ylabel("Loss")
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique sauvegardé: {save_path}")
        
        plt.show()
    
    def save_model(self, model: nn.Module, filepath: str):
        """
        Sauvegarder le modèle.
        
        Args:
            model: Modèle à sauvegarder
            filepath: Chemin de sauvegarde
        """
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_history': self.training_history,
            'model_config': {
                'input_channels': getattr(model, 'input_channels', None),
                'num_classes': getattr(model, 'num_classes', None),
                'grid_size': getattr(model, 'grid_size', None),
                'volume_size': getattr(model, 'volume_size', None),
                'input_features': getattr(model, 'input_features', None)
            }
        }, filepath)
        logger.info(f"Modèle sauvegardé: {filepath}")
    
    def load_model(self, model: nn.Module, filepath: str):
        """
        Charger un modèle sauvegardé.
        
        Args:
            model: Modèle à charger
            filepath: Chemin du fichier de sauvegarde
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint['training_history']
        logger.info(f"Modèle chargé: {filepath}")
        return model
    
    def get_augmentation_summary(self) -> Dict:
        """
        Obtenir un résumé des augmentations utilisées pendant l'entraînement.
        
        Returns:
            Résumé des augmentations
        """
        return self.augmenter.get_augmentation_summary()
    
    def validate_augmentations_for_data_type(self, augmentations: List[str], data_type: str) -> bool:
        """
        Valider que les techniques d'augmentation sont compatibles avec le type de données.
        
        Args:
            augmentations: Liste des techniques d'augmentation
            data_type: Type de données ("2d", "3d", "dataframe")
            
        Returns:
            True si les augmentations sont valides
        """
        valid_augmentations = {
            "2d": ["rotation", "flip_horizontal", "flip_vertical", "spatial_shift", 
                   "gaussian_noise", "salt_pepper_noise", "value_variation", "elastic_deformation"],
            "3d": ["rotation", "flip_horizontal", "flip_vertical", "gaussian_noise", "value_variation"],
            "dataframe": ["gaussian_noise", "value_variation", "spatial_jitter", "coordinate_perturbation"]
        }
        
        if data_type not in valid_augmentations:
            raise ValueError(f"Type de données non supporté: {data_type}")
        
        for aug in augmentations:
            if aug not in valid_augmentations[data_type]:
                logger.warning(f"Technique d'augmentation '{aug}' non supportée pour {data_type}")
                return False
        
        return True
    
    def reset_training_history(self):
        """Réinitialiser l'historique d'entraînement."""
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "epochs": []
        }
        logger.info("Historique d'entraînement réinitialisé")


if __name__ == "__main__":
    # Exemple d'utilisation
    print("Module d'entraînement géophysique chargé avec succès!")
    print("Classes disponibles:")
    print("- GeophysicalCNN2D: CNN 2D pour grilles géophysiques")
    print("- GeophysicalCNN3D: CNN 3D pour volumes géophysiques") 
    print("- GeophysicalDataFrameNet: Réseau pour DataFrames géophysiques")
    print("- GeophysicalTrainer: Entraîneur spécialisé géophysique")