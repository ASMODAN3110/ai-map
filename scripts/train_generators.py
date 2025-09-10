#!/usr/bin/env python3
"""
Script d'entraînement pour les modèles générateurs géophysiques.

Ce script entraîne les modèles U-Net 2D et VoxNet 3D conformes au cahier des charges
pour la génération de pseudo-sections 2D, cartes d'iso-résistivité/chargeabilité et modèles 3D.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from backend.model.geophysical_generators import UNet2D, VoxNet3D, GeophysicalDataProcessor
from backend.utils.logger import logger


class GeneratorTrainer:
    """
    Entraîneur pour les modèles générateurs géophysiques.
    """
    
    def __init__(self, device: str = "auto"):
        # Configuration du device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Processeur de données
        self.data_processor = GeophysicalDataProcessor()
        
        # Historique d'entraînement
        self.training_history = {
            "unet_2d": {
                "train_loss": [],
                "val_loss": [],
                "epochs": []
            },
            "voxnet_3d": {
                "train_loss": [],
                "val_loss": [],
                "epochs": []
            }
        }
        
        logger.info(f"GeneratorTrainer initialisé sur device: {self.device}")
    
    def create_synthetic_data(self, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Créer des données synthétiques pour l'entraînement.
        
        Args:
            n_samples: Nombre d'échantillons à générer
            
        Returns:
            Tuple (csv_data, 2d_targets, 3d_targets)
        """
        logger.info(f"Création de {n_samples} échantillons de données synthétiques...")
        
        # Données CSV d'entrée
        resistivity = np.random.uniform(10, 1000, n_samples)  # Ω⋅m
        chargeability = np.random.uniform(0, 50, n_samples)   # mV/V
        x_coord = np.random.uniform(0, 100, n_samples)        # m
        y_coord = np.random.uniform(0, 100, n_samples)        # m
        
        csv_data = np.column_stack([resistivity, chargeability, x_coord, y_coord])
        
        # Cibles 2D (pseudo-sections)
        targets_2d = []
        for i in range(n_samples):
            # Créer une pseudo-section de résistivité
            resistivity_target = self._create_resistivity_target(resistivity[i], x_coord[i], y_coord[i])
            # Créer une pseudo-section de chargeabilité
            chargeability_target = self._create_chargeability_target(chargeability[i], x_coord[i], y_coord[i])
            
            # Combiner en 2 canaux
            target_2d = np.stack([resistivity_target, chargeability_target], axis=0)
            targets_2d.append(target_2d)
        
        targets_2d = np.array(targets_2d)
        
        # Cibles 3D (volumes de chargeabilité)
        targets_3d = []
        for i in range(n_samples):
            # Créer un volume de chargeabilité 3D
            chargeability_volume = self._create_chargeability_volume_3d(chargeability[i], x_coord[i], y_coord[i])
            targets_3d.append(chargeability_volume)
        
        targets_3d = np.array(targets_3d)
        
        logger.info(f"✅ Données synthétiques créées:")
        logger.info(f"   - CSV: {csv_data.shape}")
        logger.info(f"   - Cibles 2D: {targets_2d.shape}")
        logger.info(f"   - Cibles 3D: {targets_3d.shape}")
        
        return csv_data, targets_2d, targets_3d
    
    def _create_resistivity_target(self, resistivity: float, x_coord: float, y_coord: float) -> np.ndarray:
        """Créer une cible de résistivité 2D."""
        # Normaliser la résistivité (log scale)
        log_resistivity = np.log10(max(resistivity, 1.0))
        normalized_resistivity = (log_resistivity - 1.0) / 3.0
        
        # Créer une grille avec variation spatiale
        x = np.linspace(0, 1, 64)
        y = np.linspace(0, 1, 64)
        X, Y = np.meshgrid(x, y)
        
        # Créer une distribution gaussienne centrée sur les coordonnées
        center_x = x_coord / 100.0
        center_y = y_coord / 100.0
        
        # Distance du centre
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # Créer la grille de résistivité
        grid = normalized_resistivity * np.exp(-distance**2 / 0.1)
        
        return grid
    
    def _create_chargeability_target(self, chargeability: float, x_coord: float, y_coord: float) -> np.ndarray:
        """Créer une cible de chargeabilité 2D."""
        # Normaliser la chargeabilité
        normalized_chargeability = chargeability / 50.0
        
        # Créer une grille avec variation spatiale
        x = np.linspace(0, 1, 64)
        y = np.linspace(0, 1, 64)
        X, Y = np.meshgrid(x, y)
        
        # Créer une distribution gaussienne centrée sur les coordonnées
        center_x = x_coord / 100.0
        center_y = y_coord / 100.0
        
        # Distance du centre
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # Créer la grille de chargeabilité
        grid = normalized_chargeability * np.exp(-distance**2 / 0.1)
        
        return grid
    
    def _create_chargeability_volume_3d(self, chargeability: float, x_coord: float, y_coord: float) -> np.ndarray:
        """Créer une cible de chargeabilité 3D."""
        # Normaliser la chargeabilité
        normalized_chargeability = chargeability / 50.0
        
        # Créer un volume 3D avec variation spatiale
        x = np.linspace(0, 1, 32)
        y = np.linspace(0, 1, 32)
        z = np.linspace(0, 1, 32)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Créer une distribution gaussienne centrée sur les coordonnées
        center_x = x_coord / 100.0
        center_y = y_coord / 100.0
        center_z = 0.5  # Centre en profondeur
        
        # Distance du centre
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2)
        
        # Créer le volume de chargeabilité
        volume = normalized_chargeability * np.exp(-distance**2 / 0.1)
        
        return volume
    
    def prepare_data(self, csv_data: np.ndarray, targets_2d: np.ndarray, targets_3d: np.ndarray,
                    test_size: float = 0.2, random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """
        Préparer les données pour l'entraînement.
        
        Args:
            csv_data: Données CSV d'entrée
            targets_2d: Cibles 2D
            targets_3d: Cibles 3D
            test_size: Proportion des données de test
            random_state: Seed pour la reproductibilité
            
        Returns:
            Tuple de DataLoaders (train_2d, val_2d, train_3d, val_3d)
        """
        logger.info("Préparation des données pour l'entraînement...")
        
        # Traiter les données CSV en grilles 2D et volumes 3D
        grids_2d = self.data_processor.process_csv_to_2d_grid(csv_data)
        volumes_3d = self.data_processor.process_csv_to_3d_volume(csv_data)
        
        # Split train/validation pour 2D
        X_train_2d, X_val_2d, y_train_2d, y_val_2d = train_test_split(
            grids_2d, targets_2d, test_size=test_size, random_state=random_state
        )
        
        # Split train/validation pour 3D
        X_train_3d, X_val_3d, y_train_3d, y_val_3d = train_test_split(
            volumes_3d, targets_3d, test_size=test_size, random_state=random_state
        )
        
        # Créer les datasets
        train_dataset_2d = TensorDataset(X_train_2d, torch.FloatTensor(y_train_2d))
        val_dataset_2d = TensorDataset(X_val_2d, torch.FloatTensor(y_val_2d))
        
        train_dataset_3d = TensorDataset(X_train_3d, torch.FloatTensor(y_train_3d))
        val_dataset_3d = TensorDataset(X_val_3d, torch.FloatTensor(y_val_3d))
        
        # Créer les DataLoaders
        train_loader_2d = DataLoader(train_dataset_2d, batch_size=16, shuffle=True)
        val_loader_2d = DataLoader(val_dataset_2d, batch_size=16, shuffle=False)
        
        train_loader_3d = DataLoader(train_dataset_3d, batch_size=8, shuffle=True)  # Batch size plus petit pour 3D
        val_loader_3d = DataLoader(val_dataset_3d, batch_size=8, shuffle=False)
        
        logger.info(f"✅ Données préparées:")
        logger.info(f"   - Train 2D: {len(train_loader_2d.dataset)} échantillons")
        logger.info(f"   - Val 2D: {len(val_loader_2d.dataset)} échantillons")
        logger.info(f"   - Train 3D: {len(train_loader_3d.dataset)} échantillons")
        logger.info(f"   - Val 3D: {len(val_loader_3d.dataset)} échantillons")
        
        return train_loader_2d, val_loader_2d, train_loader_3d, val_loader_3d
    
    def train_unet_2d(self, train_loader: DataLoader, val_loader: DataLoader,
                     num_epochs: int = 200, learning_rate: float = 1e-4,
                     weight_decay: float = 1e-5, patience: int = 20) -> Dict[str, List[float]]:
        """
        Entraîner le modèle U-Net 2D.
        
        Args:
            train_loader: DataLoader pour l'entraînement
            val_loader: DataLoader pour la validation
            num_epochs: Nombre d'époques d'entraînement
            learning_rate: Taux d'apprentissage
            weight_decay: Régularisation L2
            patience: Nombre d'époques sans amélioration avant early stopping
            
        Returns:
            Historique d'entraînement
        """
        logger.info("🚀 Début de l'entraînement U-Net 2D")
        
        # Créer le modèle
        model = UNet2D().to(self.device)
        
        # Critère et optimiseur
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Mode entraînement
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Mode évaluation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
            
            # Calcul des métriques
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Mise à jour du scheduler
            scheduler.step(avg_val_loss)
            
            # Sauvegarde de l'historique
            self.training_history["unet_2d"]["train_loss"].append(avg_train_loss)
            self.training_history["unet_2d"]["val_loss"].append(avg_val_loss)
            self.training_history["unet_2d"]["epochs"].append(epoch)
            
            # Log des métriques
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss
                }, "artifacts/models/unet_2d_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping à l'époque {epoch}")
                    break
        
        logger.info("✅ Entraînement U-Net 2D terminé")
        return self.training_history["unet_2d"]
    
    def train_voxnet_3d(self, train_loader: DataLoader, val_loader: DataLoader,
                       num_epochs: int = 200, learning_rate: float = 1e-4,
                       weight_decay: float = 1e-5, patience: int = 20) -> Dict[str, List[float]]:
        """
        Entraîner le modèle VoxNet 3D.
        
        Args:
            train_loader: DataLoader pour l'entraînement
            val_loader: DataLoader pour la validation
            num_epochs: Nombre d'époques d'entraînement
            learning_rate: Taux d'apprentissage
            weight_decay: Régularisation L2
            patience: Nombre d'époques sans amélioration avant early stopping
            
        Returns:
            Historique d'entraînement
        """
        logger.info("🚀 Début de l'entraînement VoxNet 3D")
        
        # Créer le modèle
        model = VoxNet3D().to(self.device)
        
        # Critère et optimiseur
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Mode entraînement
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Mode évaluation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
            
            # Calcul des métriques
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Mise à jour du scheduler
            scheduler.step(avg_val_loss)
            
            # Sauvegarde de l'historique
            self.training_history["voxnet_3d"]["train_loss"].append(avg_train_loss)
            self.training_history["voxnet_3d"]["val_loss"].append(avg_val_loss)
            self.training_history["voxnet_3d"]["epochs"].append(epoch)
            
            # Log des métriques
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss
                }, "artifacts/models/voxnet_3d_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping à l'époque {epoch}")
                    break
        
        logger.info("✅ Entraînement VoxNet 3D terminé")
        return self.training_history["voxnet_3d"]
    
    def plot_training_history(self, save_path: str = None):
        """
        Tracer l'historique d'entraînement.
        
        Args:
            save_path: Chemin pour sauvegarder le graphique
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # U-Net 2D Loss
        epochs_2d = self.training_history["unet_2d"]["epochs"]
        ax1.plot(epochs_2d, self.training_history["unet_2d"]["train_loss"], label="Train Loss")
        ax1.plot(epochs_2d, self.training_history["unet_2d"]["val_loss"], label="Validation Loss")
        ax1.set_title("U-Net 2D - Évolution de la Loss")
        ax1.set_xlabel("Époque")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)
        
        # VoxNet 3D Loss
        epochs_3d = self.training_history["voxnet_3d"]["epochs"]
        ax2.plot(epochs_3d, self.training_history["voxnet_3d"]["train_loss"], label="Train Loss")
        ax2.plot(epochs_3d, self.training_history["voxnet_3d"]["val_loss"], label="Validation Loss")
        ax2.set_title("VoxNet 3D - Évolution de la Loss")
        ax2.set_xlabel("Époque")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)
        
        # Comparaison des modèles
        ax3.plot(epochs_2d, self.training_history["unet_2d"]["val_loss"], label="U-Net 2D Val Loss")
        ax3.plot(epochs_3d, self.training_history["voxnet_3d"]["val_loss"], label="VoxNet 3D Val Loss")
        ax3.set_title("Comparaison des Modèles - Validation Loss")
        ax3.set_xlabel("Époque")
        ax3.set_ylabel("Validation Loss")
        ax3.legend()
        ax3.grid(True)
        
        # Vue d'ensemble
        ax4.plot(epochs_2d, self.training_history["unet_2d"]["train_loss"], label="U-Net 2D Train", alpha=0.7)
        ax4.plot(epochs_2d, self.training_history["unet_2d"]["val_loss"], label="U-Net 2D Val", alpha=0.7)
        ax4.plot(epochs_3d, self.training_history["voxnet_3d"]["train_loss"], label="VoxNet 3D Train", alpha=0.7)
        ax4.plot(epochs_3d, self.training_history["voxnet_3d"]["val_loss"], label="VoxNet 3D Val", alpha=0.7)
        ax4.set_title("Vue d'ensemble - Tous les Modèles")
        ax4.set_xlabel("Époque")
        ax4.set_ylabel("Loss")
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique sauvegardé: {save_path}")
        
        plt.show()


def main():
    """Fonction principale d'entraînement."""
    logger.info("🚀 ENTRAÎNEMENT DES MODÈLES GÉNÉRATEURS GÉOPHYSIQUES")
    logger.info("=" * 70)
    
    try:
        # Créer l'entraîneur
        trainer = GeneratorTrainer()
        
        # Créer les données synthétiques
        csv_data, targets_2d, targets_3d = trainer.create_synthetic_data(n_samples=10000)
        
        # Préparer les données
        train_loader_2d, val_loader_2d, train_loader_3d, val_loader_3d = trainer.prepare_data(
            csv_data, targets_2d, targets_3d
        )
        
        # Entraîner U-Net 2D
        logger.info("🎯 Entraînement U-Net 2D...")
        history_2d = trainer.train_unet_2d(train_loader_2d, val_loader_2d)
        
        # Entraîner VoxNet 3D
        logger.info("🎯 Entraînement VoxNet 3D...")
        history_3d = trainer.train_voxnet_3d(train_loader_3d, val_loader_3d)
        
        # Tracer l'historique
        trainer.plot_training_history("artifacts/training_history.png")
        
        # Sauvegarder l'historique
        with open("artifacts/training_history.json", "w") as f:
            json.dump(trainer.training_history, f, indent=2)
        
        logger.info("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        logger.info(f"✅ Modèles sauvegardés dans artifacts/models/")
        logger.info(f"✅ Historique sauvegardé dans artifacts/")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
