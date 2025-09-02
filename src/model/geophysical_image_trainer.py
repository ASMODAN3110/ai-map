"""
Trainer Étendu pour Modèles Hybrides Images + Données Géophysiques
==================================================================

Ce module étend le GeophysicalTrainer existant pour gérer l'entraînement
de modèles hybrides qui combinent images et données géophysiques.

Classes:
- GeophysicalImageTrainer: Trainer principal pour modèles hybrides
- ImageDataLoader: Gestionnaire de données d'images
- HybridTrainingCallback: Callbacks d'entraînement spécialisés
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import os
from datetime import datetime

# Import des modules existants
from src.model.geophysical_trainer import GeophysicalTrainer
from src.data.image_processor import GeophysicalImageProcessor, ImageAugmenter
from src.model.geophysical_hybrid_net import GeophysicalHybridNet

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageDataLoader:
    """
    Gestionnaire de données d'images géophysiques.
    
    Cette classe gère le chargement, le prétraitement et l'organisation
    des données d'images pour l'entraînement.
    """
    
    def __init__(self, image_processor: GeophysicalImageProcessor,
                 augmenter: Optional[ImageAugmenter] = None):
        """
        Initialiser le gestionnaire de données d'images.
        
        Args:
            image_processor (GeophysicalImageProcessor): Processeur d'images
            augmenter (ImageAugmenter, optional): Augmenteur d'images
        """
        self.image_processor = image_processor
        self.augmenter = augmenter or ImageAugmenter()
        self.processed_images = {}
        self.processed_features = {}
        
        logger.info("Gestionnaire de données d'images initialisé")
    
    def load_and_process_images(self, image_paths: List[str], 
                               augmentations: Optional[List[str]] = None,
                               num_augmentations: int = 0) -> torch.Tensor:
        """
        Charger et traiter un lot d'images.
        
        Args:
            image_paths (list): Liste des chemins d'images
            augmentations (list, optional): Techniques d'augmentation
            num_augmentations (int): Nombre d'augmentations par image
            
        Returns:
            torch.Tensor: Tenseur batch d'images traitées
        """
        processed_images = []
        
        for image_path in image_paths:
            try:
                # Traiter l'image
                image_tensor = self.image_processor.process_image(image_path)
                
                # Ajouter l'image originale
                processed_images.append(image_tensor)
                
                # Augmenter si demandé
                if augmentations and num_augmentations > 0:
                    # Charger l'image PIL pour l'augmentation
                    from PIL import Image
                    pil_image = Image.open(image_path).convert('RGB')
                    
                    # Générer des augmentations
                    augmented_images = self.augmenter.augment_image(
                        pil_image, augmentations, num_augmentations
                    )
                    
                    # Traiter les images augmentées
                    for aug_image in augmented_images[1:]:  # Exclure l'image originale
                        # Convertir PIL vers tensor
                        aug_tensor = self.image_processor.transform(aug_image).unsqueeze(0)
                        processed_images.append(aug_tensor)
                
            except Exception as e:
                logger.warning(f"Erreur lors du traitement de {image_path}: {e}")
                continue
        
        if not processed_images:
            raise ValueError("Aucune image n'a pu être traitée")
        
        # Concaténer en batch
        batch_tensor = torch.cat(processed_images, dim=0)
        
        logger.info(f"Images traitées: {len(processed_images)} images, forme: {batch_tensor.shape}")
        return batch_tensor
    
    def extract_features_batch(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Extraire des features d'un lot d'images.
        
        Args:
            image_paths (list): Liste des chemins d'images
            
        Returns:
            dict: Dictionnaire des features extraites
        """
        features_batch = {
            'mean_intensities': [],
            'gradient_magnitudes': [],
            'histograms': [],
            'image_sizes': []
        }
        
        for image_path in image_paths:
            try:
                features = self.image_processor.extract_geophysical_features(image_path)
                
                features_batch['mean_intensities'].append(features['mean_intensity'])
                features_batch['gradient_magnitudes'].append(features['gradient_magnitude'])
                features_batch['histograms'].append(features['histogram'])
                features_batch['image_sizes'].append(features['image_size'])
                
            except Exception as e:
                logger.warning(f"Erreur lors de l'extraction de features de {image_path}: {e}")
                continue
        
        # Convertir en arrays numpy
        for key in features_batch:
            if key == 'histograms':
                features_batch[key] = np.array(features_batch[key])
            else:
                features_batch[key] = np.array(features_batch[key])
        
        return features_batch
    
    def create_image_dataset(self, image_paths: List[str], 
                           labels: List[int],
                           augmentations: Optional[List[str]] = None,
                           num_augmentations: int = 0) -> TensorDataset:
        """
        Créer un dataset PyTorch pour les images.
        
        Args:
            image_paths (list): Chemins des images
            labels (list): Labels correspondants
            augmentations (list, optional): Techniques d'augmentation
            num_augmentations (int): Nombre d'augmentations
            
        Returns:
            TensorDataset: Dataset PyTorch
        """
        # Traiter les images
        image_tensors = self.load_and_process_images(
            image_paths, augmentations, num_augmentations
        )
        
        # Étendre les labels pour les augmentations
        extended_labels = []
        for i, label in enumerate(labels):
            extended_labels.append(label)  # Image originale
            if augmentations and num_augmentations > 0:
                extended_labels.extend([label] * num_augmentations)  # Images augmentées
        
        # Convertir en tenseurs
        label_tensors = torch.tensor(extended_labels, dtype=torch.long)
        
        # Créer le dataset
        dataset = TensorDataset(image_tensors, label_tensors)
        
        logger.info(f"Dataset d'images créé: {len(dataset)} échantillons")
        return dataset


class HybridTrainingCallback:
    """
    Callbacks d'entraînement pour modèles hybrides.
    
    Cette classe fournit des callbacks spécialisés pour l'entraînement
    de modèles hybrides images + données géophysiques.
    """
    
    def __init__(self, save_dir: str = "artifacts/models/hybrid"):
        """
        Initialiser les callbacks.
        
        Args:
            save_dir (str): Dossier de sauvegarde des modèles
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rate': []
        }
        
        logger.info(f"Callbacks d'entraînement hybride initialisés: {save_dir}")
    
    def on_epoch_end(self, epoch: int, model: GeophysicalHybridNet,
                     train_loss: float, val_loss: float,
                     train_acc: float, val_acc: float,
                     optimizer: optim.Optimizer) -> Dict[str, Any]:
        """
        Callback appelé à la fin de chaque epoch.
        
        Args:
            epoch (int): Numéro de l'epoch
            model (GeophysicalHybridNet): Modèle en cours d'entraînement
            train_loss (float): Loss d'entraînement
            val_loss (float): Loss de validation
            train_acc (float): Accuracy d'entraînement
            val_acc (float): Accuracy de validation
            optimizer (optim.Optimizer): Optimiseur
            
        Returns:
            dict: Informations sur l'epoch
        """
        # Sauvegarder l'historique
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['train_acc'].append(train_acc)
        self.training_history['val_acc'].append(val_acc)
        self.training_history['learning_rate'].append(
            optimizer.param_groups[0]['lr']
        )
        
        # Vérifier si c'est le meilleur modèle
        is_best_loss = val_loss < self.best_val_loss
        is_best_acc = val_acc > self.best_val_acc
        
        if is_best_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            
            # Sauvegarder le meilleur modèle (loss)
            best_loss_path = os.path.join(self.save_dir, "best_loss_model.pth")
            self._save_model(model, best_loss_path, epoch, val_loss, val_acc)
            logger.info(f"Meilleur modèle (loss) sauvegardé: {best_loss_path}")
        
        if is_best_acc:
            self.best_val_acc = val_acc
            
            # Sauvegarder le meilleur modèle (accuracy)
            best_acc_path = os.path.join(self.save_dir, "best_acc_model.pth")
            self._save_model(model, best_acc_path, epoch, val_loss, val_acc)
            logger.info(f"Meilleur modèle (accuracy) sauvegardé: {best_acc_path}")
        
        # Early stopping
        if not is_best_loss:
            self.patience_counter += 1
        
        epoch_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'is_best_loss': is_best_loss,
            'is_best_acc': is_best_acc,
            'patience_counter': self.patience_counter
        }
        
        return epoch_info
    
    def _save_model(self, model: GeophysicalHybridNet, filepath: str,
                   epoch: int, val_loss: float, val_acc: float):
        """Sauvegarder un modèle avec métadonnées."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'model_config': {
                'num_classes': model.num_classes,
                'image_model': model.image_model,
                'fusion_method': model.fusion_method
            },
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }, filepath)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Obtenir un résumé de l'entraînement."""
        return {
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'total_epochs': len(self.training_history['train_loss']),
            'training_history': self.training_history
        }


class GeophysicalImageTrainer(GeophysicalTrainer):
    """
    Trainer étendu pour modèles hybrides images + données géophysiques.
    
    Cette classe étend le GeophysicalTrainer existant pour gérer
    l'entraînement de modèles qui combinent images et données géophysiques.
    """
    
    def __init__(self, augmenter, device="auto"):
        """
        Initialiser le trainer d'images.
        
        Args:
            augmenter: Augmenteur de données existant
            device (str): Device pour l'entraînement
        """
        super().__init__(augmenter, device)
        
        # Initialiser les composants d'images
        self.image_processor = GeophysicalImageProcessor()
        self.image_data_loader = ImageDataLoader(self.image_processor)
        self.hybrid_callbacks = HybridTrainingCallback()
        
        logger.info("Trainer d'images géophysiques initialisé")
    
    def prepare_hybrid_data(self, image_paths: List[str], geo_data: List[List[float]],
                           labels: List[int], test_size: float = 0.2,
                           augmentations: Optional[List[str]] = None,
                           num_augmentations: int = 0) -> Tuple[DataLoader, DataLoader]:
        """
        Préparer les données hybrides (images + géophysiques).
        
        Args:
            image_paths (list): Chemins des images
            geo_data (list): Données géophysiques (listes de features)
            labels (list): Labels de classification
            test_size (float): Proportion de données de test
            augmentations (list, optional): Techniques d'augmentation d'images
            num_augmentations (int): Nombre d'augmentations par image
            
        Returns:
            tuple: (train_loader, val_loader)
        """
        from sklearn.model_selection import train_test_split
        
        # Split train/validation
        (X_img_train, X_img_val, 
         X_geo_train, X_geo_val, 
         y_train, y_val) = train_test_split(
            image_paths, geo_data, labels,
            test_size=test_size, random_state=42, stratify=labels
        )
        
        # Créer les datasets d'images
        train_img_dataset = self.image_data_loader.create_image_dataset(
            X_img_train, y_train, augmentations, num_augmentations
        )
        val_img_dataset = self.image_data_loader.create_image_dataset(
            X_img_val, y_val, augmentations, 0  # Pas d'augmentation pour la validation
        )
        
        # Créer les datasets géophysiques
        train_geo_dataset = TensorDataset(
            torch.tensor(X_geo_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_geo_dataset = TensorDataset(
            torch.tensor(X_geo_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        
        # Créer les DataLoaders
        train_loader = DataLoader(
            list(zip(train_img_dataset, train_geo_dataset)),
            batch_size=32, shuffle=True
        )
        val_loader = DataLoader(
            list(zip(val_img_dataset, val_geo_dataset)),
            batch_size=32, shuffle=False
        )
        
        logger.info(f"Données hybrides préparées:")
        logger.info(f"  - Entraînement: {len(X_img_train)} échantillons")
        logger.info(f"  - Validation: {len(X_img_val)} échantillons")
        logger.info(f"  - Augmentations: {num_augmentations}")
        
        return train_loader, val_loader
    
    def train_hybrid_model(self, model: GeophysicalHybridNet, 
                          train_loader: DataLoader, val_loader: DataLoader,
                          num_epochs: int = 100, learning_rate: float = 0.001,
                          weight_decay: float = 1e-5, patience: int = 10,
                          save_best: bool = True) -> Dict[str, Any]:
        """
        Entraîner le modèle hybride.
        
        Args:
            model (GeophysicalHybridNet): Modèle à entraîner
            train_loader (DataLoader): Loader d'entraînement
            val_loader (DataLoader): Loader de validation
            num_epochs (int): Nombre d'epochs
            learning_rate (float): Taux d'apprentissage
            weight_decay (float): Décroissance des poids
            patience (int): Patience pour early stopping
            save_best (bool): Sauvegarder le meilleur modèle
            
        Returns:
            dict: Historique d'entraînement
        """
        # Déplacer le modèle sur le device
        model = model.to(self.device)
        
        # Optimiseur
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Fonction de loss
        criterion = nn.CrossEntropyLoss()
        
        # Historique
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rate': []
        }
        
        logger.info(f"Début de l'entraînement du modèle hybride:")
        logger.info(f"  - Epochs: {num_epochs}")
        logger.info(f"  - Learning rate: {learning_rate}")
        logger.info(f"  - Device: {self.device}")
        
        for epoch in range(num_epochs):
            # Mode entraînement
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, ((img_data, img_labels), (geo_data, geo_labels)) in enumerate(train_loader):
                # Vérifier que les labels correspondent
                if not torch.equal(img_labels, geo_labels):
                    logger.warning(f"Labels incohérents dans le batch {batch_idx}")
                    continue
                
                # Déplacer sur le device
                img_data = img_data.to(self.device)
                geo_data = geo_data.to(self.device)
                labels = img_labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(img_data, geo_data)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Métriques
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Mode validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for (img_data, img_labels), (geo_data, geo_labels) in val_loader:
                    # Déplacer sur le device
                    img_data = img_data.to(self.device)
                    geo_data = geo_data.to(self.device)
                    labels = img_labels.to(self.device)
                    
                    # Forward pass
                    outputs = model(img_data, geo_data)
                    loss = criterion(outputs, labels)
                    
                    # Métriques
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculer les métriques moyennes
            train_loss_avg = train_loss / len(train_loader)
            val_loss_avg = val_loss / len(val_loader)
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            # Mettre à jour le scheduler
            scheduler.step(val_loss_avg)
            
            # Sauvegarder l'historique
            history['train_loss'].append(train_loss_avg)
            history['val_loss'].append(val_loss_avg)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Callbacks
            epoch_info = self.hybrid_callbacks.on_epoch_end(
                epoch, model, train_loss_avg, val_loss_avg,
                train_acc, val_acc, optimizer
            )
            
            # Log de l'epoch
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                logger.info(f"Epoch {epoch}/{num_epochs}")
                logger.info(f"  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%")
                logger.info(f"  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
                logger.info(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
                logger.info("-" * 50)
            
            # Early stopping
            if epoch_info['patience_counter'] >= patience:
                logger.info(f"Early stopping à l'epoch {epoch}")
                break
        
        # Résumé final
        training_summary = self.hybrid_callbacks.get_training_summary()
        logger.info("Entraînement terminé!")
        logger.info(f"Meilleur val_loss: {training_summary['best_val_loss']:.4f}")
        logger.info(f"Meilleur val_acc: {training_summary['best_val_acc']:.2f}%")
        
        return history
    
    def evaluate_hybrid_model(self, model: GeophysicalHybridNet, 
                            test_loader: DataLoader) -> Dict[str, Any]:
        """
        Évaluer le modèle hybride.
        
        Args:
            model (GeophysicalHybridNet): Modèle à évaluer
            test_loader (DataLoader): Loader de test
            
        Returns:
            dict: Métriques d'évaluation
        """
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for (img_data, img_labels), (geo_data, geo_labels) in test_loader:
                # Déplacer sur le device
                img_data = img_data.to(self.device)
                geo_data = geo_data.to(self.device)
                labels = img_labels.to(self.device)
                
                # Forward pass
                outputs = model(img_data, geo_data)
                loss = criterion(outputs, labels)
                
                # Métriques
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                # Sauvegarder les prédictions
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculer les métriques finales
        test_loss_avg = test_loss / len(test_loader)
        test_acc = 100 * test_correct / test_total
        
        # Métriques détaillées
        from sklearn.metrics import classification_report, confusion_matrix
        
        metrics = {
            'test_loss': test_loss_avg,
            'test_accuracy': test_acc,
            'classification_report': classification_report(all_labels, all_predictions),
            'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist(),
            'total_samples': test_total,
            'correct_predictions': test_correct
        }
        
        logger.info(f"Évaluation terminée:")
        logger.info(f"  Test Loss: {test_loss_avg:.4f}")
        logger.info(f"  Test Accuracy: {test_acc:.2f}%")
        logger.info(f"  Correct: {test_correct}/{test_total}")
        
        return metrics


# Fonctions utilitaires
def create_hybrid_trainer(augmenter, device="auto") -> GeophysicalImageTrainer:
    """Créer un trainer hybride avec configuration par défaut."""
    return GeophysicalImageTrainer(augmenter, device)


def train_hybrid_model_from_scratch(image_paths: List[str], geo_data: List[List[float]],
                                  labels: List[int], num_classes: int = 2,
                                  num_epochs: int = 100, **kwargs) -> Tuple[GeophysicalHybridNet, Dict[str, Any]]:
    """
    Entraîner un modèle hybride complet depuis le début.
    
    Args:
        image_paths (list): Chemins des images
        geo_data (list): Données géophysiques
        labels (list): Labels de classification
        num_classes (int): Nombre de classes
        num_epochs (int): Nombre d'epochs
        **kwargs: Arguments supplémentaires
        
    Returns:
        tuple: (modèle entraîné, historique d'entraînement)
    """
    # Créer le trainer
    from src.preprocessor.data_augmenter import GeophysicalDataAugmenter
    augmenter = GeophysicalDataAugmenter()
    trainer = create_hybrid_trainer(augmenter)
    
    # Créer le modèle
    model = GeophysicalHybridNet(num_classes=num_classes, **kwargs)
    
    # Préparer les données
    train_loader, val_loader = trainer.prepare_hybrid_data(
        image_paths, geo_data, labels
    )
    
    # Entraîner le modèle
    history = trainer.train_hybrid_model(
        model, train_loader, val_loader, num_epochs=num_epochs
    )
    
    return model, history


if __name__ == "__main__":
    # Test du trainer
    from src.preprocessor.data_augmenter import GeophysicalDataAugmenter
    
    augmenter = GeophysicalDataAugmenter()
    trainer = GeophysicalImageTrainer(augmenter)
    
    print("✅ Trainer d'images géophysiques initialisé avec succès!")
    print(f"Device: {trainer.device}")
    print(f"Image processor: {trainer.image_processor}")
    print(f"Callbacks: {trainer.hybrid_callbacks}")
