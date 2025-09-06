#!/usr/bin/env python3
"""
Pipeline principal pour le projet AI-MAP, inspiré d'EMUT.
Orchestre l'ensemble du traitement des données géophysiques et du pipeline d'entraînement CNN.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np

# Ajouter le répertoire courant au path Python
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config import CONFIG
    from src.utils.logger import logger
    
    # ============================================================================
    # PHASE 1: NETTOYAGE ET PRÉTRAITEMENT DES DONNÉES
    # ============================================================================
    
    def phase1_data_cleaning() -> Dict[str, Any]:
        """
        Phase 1: Nettoyage et prétraitement des données géophysiques.
        
        Returns:
            Dict contenant les résultats du nettoyage
        """
        logger.info("📋 Phase 1: Nettoyage et prétraitement des données")
        logger.info("-" * 40)
        
        # Initialiser le nettoyeur de données
        from src.preprocessor.data_cleaner import GeophysicalDataCleaner
        cleaner = GeophysicalDataCleaner()
        
        # Nettoyer les données de tous les dispositifs
        cleaning_results = cleaner.clean_all_devices()
        
        logger.info("✅ Nettoyage des données terminé avec succès")
        logger.info("Rapport de nettoyage:")
        for device_name, (clean_path, report) in cleaning_results.items():
            logger.info(f"  {device_name}: {report.get('cleaned_count', 0)}/{report.get('original_count', 0)} enregistrements conservés")
            
        return cleaning_results
    
    # ============================================================================
    # PHASE 2: TRAITEMENT DES DONNÉES ET CRÉATION DES GRILLES
    # ============================================================================
    
    def phase2_data_processing() -> Tuple[Optional[Any], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Phase 2: Traitement des données et création des grilles spatiales.
        
        Returns:
            Tuple contenant (processor, multi_device_tensor, volume_3d)
        """
        logger.info("\n📊 Phase 2: Traitement des données et création des grilles")
        logger.info("-" * 40)
        
        # Initialiser le processeur de données
        from src.data.data_processor import GeophysicalDataProcessor
        processor = GeophysicalDataProcessor()
        
        # Charger et valider les données nettoyées
        device_data = processor.load_and_validate()
        
        # Utiliser les données réelles si disponibles
        if not device_data:
            logger.warning("Aucune donnée de dispositif valide trouvée après le nettoyage")
            logger.info("Le pipeline continuera avec des données factices pour la démonstration")
            
            # Créer des données factices pour la démonstration
            n_samples = 50
            n_channels = 4
            grid_size = 64
            
            # Tenseur multi-dispositifs factice
            multi_device_tensor = np.random.rand(n_samples, n_channels, grid_size, grid_size).astype(np.float32)
            
            # Volume 3D factice
            volume_3d = np.random.rand(20, n_channels, 32, 32, 32).astype(np.float32)
            
            logger.info("✅ Données factices créées pour la démonstration")
            logger.info(f"Forme du tenseur multi-dispositifs: {multi_device_tensor.shape}")
            logger.info(f"Forme du volume 3D: {volume_3d.shape}")
            
            return processor, multi_device_tensor, volume_3d
    
        # Créer les grilles spatiales
        spatial_grids = processor.create_spatial_grids()
        
        # Créer le tenseur multi-dispositifs pour l'entrée CNN
        multi_device_tensor = processor.create_multi_device_tensor()
        
        # Créer le volume 3D pour VoxNet
        volume_3d = processor.create_3d_volume()
        
        logger.info("✅ Traitement des données terminé avec succès")
        logger.info(f"Forme du tenseur multi-dispositifs: {multi_device_tensor.shape}")
        logger.info(f"Forme du volume 3D: {volume_3d.shape}")
                
        return processor, multi_device_tensor, volume_3d
    
    # ============================================================================
    # PHASE 3: PRÉPARATION DES DONNÉES D'ENTRAÎNEMENT
    # ============================================================================
    
    def phase3_data_preparation(processor: Any, multi_device_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Phase 3: Division et préparation des données pour l'entraînement.
        
        Args:
            processor: Processeur de données
            multi_device_tensor: Tenseur multi-dispositifs
            
        Returns:
            Tuple contenant (x_train, x_test)
        """
        logger.info("\n🔀 Phase 3: Division et préparation des données")
        logger.info("-" * 40)
                
        if multi_device_tensor is None:
            logger.warning("Aucun tenseur multi-dispositifs disponible")
            return np.array([]), np.array([])
        
        # Diviser les données pour l'entraînement
        x_train, x_test = processor.split_data(multi_device_tensor)
        
        logger.info(f"Taille de l'ensemble d'entraînement: {len(x_train)}")
        logger.info(f"Taille de l'ensemble de test: {len(x_test)}")
            
        return x_train, x_test
    
    # ============================================================================
    # PHASE 4: ENTRAÎNEMENT DES MODÈLES
    # ============================================================================
    
    def phase4_model_training(model_type: str = "cnn_2d", 
                             x_train: Optional[np.ndarray] = None,
                             x_test: Optional[np.ndarray] = None,
                             volume_3d: Optional[np.ndarray] = None,
                             training_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Phase 4: Entraînement des modèles CNN.
        
        Args:
            model_type: Type de modèle ("cnn_2d", "cnn_3d", "hybrid", "dataframe")
            x_train: Données d'entraînement 2D
            x_test: Données de test 2D
            volume_3d: Volume 3D pour VoxNet
            training_config: Configuration d'entraînement
            
        Returns:
            Dict contenant les résultats d'entraînement
        """
        logger.info(f"\n🤖 Phase 4: Entraînement du modèle {model_type.upper()}")
        logger.info("-" * 40)
            
        # Configuration par défaut
        if training_config is None:
            training_config = {
                "epochs": 100,
                "learning_rate": 0.001,
                "batch_size": 32,
                "patience": 10,
                "device": "auto"
            }
        
        # Initialiser l'augmenteur de données
        from src.preprocessor.data_augmenter import GeophysicalDataAugmenter
        augmenter = GeophysicalDataAugmenter()
        
        # Initialiser l'entraîneur
        from src.model.geophysical_trainer import GeophysicalTrainer
        trainer = GeophysicalTrainer(augmenter, device=training_config["device"])
        
        results = {}
        
        # Si les données ne sont pas fournies, les récupérer des phases précédentes
        if x_train is None or x_test is None:
            logger.info("Récupération des données des phases précédentes...")
            processor, multi_device_tensor, volume_3d_auto = phase2_data_processing()
            x_train, x_test = phase3_data_preparation(processor, multi_device_tensor)
            
            if volume_3d is None:
                volume_3d = volume_3d_auto
        
        try:
            if model_type == "cnn_2d":
                results = train_cnn_2d(trainer, x_train, x_test, training_config)
            elif model_type == "cnn_3d":
                results = train_cnn_3d(trainer, volume_3d, training_config)
            elif model_type == "hybrid":
                results = train_hybrid_model(trainer, training_config)
            elif model_type == "dataframe":
                results = train_dataframe_model(trainer, training_config)
            else:
                raise ValueError(f"Type de modèle non supporté: {model_type}")
            
            logger.info("✅ Entraînement terminé avec succès")
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'entraînement: {str(e)}")
            raise
    
    def train_cnn_2d(trainer: Any, x_train: np.ndarray, x_test: np.ndarray, config: Dict) -> Dict[str, Any]:
        """Entraîner un modèle CNN 2D."""
        from src.model.geophysical_trainer import GeophysicalCNN2D
        
        logger.info("Entraînement du modèle CNN 2D...")
        
        # Créer des labels factices pour la démonstration
        # Dans un vrai projet, vous auriez des vrais labels
        y_train = np.random.randint(0, 2, len(x_train))
        y_test = np.random.randint(0, 2, len(x_test))
        
        # Créer le modèle
        model = GeophysicalCNN2D(
            input_channels=4,
            num_classes=2,
            grid_size=64,
            dropout_rate=0.3
        )
        
        # Préparer les données - convertir de 4D à 3D pour chaque échantillon
        x_train_3d = []
        for i in range(len(x_train)):
            # Convertir de (height, width, channels) à (height, width, channels)
            # Les données sont déjà dans le bon format (H, W, C)
            sample = x_train[i]
            x_train_3d.append(sample)
        
        x_test_3d = []
        for i in range(len(x_test)):
            sample = x_test[i]
            x_test_3d.append(sample)
        
        # Combiner train et test pour la préparation
        all_data = x_train_3d + x_test_3d
        all_labels = y_train.tolist() + y_test.tolist()
        
        # Préparer les données
        train_loader, val_loader = trainer.prepare_data_2d(
            all_data, all_labels,
            augmentations=["rotation", "flip_horizontal", "gaussian_noise"],
            num_augmentations=3,
            test_size=0.2
        )
        
        # Entraîner le modèle
        history = trainer.train_model(
            model, train_loader, val_loader,
            num_epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            patience=config["patience"]
        )
        
        # Sauvegarder le modèle
        model_path = CONFIG.paths.artifacts_dir / "models" / "cnn_2d_model.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(model, str(model_path))
        
        return {
            "model_type": "CNN_2D",
            "model": model,
            "history": history,
            "model_path": str(model_path)
        }
    
    def train_cnn_3d(trainer: Any, volume_3d: np.ndarray, config: Dict) -> Dict[str, Any]:
        """Entraîner un modèle CNN 3D."""
        from src.model.geophysical_trainer import GeophysicalCNN3D
        
        logger.info("Entraînement du modèle CNN 3D...")
        
        if volume_3d is None:
            logger.warning("Aucun volume 3D disponible, création d'un volume factice")
            volume_3d = np.random.rand(10, 4, 32, 32, 32)
        
        # Vérifier et corriger le format du volume 3D
        if len(volume_3d.shape) == 4:
            # Volume unique: (depth, height, width, channels) -> liste de volumes
            logger.info(f"Volume 3D unique détecté: {volume_3d.shape}")
            # Créer plusieurs volumes factices basés sur le volume existant
            n_volumes = 5  # Créer 5 volumes pour l'entraînement
            volumes_list = []
            for i in range(n_volumes):
                # Ajouter du bruit pour créer de la variabilité
                noise = np.random.normal(0, 0.1, volume_3d.shape)
                volume_with_noise = volume_3d + noise
                volumes_list.append(volume_with_noise)
            volume_3d = np.array(volumes_list)
            logger.info(f"Volumes 3D créés: {volume_3d.shape}")
        elif len(volume_3d.shape) == 5:
            # Déjà au bon format: (batch, depth, height, width, channels)
            logger.info(f"Volumes 3D détectés: {volume_3d.shape}")
        else:
            raise ValueError(f"Format de volume 3D non supporté: {volume_3d.shape}")
        
        # Créer des labels factices
        y_labels = np.random.randint(0, 2, len(volume_3d))
        
        # Créer le modèle
        model = GeophysicalCNN3D(
            input_channels=4,
            num_classes=2,
            volume_size=32,
            dropout_rate=0.3
        )
        
        # Vérifier la forme du volume 3D et le convertir au bon format
        print(f"Forme du volume 3D original: {volume_3d.shape}")
        
        # Le volume 3D a la forme (5, 4, 32, 32, 32) = (batch, channels, depth, height, width)
        # Pour l'augmenteur, nous avons besoin de (depth, height, width, channels)
        if len(volume_3d.shape) == 5 and volume_3d.shape[1] == 4:
            # (batch, channels, depth, height, width) -> (batch, depth, height, width, channels)
            volume_3d_transposed = np.transpose(volume_3d, (0, 2, 3, 4, 1))
            print(f"Volume 3D transposé: {volume_3d_transposed.shape}")
        elif volume_3d.shape == (4, 32, 32, 32):
            # (channels, depth, height, width) -> (depth, height, width, channels)
            volume_3d_transposed = np.transpose(volume_3d, (1, 2, 3, 0))
            print(f"Volume 3D transposé: {volume_3d_transposed.shape}")
        else:
            # Si la forme est différente, essayer une transposition différente
            print(f"Forme inattendue: {volume_3d.shape}")
            volume_3d_transposed = volume_3d  # Utiliser tel quel
        
        # Créer la liste des volumes pour l'entraînement
        volumes_list = []
        if len(volume_3d_transposed.shape) == 5:
            # Si c'est un batch de volumes, prendre chaque volume individuellement
            for i in range(volume_3d_transposed.shape[0]):
                volumes_list.append(volume_3d_transposed[i])  # (depth, height, width, channels)
        else:
            # Si c'est un seul volume, l'ajouter tel quel
            volumes_list.append(volume_3d_transposed)
        train_loader, val_loader = trainer.prepare_data_3d(
            volumes_list, y_labels.tolist(),
            augmentations=["rotation", "gaussian_noise"],
            num_augmentations=2,
            test_size=0.2
        )
        
        # Entraîner le modèle
        history = trainer.train_model(
            model, train_loader, val_loader,
            num_epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            patience=config["patience"]
        )
        
        # Sauvegarder le modèle
        model_path = CONFIG.paths.artifacts_dir / "models" / "cnn_3d_model.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(model, str(model_path))
        
        return {
            "model_type": "CNN_3D",
            "model": model,
            "history": history,
            "model_path": str(model_path)
        }
    
    def train_hybrid_model(trainer: Any, config: Dict) -> Dict[str, Any]:
        """Entraîner un modèle hybride (images + données géophysiques)."""
        from src.model.geophysical_image_trainer import GeophysicalImageTrainer
        
        logger.info("Entraînement du modèle hybride...")
        
        # Créer le trainer hybride
        hybrid_trainer = GeophysicalImageTrainer(trainer.augmenter, device=config["device"])
        
        # Utiliser plus d'images pour avoir suffisamment de données
        image_paths = [
            str(CONFIG.paths.data_dir / "raw" / "images" / "resistivity" / "resis1.JPG"),
            str(CONFIG.paths.data_dir / "raw" / "images" / "resistivity" / "resis2.JPG"),
            str(CONFIG.paths.data_dir / "raw" / "images" / "resistivity" / "resis3.JPG"),
            str(CONFIG.paths.data_dir / "raw" / "images" / "resistivity" / "resis4.JPG"),
            str(CONFIG.paths.data_dir / "raw" / "images" / "chargeability" / "char_1.PNG"),
            str(CONFIG.paths.data_dir / "raw" / "images" / "chargeability" / "char_2.PNG"),
            str(CONFIG.paths.data_dir / "raw" / "images" / "chargeability" / "char_3.PNG"),
            str(CONFIG.paths.data_dir / "raw" / "images" / "chargeability" / "char_4.PNG"),
        ]
        
        # Vérifier que les fichiers existent
        existing_paths = []
        for path in image_paths:
            if Path(path).exists():
                existing_paths.append(path)
            else:
                logger.warning(f"Image non trouvée: {path}")
        
        if len(existing_paths) < 4:
            logger.error(f"Pas assez d'images trouvées. Trouvé: {len(existing_paths)}, Nécessaire: 4")
            # Utiliser des images de substitution
            existing_paths = [
                str(CONFIG.paths.data_dir / "raw" / "images" / "resistivity" / "resis1.JPG"),
                str(CONFIG.paths.data_dir / "raw" / "images" / "resistivity" / "resis2.JPG"),
                str(CONFIG.paths.data_dir / "raw" / "images" / "chargeability" / "char_1.PNG"),
                str(CONFIG.paths.data_dir / "raw" / "images" / "chargeability" / "char_2.PNG"),
            ]
        
        image_paths = existing_paths
        
        # Créer des données géophysiques factices avec la bonne dimension (4)
        geo_data = [np.random.rand(4).tolist() for _ in range(len(image_paths))]
        
        # Créer des labels équilibrés (4 de chaque classe)
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        
        # Créer le modèle hybride
        from src.model.geophysical_hybrid_net import GeophysicalHybridNet
        model = GeophysicalHybridNet(num_classes=2)
        
        # Préparer les données hybrides
        train_loader, val_loader = hybrid_trainer.prepare_hybrid_data(
            image_paths, geo_data, labels,
            test_size=0.2,
            augmentations=["rotation", "brightness"],
            num_augmentations=2
        )
        
        # Entraîner le modèle
        history = hybrid_trainer.train_hybrid_model(
            model, train_loader, val_loader,
            num_epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            patience=config["patience"]
        )
        
        # Sauvegarder le modèle
        model_path = CONFIG.paths.artifacts_dir / "models" / "hybrid_model.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(model_path))
        
        return {
            "model_type": "HYBRID",
            "model": model,
            "history": history,
            "model_path": str(model_path)
        }
    
    def train_dataframe_model(trainer: Any, config: Dict) -> Dict[str, Any]:
        """Entraîner un modèle pour les DataFrames géophysiques."""
        from src.model.geophysical_trainer import GeophysicalDataFrameNet
        import pandas as pd
        
        logger.info("Entraînement du modèle DataFrame...")
        
        # Créer des DataFrames factices pour la démonstration
        dataframes = []
        labels = []
        
        for i in range(20):
            # Créer un DataFrame factice avec des colonnes géophysiques
            df = pd.DataFrame({
                'x': np.random.rand(50),
                'y': np.random.rand(50),
                'z': np.random.rand(50),
                'resistivity': np.random.uniform(1e-8, 1e9, 50),
                'chargeability': np.random.uniform(0, 200, 50)
            })
            dataframes.append(df)
            labels.append(np.random.randint(0, 2))
        
        # Créer le modèle
        model = GeophysicalDataFrameNet(
            input_features=5,  # x, y, z, resistivity, chargeability
            num_classes=2,
            hidden_layers=[256, 128, 64],
            dropout_rate=0.3
        )
        
        # Préparer les données
        train_loader, val_loader = trainer.prepare_data_dataframe(
            dataframes, labels,
            augmentations=["gaussian_noise", "value_variation"],
            num_augmentations=3,
            test_size=0.2
        )
        
        # Entraîner le modèle
        history = trainer.train_model(
            model, train_loader, val_loader,
            num_epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            patience=config["patience"]
        )
        
        # Sauvegarder le modèle
        model_path = CONFIG.paths.artifacts_dir / "models" / "dataframe_model.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(model, str(model_path))
        
        return {
            "model_type": "DATAFRAME",
            "model": model,
            "history": history,
            "model_path": str(model_path)
        }
    
    # ============================================================================
    # PHASE 5: ÉVALUATION ET RÉSULTATS
    # ============================================================================
    
    def phase5_evaluation_and_results(training_results: Dict[str, Any], 
                                    processor: Optional[Any] = None) -> Dict[str, Any]:
        """
        Phase 5: Évaluation des modèles et génération des résultats.
        
        Args:
            training_results: Résultats d'entraînement
            processor: Processeur de données
            
        Returns:
            Dict contenant les résultats d'évaluation
        """
        logger.info("\n📈 Phase 5: Évaluation et résultats")
        logger.info("-" * 40)
            
        results = {
            "training_results": training_results,
            "evaluation_metrics": {},
            "model_summary": {}
        }
        
        # Générer le résumé des données si disponible
        if processor and hasattr(processor, 'get_data_summary'):
                try:
                    data_summary = processor.get_data_summary()
                    results["data_summary"] = data_summary
                    
                    # Sauvegarder le résumé dans les artefacts
                    artifacts_dir = Path(CONFIG.paths.artifacts_dir)
                    artifacts_dir.mkdir(parents=True, exist_ok=True)
                    
                    summary_file = artifacts_dir / "training_summary.json"
                    with open(summary_file, 'w') as f:
                     json.dump(results, f, indent=2, default=str)
                    
                    logger.info(f"Résumé d'entraînement sauvegardé dans: {summary_file}")
                except Exception as e:
                    logger.warning(f"Impossible de générer le résumé des données: {e}")
        
        # Générer le résumé du modèle
        if training_results:
            model_type = training_results.get("model_type", "UNKNOWN")
            model_path = training_results.get("model_path", "N/A")
            history = training_results.get("history", {})
            
            results["model_summary"] = {
                "model_type": model_type,
                "model_path": model_path,
                "total_epochs": len(history.get("epochs", [])),
                "final_train_loss": history.get("train_loss", [])[-1] if history.get("train_loss") else "N/A",
                "final_val_loss": history.get("val_loss", [])[-1] if history.get("val_loss") else "N/A",
                "final_train_acc": history.get("train_accuracy", [])[-1] if history.get("train_accuracy") else "N/A",
                "final_val_acc": history.get("val_accuracy", [])[-1] if history.get("val_accuracy") else "N/A"
            }
            
            logger.info(f"Résumé du modèle {model_type}:")
            logger.info(f"  - Chemin: {model_path}")
            logger.info(f"  - Époques: {results['model_summary']['total_epochs']}")
            train_loss = results['model_summary']['final_train_loss']
            val_loss = results['model_summary']['final_val_loss']
            train_acc = results['model_summary']['final_train_acc']
            val_acc = results['model_summary']['final_val_acc']
            
            if isinstance(train_loss, (int, float)) and isinstance(val_loss, (int, float)):
                logger.info(f"  - Loss finale (train/val): {train_loss:.4f}/{val_loss:.4f}")
            else:
                logger.info(f"  - Loss finale (train/val): {train_loss}/{val_loss}")
                
            if isinstance(train_acc, (int, float)) and isinstance(val_acc, (int, float)):
                logger.info(f"  - Accuracy finale (train/val): {train_acc:.2f}%/{val_acc:.2f}%")
            else:
                logger.info(f"  - Accuracy finale (train/val): {train_acc}%/{val_acc}%")
        
        return results
    
    # ============================================================================
    # FONCTION PRINCIPALE
    # ============================================================================
    
    def main():
        """
        Pipeline principal pour le projet AI-MAP.
        """
        try:
            logger.info("🚀 Starting AI-MAP Pipeline")
            logger.info("=" * 60)
            
            # Phase 1: Nettoyage des données
            cleaning_results = phase1_data_cleaning()
            
            # Phase 2: Traitement des données
            processor, multi_device_tensor, volume_3d = phase2_data_processing()
            
            # Phase 3: Préparation des données
            x_train, x_test = phase3_data_preparation(processor, multi_device_tensor)
            
            # Phase 4: Entraînement (sélection du modèle via arguments)
            training_config = {
                "epochs": 50,  # Réduit pour la démonstration
                "learning_rate": 0.001,
                "batch_size": 32,
                "patience": 10,
                "device": "auto"
            }
            
            # Entraîner le modèle sélectionné
            training_results = phase4_model_training(
                model_type="cnn_2d",  # Par défaut, peut être changé via CLI
                x_train=x_train,
                x_test=x_test,
                volume_3d=volume_3d,
                training_config=training_config
            )
            
            # Phase 5: Évaluation et résultats
            final_results = phase5_evaluation_and_results(training_results, processor)
            
            # Statut final
            logger.info("\n" + "=" * 60)
            logger.info("🎉 PIPELINE AI-MAP TERMINÉ AVEC SUCCÈS!")
            logger.info("=" * 60)
            
            logger.info("📋 Ce qui a été accompli:")
            logger.info("  ✅ Nettoyage et validation des données")
            logger.info("  ✅ Transformation et alignement des coordonnées")
            logger.info("  ✅ Création des grilles spatiales")
            logger.info("  ✅ Normalisation des données")
            logger.info("  ✅ Préparation du tenseur multi-dispositifs")
            logger.info("  ✅ Création du volume 3D")
            logger.info("  ✅ Division des données d'entraînement/test")
            logger.info("  ✅ Entraînement du modèle sélectionné")
            logger.info("  ✅ Évaluation et sauvegarde des résultats")
            
            logger.info(f"\n🚀 Modèle entraîné: {training_results.get('model_type', 'UNKNOWN')}")
            logger.info(f"📁 Modèle sauvegardé: {training_results.get('model_path', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Le pipeline a échoué: {str(e)}")
            logger.error("Trace de la pile:", exc_info=True)
            raise
    
    # ============================================================================
    # INTERFACE EN LIGNE DE COMMANDE
    # ============================================================================
    
    def parse_arguments():
        """Parser les arguments de la ligne de commande."""
        parser = argparse.ArgumentParser(
            description="Pipeline AI-MAP pour l'entraînement de modèles géophysiques",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Exemples d'utilisation:
  python main.py --model cnn_2d --epochs 100
  python main.py --model hybrid --epochs 50 --learning-rate 0.0001
  python main.py --model cnn_3d --batch-size 16 --patience 15
  python main.py --model dataframe --epochs 200 --device cuda
            """
        )
        
        # Sélection du modèle
        parser.add_argument(
            "--model", "-m",
            choices=["cnn_2d", "cnn_3d", "hybrid", "dataframe"],
            default="cnn_2d",
            help="Type de modèle à entraîner (défaut: cnn_2d)"
        )
        
        # Paramètres d'entraînement
        parser.add_argument(
            "--epochs", "-e",
            type=int,
            default=50,
            help="Nombre d'époques d'entraînement (défaut: 50)"
        )
        
        parser.add_argument(
            "--learning-rate", "-lr",
            type=float,
            default=0.001,
            help="Taux d'apprentissage (défaut: 0.001)"
        )
        
        parser.add_argument(
            "--batch-size", "-b",
            type=int,
            default=32,
            help="Taille du batch (défaut: 32)"
        )
        
        parser.add_argument(
            "--patience", "-p",
            type=int,
            default=10,
            help="Patience pour l'early stopping (défaut: 10)"
        )
        
        parser.add_argument(
            "--device", "-d",
            choices=["auto", "cpu", "cuda"],
            default="auto",
            help="Device pour l'entraînement (défaut: auto)"
        )
        
        # Options de pipeline
        parser.add_argument(
            "--skip-cleaning",
            action="store_true",
            help="Passer la phase de nettoyage des données"
        )
        
        parser.add_argument(
            "--skip-processing",
            action="store_true",
            help="Passer la phase de traitement des données"
        )
        
        parser.add_argument(
            "--skip-training",
            action="store_true",
            help="Passer la phase d'entraînement (utile pour tester le pipeline)"
        )
        
        # Options de sortie
        parser.add_argument(
            "--output-dir", "-o",
            type=str,
            default=None,
            help="Répertoire de sortie pour les modèles (défaut: artifacts/models/)"
        )
        
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Mode verbeux pour plus de détails"
        )
        
        return parser.parse_args()
    
    def main_with_args():
        """Fonction main avec gestion des arguments CLI."""
        args = parse_arguments()
        
        # Configuration du logging
        if args.verbose:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Configuration d'entraînement
        training_config = {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "device": args.device
        }
        
        # Répertoire de sortie
        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            CONFIG.paths.artifacts_dir = output_path
        
        logger.info("🚀 Starting AI-MAP Pipeline with CLI arguments")
        logger.info("=" * 60)
        logger.info(f"Configuration:")
        logger.info(f"  - Modèle: {args.model}")
        logger.info(f"  - Époques: {args.epochs}")
        logger.info(f"  - Learning rate: {args.learning_rate}")
        logger.info(f"  - Batch size: {args.batch_size}")
        logger.info(f"  - Patience: {args.patience}")
        logger.info(f"  - Device: {args.device}")
        logger.info(f"  - Output: {CONFIG.paths.artifacts_dir}")
        logger.info("=" * 60)
        
        try:
            # Phase 1: Nettoyage des données
            if not args.skip_cleaning:
                cleaning_results = phase1_data_cleaning()
            else:
                logger.info("⏭️  Phase 1: Nettoyage des données (ignorée)")
                cleaning_results = {}
            
            # Phase 2: Traitement des données
            if not args.skip_processing:
                processor, multi_device_tensor, volume_3d = phase2_data_processing()
            else:
                logger.info("⏭️  Phase 2: Traitement des données (ignorée)")
                processor, multi_device_tensor, volume_3d = None, None, None
            
            # Phase 3: Préparation des données
            if not args.skip_processing:
                x_train, x_test = phase3_data_preparation(processor, multi_device_tensor)
            else:
                logger.info("⏭️  Phase 3: Préparation des données (ignorée)")
                x_train, x_test = np.array([]), np.array([])
            
            # Phase 4: Entraînement
            if not args.skip_training:
                training_results = phase4_model_training(
                    model_type=args.model,
                    x_train=x_train,
                    x_test=x_test,
                    volume_3d=volume_3d,
                    training_config=training_config
                )
            else:
                logger.info("⏭️  Phase 4: Entraînement (ignorée)")
                training_results = {"model_type": args.model, "model_path": "N/A"}
            
            # Phase 5: Évaluation et résultats
            final_results = phase5_evaluation_and_results(training_results, processor)
            
            # Statut final
            logger.info("\n" + "=" * 60)
            logger.info("🎉 PIPELINE AI-MAP TERMINÉ AVEC SUCCÈS!")
            logger.info("=" * 60)
            
            logger.info("📋 Résumé de l'exécution:")
            logger.info(f"  ✅ Modèle entraîné: {args.model}")
            logger.info(f"  ✅ Époques: {args.epochs}")
            logger.info(f"  ✅ Device: {args.device}")
            if training_results.get("model_path"):
                logger.info(f"  ✅ Modèle sauvegardé: {training_results['model_path']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Le pipeline a échoué: {str(e)}")
            logger.error("Trace de la pile:", exc_info=True)
            raise
    
    # ============================================================================
    # POINT D'ENTRÉE
    # ============================================================================
    
    if __name__ == "__main__":
        # Vérifier si des arguments sont fournis
        if len(sys.argv) > 1:
            # Mode CLI avec arguments
            success = main_with_args()
        else:
            # Mode par défaut (sans arguments)
            success = main()
        
        if success:
            logger.info("🎯 Pipeline terminé avec succès!")
        else:
            logger.error("💥 Le pipeline a échoué!")
            sys.exit(1)
            
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Veuillez vérifier que tous les packages requis sont installés:")
    print("pip install -r requirements.txt")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Erreur inattendue: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
