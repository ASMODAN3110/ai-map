#!/usr/bin/env python3
"""
Pipeline principal pour le projet AI-MAP, inspiré d'EMUT.
Orchestre l'ensemble du traitement des données géophysiques et du pipeline d'entraînement des modèles générateurs.
Conforme au cahier des charges avec U-Net 2D et VoxNet 3D.
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
    from backend.utils.logger import logger
    
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
        from backend.preprocessor.data_cleaner import GeophysicalDataCleaner
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
        from backend.data.data_processor import GeophysicalDataProcessor
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
    # PHASE 4: ENTRAÎNEMENT DES MODÈLES GÉNÉRATEURS
    # ============================================================================
    
    def phase4_model_training(model_type: str = "unet_2d", 
                             training_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Phase 4: Entraînement des modèles générateurs conformes au cahier des charges.
        
        Args:
            model_type: Type de modèle ("unet_2d", "voxnet_3d", "integrated")
            training_config: Configuration d'entraînement
            
        Returns:
            Dict contenant les résultats d'entraînement
        """
        logger.info(f"\n🤖 Phase 4: Entraînement du modèle générateur {model_type.upper()}")
        logger.info("-" * 40)
            
        # Configuration par défaut
        if training_config is None:
            training_config = {
                "epochs": 200,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "patience": 20,
                "device": "auto"
            }
        
        try:
            if model_type == "unet_2d":
                results = train_unet_2d(training_config)
            elif model_type == "voxnet_3d":
                results = train_voxnet_3d(training_config)
            elif model_type == "integrated":
                results = train_integrated_generators(training_config)
            else:
                raise ValueError(f"Type de modèle non supporté: {model_type}")
            
            logger.info("✅ Entraînement terminé avec succès")
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'entraînement: {str(e)}")
            raise
    
    def train_unet_2d(config: Dict) -> Dict[str, Any]:
        """Entraîner le modèle U-Net 2D."""
        from train_generators import GeneratorTrainer
        
        logger.info("Entraînement du modèle U-Net 2D...")
        
        # Créer l'entraîneur
        trainer = GeneratorTrainer(device=config["device"])
        
        # Créer des données synthétiques
        csv_data, targets_2d, targets_3d = trainer.create_synthetic_data(n_samples=10000)
        
        # Préparer les données
        train_loader_2d, val_loader_2d, _, _ = trainer.prepare_data(csv_data, targets_2d, targets_3d)
        
        # Entraîner le modèle
        history = trainer.train_unet_2d(
            train_loader_2d, val_loader_2d,
            num_epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            patience=config["patience"]
        )
        
        # Chemin de sauvegarde
        model_path = CONFIG.paths.artifacts_dir / "models" / "unet_2d_model.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        return {
            "model_type": "U-Net_2D",
            "history": history,
            "model_path": str(model_path)
        }
    
    def train_voxnet_3d(config: Dict) -> Dict[str, Any]:
        """Entraîner le modèle VoxNet 3D."""
        from train_generators import GeneratorTrainer
        
        logger.info("Entraînement du modèle VoxNet 3D...")
        
        # Créer l'entraîneur
        trainer = GeneratorTrainer(device=config["device"])
        
        # Créer des données synthétiques
        csv_data, targets_2d, targets_3d = trainer.create_synthetic_data(n_samples=10000)
        
        # Préparer les données
        _, _, train_loader_3d, val_loader_3d = trainer.prepare_data(csv_data, targets_2d, targets_3d)
        
        # Entraîner le modèle
        history = trainer.train_voxnet_3d(
            train_loader_3d, val_loader_3d,
            num_epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            patience=config["patience"]
        )
        
        # Chemin de sauvegarde
        model_path = CONFIG.paths.artifacts_dir / "models" / "voxnet_3d_model.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        return {
            "model_type": "VoxNet_3D",
            "history": history,
            "model_path": str(model_path)
        }
    
    def train_integrated_generators(config: Dict) -> Dict[str, Any]:
        """Entraîner les modèles générateurs intégrés (U-Net 2D + VoxNet 3D)."""
        from train_generators import GeneratorTrainer
        
        logger.info("Entraînement des modèles générateurs intégrés...")
        
        # Créer l'entraîneur
        trainer = GeneratorTrainer(device=config["device"])
        
        # Créer des données synthétiques
        csv_data, targets_2d, targets_3d = trainer.create_synthetic_data(n_samples=10000)
        
        # Préparer les données
        train_loader_2d, val_loader_2d, train_loader_3d, val_loader_3d = trainer.prepare_data(csv_data, targets_2d, targets_3d)
        
        # Entraîner les deux modèles
        history_2d = trainer.train_unet_2d(
            train_loader_2d, val_loader_2d,
            num_epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            patience=config["patience"]
        )
        
        history_3d = trainer.train_voxnet_3d(
            train_loader_3d, val_loader_3d,
            num_epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            patience=config["patience"]
        )
        
        # Chemins de sauvegarde
        model_path_2d = CONFIG.paths.artifacts_dir / "models" / "unet_2d_model.pth"
        model_path_3d = CONFIG.paths.artifacts_dir / "models" / "voxnet_3d_model.pth"
        model_path_2d.parent.mkdir(parents=True, exist_ok=True)
        model_path_3d.parent.mkdir(parents=True, exist_ok=True)
        
        return {
            "model_type": "INTEGRATED_GENERATORS",
            "history_2d": history_2d,
            "history_3d": history_3d,
            "model_path_2d": str(model_path_2d),
            "model_path_3d": str(model_path_3d)
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
        Pipeline principal pour le projet AI-MAP avec modèles générateurs.
        """
        try:
            logger.info("🚀 Starting AI-MAP Pipeline (Generators)")
            logger.info("=" * 60)
            
            # Phase 1: Nettoyage des données
            cleaning_results = phase1_data_cleaning()
            
            # Phase 2: Traitement des données
            processor, multi_device_tensor, volume_3d = phase2_data_processing()
            
            # Phase 3: Préparation des données
            x_train, x_test = phase3_data_preparation(processor, multi_device_tensor)
            
            # Phase 4: Entraînement (sélection du modèle via arguments)
            training_config = {
                "epochs": 200,  # Conforme au cahier des charges
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "patience": 20,
                "device": "auto"
            }
            
            # Entraîner le modèle sélectionné
            training_results = phase4_model_training(
                model_type="unet_2d",  # Par défaut, peut être changé via CLI
                training_config=training_config
            )
            
            # Phase 5: Évaluation et résultats
            final_results = phase5_evaluation_and_results(training_results, processor)
            
            # Statut final
            logger.info("\n" + "=" * 60)
            logger.info("🎉 PIPELINE AI-MAP (GÉNÉRATEURS) TERMINÉ AVEC SUCCÈS!")
            logger.info("=" * 60)
            
            logger.info("📋 Ce qui a été accompli:")
            logger.info("  ✅ Nettoyage et validation des données")
            logger.info("  ✅ Transformation et alignement des coordonnées")
            logger.info("  ✅ Création des grilles spatiales")
            logger.info("  ✅ Normalisation des données")
            logger.info("  ✅ Préparation du tenseur multi-dispositifs")
            logger.info("  ✅ Création du volume 3D")
            logger.info("  ✅ Division des données d'entraînement/test")
            logger.info("  ✅ Entraînement du modèle générateur sélectionné")
            logger.info("  ✅ Évaluation et sauvegarde des résultats")
            
            logger.info(f"\n🚀 Modèle entraîné: {training_results.get('model_type', 'UNKNOWN')}")
            logger.info(f"📁 Modèle sauvegardé: {training_results.get('model_path', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Le pipeline générateurs a échoué: {str(e)}")
            logger.error("Trace de la pile:", exc_info=True)
            raise
    
    # ============================================================================
    # INTERFACE EN LIGNE DE COMMANDE
    # ============================================================================
    
    def parse_arguments():
        """Parser les arguments de la ligne de commande."""
        parser = argparse.ArgumentParser(
            description="Pipeline AI-MAP pour l'entraînement de modèles générateurs géophysiques",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Exemples d'utilisation:
  python main.py --model unet_2d --epochs 200
  python main.py --model voxnet_3d --epochs 150 --learning-rate 0.0001
  python main.py --model integrated --epochs 200 --patience 20
  python main.py --model unet_2d --epochs 100 --device cuda
            """
        )
        
        # Sélection du modèle
        parser.add_argument(
            "--model", "-m",
            choices=["unet_2d", "voxnet_3d", "integrated"],
            default="unet_2d",
            help="Type de modèle générateur à entraîner (défaut: unet_2d)"
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
        """Fonction main avec gestion des arguments CLI pour les modèles générateurs."""
        args = parse_arguments()
        
        # Configuration du logging
        if args.verbose:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Configuration d'entraînement
        training_config = {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": 1e-5,  # Ajouté pour les générateurs
            "patience": args.patience,
            "device": args.device
        }
        
        # Répertoire de sortie
        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            CONFIG.paths.artifacts_dir = output_path
        
        logger.info("🚀 Starting AI-MAP Pipeline with CLI arguments (Generators)")
        logger.info("=" * 60)
        logger.info(f"Configuration:")
        logger.info(f"  - Modèle générateur: {args.model}")
        logger.info(f"  - Époques: {args.epochs}")
        logger.info(f"  - Learning rate: {args.learning_rate}")
        logger.info(f"  - Weight decay: {training_config['weight_decay']}")
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
                    training_config=training_config
                )
            else:
                logger.info("⏭️  Phase 4: Entraînement (ignorée)")
                training_results = {"model_type": args.model, "model_path": "N/A"}
            
            # Phase 5: Évaluation et résultats
            final_results = phase5_evaluation_and_results(training_results, processor)
            
            # Statut final
            logger.info("\n" + "=" * 60)
            logger.info("🎉 PIPELINE AI-MAP (GÉNÉRATEURS) TERMINÉ AVEC SUCCÈS!")
            logger.info("=" * 60)
            
            logger.info("📋 Résumé de l'exécution:")
            logger.info(f"  ✅ Modèle générateur entraîné: {args.model}")
            logger.info(f"  ✅ Époques: {args.epochs}")
            logger.info(f"  ✅ Device: {args.device}")
            if training_results.get("model_path"):
                logger.info(f"  ✅ Modèle sauvegardé: {training_results['model_path']}")
            elif training_results.get("model_path_2d"):
                logger.info(f"  ✅ Modèle 2D sauvegardé: {training_results['model_path_2d']}")
                logger.info(f"  ✅ Modèle 3D sauvegardé: {training_results['model_path_3d']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Le pipeline générateurs a échoué: {str(e)}")
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
            logger.info("🎯 Pipeline générateurs terminé avec succès!")
        else:
            logger.error("💥 Le pipeline générateurs a échoué!")
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
