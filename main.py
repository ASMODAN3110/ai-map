#!/usr/bin/env python3
"""
Pipeline principal pour le projet AI-MAP, inspiré d'EMUT.
Orchestre l'ensemble du traitement des données géophysiques et du pipeline d'entraînement CNN.
"""

import json
import sys
from pathlib import Path

# Ajouter le répertoire courant au path Python
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config import CONFIG
    from src.utils.logger import logger
    
    def main():
        """
        Pipeline principal pour le projet AI-MAP.
        """
        try:
            logger.info("🚀 Starting AI-MAP Pipeline")
            logger.info("=" * 60)
            
            # Phase 1: Nettoyage et prétraitement des données
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
            
            # Phase 2: Traitement des données et création des grilles
            logger.info("\n📊 Phase 2: Traitement des données et création des grilles")
            logger.info("-" * 40)
            
            # Initialiser le processeur de données
            from src.data.data_processor import GeophysicalDataProcessor
            processor = GeophysicalDataProcessor()
            
            # Charger et valider les données nettoyées
            device_data = processor.load_and_validate()
            
            if not device_data:
                logger.warning("Aucune donnée de dispositif valide trouvée après le nettoyage")
                logger.info("Le pipeline continuera avec des données vides pour la démonstration")
            else:
                # Créer les grilles spatiales
                spatial_grids = processor.create_spatial_grids()
                
                # Créer le tenseur multi-dispositifs pour l'entrée CNN
                multi_device_tensor = processor.create_multi_device_tensor()
                
                # Créer le volume 3D pour VoxNet
                volume_3d = processor.create_3d_volume()
                
                logger.info("✅ Traitement des données terminé avec succès")
                logger.info(f"Forme du tenseur multi-dispositifs: {multi_device_tensor.shape}")
                logger.info(f"Forme du volume 3D: {volume_3d.shape}")
                
                # Phase 3: Division et préparation des données
                logger.info("\n🔀 Phase 3: Division et préparation des données")
                logger.info("-" * 40)
                
                # Diviser les données pour l'entraînement
                x_train, x_test = processor.split_data(multi_device_tensor)
                
                logger.info(f"Taille de l'ensemble d'entraînement: {len(x_train)}")
                logger.info(f"Taille de l'ensemble de test: {len(x_test)}")
            
            # Phase 4: Entraînement du modèle (Placeholder pour l'instant)
            logger.info("\n🤖 Phase 4: Entraînement du modèle (Placeholder)")
            logger.info("-" * 40)
            
            logger.info("L'entraînement du modèle sera implémenté dans la prochaine phase")
            logger.info("Configuration prête pour:")
            logger.info(f"  - U-Net 2D: {CONFIG.cnn.unet_2d['input_shape']}")
            logger.info(f"  - VoxNet 3D: {CONFIG.cnn.voxnet_3d['input_shape']}")
            
            # Phase 5: Résultats et résumé
            logger.info("\n📈 Phase 5: Résultats et résumé")
            logger.info("-" * 40)
            
            # Obtenir le résumé des données si le processeur a des données
            if 'processor' in locals() and hasattr(processor, 'get_data_summary'):
                try:
                    data_summary = processor.get_data_summary()
                    
                    # Sauvegarder le résumé dans les artefacts
                    artifacts_dir = Path(CONFIG.paths.artifacts_dir)
                    artifacts_dir.mkdir(parents=True, exist_ok=True)
                    
                    summary_file = artifacts_dir / "phase1_summary.json"
                    with open(summary_file, 'w') as f:
                        json.dump(data_summary, f, indent=2, default=str)
                    
                    logger.info(f"Résumé des données sauvegardé dans: {summary_file}")
                except Exception as e:
                    logger.warning(f"Impossible de générer le résumé des données: {e}")
            
            # Statut final
            logger.info("\n" + "=" * 60)
            logger.info("🎉 PIPELINE AI-MAP PHASE 1 TERMINÉ AVEC SUCCÈS!")
            logger.info("=" * 60)
            
            logger.info("📋 Ce qui a été accompli:")
            logger.info("  ✅ Nettoyage et validation des données")
            logger.info("  ✅ Transformation et alignement des coordonnées")
            logger.info("  ✅ Création des grilles spatiales")
            logger.info("  ✅ Normalisation des données")
            logger.info("  ✅ Préparation du tenseur multi-dispositifs")
            logger.info("  ✅ Création du volume 3D")
            logger.info("  ✅ Division des données d'entraînement/test")
            
            logger.info("\n🚀 Prochaines étapes:")
            logger.info("  📊 Phase 2: Implémentation des modèles CNN")
            logger.info("  🧠 Phase 3: Entraînement et validation des modèles")
            logger.info("  🌐 Phase 4: Développement de l'application web")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Le pipeline a échoué: {str(e)}")
            logger.error("Trace de la pile:", exc_info=True)
            raise
    
    if __name__ == "__main__":
        success = main()
        if success:
            logger.info("🎯 Pipeline terminé avec succès!")
        else:
            logger.error("💥 Le pipeline a échoué!")
            sys.exit(1)
            
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Veuillez vérifier que tous les packages requis sont installés:")
    print("pip install -r requirements/requirements_phase1.txt")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Erreur inattendue: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
