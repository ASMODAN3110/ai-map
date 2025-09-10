#!/usr/bin/env python3
"""
Script de test pour les modèles générateurs géophysiques.

Ce script teste les modèles U-Net 2D et VoxNet 3D entraînés
et génère des visualisations pour validation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple

from backend.model.geophysical_generators import UNet2D, VoxNet3D, GeophysicalImageGenerator
from backend.utils.logger import logger


class GeneratorTester:
    """
    Testeur pour les modèles générateurs géophysiques.
    """
    
    def __init__(self, model_path_2d: str = "artifacts/models/unet_2d_model.pth",
                 model_path_3d: str = "artifacts/models/voxnet_3d_model.pth"):
        self.model_path_2d = model_path_2d
        self.model_path_3d = model_path_3d
        
        # Vérifier la disponibilité des modèles
        self.has_unet_2d = Path(model_path_2d).exists()
        self.has_voxnet_3d = Path(model_path_3d).exists()
        
        logger.info(f"GeneratorTester initialisé:")
        logger.info(f"   - U-Net 2D: {'✅' if self.has_unet_2d else '❌'} {model_path_2d}")
        logger.info(f"   - VoxNet 3D: {'✅' if self.has_voxnet_3d else '❌'} {model_path_3d}")
    
    def test_unet_2d(self, n_samples: int = 5) -> Dict[str, any]:
        """
        Tester le modèle U-Net 2D.
        
        Args:
            n_samples: Nombre d'échantillons de test
            
        Returns:
            Résultats du test
        """
        if not self.has_unet_2d:
            logger.warning("❌ Modèle U-Net 2D non trouvé, test ignoré")
            return {"status": "skipped", "reason": "Model not found"}
        
        logger.info(f"🧪 Test du modèle U-Net 2D avec {n_samples} échantillons...")
        
        try:
            # Charger le modèle
            model = UNet2D()
            checkpoint = torch.load(self.model_path_2d, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Créer des données de test
            test_data = self._create_test_data_2d(n_samples)
            
            # Générer les prédictions
            start_time = time.time()
            with torch.no_grad():
                predictions = model(test_data)
            generation_time = time.time() - start_time
            
            # Analyser les résultats
            results = self._analyze_2d_results(test_data, predictions)
            results.update({
                "status": "success",
                "n_samples": n_samples,
                "generation_time": generation_time,
                "avg_time_per_sample": generation_time / n_samples
            })
            
            logger.info(f"✅ Test U-Net 2D réussi:")
            logger.info(f"   - Temps de génération: {generation_time:.3f}s")
            logger.info(f"   - Temps moyen par échantillon: {generation_time/n_samples:.3f}s")
            logger.info(f"   - Forme des prédictions: {predictions.shape}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du test U-Net 2D: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_voxnet_3d(self, n_samples: int = 3) -> Dict[str, any]:
        """
        Tester le modèle VoxNet 3D.
        
        Args:
            n_samples: Nombre d'échantillons de test
            
        Returns:
            Résultats du test
        """
        if not self.has_voxnet_3d:
            logger.warning("❌ Modèle VoxNet 3D non trouvé, test ignoré")
            return {"status": "skipped", "reason": "Model not found"}
        
        logger.info(f"🧪 Test du modèle VoxNet 3D avec {n_samples} échantillons...")
        
        try:
            # Charger le modèle
            model = VoxNet3D()
            checkpoint = torch.load(self.model_path_3d, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Créer des données de test
            test_data = self._create_test_data_3d(n_samples)
            
            # Générer les prédictions
            start_time = time.time()
            with torch.no_grad():
                predictions = model(test_data)
            generation_time = time.time() - start_time
            
            # Analyser les résultats
            results = self._analyze_3d_results(test_data, predictions)
            results.update({
                "status": "success",
                "n_samples": n_samples,
                "generation_time": generation_time,
                "avg_time_per_sample": generation_time / n_samples
            })
            
            logger.info(f"✅ Test VoxNet 3D réussi:")
            logger.info(f"   - Temps de génération: {generation_time:.3f}s")
            logger.info(f"   - Temps moyen par échantillon: {generation_time/n_samples:.3f}s")
            logger.info(f"   - Forme des prédictions: {predictions.shape}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du test VoxNet 3D: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_integrated_generator(self, n_samples: int = 3) -> Dict[str, any]:
        """
        Tester le générateur intégré.
        
        Args:
            n_samples: Nombre d'échantillons de test
            
        Returns:
            Résultats du test
        """
        logger.info(f"🧪 Test du générateur intégré avec {n_samples} échantillons...")
        
        try:
            # Créer le générateur
            generator = GeophysicalImageGenerator(
                model_path_2d=self.model_path_2d if self.has_unet_2d else None,
                model_path_3d=self.model_path_3d if self.has_voxnet_3d else None
            )
            
            # Créer des données CSV de test
            csv_data = self._create_test_csv_data(n_samples)
            
            # Générer les pseudo-sections 2D
            start_time = time.time()
            pseudo_sections = generator.generate_pseudo_sections(csv_data, method="pole-dipole")
            time_2d = time.time() - start_time
            
            # Générer les modèles 3D
            start_time = time.time()
            models_3d = generator.generate_3d_models(csv_data, method="pole-dipole")
            time_3d = time.time() - start_time
            
            # Analyser les résultats
            results = {
                "status": "success",
                "n_samples": n_samples,
                "pseudo_sections": {
                    "count": len(pseudo_sections),
                    "generation_time": time_2d,
                    "avg_time_per_sample": time_2d / n_samples
                },
                "models_3d": {
                    "count": len(models_3d),
                    "generation_time": time_3d,
                    "avg_time_per_sample": time_3d / n_samples
                },
                "total_time": time_2d + time_3d
            }
            
            logger.info(f"✅ Test générateur intégré réussi:")
            logger.info(f"   - Pseudo-sections 2D: {len(pseudo_sections)} en {time_2d:.3f}s")
            logger.info(f"   - Modèles 3D: {len(models_3d)} en {time_3d:.3f}s")
            logger.info(f"   - Temps total: {time_2d + time_3d:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du test générateur intégré: {e}")
            return {"status": "error", "error": str(e)}
    
    def _create_test_data_2d(self, n_samples: int) -> torch.Tensor:
        """Créer des données de test 2D."""
        # Données CSV simulées
        resistivity = np.random.uniform(10, 1000, n_samples)
        chargeability = np.random.uniform(0, 50, n_samples)
        x_coord = np.random.uniform(0, 100, n_samples)
        y_coord = np.random.uniform(0, 100, n_samples)
        
        csv_data = np.column_stack([resistivity, chargeability, x_coord, y_coord])
        
        # Traiter en grilles 2D
        from backend.model.geophysical_generators import GeophysicalDataProcessor
        processor = GeophysicalDataProcessor()
        grids_2d = processor.process_csv_to_2d_grid(csv_data)
        
        return grids_2d
    
    def _create_test_data_3d(self, n_samples: int) -> torch.Tensor:
        """Créer des données de test 3D."""
        # Données CSV simulées
        resistivity = np.random.uniform(10, 1000, n_samples)
        chargeability = np.random.uniform(0, 50, n_samples)
        x_coord = np.random.uniform(0, 100, n_samples)
        y_coord = np.random.uniform(0, 100, n_samples)
        
        csv_data = np.column_stack([resistivity, chargeability, x_coord, y_coord])
        
        # Traiter en volumes 3D
        from backend.model.geophysical_generators import GeophysicalDataProcessor
        processor = GeophysicalDataProcessor()
        volumes_3d = processor.process_csv_to_3d_volume(csv_data)
        
        return volumes_3d
    
    def _create_test_csv_data(self, n_samples: int) -> np.ndarray:
        """Créer des données CSV de test."""
        # Données géophysiques simulées
        resistivity = np.random.uniform(10, 1000, n_samples)  # Ω⋅m
        chargeability = np.random.uniform(0, 50, n_samples)   # mV/V
        x_coord = np.random.uniform(0, 100, n_samples)        # m
        y_coord = np.random.uniform(0, 100, n_samples)        # m
        
        csv_data = np.column_stack([resistivity, chargeability, x_coord, y_coord])
        
        return csv_data.astype(np.float32)
    
    def _analyze_2d_results(self, input_data: torch.Tensor, predictions: torch.Tensor) -> Dict[str, any]:
        """Analyser les résultats 2D."""
        # Statistiques de base
        input_stats = {
            "mean": input_data.mean().item(),
            "std": input_data.std().item(),
            "min": input_data.min().item(),
            "max": input_data.max().item()
        }
        
        output_stats = {
            "mean": predictions.mean().item(),
            "std": predictions.std().item(),
            "min": predictions.min().item(),
            "max": predictions.max().item()
        }
        
        # Vérifier la forme
        expected_shape = (input_data.size(0), 2, 64, 64)
        shape_correct = predictions.shape == expected_shape
        
        return {
            "input_stats": input_stats,
            "output_stats": output_stats,
            "shape_correct": shape_correct,
            "expected_shape": expected_shape,
            "actual_shape": list(predictions.shape)
        }
    
    def _analyze_3d_results(self, input_data: torch.Tensor, predictions: torch.Tensor) -> Dict[str, any]:
        """Analyser les résultats 3D."""
        # Statistiques de base
        input_stats = {
            "mean": input_data.mean().item(),
            "std": input_data.std().item(),
            "min": input_data.min().item(),
            "max": input_data.max().item()
        }
        
        output_stats = {
            "mean": predictions.mean().item(),
            "std": predictions.std().item(),
            "min": predictions.min().item(),
            "max": predictions.max().item()
        }
        
        # Vérifier la forme
        expected_shape = (input_data.size(0), 1, 32, 32, 32)
        shape_correct = predictions.shape == expected_shape
        
        return {
            "input_stats": input_stats,
            "output_stats": output_stats,
            "shape_correct": shape_correct,
            "expected_shape": expected_shape,
            "actual_shape": list(predictions.shape)
        }
    
    def run_comprehensive_test(self) -> Dict[str, any]:
        """
        Exécuter une suite de tests complète.
        
        Returns:
            Résultats de tous les tests
        """
        logger.info("🧪 DÉBUT DES TESTS COMPLETS")
        logger.info("=" * 50)
        
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": {}
        }
        
        # Test U-Net 2D
        logger.info("1️⃣ Test U-Net 2D...")
        results["tests"]["unet_2d"] = self.test_unet_2d(n_samples=5)
        
        # Test VoxNet 3D
        logger.info("2️⃣ Test VoxNet 3D...")
        results["tests"]["voxnet_3d"] = self.test_voxnet_3d(n_samples=3)
        
        # Test générateur intégré
        logger.info("3️⃣ Test générateur intégré...")
        results["tests"]["integrated_generator"] = self.test_integrated_generator(n_samples=3)
        
        # Résumé des résultats
        successful_tests = sum(1 for test in results["tests"].values() if test.get("status") == "success")
        total_tests = len(results["tests"])
        
        results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0
        }
        
        logger.info("📊 RÉSUMÉ DES TESTS:")
        logger.info(f"   - Tests réussis: {successful_tests}/{total_tests}")
        logger.info(f"   - Taux de succès: {results['summary']['success_rate']:.1%}")
        
        return results


def main():
    """Fonction principale de test."""
    logger.info("🧪 TEST DES MODÈLES GÉNÉRATEURS GÉOPHYSIQUES")
    logger.info("=" * 60)
    
    try:
        # Créer le testeur
        tester = GeneratorTester()
        
        # Exécuter les tests complets
        results = tester.run_comprehensive_test()
        
        # Sauvegarder les résultats
        results_path = "artifacts/test_results.json"
        Path("artifacts").mkdir(exist_ok=True)
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Résultats sauvegardés: {results_path}")
        
        # Afficher le résumé final
        summary = results["summary"]
        if summary["success_rate"] == 1.0:
            logger.info("🎉 TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS!")
        elif summary["success_rate"] >= 0.5:
            logger.info("⚠️ CERTAINS TESTS ONT ÉCHOUÉ, MAIS LA MAJORITÉ RÉUSSIT")
        else:
            logger.info("❌ LA MAJORITÉ DES TESTS ONT ÉCHOUÉ")
        
        return summary["success_rate"] == 1.0
        
    except Exception as e:
        logger.error(f"❌ Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
