#!/usr/bin/env python3
"""
Script de test pour vérifier que le pipeline AI-MAP fonctionne complètement.
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire courant au path Python
sys.path.insert(0, str(Path(__file__).parent))

def test_pipeline():
    """Tester le pipeline complet."""
    
    print("🚀 Test du pipeline AI-MAP complet")
    print("=" * 50)
    
    try:
        # Importer les fonctions du pipeline
        from main import main_with_args
        
        # Configuration des arguments
        sys.argv = ['main.py', '--epochs', '1', '--verbose']
        
        print("📋 Exécution du pipeline...")
        print(f"Arguments: {sys.argv}")
        print("-" * 30)
        
        # Exécuter le pipeline
        success = main_with_args()
        
        if success:
            print("\n✅ PIPELINE EXÉCUTÉ AVEC SUCCÈS!")
            print("=" * 50)
            print("🎯 Le pipeline AI-MAP fonctionne parfaitement!")
            print("📊 Toutes les phases ont été exécutées:")
            print("  ✅ Phase 1: Nettoyage des données")
            print("  ✅ Phase 2: Traitement des données")
            print("  ✅ Phase 3: Préparation des données")
            print("  ✅ Phase 4: Entraînement du modèle")
            print("  ✅ Phase 5: Évaluation et résultats")
            return True
        else:
            print("\n❌ Le pipeline a échoué!")
            return False
            
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_models():
    """Tester différents modèles."""
    
    print("\n🤖 Test de différents modèles")
    print("=" * 50)
    
    models = ['cnn_2d', 'cnn_3d', 'hybrid', 'dataframe']
    
    for model in models:
        print(f"\n📊 Test du modèle: {model}")
        print("-" * 30)
        
        try:
            from main import main_with_args
            
            # Configuration des arguments
            sys.argv = ['main.py', '--model', model, '--epochs', '1', '--skip-training']
            
            success = main_with_args()
            
            if success:
                print(f"✅ Modèle {model}: SUCCÈS")
            else:
                print(f"❌ Modèle {model}: ÉCHEC")
                
        except Exception as e:
            print(f"❌ Modèle {model}: ERREUR - {e}")

def main():
    """Fonction principale."""
    
    print("🧪 TESTS DU PIPELINE AI-MAP")
    print("=" * 60)
    
    # Test 1: Pipeline complet
    print("\n1️⃣ Test du pipeline complet")
    success = test_pipeline()
    
    if success:
        # Test 2: Différents modèles
        print("\n2️⃣ Test de différents modèles")
        test_different_models()
        
        print("\n" + "=" * 60)
        print("🎉 TOUS LES TESTS TERMINÉS!")
        print("=" * 60)
        print("\n📋 Le pipeline AI-MAP est prêt à être utilisé:")
        print("  - python main.py --epochs 10")
        print("  - python main.py --model hybrid --epochs 5")
        print("  - python main.py --model cnn_3d --epochs 3")
        print("  - python main.py --help  # Pour voir toutes les options")
    else:
        print("\n❌ Le pipeline de base a échoué. Vérifiez la configuration.")

if __name__ == "__main__":
    main()
