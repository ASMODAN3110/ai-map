#!/usr/bin/env python3
"""
Script de test pour vérifier l'intégration API avec des données CSV réelles.
"""

import requests
import json
import time
from pathlib import Path

def test_api_health():
    """Tester la santé de l'API."""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health: {data}")
            return True
        else:
            print(f"❌ API Health failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Health error: {e}")
        return False

def test_generate_sample_images():
    """Tester la génération d'images d'exemple."""
    try:
        print("\n🧪 Test de génération d'images d'exemple...")
        
        data = {
            'method': 'pole-dipole',
            'samples': '3'
        }
        
        response = requests.post(
            "http://localhost:8000/api/generate-sample-images",
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Génération d'exemple réussie:")
            print(f"   - Méthode: {result['method']}")
            print(f"   - Échantillons: {result['num_samples']}")
            print(f"   - Pseudo-sections: {len(result['visualizations']['pseudo_sections'])}")
            print(f"   - Cartes de chargeabilité: {len(result['visualizations']['chargeability_maps'])}")
            print(f"   - Modèle 3D: {'Oui' if result['visualizations']['model_3d'] else 'Non'}")
            return True
        else:
            print(f"❌ Génération d'exemple échouée: {response.status_code}")
            print(f"   Erreur: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du test d'exemple: {e}")
        return False

def test_generate_images_from_csv():
    """Tester la génération d'images à partir d'un fichier CSV réel."""
    try:
        print("\n📊 Test de génération d'images avec données CSV réelles...")
        
        # Vérifier qu'un fichier CSV existe
        csv_files = [
            "data/processed/profil_1_cleaned.csv",
            "data/processed/profil_2_cleaned.csv",
            "data/processed/profil_3_cleaned.csv"
        ]
        
        csv_file = None
        for file_path in csv_files:
            if Path(file_path).exists():
                csv_file = file_path
                break
        
        if not csv_file:
            print("⚠️  Aucun fichier CSV trouvé, test ignoré")
            return True
        
        print(f"📁 Utilisation du fichier: {csv_file}")
        
        # Lire le fichier CSV
        with open(csv_file, 'rb') as f:
            files = {'file': f}
            data = {
                'method': 'pole-dipole',
                'samples': '3'
            }
            
            response = requests.post(
                "http://localhost:8000/api/generate-images",
                files=files,
                data=data
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Génération CSV réussie:")
            print(f"   - Méthode: {result['method']}")
            print(f"   - Échantillons: {result['num_samples']}")
            print(f"   - Colonnes utilisées: {result['metadata'].get('columns_used', 'N/A')}")
            print(f"   - Forme des données: {result['metadata']['data_shape']}")
            print(f"   - Pseudo-sections: {len(result['visualizations']['pseudo_sections'])}")
            print(f"   - Cartes de chargeabilité: {len(result['visualizations']['chargeability_maps'])}")
            print(f"   - Modèle 3D: {'Oui' if result['visualizations']['model_3d'] else 'Non'}")
            return True
        else:
            print(f"❌ Génération CSV échouée: {response.status_code}")
            print(f"   Erreur: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du test CSV: {e}")
        return False

def test_api_methods():
    """Tester les endpoints d'information."""
    try:
        print("\n📋 Test des endpoints d'information...")
        
        # Test des méthodes disponibles
        response = requests.get("http://localhost:8000/api/methods")
        if response.status_code == 200:
            methods = response.json()
            print(f"✅ Méthodes disponibles: {len(methods['methods'])}")
            for method in methods['methods']:
                print(f"   - {method['name']}: {method['description']}")
        else:
            print(f"❌ Erreur méthodes: {response.status_code}")
            return False
        
        # Test des modèles disponibles
        response = requests.get("http://localhost:8000/api/models")
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Modèles disponibles: {len(models['models'])}")
            for model in models['models']:
                print(f"   - {model['name']}: {model['description']}")
        else:
            print(f"❌ Erreur modèles: {response.status_code}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test des endpoints: {e}")
        return False

def main():
    """Fonction principale de test."""
    print("🚀 TEST D'INTÉGRATION API AI-MAP")
    print("=" * 50)
    
    # Attendre que l'API soit prête
    print("⏳ Attente du démarrage de l'API...")
    time.sleep(3)
    
    tests = [
        ("Health Check", test_api_health),
        ("Endpoints d'information", test_api_methods),
        ("Génération d'exemple", test_generate_sample_images),
        ("Génération CSV", test_generate_images_from_csv)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Test: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erreur inattendue: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    print("\n📊 RÉSUMÉ DES TESTS:")
    print("-" * 30)
    passed = 0
    for test_name, result in results:
        status = "✅ PASSÉ" if result else "❌ ÉCHOUÉ"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Résultat: {passed}/{len(results)} tests passés")
    
    if passed == len(results):
        print("🎉 TOUS LES TESTS SONT PASSÉS!")
        print("✅ L'API est prête pour l'intégration frontend")
    else:
        print("⚠️  Certains tests ont échoué")
        print("🔧 Vérifiez les logs du serveur API")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
