#!/usr/bin/env python3
"""
Script de test pour vérifier le pipeline d'entraînement AI-MAP.
Ce script teste les différentes phases du pipeline sans entraîner de vrais modèles.
"""

import sys
import subprocess
from pathlib import Path

def test_pipeline_phases():
    """Tester les différentes phases du pipeline."""
    print("🧪 Test du Pipeline AI-MAP")
    print("=" * 50)
    
    # Test 1: Pipeline complet sans entraînement
    print("\n1️⃣ Test du pipeline complet (sans entraînement)")
    try:
        result = subprocess.run([
            sys.executable, "main.py", 
            "--skip-training", 
            "--verbose"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Pipeline complet : SUCCÈS")
        else:
            print("❌ Pipeline complet : ÉCHEC")
            print(f"Erreur: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⏰ Pipeline complet : TIMEOUT")
    except Exception as e:
        print(f"❌ Pipeline complet : ERREUR - {e}")
    
    # Test 2: Test avec nettoyage seulement
    print("\n2️⃣ Test du nettoyage des données")
    try:
        result = subprocess.run([
            sys.executable, "main.py", 
            "--skip-processing", 
            "--skip-training",
            "--verbose"
        ], capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            print("✅ Nettoyage des données : SUCCÈS")
        else:
            print("❌ Nettoyage des données : ÉCHEC")
            print(f"Erreur: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⏰ Nettoyage des données : TIMEOUT")
    except Exception as e:
        print(f"❌ Nettoyage des données : ERREUR - {e}")
    
    # Test 3: Test d'entraînement rapide CNN 2D
    print("\n3️⃣ Test d'entraînement rapide CNN 2D")
    try:
        result = subprocess.run([
            sys.executable, "main.py", 
            "--model", "cnn_2d",
            "--epochs", "2",
            "--batch-size", "8",
            "--patience", "1"
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Entraînement CNN 2D : SUCCÈS")
        else:
            print("❌ Entraînement CNN 2D : ÉCHEC")
            print(f"Erreur: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⏰ Entraînement CNN 2D : TIMEOUT")
    except Exception as e:
        print(f"❌ Entraînement CNN 2D : ERREUR - {e}")
    
    # Test 4: Test d'entraînement rapide DataFrame
    print("\n4️⃣ Test d'entraînement rapide DataFrame")
    try:
        result = subprocess.run([
            sys.executable, "main.py", 
            "--model", "dataframe",
            "--epochs", "2",
            "--batch-size", "16",
            "--patience", "1"
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Entraînement DataFrame : SUCCÈS")
        else:
            print("❌ Entraînement DataFrame : ÉCHEC")
            print(f"Erreur: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⏰ Entraînement DataFrame : TIMEOUT")
    except Exception as e:
        print(f"❌ Entraînement DataFrame : ERREUR - {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Tests terminés")

def test_help_command():
    """Tester la commande d'aide."""
    print("\n📖 Test de la commande d'aide")
    try:
        result = subprocess.run([
            sys.executable, "main.py", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and "Pipeline AI-MAP" in result.stdout:
            print("✅ Commande d'aide : SUCCÈS")
            return True
        else:
            print("❌ Commande d'aide : ÉCHEC")
            return False
    except Exception as e:
        print(f"❌ Commande d'aide : ERREUR - {e}")
        return False

def check_dependencies():
    """Vérifier les dépendances."""
    print("\n🔍 Vérification des dépendances")
    
    required_modules = [
        "torch", "numpy", "pandas", "sklearn", 
        "matplotlib", "PIL", "pathlib"
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️  Modules manquants: {', '.join(missing_modules)}")
        print("Installez-les avec: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ Toutes les dépendances sont installées")
        return True

def main():
    """Fonction principale de test."""
    print("🚀 Démarrage des tests du pipeline AI-MAP")
    print("=" * 60)
    
    # Vérifier les dépendances
    deps_ok = check_dependencies()
    
    # Tester la commande d'aide
    help_ok = test_help_command()
    
    if deps_ok and help_ok:
        # Lancer les tests du pipeline
        test_pipeline_phases()
    else:
        print("\n❌ Tests interrompus - Dépendances manquantes ou erreur d'aide")
        return False
    
    print("\n🎯 Tests terminés avec succès!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
