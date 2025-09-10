#!/usr/bin/env python3
"""
Script principal pour démarrer le projet AI-MAP.
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Fonction principale pour démarrer le projet."""
    print("🚀 Démarrage du projet AI-MAP...")
    print("=" * 50)
    
    # Vérifier que nous sommes dans le bon répertoire
    project_root = Path(__file__).parent.parent
    
    # Options de démarrage
    print("Choisissez une option:")
    print("1. Démarrer le serveur API")
    print("2. Entraîner les modèles générateurs")
    print("3. Tester les modèles générateurs")
    print("4. Démonstration des modèles")
    print("5. Tests d'intégration API")
    print("6. Pipeline complet")
    
    choice = input("\nVotre choix (1-6): ").strip()
    
    if choice == "1":
        print("🌐 Démarrage du serveur API...")
        subprocess.run([sys.executable, "run_api_server.py"], cwd=project_root / "scripts")
    elif choice == "2":
        print("🤖 Entraînement des modèles générateurs...")
        subprocess.run([sys.executable, "train_generators.py"], cwd=project_root / "scripts")
    elif choice == "3":
        print("🧪 Test des modèles générateurs...")
        subprocess.run([sys.executable, "test_generators.py"], cwd=project_root / "scripts")
    elif choice == "4":
        print("🎯 Démonstration des modèles...")
        subprocess.run([sys.executable, "demo_generators.py"], cwd=project_root / "scripts")
    elif choice == "5":
        print("🔗 Tests d'intégration API...")
        subprocess.run([sys.executable, "test_api_integration.py"], cwd=project_root / "scripts")
    elif choice == "6":
        print("🔄 Pipeline complet...")
        subprocess.run([sys.executable, "main.py"], cwd=project_root / "scripts")
    else:
        print("❌ Choix invalide!")

if __name__ == "__main__":
    main()
