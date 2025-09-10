#!/usr/bin/env python3
"""
Script rapide pour identifier les tests qui échouent
"""

import sys
import os
import unittest
import subprocess
from pathlib import Path

# Configuration
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

def run_test_quick(test_file):
    """Exécute un test rapidement et retourne le statut"""
    try:
        result = subprocess.run([
            sys.executable, str(test_file)
        ], capture_output=True, text=True, timeout=30)
        
        return {
            'file': test_file.name,
            'status': 'SUCCESS' if result.returncode == 0 else 'FAILED',
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            'file': test_file.name,
            'status': 'TIMEOUT',
            'returncode': -1,
            'stdout': '',
            'stderr': 'Timeout après 30 secondes'
        }
    except Exception as e:
        return {
            'file': test_file.name,
            'status': 'ERROR',
            'returncode': -1,
            'stdout': '',
            'stderr': str(e)
        }

def main():
    print("🔍 VÉRIFICATION RAPIDE DES TESTS")
    print("=" * 50)
    
    # Trouver tous les fichiers de tests
    test_dir = Path("test/unit/preprocessor")
    test_files = list(test_dir.glob("test_data_cleaner_*.py"))
    
    print(f"📁 {len(test_files)} fichiers de tests trouvés")
    print()
    
    failed_tests = []
    successful_tests = []
    
    # Exécuter chaque test rapidement
    for i, test_file in enumerate(test_files, 1):
        print(f"[{i:2d}/{len(test_files)}] {test_file.name}...", end=" ")
        
        result = run_test_quick(test_file)
        
        if result['status'] == 'SUCCESS':
            print("✅")
            successful_tests.append(result)
        else:
            print("❌")
            failed_tests.append(result)
    
    # Résumé
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ")
    print("=" * 50)
    print(f"✅ Tests réussis: {len(successful_tests)}")
    print(f"❌ Tests échoués: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\n❌ TESTS QUI ÉCHOUENT:")
        print("-" * 30)
        for result in failed_tests:
            print(f"• {result['file']} (code: {result['returncode']})")
            if result['stderr']:
                # Afficher la première ligne d'erreur
                error_line = result['stderr'].split('\n')[0]
                if error_line:
                    print(f"  → {error_line}")
    
    return failed_tests

if __name__ == "__main__":
    failed_tests = main()
