# 🧪 Tests du Projet AI-MAP

Ce dossier contient tous les tests pour le projet AI-MAP (Intelligence Artificielle pour l'Analyse Géophysique).

## 📁 Structure des Tests

```
test/
├── __init__.py                 # Package principal des tests
├── README.md                   # Ce fichier
├── run_tests.py                # Script principal d'exécution des tests
├── unit/                       # Tests unitaires
│   ├── __init__.py
│   ├── preprocessor/           # Tests du module preprocessor
│   │   ├── __init__.py
│   │   └── test_data_cleaner_init.py
│   ├── data/                   # Tests du module data
│   ├── model/                  # Tests du module model
│   └── utils/                  # Tests du module utils
├── integration/                # Tests d'intégration
│   └── __init__.py
├── e2e/                        # Tests end-to-end
│   └── __init__.py
└── fixtures/                   # Données de test
    └── __init__.py
```

## 🎯 Types de Tests

### 🧪 Tests Unitaires (`unit/`)
- **Objectif** : Tester chaque composant individuellement
- **Portée** : Fonctions, classes, méthodes isolées
- **Exemple** : Test de la fonction `__init__` de `GeophysicalDataCleaner`

### 🔗 Tests d'Intégration (`integration/`)
- **Objectif** : Tester l'interaction entre composants
- **Portée** : Modules et sous-systèmes
- **Exemple** : Test du pipeline de prétraitement complet

### 🌐 Tests End-to-End (`e2e/`)
- **Objectif** : Tester le système complet
- **Portée** : Workflow complet de bout en bout
- **Exemple** : Test du pipeline complet AI-MAP

## 🚀 Exécution des Tests

### Exécuter tous les tests
```bash
python test/run_tests.py
```

### Exécuter uniquement les tests unitaires
```bash
python test/run_tests.py --unit
```

### Exécuter uniquement les tests d'intégration
```bash
python test/run_tests.py --integration
```

### Exécuter uniquement les tests end-to-end
```bash
python test/run_tests.py --e2e
```

### Mode verbeux
```bash
python test/run_tests.py --verbose
```

## 📝 Création de Nouveaux Tests

### 1. Test Unitaire
```python
# test/unit/preprocessor/test_nouvelle_fonction.py
import unittest
from src.preprocessor.nouvelle_fonction import NouvelleFonction

class TestNouvelleFonction(unittest.TestCase):
    def test_fonction_basique(self):
        """Test de base de la nouvelle fonction"""
        result = NouvelleFonction()
        self.assertIsNotNone(result)
```

### 2. Test d'Intégration
```python
# test/integration/test_pipeline_complet.py
import unittest
from src.preprocessor.data_cleaner import GeophysicalDataCleaner
from src.data.data_processor import GeophysicalDataProcessor

class TestPipelineIntegration(unittest.TestCase):
    def test_pipeline_complet(self):
        """Test du pipeline complet"""
        # Test d'intégration...
```

## 🔧 Configuration des Tests

### Variables d'Environnement
- `PYTHONPATH` : Doit inclure le répertoire racine du projet
- `TEST_DATA_DIR` : Répertoire des données de test (optionnel)

### Dépendances
- `unittest` : Framework de test standard Python
- `pytest` : Framework alternatif (optionnel)
- `coverage` : Mesure de couverture de code (optionnel)

## 📊 Couverture de Code

Pour mesurer la couverture de code :
```bash
# Installer coverage
pip install coverage

# Exécuter les tests avec coverage
coverage run test/run_tests.py

# Générer le rapport
coverage report

# Générer le rapport HTML
coverage html
```

## 🐛 Débogage des Tests

### Mode Verbose
```bash
python test/run_tests.py --verbose
```

### Test d'un Fichier Spécifique
```bash
python -m unittest test.unit.preprocessor.test_data_cleaner_init
```

### Test d'une Classe Spécifique
```bash
python -m unittest test.unit.preprocessor.test_data_cleaner_init.TestGeophysicalDataCleanerInit
```

### Test d'une Méthode Spécifique
```bash
python -m unittest test.unit.preprocessor.test_data_cleaner_init.TestGeophysicalDataCleanerInit.test_import_class
```

## 📋 Bonnes Pratiques

1. **Nommage** : Préfixer tous les fichiers de test par `test_`
2. **Organisation** : Organiser les tests par module et type
3. **Isolation** : Chaque test doit être indépendant
4. **Nettoyage** : Utiliser `setUp()` et `tearDown()` pour la configuration
5. **Assertions** : Utiliser des assertions spécifiques et descriptives
6. **Documentation** : Documenter chaque test avec des docstrings claires

## 🎉 Exemple de Test Réussi

```
🧪 EXÉCUTION DES TESTS UNITAIRES
==================================================
📊 14 tests unitaires découverts
...............
----------------------------------------------------------------------
Ran 14 tests in 0.076s

OK
📋 RÉSUMÉ DES TESTS
==============================
🧪 Tests unitaires: ✅ SUCCÈS
🔗 Tests d'intégration: ✅ SUCCÈS
🌐 Tests end-to-end: ✅ SUCCÈS

🎉 TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS!
```

## 🤝 Contribution

Pour ajouter de nouveaux tests :
1. Créer le fichier de test dans le bon répertoire
2. Suivre les conventions de nommage
3. Ajouter des tests pour les cas normaux et d'erreur
4. Vérifier que tous les tests passent
5. Documenter les nouveaux tests

---

**Note** : Ce système de tests suit les standards Python et les bonnes pratiques de développement logiciel.
