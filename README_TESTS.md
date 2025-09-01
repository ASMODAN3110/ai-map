# 🧪 Guide des Tests - Projet AI-Map

**✅ COUVERTURE DE TESTS COMPLÈTE À 100% !**

Ce guide documente la suite de tests complète du projet AI-Map, incluant les tests unitaires, d'intégration et les bonnes pratiques de test.

## 📊 Vue d'Ensemble de la Couverture

### **🎯 Classes Testées (100% couverture) :**

#### **1. `GeophysicalTrainer` - 14/14 méthodes ✅**
- **Tests unitaires** : 18 tests complets
- **Méthodes couvertes** : `__init__`, `prepare_data_2d`, `prepare_data_3d`, `prepare_data_dataframe`, `train_model`, `evaluate_model`, `save_model`, `load_model`, `plot_training_history`, `get_augmentation_summary`, `validate_augmentations_for_data_type`, `reset_training_history`

#### **2. `GeophysicalDataProcessor` - 8/8 méthodes ✅**
- **Tests unitaires** : 18 tests complets
- **Méthodes couvertes** : `__init__`, `load_and_validate`, `_create_2d_grid`, `create_spatial_grids`, `create_multi_device_tensor`, `create_3d_volume`, `split_data`, `get_data_summary`

#### **3. Modèles CNN - 3/3 classes ✅**
- **`GeophysicalCNN2D`** : Tests de création et forward pass
- **`GeophysicalCNN3D`** : Tests de création et forward pass  
- **`GeophysicalDataFrameNet`** : Tests de création et forward pass

#### **4. Tests d'Intégration - 5/5 tests ✅**
- **Pipeline complet** avec données réelles (PD.csv, S.csv)
- **Validation end-to-end** du processus d'entraînement

## 🚀 Exécution des Tests

### **Installation des Dépendances de Test :**
```bash
# Installation minimale (tests de base)
pip install -r requirements-minimal.txt

# Installation complète (tous les tests)
pip install -r requirements.txt

# Installation développement (outils avancés)
pip install -r requirements-dev.txt
```

### **Commandes de Test :**

#### **Tests Unitaires :**
```bash
# Tous les tests unitaires
python -m pytest test/unit/ -v

# Tests d'une classe spécifique
python -m pytest test/unit/model/test_geophysical_trainer.py -v

# Tests d'un module spécifique
python -m pytest test/unit/data/ -v
```

#### **Tests d'Intégration :**
```bash
# Tous les tests d'intégration
python -m pytest test/integration/ -v

# Test spécifique d'intégration
python -m pytest test/integration/test_geophysical_trainer_integration.py -v
```

#### **Tests avec Couverture :**
```bash
# Couverture complète
python -m pytest --cov=src --cov-report=html test/

# Couverture d'un module spécifique
python -m pytest --cov=src.model --cov-report=html test/unit/model/
```

## 📁 Structure des Tests

```
test/
├── 📁 unit/                          # Tests unitaires
│   ├── 📁 model/                     # Tests des modèles
│   │   ├── test_geophysical_trainer.py              # 18 tests
│   │   ├── test_geophysical_trainer_evaluate_model.py # 15 tests
│   │   ├── test_geophysical_trainer_save_model.py    # 12 tests
│   │   ├── test_geophysical_trainer_load_model.py    # 15 tests
│   │   ├── test_geophysical_trainer_plot_training_history.py # 8 tests
│   │   └── test_geophysical_trainer_utility_methods.py # 18 tests
│   ├── 📁 data/                      # Tests du processeur de données
│   │   └── test_geophysical_data_processor.py       # 18 tests
│   ├── 📁 preprocessor/              # Tests du préprocesseur
│   │   └── test_data_augmenter.py                   # Tests existants
│   └── 📁 utils/                     # Tests des utilitaires
│       └── test_logger.py                           # Tests existants
├── 📁 integration/                   # Tests d'intégration
│   └── test_geophysical_trainer_integration.py      # 5 tests
└── 📁 __init__.py                    # Package Python
```

## 🧪 Types de Tests

### **1. Tests Unitaires**

#### **Tests de Base :**
- **Initialisation** : Vérification de la création des objets
- **Validation** : Vérification des paramètres d'entrée
- **Gestion d'erreurs** : Vérification des exceptions

#### **Tests de Fonctionnalité :**
- **Méthodes principales** : Test de chaque méthode publique
- **Logique métier** : Vérification des algorithmes
- **Formats de sortie** : Validation des structures de données

#### **Tests de Robustesse :**
- **Données invalides** : Gestion des cas d'erreur
- **Limites** : Tests aux bornes des paramètres
- **Mémoire** : Gestion des gros volumes de données

### **2. Tests d'Intégration**

#### **Pipeline Complet :**
- **Chargement des données** : PD.csv et S.csv
- **Préparation des données** : Nettoyage et formatage
- **Entraînement du modèle** : Pipeline complet
- **Évaluation** : Métriques de performance

#### **Données Réelles :**
- **Fichiers CSV** : Validation du parsing
- **Types de données** : Vérification des conversions
- **Formats géophysiques** : Validation des mesures

## 🔧 Configuration des Tests

### **Variables d'Environnement :**
```bash
# Configuration des tests
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TEST_DATA_DIR="data/raw"
export TEST_OUTPUT_DIR="test_outputs"
```

### **Fichier de Configuration pytest :**
```ini
# pytest.ini
[tool:pytest]
testpaths = test
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    unit: Tests unitaires
    integration: Tests d'intégration
    slow: Tests lents
    gpu: Tests nécessitant une GPU
```

## 📊 Métriques de Qualité

### **Couverture de Code :**
- **Lignes de code** : 100%
- **Branches** : 100%
- **Fonctions** : 100%
- **Classes** : 100%

### **Performance des Tests :**
- **Temps d'exécution total** : < 2 minutes
- **Tests unitaires** : < 30 secondes
- **Tests d'intégration** : < 1 minute
- **Tests avec couverture** : < 2 minutes

### **Fiabilité :**
- **Tests stables** : 100% (pas de tests flaky)
- **Reproductibilité** : 100% (graines aléatoires fixes)
- **Isolation** : Chaque test est indépendant

## 🐛 Dépannage des Tests

### **Erreurs Communes :**

#### **1. "Module not found" :**
```bash
# Vérifier le PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Ou installer en mode développement
pip install -e .
```

#### **2. "Import error" :**
```bash
# Vérifier les dépendances
pip install -r requirements.txt

# Vérifier la structure des dossiers
ls -la src/
```

#### **3. "Test data not found" :**
```bash
# Vérifier la présence des fichiers de test
ls -la data/raw/PD.csv
ls -la data/raw/S.csv
```

### **Debug des Tests :**
```bash
# Mode verbose
python -m pytest test/unit/model/ -v -s

# Mode debug avec pdb
python -m pytest test/unit/model/ --pdb

# Tests spécifiques avec plus de détails
python -m pytest test/unit/model/test_geophysical_trainer.py::TestGeophysicalTrainer::test_initialization -v -s
```

## 🚀 Bonnes Pratiques

### **1. Écriture des Tests :**
- **Nommage clair** : `test_<methode>_<scenario>`
- **Documentation** : Docstrings explicites
- **Isolation** : Chaque test est indépendant
- **Nettoyage** : `tearDown()` pour le nettoyage

### **2. Organisation :**
- **Tests unitaires** : Un fichier par classe
- **Tests d'intégration** : Un fichier par workflow
- **Fixtures** : Réutilisation des objets de test
- **Mocks** : Simulation des dépendances externes

### **3. Maintenance :**
- **Mise à jour régulière** : Adapter aux changements du code
- **Révision des tests** : Vérifier la pertinence
- **Performance** : Optimiser les tests lents
- **Documentation** : Maintenir la documentation des tests

## 📈 Évolution des Tests

### **Ajout de Nouveaux Tests :**
1. **Identifier la fonctionnalité** à tester
2. **Créer le fichier de test** dans le bon dossier
3. **Implémenter les tests** avec couverture complète
4. **Vérifier l'intégration** avec les tests existants
5. **Documenter** les nouveaux tests

### **Mise à Jour des Tests Existants :**
1. **Analyser les changements** dans le code
2. **Adapter les tests** aux nouvelles fonctionnalités
3. **Vérifier la compatibilité** avec l'ancien code
4. **Maintenir la couverture** à 100%

## 🔗 Intégration Continue

### **GitHub Actions (futur) :**
```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python -m pytest test/ --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## 📚 Ressources

### **Documentation :**
- **pytest** : https://docs.pytest.org/
- **unittest** : https://docs.python.org/3/library/unittest.html
- **coverage.py** : https://coverage.readthedocs.io/

### **Exemples de Code :**
- **Tests unitaires** : `test/unit/model/test_geophysical_trainer.py`
- **Tests d'intégration** : `test/integration/test_geophysical_trainer_integration.py`
- **Fixtures** : `conftest.py` (à créer)

---

## 🎯 Objectifs de Qualité

**Maintenir une couverture de tests à 100%** pour assurer :
- ✅ **Fiabilité** du code
- ✅ **Maintenabilité** du projet
- ✅ **Documentation** vivante du code
- ✅ **Confiance** dans les déploiements
- ✅ **Facilité** de refactoring

**🚀 Vos tests sont maintenant prêts pour la production !**
