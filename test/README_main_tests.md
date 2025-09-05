# Tests pour le fichier main.py

Ce répertoire contient tous les tests pour le fichier `main.py` du projet AI-MAP.

## Structure des tests

```
test/
├── unit/
│   └── test_main.py              # Tests unitaires pour main.py
├── integration/
│   └── test_main_integration.py  # Tests d'intégration pour main.py
├── test_main_runner.py           # Script pour exécuter les tests
└── README_main_tests.md          # Ce fichier
```

## Types de tests

### Tests unitaires (`test/unit/test_main.py`)

Les tests unitaires testent chaque fonction individuellement avec des mocks :

- **TestMainPipeline** : Tests pour les fonctions principales du pipeline
  - `test_phase1_data_cleaning_success()` : Test de la phase 1 de nettoyage
  - `test_phase2_data_processing_success()` : Test de la phase 2 de traitement
  - `test_phase3_data_preparation_with_data()` : Test de la phase 3 de préparation
  - `test_phase4_model_training_cnn_2d()` : Test de la phase 4 d'entraînement
  - `test_phase5_evaluation_and_results()` : Test de la phase 5 d'évaluation
  - `test_train_cnn_2d()` : Test de l'entraînement CNN 2D
  - `test_train_cnn_3d()` : Test de l'entraînement CNN 3D
  - `test_train_hybrid_model()` : Test de l'entraînement hybride
  - `test_train_dataframe_model()` : Test de l'entraînement DataFrame
  - `test_main_success()` : Test de la fonction main()
  - `test_main_failure()` : Test de la gestion d'erreurs
  - `test_main_with_args_success()` : Test de main_with_args()
  - `test_parse_arguments_default()` : Test du parsing d'arguments
  - `test_parse_arguments_custom()` : Test du parsing avec arguments personnalisés

- **TestMainIntegration** : Tests d'intégration basiques
  - `test_main_module_import()` : Test d'import du module
  - `test_parse_arguments_help()` : Test de l'aide des arguments
  - `test_parse_arguments_invalid_model()` : Test avec modèle invalide

### Tests d'intégration (`test/integration/test_main_integration.py`)

Les tests d'intégration testent le pipeline complet avec des données simulées :

- **TestMainIntegration** : Tests d'intégration complets
  - `test_main_pipeline_complete_simulation()` : Test du pipeline complet
  - `test_main_with_args_cnn_2d()` : Test avec modèle CNN 2D
  - `test_main_with_args_skip_phases()` : Test avec phases ignorées
  - `test_main_with_args_hybrid_model()` : Test avec modèle hybride
  - `test_main_with_args_dataframe_model()` : Test avec modèle DataFrame
  - `test_main_with_args_cnn_3d()` : Test avec modèle CNN 3D
  - `test_parse_arguments_all_models()` : Test de tous les modèles
  - `test_parse_arguments_all_devices()` : Test de tous les devices
  - `test_parse_arguments_numeric_values()` : Test des valeurs numériques
  - `test_parse_arguments_boolean_flags()` : Test des flags booléens
  - `test_parse_arguments_output_dir()` : Test du répertoire de sortie

## Exécution des tests

### Exécuter tous les tests

```bash
# Depuis la racine du projet
python test/test_main_runner.py

# Ou depuis le répertoire test
cd test
python test_main_runner.py
```

### Exécuter une classe de test spécifique

```bash
# Tests unitaires
python test/test_main_runner.py TestMainPipeline

# Tests d'intégration
python test/test_main_runner.py TestMainIntegration
```

### Exécuter une méthode de test spécifique

```bash
# Test spécifique
python test/test_main_runner.py TestMainPipeline test_phase1_data_cleaning_success
```

### Exécuter avec unittest directement

```bash
# Tests unitaires
python -m unittest test.unit.test_main -v

# Tests d'intégration
python -m unittest test.integration.test_main_integration -v

# Tous les tests
python -m unittest discover test -p "test_main*.py" -v
```

## Couverture des tests

Les tests couvrent :

### Fonctions principales
- ✅ `phase1_data_cleaning()`
- ✅ `phase2_data_processing()`
- ✅ `phase3_data_preparation()`
- ✅ `phase4_model_training()`
- ✅ `phase5_evaluation_and_results()`
- ✅ `main()`
- ✅ `main_with_args()`
- ✅ `parse_arguments()`

### Fonctions d'entraînement
- ✅ `train_cnn_2d()`
- ✅ `train_cnn_3d()`
- ✅ `train_hybrid_model()`
- ✅ `train_dataframe_model()`

### Types de modèles
- ✅ CNN 2D
- ✅ CNN 3D
- ✅ Modèle hybride
- ✅ Modèle DataFrame

### Arguments CLI
- ✅ Tous les modèles supportés
- ✅ Tous les devices supportés
- ✅ Valeurs numériques
- ✅ Flags booléens
- ✅ Répertoire de sortie

### Gestion d'erreurs
- ✅ Erreurs d'import
- ✅ Erreurs d'entraînement
- ✅ Arguments invalides
- ✅ Données manquantes

## Dépendances des tests

Les tests utilisent les mocks suivants :
- `unittest.mock.Mock` : Pour mocker les dépendances
- `unittest.mock.patch` : Pour patcher les imports
- `tempfile` : Pour créer des répertoires temporaires
- `shutil` : Pour le nettoyage des fichiers temporaires

## Notes importantes

1. **Mocks** : Tous les tests utilisent des mocks pour éviter les dépendances externes
2. **Données factices** : Les tests créent des données factices pour simuler les entrées
3. **Nettoyage** : Chaque test nettoie ses fichiers temporaires
4. **Isolation** : Les tests sont isolés et ne dépendent pas les uns des autres
5. **Couverture** : Les tests couvrent les cas de succès et d'échec

## Ajout de nouveaux tests

Pour ajouter de nouveaux tests :

1. **Tests unitaires** : Ajouter dans `test/unit/test_main.py`
2. **Tests d'intégration** : Ajouter dans `test/integration/test_main_integration.py`
3. **Nommage** : Suivre la convention `test_nom_de_la_fonction`
4. **Documentation** : Ajouter une docstring explicative
5. **Mocks** : Utiliser des mocks appropriés pour les dépendances

## Exécution en CI/CD

Pour exécuter les tests dans un pipeline CI/CD :

```bash
# Installation des dépendances
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Exécution des tests
python test/test_main_runner.py

# Vérification du code de sortie
echo $?
```
