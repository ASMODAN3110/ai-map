# 🚀 Module d'Augmentation de Données Géophysiques

Ce module fournit des techniques d'augmentation de données spécifiquement adaptées aux données géophysiques (résistivité, chargeabilité) pour améliorer l'entraînement des modèles CNN.

## 📋 Fonctionnalités

### 🎯 Techniques d'Augmentation 2D
- **Rotation** : Rotation aléatoire de 90°, 180° ou 270°
- **Retournement** : Horizontal et vertical
- **Décalage spatial** : Déplacement aléatoire avec remplissage par zéros
- **Bruit gaussien** : Ajout de bruit pour simuler les erreurs de mesure
- **Bruit poivre et sel** : Simulation d'artefacts de mesure
- **Variation des valeurs** : Légères variations de résistivité et chargeabilité
- **Déformation élastique** : Déformation spatiale réaliste

### 🎯 Techniques d'Augmentation 3D
- **Rotation** : Rotation autour des axes X, Y, Z
- **Retournement** : Horizontal et vertical
- **Bruit gaussien** : Ajout de bruit volumétrique
- **Variation des valeurs** : Variations dans l'espace 3D

### 🎯 Techniques d'Augmentation DataFrame
- **Bruit gaussien** : Ajout de bruit aux colonnes numériques
- **Variation des valeurs** : Variations des mesures géophysiques
- **Jitter spatial** : Perturbation des coordonnées spatiales
- **Perturbation des coordonnées** : Variations mineures des positions

## 🚀 Installation et Utilisation

### Import du module
```python
from src.preprocessor.data_augmenter import GeophysicalDataAugmenter
```

### Initialisation
```python
# Avec graine aléatoire pour la reproductibilité
augmenter = GeophysicalDataAugmenter(random_seed=42)

# Sans graine (aléatoire)
augmenter = GeophysicalDataAugmenter()
```

## 📊 Exemples d'Utilisation

### 1. Augmentation de Grilles 2D
```python
import numpy as np

# Créer une grille de test (64x64x4)
grid_2d = np.random.rand(64, 64, 4)

# Techniques d'augmentation recommandées
augmentations = ["rotation", "flip_horizontal", "gaussian_noise", "spatial_shift"]

# Générer 3 augmentations
augmented_grids = augmenter.augment_2d_grid(
    grid_2d, 
    augmentations, 
    num_augmentations=3
)

print(f"Généré {len(augmented_grids)} grilles augmentées")
```

### 2. Augmentation de Volumes 3D
```python
# Créer un volume de test (32x32x32x4)
volume_3d = np.random.rand(32, 32, 32, 4)

# Augmenter le volume
augmented_volumes = augmenter.augment_3d_volume(
    volume_3d, 
    ["rotation", "flip_horizontal"], 
    num_augmentations=2
)
```

### 3. Augmentation de DataFrames
```python
import pandas as pd

# Créer un DataFrame de test
df = pd.DataFrame({
    'x': np.random.rand(100) * 1000,
    'y': np.random.rand(100) * 1000,
    'resistivity': np.random.rand(100) * 1000,
    'chargeability': np.random.rand(100) * 200
})

# Augmenter le DataFrame
augmented_dfs = augmenter.augment_dataframe(
    df, 
    ["gaussian_noise", "spatial_jitter"], 
    num_augmentations=2
)
```

## 🔧 Configuration Avancée

### Obtenir les Recommandations
```python
# Recommandations pour grilles 2D
recommendations_2d = augmenter.get_recommended_augmentations("2d_grid")
print(f"Techniques 2D recommandées: {recommendations_2d}")

# Recommandations pour volumes 3D
recommendations_3d = augmenter.get_recommended_augmentations("3d_volume")
print(f"Techniques 3D recommandées: {recommendations_3d}")

# Recommandations pour DataFrames
recommendations_df = augmenter.get_recommended_augmentations("dataframe")
print(f"Techniques DataFrame recommandées: {recommendations_df}")
```

### Validation des Paramètres
```python
# Valider les techniques d'augmentation
augmentations = ["rotation", "flip_horizontal", "invalid_technique"]
is_valid = augmenter.validate_augmentation_parameters(augmentations, "2d_grid")
print(f"Paramètres valides: {is_valid}")
```

### Suivi des Augmentations
```python
# Obtenir un résumé des augmentations effectuées
summary = augmenter.get_augmentation_summary()
print(f"Total des augmentations: {summary['total_augmentations']}")
print(f"Types utilisés: {summary['augmentation_types']}")

# Réinitialiser l'historique
augmenter.reset_history()
```

## 🎨 Intégration dans le Pipeline

### Avec le DataProcessor
```python
from src.data.data_processor import GeophysicalDataProcessor
from src.preprocessor.data_augmenter import GeophysicalDataAugmenter

# Initialiser les composants
processor = GeophysicalDataProcessor()
augmenter = GeophysicalDataAugmenter(random_seed=42)

# Charger et traiter les données
device_data = processor.loadAnd_validate()
multi_device_tensor = processor.create_multi_device_tensor()

# Augmenter les données avant l'entraînement
augmented_tensors = augmenter.augment_2d_grid(
    multi_device_tensor[0],  # Première grille
    ["rotation", "flip_horizontal", "gaussian_noise"],
    num_augmentations=5
)

# Combiner les données originales et augmentées
training_data = np.vstack([multi_device_tensor, np.array(augmented_tensors)])
print(f"Données d'entraînement: {training_data.shape}")
```

## ⚙️ Paramètres Recommandés

### Pour l'Entraînement CNN
```python
# Augmentations modérées pour éviter la sur-augmentation
moderate_augmentations = [
    "rotation",           # Rotation géométrique
    "flip_horizontal",    # Retournement horizontal
    "gaussian_noise",     # Bruit réaliste
    "spatial_shift"       # Décalage mineur
]

# Augmentations agressives pour plus de variabilité
aggressive_augmentations = [
    "rotation", "flip_horizontal", "flip_vertical",
    "gaussian_noise", "salt_pepper_noise", "spatial_shift",
    "value_variation", "elastic_deformation"
]

# Augmentations pour DataFrames
dataframe_augmentations = [
    "gaussian_noise", "value_variation", 
    "spatial_jitter", "coordinate_perturbation"
]
```

## 🔍 Débogage et Surveillance

### Vérifier l'État de l'Augmenteur
```python
# Vérifier l'historique
print(f"Historique: {len(augmenter.augmentation_history)} augmentations")

# Vérifier la graine aléatoire
if hasattr(augmenter, '_random_seed'):
    print(f"Graine aléatoire: {augmenter._random_seed}")

# Obtenir un résumé détaillé
summary = augmenter.get_augmentation_summary()
for key, value in summary.items():
    print(f"{key}: {value}")
```

### Gestion des Erreurs
```python
try:
    # Tentative d'augmentation
    augmented_data = augmenter.augment_2d_grid(
        invalid_grid,  # Grille de mauvaise dimension
        ["rotation"]
    )
except ValueError as e:
    print(f"Erreur d'augmentation: {e}")
    # Gérer l'erreur appropriée
```

## 📈 Performance et Optimisation

### Mesurer les Performances
```python
import time

start_time = time.time()
augmented_grids = augmenter.augment_2d_grid(
    large_grid, 
    ["rotation", "flip_horizontal", "gaussian_noise"], 
    num_augmentations=10
)
execution_time = time.time() - start_time

print(f"10 augmentations en {execution_time:.3f} secondes")
print(f"Taux: {10/execution_time:.1f} augmentations/seconde")
```

### Optimisations Recommandées
- **Graine aléatoire** : Utiliser une graine fixe pour la reproductibilité
- **Taille des données** : Augmenter par lots pour les gros volumes
- **Techniques sélectives** : Choisir les techniques les plus pertinentes
- **Cache des résultats** : Stocker les augmentations fréquemment utilisées

## 🧪 Tests et Validation

### Exécuter les Tests
```bash
# Tests unitaires
python test/unit/preprocessor/test_data_augmenter.py

# Tests avec couverture (si disponible)
python -m pytest test/unit/preprocessor/test_data_augmenter.py --cov=src.preprocessor.data_augmenter
```

### Exemple de Démonstration
```bash
# Lancer l'exemple complet
python examples/data_augmentation_example.py
```

## 🔮 Extensions Futures

### Techniques d'Augmentation Avancées
- **Augmentation conditionnelle** : Basée sur les caractéristiques géologiques
- **Augmentation adaptative** : Ajustement automatique des paramètres
- **Augmentation multi-échelle** : Variations à différentes résolutions
- **Augmentation physique** : Basée sur les lois de la géophysique

### Intégration avec d'Autres Frameworks
- **TensorFlow/Keras** : Générateurs de données personnalisés
- **PyTorch** : Transforms personnalisés
- **Scikit-learn** : Pipelines de prétraitement

## 📚 Références et Ressources

- **Documentation NumPy** : https://numpy.org/doc/
- **Documentation SciPy** : https://scipy.org/docs/
- **Documentation Pandas** : https://pandas.pydata.org/docs/
- **Techniques d'Augmentation en Vision par Ordinateur** : Papers sur arXiv

---

## 🤝 Contribution

Pour contribuer à ce module :
1. Ajouter de nouvelles techniques d'augmentation
2. Améliorer la documentation
3. Ajouter des tests unitaires
4. Optimiser les performances
5. Corriger les bugs

## 📄 Licence

Ce module fait partie du projet AI-MAP et suit la même licence que le projet principal.
