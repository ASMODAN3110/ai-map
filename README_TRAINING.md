# Module d'Entraînement Géophysique

Ce module fournit des fonctionnalités complètes d'entraînement pour les modèles de deep learning spécialisés dans l'analyse de données géophysiques, avec intégration automatique de l'augmentation de données et support des modèles hybrides.

## 🎯 Modèles Disponibles

### **✅ Modèles Opérationnels**
- **CNN 2D** : `cnn_2d_model.pth` - Modèle géophysique 2D
- **CNN 3D** : `cnn_3d_model.pth` - Modèle géophysique 3D  
- **Modèle Hybride** : `hybrid_model.pth` - Images + Données géophysiques

### **🚀 Scripts d'Exécution**
- **run_cnn_2d_model.py** : Exécution du modèle CNN 2D
- **run_cnn_3d_model.py** : Exécution du modèle CNN 3D
- **run_hybrid_model.py** : Exécution du modèle hybride

## 🏗️ Architecture

### Classes Principales

#### 1. **GeophysicalCNN2D** (cnn_2d_model.pth)
- **Réseau convolutif 2D** optimisé pour les grilles géophysiques
- **Architecture** : 4 couches de convolution + couches fully connected
- **Entrée** : Grilles 2D (4, 64, 64) - 4 canaux pour les dispositifs
- **Sortie** : Classification binaire (2 classes)
- **Paramètres** : ~2M paramètres entraînables

#### 2. **GeophysicalCNN3D** (cnn_3d_model.pth)
- **Réseau convolutif 3D** pour les volumes géophysiques
- **Architecture** : 3 couches de convolution 3D + couches fully connected
- **Entrée** : Volumes 3D (4, 32, 32, 32) - 4 canaux multi-dispositifs
- **Sortie** : Classification binaire (2 classes)
- **Paramètres** : ~1.5M paramètres entraînables

#### 3. **GeophysicalHybridNet** (hybrid_model.pth)
- **Modèle hybride** combinant images et données géophysiques
- **Architecture** : ResNet18 + Encodeur géophysique + Module de fusion
- **Entrée** : Images (3, 64, 64) + Données géophysiques (4,)
- **Sortie** : Classification binaire (2 classes)
- **Paramètres** : ~12M paramètres entraînables

#### 4. **GeophysicalTrainer**
- **Entraîneur spécialisé** intégrant l'augmenteur de données
- **Fonctionnalités** : Préparation, entraînement, évaluation, sauvegarde
- **Support multi-device** : CPU/GPU automatique

## 🚀 Fonctionnalités

### **🎯 Exécution des Modèles Sauvegardés**

#### **CNN 2D**
```bash
# Exécution simple
python run_cnn_2d_model.py

# Avec données réelles
python run_cnn_2d_model.py --real-data

# Mode verbose
python run_cnn_2d_model.py --verbose
```

#### **CNN 3D**
```bash
# Exécution simple
python run_cnn_3d_model.py

# Avec données réelles
python run_cnn_3d_model.py --real-data

# Mode verbose
python run_cnn_3d_model.py --verbose
```

#### **Modèle Hybride**
```bash
# Exécution simple
python run_hybrid_model.py

# Avec données réelles
python run_hybrid_model.py --real-data

# Mode verbose
python run_hybrid_model.py --verbose
```

### **🚀 Entraînement de Nouveaux Modèles**

#### **Via Pipeline Principal**
```bash
# CNN 2D
python main.py --model cnn_2d --epochs 50 --batch-size 16

# CNN 3D
python main.py --model cnn_3d --epochs 80 --batch-size 16

# Modèle Hybride
python main.py --model hybrid --epochs 60 --learning-rate 0.0005
```

### Préparation des Données avec Augmentation

#### `prepare_data_2d()`
```python
train_loader, val_loader = trainer.prepare_data_2d(
    grids=grids_2d,
    labels=labels,
    augmentations=["rotation", "flip_horizontal", "gaussian_noise"],
    num_augmentations=5,
    test_size=0.2
)
```

#### `prepare_data_3d()`
```python
train_loader, val_loader = trainer.prepare_data_3d(
    volumes=volumes_3d,
    labels=labels,
    augmentations=["rotation", "gaussian_noise"],
    num_augmentations=3,
    test_size=0.2
)
```

#### `prepare_data_dataframe()`
```python
train_loader, val_loader = trainer.prepare_data_dataframe(
    dataframes=dataframes,
    labels=labels,
    augmentations=["gaussian_noise", "value_variation"],
    num_augmentations=5,
    test_size=0.2
)
```

### Entraînement du Modèle

#### `train_model()`
```python
history = trainer.train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    learning_rate=0.001,
    weight_decay=1e-5,
    patience=10
)
```

**Fonctionnalités d'entraînement :**
- ✅ **Early stopping** automatique
- ✅ **Learning rate scheduling** adaptatif
- ✅ **Métriques en temps réel** (loss, accuracy)
- ✅ **Sauvegarde automatique** du meilleur modèle
- ✅ **Logging détaillé** des performances

### Évaluation et Analyse

#### `evaluate_model()`
```python
metrics = trainer.evaluate_model(model, test_loader)
# Retourne: test_loss, test_accuracy, classification_report, confusion_matrix
```

#### `plot_training_history()`
```python
trainer.plot_training_history(save_path="training_history.png")
# Génère des graphiques d'évolution des métriques
```

### Gestion des Modèles

#### `save_model()` / `load_model()`
```python
# Sauvegarde complète (poids + historique + configuration)
trainer.save_model(model, "model_complet.pth")

# Chargement du modèle
model = trainer.load_model(model, "model_complet.pth")
```

## 📊 Techniques d'Augmentation Supportées

### Données 2D
- **Géométriques** : Rotation, Flip horizontal/vertical, Décalage spatial
- **Bruit** : Bruit gaussien, Bruit sel-et-poivre
- **Variations** : Variation des valeurs, Déformation élastique

### Données 3D
- **Géométriques** : Rotation, Flip horizontal/vertical
- **Bruit** : Bruit gaussien
- **Variations** : Variation des valeurs

### DataFrames
- **Bruit** : Bruit gaussien
- **Variations** : Variation des valeurs, Jitter spatial, Perturbation des coordonnées

## 🔧 Utilisation

### Installation des Dépendances

#### **Installation Minimale (recommandée) :**
```bash
pip install -r requirements-minimal.txt
```

#### **Installation Complète :**
```bash
pip install -r requirements.txt
```

#### **Installation Développement :**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### **✅ Tests et Validation**

**Couverture de tests complète à 100% !**

```bash
# Lancer tous les tests unitaires
python -m pytest test/unit/model/ -v

# Lancer les tests d'intégration
python -m pytest test/integration/ -v

# Tests avec couverture
python -m pytest --cov=src --cov-report=html test/
```

**Tests disponibles :**
- **108 tests unitaires** pour toutes les méthodes du trainer
- **5 tests d'intégration** avec données réelles
- **Validation complète** du pipeline d'entraînement
- **Tests spécialisés** pour chaque fonctionnalité (évaluation, sauvegarde, chargement, etc.)

### Exemple Complet
```python
from src.preprocessor.data_augmenter import GeophysicalDataAugmenter
from src.model.geophysical_trainer import GeophysicalTrainer, GeophysicalCNN2D

# 1. Initialiser
augmenter = GeophysicalDataAugmenter(random_seed=42)
trainer = GeophysicalTrainer(augmenter, device="auto")

# 2. Créer le modèle
model = GeophysicalCNN2D(input_channels=4, num_classes=2, grid_size=64)

# 3. Préparer les données avec augmentation
train_loader, val_loader = trainer.prepare_data_2d(
    grids=mes_grilles,
    labels=mes_labels,
    augmentations=["rotation", "gaussian_noise"],
    num_augmentations=5
)

# 4. Entraîner
history = trainer.train_model(model, train_loader, val_loader, num_epochs=50)

# 5. Évaluer
metrics = trainer.evaluate_model(model, test_loader)
print(f"Accuracy: {metrics['test_accuracy']:.2f}%")
```

## 📈 Métriques et Monitoring

### Métriques d'Entraînement
- **Loss** : Entraînement et validation
- **Accuracy** : Entraînement et validation
- **Learning rate** : Évolution automatique

### Graphiques Automatiques
- 📊 Évolution de la loss
- 📈 Évolution de l'accuracy
- 🔄 Évolution du learning rate
- 📋 Vue d'ensemble des métriques

## 🎯 Cas d'Usage

### 1. **Classification de Résistivité**
- Entrée : Grilles 2D de résistivité
- Sortie : Classification des types de sol
- Augmentation : Rotation, bruit, variations

### 2. **Détection d'Anomalies 3D**
- Entrée : Volumes 3D géophysiques
- Sortie : Localisation des anomalies
- Augmentation : Rotation, bruit

### 3. **Analyse de Données Tabulaire**
- Entrée : Features extraites des sondages
- Sortie : Classification des formations
- Augmentation : Bruit, variations

## ⚠️ Bonnes Pratiques

### 1. **Validation des Augmentations**
```python
# Valider avant utilisation
is_valid = trainer.validate_augmentations_for_data_type(
    augmentations=["rotation", "gaussian_noise"], 
    data_type="2d"
)
```

### 2. **Gestion de la Mémoire**
- **2D** : Batch size 32-64
- **3D** : Batch size 16-32 (plus petit pour la mémoire)
- **DataFrame** : Batch size 64-128

### 3. **Reproductibilité**
```python
# Toujours définir un seed
augmenter = GeophysicalDataAugmenter(random_seed=42)
trainer = GeophysicalTrainer(augmenter, device="auto")
```

## 🐛 Dépannage

### Erreurs Communes

#### 1. **"CUDA out of memory"**
- Réduire le batch size
- Utiliser des grilles plus petites
- Activer le gradient checkpointing

#### 2. **"Input shape mismatch"**
- Vérifier `input_channels` et `grid_size`
- S'assurer que les données sont dans le bon format
- Utiliser `model._calculate_feature_size()` pour vérifier

#### 3. **"Augmentation not supported"**
- Utiliser `validate_augmentations_for_data_type()`
- Vérifier la liste des augmentations supportées
- Adapter les augmentations au type de données

## 📚 Exemples Supplémentaires

Consultez le fichier `examples/training_example.py` pour un exemple complet d'utilisation avec données simulées.

## 🔗 Intégration

Ce module s'intègre parfaitement avec :
- ✅ **DataCleaner** : Nettoyage des données brutes
- ✅ **DataAugmenter** : Augmentation automatique
- ✅ **DataProcessor** : Traitement des données
- ✅ **Logger** : Suivi des opérations

---

**Note** : Ce module est conçu pour être extensible. Vous pouvez facilement ajouter de nouveaux types de modèles ou techniques d'augmentation en héritant des classes existantes.
