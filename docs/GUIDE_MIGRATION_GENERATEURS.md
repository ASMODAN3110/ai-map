# Guide de Migration - Modèles Générateurs Géophysiques

## 📋 **Vue d'ensemble**

Ce guide documente la migration des modèles de classification vers des modèles générateurs conformes au cahier des charges du projet AI-MAP.

## 🔄 **Changements Majeurs**

### **Avant (Modèles de Classification)**
- **CNN 2D** : ~2M paramètres, classification binaire (2 classes)
- **CNN 3D** : ~1.5M paramètres, classification binaire (2 classes)
- **Modèle Hybride** : ~12M paramètres, classification binaire (2 classes)
- **Objectif** : Classification des données géophysiques

### **Après (Modèles Générateurs)**
- **U-Net 2D** : ~31M paramètres, génération de pseudo-sections 2D
- **VoxNet 3D** : ~15M paramètres, génération de modèles 3D
- **Générateur Intégré** : ~46M paramètres total, pipeline complet
- **Objectif** : Génération d'images géophysiques à partir de données CSV

## 🏗️ **Architecture des Nouveaux Modèles**

### **U-Net 2D**
```python
# Spécifications conformes au cahier des charges
- Entrée: Tenseur 4D (64×64×4) - 4 canaux pour les dispositifs
- Encodeur: 4 blocs convolutionnels (64→128→256→512→1024 filtres)
- Décodeur: 4 blocs de déconvolution avec connexions résiduelles
- Sortie: 2 canaux (résistivité vraie, chargeabilité vraie)
- Paramètres: ~31M paramètres entraînables
```

### **VoxNet 3D**
```python
# Spécifications conformes au cahier des charges
- Entrée: Tenseur 5D (32×32×32×4) - Volume 3D multi-canaux
- Convolutions 3D: 3 couches (32→64→128 filtres)
- Déconvolutions 3D: Reconstruction volumétrique
- Sortie: Volume 3D de chargeabilité
- Paramètres: ~15M paramètres entraînables
```

## 📁 **Nouveaux Fichiers**

### **1. `backend/model/geophysical_generators.py`**
- **UNet2D** : Modèle U-Net 2D conforme au cahier des charges
- **VoxNet3D** : Modèle VoxNet 3D conforme au cahier des charges
- **GeophysicalDataProcessor** : Processeur de données pour préparer les entrées
- **GeophysicalImageGenerator** : Générateur intégré utilisant les deux modèles

### **2. `train_generators.py`**
- Script d'entraînement pour les nouveaux modèles générateurs
- Génération de données synthétiques (10,000 échantillons)
- Entraînement avec early stopping et validation croisée
- Sauvegarde des modèles dans `artifacts/models/`

### **3. `test_generators.py`**
- Script de test pour valider les modèles générateurs
- Tests de performance et de génération
- Validation de l'architecture et des sorties
- Génération de rapports de test

### **4. `demo_generators.py`**
- Script de démonstration des nouveaux modèles
- Exemples d'utilisation et de génération
- Validation des performances et de l'architecture
- Guide d'utilisation pour les développeurs

## 🔧 **Modifications des Fichiers Existants**

### **`api_server.py`**
```python
# Avant
from backend.model.image_generator import GeophysicalVisualizationGenerator
image_generator = GeophysicalVisualizationGenerator()

# Après
from backend.model.geophysical_generators import GeophysicalImageGenerator
image_generator = GeophysicalImageGenerator()
```

### **Endpoints API Mis à Jour**
- **`/api/models`** : Informations sur les nouveaux modèles générateurs
- **`/api/generate-images`** : Utilise les nouveaux générateurs
- **`/api/generate-sample-images`** : Génération d'exemples avec les nouveaux modèles

## 🚀 **Utilisation des Nouveaux Modèles**

### **1. Génération de Pseudo-sections 2D**
```python
from backend.model.geophysical_generators import GeophysicalImageGenerator

# Créer le générateur
generator = GeophysicalImageGenerator()

# Données CSV d'entrée
csv_data = np.array([
    [resistivity, chargeability, x_coord, y_coord],
    # ... plus d'échantillons
])

# Générer les pseudo-sections
pseudo_sections = generator.generate_pseudo_sections(csv_data, method="pole-dipole")
```

### **2. Génération de Modèles 3D**
```python
# Générer les modèles 3D
models_3d = generator.generate_3d_models(csv_data, method="pole-dipole")
```

### **3. Utilisation Directe des Modèles**
```python
from backend.model.geophysical_generators import UNet2D, VoxNet3D

# U-Net 2D
unet_2d = UNet2D()
output_2d = unet_2d(input_grids_2d)  # (batch_size, 2, 64, 64)

# VoxNet 3D
voxnet_3d = VoxNet3D()
output_3d = voxnet_3d(input_volumes_3d)  # (batch_size, 1, 32, 32, 32)
```

## 📊 **Données d'Entrée et de Sortie**

### **Format d'Entrée (CSV)**
```csv
resistivity,chargeability,x_coord,y_coord
100.5,15.2,50.0,75.0
250.8,8.7,25.0,60.0
...
```

### **Format de Sortie (Images Base64)**
```python
{
    "pseudo_sections": [
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
        # ... plus d'images
    ],
    "models_3d": [
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
        # ... plus d'images
    ]
}
```

## 🧪 **Tests et Validation**

### **1. Test des Modèles Individuels**
```bash
python test_generators.py
```

### **2. Démonstration Complète**
```bash
python demo_generators.py
```

### **3. Entraînement des Modèles**
```bash
python train_generators.py
```

## 📈 **Performances Attendues**

### **U-Net 2D**
- **Temps de génération** : < 1 seconde par échantillon
- **Résolution** : 64×64 pixels
- **Canaux de sortie** : 2 (résistivité + chargeabilité)

### **VoxNet 3D**
- **Temps de génération** : < 2 secondes par échantillon
- **Résolution** : 32×32×32 voxels
- **Canaux de sortie** : 1 (chargeabilité 3D)

### **Générateur Intégré**
- **Temps total** : < 3 secondes par échantillon
- **Sorties** : Pseudo-sections 2D + Modèles 3D
- **Format** : Images Base64 prêtes pour le frontend

## 🔄 **Migration du Frontend**

### **Aucun Changement Requis**
Le frontend existant est compatible avec les nouveaux modèles car :
- Les endpoints API restent identiques
- Le format de réponse (Base64) est conservé
- Les noms des champs de réponse sont maintenus

### **Améliorations Possibles**
- Affichage des informations sur les nouveaux modèles
- Visualisation des métriques de performance
- Interface pour l'entraînement des modèles

## 📚 **Documentation Supplémentaire**

### **Guides Disponibles**
- **`GUIDE_GENERATION_IMAGES.md`** : Guide détaillé de génération d'images
- **`README_TRAINING_GUIDE.md`** : Guide d'entraînement des modèles
- **`README_TESTS.md`** : Guide des tests et validation

### **Exemples de Code**
- **`demo_generators.py`** : Exemples d'utilisation complets
- **`test_generators.py`** : Tests de validation
- **`train_generators.py`** : Script d'entraînement

## ⚠️ **Points d'Attention**

### **1. Ressources Système**
- **Mémoire** : Les nouveaux modèles nécessitent plus de RAM
- **GPU** : Recommandé pour l'entraînement et l'inférence
- **Stockage** : Les modèles entraînés sont plus volumineux

### **2. Compatibilité**
- **Python** : 3.8+ requis
- **PyTorch** : 1.9+ requis
- **Dépendances** : Voir `requirements.txt`

### **3. Données d'Entrée**
- **Format CSV** : Colonnes numériques requises
- **Normalisation** : Effectuée automatiquement
- **Validation** : Vérification des formats et valeurs

## 🎯 **Prochaines Étapes**

### **1. Entraînement des Modèles**
```bash
# Entraîner les modèles sur des données synthétiques
python train_generators.py
```

### **2. Validation des Performances**
```bash
# Tester les modèles entraînés
python test_generators.py
```

### **3. Intégration avec le Frontend**
```bash
# Démarrer l'API avec les nouveaux modèles
python api_server.py
```

### **4. Tests End-to-End**
```bash
# Tester l'intégration complète
python test_api_integration.py
```

## 📞 **Support et Assistance**

Pour toute question ou problème lors de la migration :
1. Consultez les logs détaillés dans `logs/`
2. Exécutez les scripts de test et démonstration
3. Vérifiez la compatibilité des dépendances
4. Consultez la documentation des modèles

---

**✅ Migration Terminée** : Les modèles générateurs sont maintenant conformes au cahier des charges et prêts pour la génération d'images géophysiques professionnelles.
