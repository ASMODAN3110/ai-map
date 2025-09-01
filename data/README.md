# 📁 Organisation des Données Géophysiques

Ce dossier contient toutes les données organisées pour l'entraînement et les tests du modèle AI-MAP.

## 🏗️ Structure des Dossiers

```
data/
├── raw/                          # Données brutes originales
│   ├── images/                   # Images géophysiques
│   │   ├── resistivity/         # Images de résistivité (20 fichiers)
│   │   ├── chargeability/       # Images de chargeabilité (11 fichiers)
│   │   └── profiles/            # Images de profils (6 fichiers)
│   └── csv/                     # Données CSV
│       └── profiles/            # Profils géophysiques (14 fichiers)
├── training/                     # Données d'entraînement (80%)
│   ├── images/                  # Images pour l'entraînement
│   │   ├── resistivity/         # Images de résistivité d'entraînement
│   │   ├── chargeability/       # Images de chargeabilité d'entraînement
│   │   └── profiles/            # Images de profils d'entraînement
│   └── csv/                     # CSV pour l'entraînement
├── test/                        # Données de test (20%)
│   ├── images/                  # Images pour les tests
│   │   ├── resistivity/         # Images de résistivité de test
│   │   ├── chargeability/       # Images de chargeabilité de test
│   │   └── profiles/            # Images de profils de test
│   └── csv/                     # CSV pour les tests
├── processed/                    # Données traitées
│   ├── schlumberger_cleaned.csv # Données Schlumberger nettoyées
│   └── pole_dipole_cleaned.csv  # Données Pole-Dipole nettoyées
├── metadata.json                 # Métadonnées du projet
└── README.md                     # Ce fichier
```

## 📊 Types de Données

### 🖼️ Images Géophysiques

#### **Résistivité (Resistivity)**
- **Format** : JPG
- **Quantité** : 20 images
- **Description** : Mesures de résistivité électrique du sous-sol
- **Profondeurs** : 0m, 20m, 30m, 50m, 70m, 85m, 100m, 150m
- **Usage** : Détection de structures géologiques, nappes phréatiques

#### **Chargeabilité (Chargeability)**
- **Format** : JPG, PNG
- **Quantité** : 11 images
- **Description** : Mesures de chargeabilité électrique du sous-sol
- **Usage** : Détection de minéralisation, argiles, structures polarisables

#### **Profils Géologiques**
- **Format** : JPG
- **Quantité** : 6 images
- **Description** : Profils combinant résistivité et chargeabilité
- **Usage** : Analyse intégrée des propriétés géophysiques

### 📈 Données CSV

#### **Profils Géophysiques**
- **Format** : CSV
- **Quantité** : 14 fichiers
- **Description** : Données numériques des profils géophysiques
- **Colonnes** : Coordonnées, résistivité, chargeabilité, profondeur

## 🎯 Utilisation pour l'IA

### **Entraînement (80% des données)**
- **Objectif** : Entraîner les modèles CNN
- **Augmentation** : Utilisation de l'ImageAugmenter
- **Techniques** : Rotation, retournement, bruit, déformation élastique

### **Test (20% des données)**
- **Objectif** : Évaluer les performances du modèle
- **Validation** : Mesurer la précision et la robustesse
- **Généralisation** : Tester sur des données non vues

## 🚀 Chargement des Données

### **Exemple Python**

```python
from pathlib import Path
from src.data.image_processor import ImageAugmenter

# Charger les données d'entraînement
train_path = Path("data/training/images")
resistivity_files = list(train_path.glob("resistivity/*.JPG"))
chargeability_files = list(train_path.glob("chargeability/*.JPG"))

# Initialiser l'augmenteur
augmenter = ImageAugmenter(random_seed=42)

# Augmenter une image
from PIL import Image
image = Image.open(resistivity_files[0])
augmented = augmenter.augment_image(
    image, 
    ["rotation", "gaussian_noise", "elastic_deformation"]
)
```

### **Script d'Exemple Complet**

```bash
# Lancer l'exemple d'utilisation
python examples/data_loader_example.py
```

## 🔧 Organisation Automatique

### **Script d'Organisation**

```bash
# Réorganiser les données
python organize_data.py
```

Ce script :
- Répartit automatiquement les données (80% entraînement, 20% test)
- Maintient la cohérence entre images et CSV
- Utilise une graine aléatoire pour la reproductibilité

## 📋 Métadonnées

Le fichier `metadata.json` contient :
- Description complète du projet
- Structure détaillée des données
- Informations géophysiques
- Techniques d'augmentation disponibles
- Utilisation prévue pour l'IA

## 🎨 Augmentation des Données

### **Techniques Disponibles**

#### **Techniques de Base**
- `rotation` : Rotation aléatoire (-15° à +15°)
- `flip_horizontal` : Retournement horizontal
- `flip_vertical` : Retournement vertical
- `brightness` : Variation de luminosité
- `contrast` : Variation de contraste

#### **Techniques Avancées**
- `elastic_deformation` : Déformation élastique (plis géologiques)
- `color_jittering` : Variation de couleur
- `gaussian_noise` : Bruit gaussien réaliste
- `blur_sharpen` : Flou ou aiguisage
- `perspective_transform` : Transformation de perspective
- `cutout` : Masquage de zones

#### **Techniques Géophysiques**
- `geological_stratification` : Couches géologiques
- `fracture_patterns` : Motifs de fractures
- `mineral_inclusions` : Inclusions minérales
- `weathering_effects` : Effets d'altération
- `sedimentary_layers` : Couches sédimentaires

## 🔍 Validation des Données

### **Vérifications Automatiques**
- Format des fichiers
- Cohérence des dimensions
- Intégrité des données CSV
- Correspondance images/CSV

### **Tests de Qualité**
- Résolution des images
- Plages de valeurs géophysiques
- Couverture spatiale
- Métadonnées complètes

## 📈 Prochaines Étapes

1. **Entraînement du Modèle**
   - Utiliser les données d'entraînement organisées
   - Appliquer l'augmentation pour enrichir le dataset
   - Valider sur les données de test

2. **Optimisation**
   - Ajuster les paramètres d'augmentation
   - Équilibrer les classes de données
   - Améliorer la qualité des images

3. **Déploiement**
   - Intégrer dans le pipeline de production
   - Automatiser le traitement des nouvelles données
   - Surveiller les performances en continu

## 🤝 Contribution

Pour ajouter de nouvelles données :
1. Placer les fichiers dans le dossier `raw` approprié
2. Mettre à jour `metadata.json`
3. Réexécuter `organize_data.py`
4. Tester la cohérence des données

## 📞 Support

En cas de questions sur l'organisation des données :
- Consulter `metadata.json` pour les détails
- Vérifier les exemples dans `examples/`
- Consulter la documentation du projet
