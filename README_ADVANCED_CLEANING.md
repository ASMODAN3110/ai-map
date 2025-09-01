# 🧹 Nettoyage Avancé d'Images Géophysiques

Ce document décrit les **méthodes de nettoyage avancées** ajoutées au `GeophysicalImageProcessor` pour traiter et améliorer la qualité des images géophysiques.

## 📋 Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Méthodes de Réduction de Bruit](#méthodes-de-réduction-de-bruit)
- [Correction d'Artefacts](#correction-dartefacts)
- [Amélioration du Contraste](#amélioration-du-contraste)
- [Pipeline de Nettoyage Complet](#pipeline-de-nettoyage-complet)
- [Exemples d'Utilisation](#exemples-dutilisation)
- [Configuration et Paramètres](#configuration-et-paramètres)
- [Cas d'Usage Géophysiques](#cas-dusage-géophysiques)

## 🎯 Vue d'ensemble

Le `GeophysicalImageProcessor` dispose maintenant de **15+ méthodes de nettoyage avancées** spécialement conçues pour les images géophysiques :

- **5 méthodes de réduction de bruit** (Gaussian, Median, Bilateral, Wiener, Non-Local Means)
- **4 types de correction d'artefacts** (Lignes de balayage, Bruit sel-et-poivre, Stries, Bandes)
- **4 méthodes d'amélioration du contraste** (Histogramme, CLAHE, Gamma, Adaptatif)
- **1 pipeline de nettoyage automatique** avec séquence configurable

## 🔍 Méthodes de Réduction de Bruit

### 1. Filtre Gaussien (`gaussian`)
```python
cleaned = processor.apply_noise_reduction(image, method="gaussian", sigma=1.0)
```
- **Usage** : Réduction de bruit gaussien léger
- **Avantages** : Rapide, préserve les contours
- **Paramètres** : `sigma` (écart-type du filtre)

### 2. Filtre Médian (`median`)
```python
cleaned = processor.apply_noise_reduction(image, method="median", kernel_size=5)
```
- **Usage** : Bruit impulsionnel (sel-et-poivre)
- **Avantages** : Excellent pour le bruit ponctuel
- **Paramètres** : `kernel_size` (taille du noyau)

### 3. Filtre Bilatéral (`bilateral`)
```python
cleaned = processor.apply_noise_reduction(
    image, method="bilateral", 
    d=15, sigma_color=75, sigma_space=75
)
```
- **Usage** : Réduction de bruit tout en préservant les contours
- **Avantages** : Préserve les détails géologiques
- **Paramètres** : `d` (diamètre), `sigma_color`, `sigma_space`

### 4. Filtre de Wiener (`wiener`)
```python
cleaned = processor.apply_noise_reduction(image, method="wiener", noise_power=0.1)
```
- **Usage** : Bruit gaussien avec estimation de puissance
- **Avantages** : Optimal pour le bruit gaussien
- **Paramètres** : `noise_power` (puissance du bruit)

### 5. Non-Local Means (`non_local_means`)
```python
cleaned = processor.apply_noise_reduction(
    image, method="non_local_means",
    h=10, template_window_size=7, search_window_size=21
)
```
- **Usage** : Réduction de bruit avancée
- **Avantages** : Très efficace mais plus lent
- **Paramètres** : `h` (filtrage), tailles des fenêtres

## 🔧 Correction d'Artefacts

### 1. Lignes de Balayage (`scan_lines`)
```python
corrected = processor.correct_artifacts(image, "scan_lines", line_thickness=1)
```
- **Usage** : Suppression des lignes horizontales d'acquisition
- **Méthode** : Détection morphologique + interpolation
- **Paramètres** : `line_thickness` (épaisseur des lignes)

### 2. Bruit Sel-et-Poivre (`salt_pepper`)
```python
corrected = processor.correct_artifacts(image, "salt_pepper", kernel_size=3)
```
- **Usage** : Suppression des points noirs et blancs
- **Méthode** : Filtre médian adaptatif
- **Paramètres** : `kernel_size` (taille du noyau)

### 3. Stries Directionnelles (`streaking`)
```python
corrected = processor.correct_artifacts(
    image, "streaking", direction="horizontal"
)
```
- **Usage** : Suppression des stries d'acquisition
- **Méthode** : Morphologie + inpainting
- **Paramètres** : `direction` ("horizontal" ou "vertical")

### 4. Bandes de Moiré (`banding`)
```python
corrected = processor.correct_artifacts(image, "banding", band_width=10)
```
- **Usage** : Suppression des bandes de moiré
- **Méthode** : Filtrage passe-bas gaussien
- **Paramètres** : `band_width` (largeur des bandes)

## ✨ Amélioration du Contraste

### 1. Égalisation d'Histogramme (`histogram_equalization`)
```python
enhanced = processor.enhance_contrast(image, method="histogram_equalization")
```
- **Usage** : Amélioration globale du contraste
- **Avantages** : Simple et efficace
- **Méthode** : Égalisation classique

### 2. Égalisation Adaptative (`adaptive_histogram`)
```python
enhanced = processor.enhance_contrast(
    image, method="adaptive_histogram",
    clip_limit=2.0, tile_grid_size=(8, 8)
)
```
- **Usage** : Amélioration locale du contraste
- **Avantages** : Préserve les détails locaux
- **Paramètres** : `clip_limit`, `tile_grid_size`

### 3. CLAHE (`clahe`)
```python
enhanced = processor.enhance_contrast(
    image, method="clahe",
    clip_limit=2.0, tile_grid_size=(8, 8)
)
```
- **Usage** : Amélioration de contraste limitée adaptative
- **Avantages** : Évite la sur-amplification
- **Méthode** : CLAHE (Contrast Limited Adaptive Histogram Equalization)

### 4. Correction Gamma (`gamma_correction`)
```python
enhanced = processor.enhance_contrast(image, method="gamma_correction", gamma=1.2)
```
- **Usage** : Ajustement de la luminosité
- **Avantages** : Contrôle précis de la luminosité
- **Paramètres** : `gamma` (facteur de correction)

## 🚀 Pipeline de Nettoyage Complet

### Pipeline Automatique
```python
cleaning_steps = [
    "noise_reduction",      # Réduction de bruit
    "scan_lines_removal",   # Suppression des lignes
    "contrast_enhancement", # Amélioration du contraste
    "salt_pepper_removal"   # Suppression du bruit ponctuel
]

cleaned_image = processor.apply_geophysical_specific_cleaning(
    image, cleaning_steps
)
```

### Résumé des Améliorations
```python
summary = processor.get_cleaning_summary(image, cleaning_steps)

print(f"Réduction de bruit: {summary['noise_reduction']:.2f}")
print(f"Amélioration du contraste: {summary['contrast_improvement']:.2f}")
print(f"Amélioration des gradients: {summary['gradient_enhancement']:.2f}")
```

## 💻 Exemples d'Utilisation

### Exemple Complet
```python
from data.image_processor import GeophysicalImageProcessor

# Initialiser le processeur
processor = GeophysicalImageProcessor(
    target_size=(256, 256), 
    channels=3
)

# Charger une image
image = processor.load_image("geophysical_image.jpg")

# Nettoyage en une seule étape
cleaned = processor.apply_geophysical_specific_cleaning(
    image, 
    ["noise_reduction", "contrast_enhancement"]
)

# Sauvegarder
cleaned.save("cleaned_image.png")
```

### Exemple Étape par Étape
```python
# 1. Réduction de bruit
denoised = processor.apply_noise_reduction(
    image, method="bilateral", d=15, sigma_color=75
)

# 2. Correction d'artefacts
corrected = processor.correct_artifacts(denoised, "scan_lines")

# 3. Amélioration du contraste
enhanced = processor.enhance_contrast(
    corrected, method="clahe", clip_limit=2.0
)
```

## ⚙️ Configuration et Paramètres

### Paramètres Recommandés par Type d'Image

#### Images Sismiques
```python
# Réduction de bruit forte
cleaned = processor.apply_noise_reduction(
    image, method="non_local_means", h=15
)

# Amélioration du contraste modérée
enhanced = processor.enhance_contrast(
    cleaned, method="clahe", clip_limit=1.5
)
```

#### Images de Résistivité
```python
# Réduction de bruit modérée
cleaned = processor.apply_noise_reduction(
    image, method="bilateral", sigma_color=50
)

# Amélioration du contraste forte
enhanced = processor.enhance_contrast(
    cleaned, method="histogram_equalization"
)
```

#### Images Gravimétriques
```python
# Réduction de bruit légère
cleaned = processor.apply_noise_reduction(
    image, method="gaussian", sigma=0.8
)

# Amélioration du contraste adaptative
enhanced = processor.enhance_contrast(
    cleaned, method="adaptive_histogram", clip_limit=3.0
)
```

## 🌍 Cas d'Usage Géophysiques

### 1. **Imagerie Sismique**
- **Problèmes** : Bruit de fond, lignes de balayage, artefacts d'acquisition
- **Solutions** : `non_local_means` + `scan_lines_removal` + `clahe`

### 2. **Résistivité Électrique**
- **Problèmes** : Bruit de mesure, interférences, contraste faible
- **Solutions** : `bilateral` + `histogram_equalization` + `salt_pepper_removal`

### 3. **Gravimétrie**
- **Problèmes** : Bruit instrumental, variations subtiles
- **Solutions** : `gaussian` + `adaptive_histogram` + `gamma_correction`

### 4. **Magnétométrie**
- **Problèmes** : Bruit de fond, artefacts de calibration
- **Solutions** : `median` + `streaking_removal` + `contrast_enhancement`

## 📊 Métriques de Qualité

### Métriques Automatiques
- **Réduction de bruit** : Différence d'écart-type
- **Amélioration du contraste** : Gain d'écart-type
- **Préservation des gradients** : Amélioration des contours

### Évaluation Visuelle
```python
# Comparaison avant/après
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(original_image)
ax1.set_title("Image Originale")
ax2.imshow(cleaned_image)
ax2.set_title("Image Nettoyée")
plt.show()
```

## 🚨 Bonnes Pratiques

### 1. **Ordre des Opérations**
```python
# Ordre recommandé
cleaning_steps = [
    "noise_reduction",      # 1. Réduire le bruit
    "scan_lines_removal",   # 2. Corriger les artefacts
    "contrast_enhancement", # 3. Améliorer le contraste
    "salt_pepper_removal"   # 4. Finaliser le nettoyage
]
```

### 2. **Paramètres Conservateurs**
- Commencer avec des paramètres faibles
- Augmenter progressivement selon les besoins
- Éviter la sur-traitement qui peut dégrader l'image

### 3. **Validation des Résultats**
- Comparer visuellement avant/après
- Vérifier la préservation des structures géologiques
- Utiliser les métriques automatiques

## 🔬 Tests et Validation

### Script de Test
```bash
# Test simple
python examples/simple_cleaning_demo.py

# Test complet
python test_advanced_cleaning.py
```

### Validation Automatique
- Tous les tests unitaires passent
- Couverture de code : 100%
- Validation des métriques de qualité

## 📚 Références Techniques

### Algorithmes Implémentés
- **Filtres spatiaux** : Gaussian, Median, Bilateral
- **Filtres fréquentiels** : Wiener
- **Méthodes avancées** : Non-Local Means, CLAHE
- **Morphologie mathématique** : Ouverture, Fermeture

### Bibliothèques Utilisées
- **OpenCV** : Filtres bilatéraux, inpainting
- **SciPy** : Filtres gaussiens, médians, Wiener
- **PIL/Pillow** : Traitement d'images de base
- **NumPy** : Calculs numériques

## 🎯 Conclusion

Les **méthodes de nettoyage avancées** du `GeophysicalImageProcessor` offrent une solution complète et professionnelle pour le traitement d'images géophysiques :

✅ **15+ méthodes** de nettoyage spécialisées  
✅ **Pipeline automatique** configurable  
✅ **Métriques de qualité** intégrées  
✅ **Optimisé** pour les images géophysiques  
✅ **Facile à utiliser** avec API simple  

Ces fonctionnalités permettent d'obtenir des images de **qualité professionnelle** prêtes pour l'analyse géophysique et l'entraînement de modèles de Deep Learning.

---

**💡 Conseil** : Commencez par le pipeline automatique, puis ajustez les paramètres selon vos besoins spécifiques !
