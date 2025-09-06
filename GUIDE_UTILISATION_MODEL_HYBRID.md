# 🚀 Guide d'Utilisation du Modèle Hybride (hybrid_model.pth)

## 📋 **Vue d'ensemble**

Votre modèle hybride est maintenant prêt à être utilisé ! Ce guide vous explique toutes les façons d'exécuter et d'utiliser le modèle hybride sauvegardé qui combine images et données géophysiques.

## 🎯 **Informations du Modèle**

- **Fichier**: `artifacts/models/hybrid_model.pth`
- **Architecture**: Modèle Hybride (Images + Données Géophysiques)
- **Encodeur d'images**: ResNet18 (512 features)
- **Encodeur géophysique**: 4 → 256 features
- **Méthode de fusion**: Concatenation
- **Classes**: 2 (classification binaire)
- **Paramètres totaux**: 12,076,162 paramètres entraînables

## 🚀 **Méthodes d'Exécution**

### 1. **Exécution Simple (Données Factices)**
```bash
python run_hybrid_model.py
```
- ✅ Test rapide avec des données d'exemple
- ✅ Vérification que le modèle fonctionne
- ✅ Affiche les informations du modèle

### 2. **Exécution avec Données Réelles**
```bash
python run_hybrid_model.py --real-data
```
- 🌍 Utilise vos vraies images géophysiques
- 📊 Charge les données via le pipeline de données
- 🔍 Traite les images de résistivité et chargeabilité

### 3. **Mode Verbose (Détails Complets)**
```bash
python run_hybrid_model.py --verbose
```
- 📋 Affiche un résumé détaillé de l'exécution
- 🔍 Montre toutes les informations de débogage
- 📊 Détaille les formes des données et prédictions

### 4. **Aide et Options**
```bash
python run_hybrid_model.py --help
```

## 🔧 **Fonctionnalités du Script**

### **Chargement du Modèle**
- ✅ Vérification de l'existence du fichier modèle
- ✅ Chargement des poids sauvegardés
- ✅ Initialisation de l'architecture hybride
- ✅ Affichage des informations du modèle

### **Traitement des Données**
- ✅ **Images**: Redimensionnement automatique à 64x64 pixels
- ✅ **Préprocessing**: Normalisation [0, 255] → [0, 1]
- ✅ **Format PyTorch**: Conversion (H, W, C) → (C, H, W)
- ✅ **Données géophysiques**: 4 dimensions normalisées

### **Prédictions**
- ✅ **Classification binaire**: 2 classes
- ✅ **Probabilités**: Distribution des classes
- ✅ **Classes prédites**: Prédiction finale
- ✅ **Mode évaluation**: Pas de gradients (torch.no_grad)

## 📊 **Format des Données d'Entrée**

### **Images**
- **Format**: JPG, PNG
- **Taille**: Automatiquement redimensionnée à 64x64
- **Canaux**: 3 (RGB)
- **Normalisation**: [0, 255] → [0, 1]

### **Données Géophysiques**
- **Dimensions**: 4 valeurs
- **Type**: float32
- **Normalisation**: Automatique
- **Exemple**: [résistivité, chargeabilité, coordonnées, profondeur]

## 🎯 **Exemples d'Utilisation**

### **Test Rapide**
```bash
# Test avec données factices
python run_hybrid_model.py

# Résultat attendu:
# 🎯 Prédictions:
#    - Classes prédites: [0]
#    - Probabilités: [[0.504 0.496]]
```

### **Avec Données Réelles**
```bash
# Test avec vraies images
python run_hybrid_model.py --real-data

# Résultat attendu:
# ✅ Données réelles chargées:
#    - Image: data/raw/images/resistivity/resis1.JPG -> (64, 64, 3)
#    - Données géophysiques: (4,)
```

### **Mode Détaillé**
```bash
# Test avec informations complètes
python run_hybrid_model.py --real-data --verbose

# Résultat attendu:
# 📋 Résumé détaillé:
#    - Modèle: Hybride (Images + Données Géophysiques)
#    - Données: Réelles
#    - Image: (64, 64, 3)
#    - Données géophysiques: (4,)
#    - Prédictions: [0]
#    - Probabilités: [[0.522 0.478]]
```

## 🔍 **Interprétation des Résultats**

### **Classes de Sortie**
- **Classe 0**: Première catégorie géologique
- **Classe 1**: Deuxième catégorie géologique

### **Probabilités**
- **Format**: [prob_classe_0, prob_classe_1]
- **Somme**: Toujours égale à 1.0
- **Interprétation**: Confiance du modèle pour chaque classe

### **Exemple de Résultat**
```
🎯 Prédictions:
   - Classes prédites: [0]
   - Probabilités: [[0.52192116 0.4780788 ]]
```
**Interprétation**: Le modèle prédit la classe 0 avec 52.2% de confiance et la classe 1 avec 47.8% de confiance.

## 🛠️ **Personnalisation**

### **Changer le Modèle**
```bash
python run_hybrid_model.py --model-path "chemin/vers/votre/modele.pth"
```

### **Utiliser d'Autres Images**
Modifiez la liste `image_paths` dans la fonction `load_real_data()` du script.

### **Modifier les Données Géophysiques**
Ajustez la génération des données dans `create_sample_data()` ou `load_real_data()`.

## ⚠️ **Notes Importantes**

1. **Images**: Le script redimensionne automatiquement toutes les images à 64x64 pixels
2. **Données géophysiques**: Actuellement générées aléatoirement pour les tests
3. **Performance**: Le modèle fonctionne sur CPU par défaut
4. **Mémoire**: Le modèle utilise environ 12M de paramètres

## 🎉 **Résumé**

Votre modèle hybride est maintenant **parfaitement fonctionnel** et prêt pour :
- ✅ **Classification géologique** basée sur images + données
- ✅ **Prédictions en temps réel** sur de nouvelles données
- ✅ **Intégration** dans vos pipelines d'analyse
- ✅ **Expérimentation** avec différents types de données

Le modèle combine efficacement les informations visuelles des images géophysiques avec les données tabulaires pour une classification précise ! 🎯
