# 🚀 Guide d'Utilisation du Modèle CNN 2D (cnn_2d_model.pth)

## 📋 **Vue d'ensemble**

Votre modèle CNN 2D est maintenant prêt à être utilisé ! Ce guide vous explique toutes les façons d'exécuter et d'utiliser le modèle sauvegardé.

## 🎯 **Informations du Modèle**

- **Fichier**: `artifacts/models/cnn_2d_model.pth`
- **Architecture**: CNN 2D géophysique
- **Entrée**: 4 canaux, grille 64x64
- **Sortie**: 2 classes (classification binaire)
- **Époques d'entraînement**: 24
- **Dropout**: 0.3

## 🚀 **Méthodes d'Exécution**

### 1. **Exécution Simple (Données Factices)**
```bash
python run_cnn_2d_model.py
```
- ✅ Test rapide avec des données d'exemple
- ✅ Vérification que le modèle fonctionne
- ✅ Affiche les informations du modèle

### 2. **Exécution avec Données Réelles**
```bash
python run_cnn_2d_model.py --real-data
```
- 🌍 Utilise vos vraies données géophysiques
- 📊 Charge les données via le pipeline complet
- 🔮 Fait des prédictions sur 2 échantillons réels

### 3. **Exécution avec Modèle Personnalisé**
```bash
python run_cnn_2d_model.py --model-path "chemin/vers/votre/modele.pth"
```

## 📊 **Résultats d'Exécution**

### **Exemple de Sortie:**
```
🚀 EXÉCUTION DU MODÈLE CNN 2D
==================================================
🔄 Chargement du modèle depuis: artifacts/models/cnn_2d_model.pth
✅ Modèle CNN 2D chargé avec succès!

📋 INFORMATIONS DU MODÈLE:
------------------------------
📊 Époques d'entraînement: 24

🧪 TEST AVEC DES DONNÉES D'EXEMPLE:
----------------------------------------
📊 Création de données d'exemple...
✅ Données d'exemple créées: (4, 64, 64)

🔮 Prédiction sur des données de forme: (4, 64, 64)
📊 Données formatées pour le modèle: torch.Size([1, 4, 64, 64])
📈 Prédictions: tensor([[ 18.9407, -17.3606]])
🎯 Classes prédites: tensor([0])
📊 Probabilités: tensor([[1.0000e+00, 1.7160e-16]])

🎯 RÉSULTATS:
--------------------
✅ Prédiction réussie!
📊 Classes prédites: [0]
📈 Probabilités: [[1.000000e+00 1.716001e-16]]

==================================================
🎉 EXÉCUTION DU MODÈLE TERMINÉE AVEC SUCCÈS!
==================================================
```

## 🔧 **Utilisation Programmatique**

### **Charger le Modèle dans Votre Code:**
```python
import torch
import numpy as np
from src.model.geophysical_trainer import GeophysicalCNN2D

# 1. Créer le modèle avec la même architecture
model = GeophysicalCNN2D(
    input_channels=4,
    num_classes=2,
    grid_size=64,
    dropout_rate=0.3
)

# 2. Charger les poids
checkpoint = torch.load("artifacts/models/cnn_2d_model.pth", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# 3. Mettre en mode évaluation
model.eval()

# 4. Faire des prédictions
with torch.no_grad():
    # Vos données doivent être de forme (batch_size, 4, 64, 64)
    predictions = model(your_data)
    probabilities = torch.softmax(predictions, dim=1)
    predicted_classes = torch.argmax(predictions, dim=1)
```

### **Format des Données d'Entrée:**
- **Forme attendue**: `(batch_size, 4, 64, 64)`
- **4 canaux**: Représentent les différents dispositifs géophysiques
- **64x64**: Grille spatiale normalisée
- **Type**: `torch.FloatTensor` ou `numpy.ndarray`

## 📈 **Interprétation des Résultats**

### **Prédictions:**
- **Valeurs brutes**: Scores logits pour chaque classe
- **Classes prédites**: 0 ou 1 (classification binaire)
- **Probabilités**: Probabilité d'appartenance à chaque classe

### **Exemple d'Interprétation:**
```
📈 Prédictions: tensor([[ 18.9407, -17.3606]])
🎯 Classes prédites: tensor([0])
📊 Probabilités: tensor([[1.0000e+00, 1.7160e-16]])
```
- **Classe 0**: Probabilité ~100% (très confiant)
- **Classe 1**: Probabilité ~0% (très peu probable)

## 🛠️ **Dépannage**

### **Erreur de Format de Données:**
```
❌ Erreur: Forme d'entrée attendue: (batch_size, 4, 64, 64)
```
**Solution**: Le script gère automatiquement la conversion des formats.

### **Modèle Non Trouvé:**
```
❌ FileNotFoundError: Le fichier modèle n'existe pas
```
**Solution**: Vérifiez que le fichier `artifacts/models/cnn_2d_model.pth` existe.

### **Erreur de Mémoire:**
```
❌ CUDA out of memory
```
**Solution**: Le modèle utilise automatiquement le CPU par défaut.

## 🎯 **Cas d'Usage Avancés**

### **1. Prédiction sur Plusieurs Échantillons:**
```python
# Charger plusieurs échantillons
batch_data = torch.FloatTensor(np.random.rand(5, 4, 64, 64))
predictions = model(batch_data)
```

### **2. Utilisation avec GPU:**
```python
# Déplacer le modèle sur GPU
model = model.cuda()
data = data.cuda()
predictions = model(data)
```

### **3. Sauvegarde des Prédictions:**
```python
results = {
    'predictions': predictions.cpu().numpy(),
    'probabilities': probabilities.cpu().numpy(),
    'predicted_classes': predicted_classes.cpu().numpy()
}
np.save('predictions.npy', results)
```

## 📚 **Fichiers Associés**

- **Modèle principal**: `artifacts/models/cnn_2d_model.pth`
- **Script d'exécution**: `run_cnn_2d_model.py`
- **Architecture**: `src/model/geophysical_trainer.py` (classe `GeophysicalCNN2D`)
- **Configuration**: `config.py`

## 🎉 **Félicitations !**

Votre modèle CNN 2D est maintenant entièrement fonctionnel et prêt pour la production ! Vous pouvez l'utiliser pour faire des prédictions sur de nouvelles données géophysiques.

---

**💡 Conseil**: Commencez par tester avec `python run_cnn_2d_model.py` pour vérifier que tout fonctionne, puis utilisez `--real-data` pour des prédictions sur vos vraies données.
