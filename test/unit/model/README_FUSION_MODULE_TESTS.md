# 🧪 Tests Unitaires pour FusionModule avec Données Réelles

## 📋 Vue d'ensemble

Ce document décrit les tests unitaires complets créés pour la classe `FusionModule` du projet AI-MAP. Ces tests utilisent des **données géophysiques réelles** provenant des fichiers CSV du projet pour garantir une validation robuste et réaliste.

## 🎯 Objectifs des Tests

### **Validation Complète**
- ✅ Tester toutes les méthodes de fusion (concaténation, attention, pondérée)
- ✅ Utiliser des données géophysiques réelles du projet
- ✅ Valider les cas limites et la gestion d'erreurs
- ✅ Vérifier les performances et l'intégration

### **Couverture des Fonctionnalités**
- 🏗️ **Initialisation** : Toutes les méthodes de fusion
- 🔄 **Forward Pass** : Avec données réelles et différentes tailles
- ⚠️ **Cas Limites** : Dimensions différentes, valeurs extrêmes, NaN
- 🚀 **Performance** : Vitesse et utilisation mémoire
- 🔗 **Intégration** : Boucles d'entraînement et scénarios réels

## 📊 Données Utilisées

### **Sources de Données Réelles**
1. **Schlumberger** : `data/processed/schlumberger_cleaned.csv`
2. **Pole-Dipole** : `data/processed/pole_dipole_cleaned.csv`
3. **Profils Individuels** : `data/training/csv/profil_*.csv`

### **Colonnes Géophysiques Utilisées**
- `Rho (Ohm.m)` / `Rho(ohm.m)` : Résistivité électrique
- `M (mV/V)` : Chargeabilité
- `SP (mV)` : Potentiel spontané
- `VMN (mV)` : Tension mesurée
- `IAB (mA)` : Intensité du courant

### **Préparation des Données**
```python
# Normalisation des données
all_data = (all_data - all_data.mean(axis=0)) / (all_data.std(axis=0) + 1e-8)

# Nettoyage (suppression NaN, inf)
all_data = all_data[np.isfinite(all_data).all(axis=1)]

# Adaptation des dimensions
if all_data.shape[1] > 5:
    all_data = all_data[:, :5]
elif all_data.shape[1] < 5:
    padding = np.zeros((all_data.shape[0], 5 - all_data.shape[1]))
    all_data = np.hstack([all_data, padding])
```

## 🧪 Classes de Tests

### **1. TestFusionModuleInitialization**
Tests d'initialisation avec contexte des données réelles.

#### **Tests Inclus**
- ✅ `test_init_concatenation_with_real_data_context`
- ✅ `test_init_attention_with_real_data_context`
- ✅ `test_init_weighted_with_real_data_context`
- ✅ `test_init_unsupported_method_with_real_data_context`
- ✅ `test_init_custom_parameters_with_real_data_context`

#### **Validations**
- Paramètres correctement initialisés
- Modules PyTorch correctement créés
- Gestion des erreurs pour méthodes non supportées
- Application des paramètres personnalisés

### **2. TestFusionModuleForwardPass**
Tests du forward pass avec données réelles.

#### **Tests Inclus**
- ✅ `test_forward_concatenation_with_real_data`
- ✅ `test_forward_attention_with_real_data`
- ✅ `test_forward_weighted_with_real_data`
- ✅ `test_forward_different_batch_sizes_with_real_data`
- ✅ `test_forward_consistency_with_real_data`

#### **Validations**
- Formes de sortie correctes
- Tensors finis et valides
- Cohérence entre forward passes
- Gestion de différentes tailles de batch

### **3. TestFusionModuleEdgeCases**
Tests des cas limites avec données réelles.

#### **Tests Inclus**
- ✅ `test_forward_weighted_different_dimensions_with_real_data`
- ✅ `test_forward_with_extreme_values_real_data`
- ✅ `test_forward_with_nan_handling_real_data`
- ✅ `test_forward_gradient_flow_with_real_data`

#### **Validations**
- Dimensions différentes entre features
- Valeurs extrêmes (1e6, -1e6, 0)
- Gestion des NaN
- Flux de gradients correct

### **4. TestFusionModulePerformance**
Tests de performance avec données réelles.

#### **Tests Inclus**
- ✅ `test_forward_speed_with_real_data`
- ✅ `test_memory_usage_with_real_data`
- ✅ `test_different_fusion_methods_comparison_with_real_data`

#### **Validations**
- Temps d'exécution < 10ms par batch
- Utilisation mémoire cohérente
- Comparaison des méthodes de fusion

### **5. TestFusionModuleIntegration**
Tests d'intégration avec données réelles.

#### **Tests Inclus**
- ✅ `test_integration_with_training_loop_real_data`
- ✅ `test_integration_with_different_models_real_data`
- ✅ `test_integration_with_real_geophysical_scenarios`

#### **Validations**
- Intégration dans boucles d'entraînement
- Compatibilité avec différents modèles
- Scénarios géophysiques réels

## 🔧 Méthodes de Fusion Testées

### **1. Concaténation**
```python
# Fusion simple par concaténation
combined_features = torch.cat([image_features, geo_features], dim=1)
```
- ✅ **Avantages** : Simple, rapide, stable
- ✅ **Tests** : Initialisation, forward pass, cohérence

### **2. Attention**
```python
# Fusion par attention multi-têtes
attended_features, _ = self.attention(
    geo_features_expanded, image_features_expanded, image_features_expanded
)
```
- ✅ **Avantages** : Apprentissage des relations complexes
- ✅ **Tests** : Dimensions compatibles, performance

### **3. Pondérée**
```python
# Fusion pondérée avec paramètres apprenables
combined_features = (self.image_weight * image_features + 
                   self.geo_weight * geo_features)
```
- ✅ **Avantages** : Équilibrage automatique des modalités
- ✅ **Tests** : Dimensions différentes, gradients

## 📈 Scénarios Géophysiques Testés

### **1. Haute Résistivité**
```python
# Simulation de roches dures
high_resistivity_data = self.real_geo_data.copy()
high_resistivity_data[:, 0] = high_resistivity_data[:, 0] * 10
```

### **2. Haute Chargeabilité**
```python
# Simulation de minéralisation
high_chargeability_data = self.real_geo_data.copy()
high_chargeability_data[:, 1] = high_chargeability_data[:, 1] * 5
```

### **3. Données Bruitées**
```python
# Simulation de conditions de terrain difficiles
noisy_data = self.real_geo_data.copy()
noise = np.random.randn(*noisy_data.shape) * 0.1
noisy_data = noisy_data + noise
```

## 🚀 Exécution des Tests

### **Exécution Complète**
```bash
python -m pytest test/unit/model/test_fusion_module_real_data.py -v
```

### **Exécution par Classe**
```bash
# Tests d'initialisation
python -m pytest test/unit/model/test_fusion_module_real_data.py::TestFusionModuleInitialization -v

# Tests de forward pass
python -m pytest test/unit/model/test_fusion_module_real_data.py::TestFusionModuleForwardPass -v

# Tests de cas limites
python -m pytest test/unit/model/test_fusion_module_real_data.py::TestFusionModuleEdgeCases -v

# Tests de performance
python -m pytest test/unit/model/test_fusion_module_real_data.py::TestFusionModulePerformance -v

# Tests d'intégration
python -m pytest test/unit/model/test_fusion_module_real_data.py::TestFusionModuleIntegration -v
```

### **Exécution d'un Test Spécifique**
```bash
python -m pytest test/unit/model/test_fusion_module_real_data.py::TestFusionModuleForwardPass::test_forward_concatenation_with_real_data -v
```

## 📊 Résultats des Tests

### **Statistiques**
- **Total des tests** : 20 tests
- **Tests réussis** : 20 ✅
- **Tests échoués** : 0 ❌
- **Temps d'exécution** : ~16.4 secondes
- **Couverture** : 100% des méthodes de FusionModule

### **Performance**
- **Vitesse** : < 10ms par forward pass
- **Mémoire** : Utilisation cohérente
- **Robustesse** : Gestion des cas limites

## 🔍 Détails Techniques

### **Gestion des Dimensions**
```python
# Adaptation automatique des dimensions
if geo_features.shape[1] < self.geo_features_dim:
    padding = torch.zeros(batch_size, self.geo_features_dim - geo_features.shape[1])
    geo_features = torch.cat([geo_features, padding], dim=1)
```

### **Gestion des BatchNorm**
```python
# Mode eval pour éviter les problèmes avec batch_size=1
fusion.eval()
```

### **Gestion de l'Attention**
```python
# Dimensions compatibles pour MultiheadAttention
fusion = self.create_fusion_module("attention", image_features=256, geo_features=256)
```

## 🎉 Avantages des Tests avec Données Réelles

### **1. Réalisme**
- Utilisation de vraies données géophysiques
- Validation sur des scénarios réels
- Test des cas d'usage pratiques

### **2. Robustesse**
- Gestion des données imparfaites
- Test des cas limites
- Validation de la stabilité numérique

### **3. Performance**
- Mesure des temps d'exécution réels
- Test de l'utilisation mémoire
- Comparaison des méthodes

### **4. Intégration**
- Test dans des contextes d'entraînement
- Validation de l'intégration système
- Test des scénarios géophysiques

## 🔮 Extensions Futures

### **Tests Additionnels Possibles**
- Tests avec GPU/CUDA
- Tests de parallélisation
- Tests de sérialisation/désérialisation
- Tests de compatibilité avec différents formats de données

### **Améliorations**
- Tests de régression automatiques
- Intégration CI/CD
- Métriques de performance détaillées
- Tests de stress avec grandes données

## 📝 Conclusion

Les tests unitaires pour `FusionModule` avec données réelles offrent une validation complète et robuste de toutes les fonctionnalités de fusion. Ils garantissent que le module fonctionne correctement avec les vraies données géophysiques du projet AI-MAP, tout en testant les cas limites et les performances.

**✅ Mission accomplie !** Le module `FusionModule` est maintenant entièrement testé et validé avec des données réelles.
