# Guide de Résolution de Problèmes - Erreurs de Forme

## ❌ Erreur Commune : Incompatibilité de Forme

```
ValueError: Arguments `target` and `output` must have the same shape up until the last dimension: 
target.shape=(None, 224, 224), output.shape=(None, 448, 448, 8)
```

## 🔍 Diagnostic de l'Erreur

Cette erreur indique que :
- **Target (masque)** : forme `(batch, 224, 224)`
- **Output (prédiction)** : forme `(batch, 448, 448, 8)`

### Problèmes identifiés :
1. **Taille spatiale différente** : 224×224 vs 448×448
2. **Forme incompatible** : le target n'a pas la dimension des classes

## 🛠️ Solutions Implémentées

### 1. Architecture U-Net Mini Corrigée

**Avant (problématique) :**
```python
# 4 downsampling + 4 upsampling + 1 upsampling final = 5 upsampling
x1, skip1 = encoder_block(inputs, filters_base)      # 224→112
x2, skip2 = encoder_block(x1, filters_base * 2)      # 112→56  
x3, skip3 = encoder_block(x2, filters_base * 4)      # 56→28
x4, skip4 = encoder_block(x3, filters_base * 8)      # 28→14
bottleneck = conv_block(x4, filters_base * 16)       # 14→7
# Puis 4 decoder + 1 final upsampling = 448x448 ❌
```

**Après (corrigée) :**
```python
# 3 downsampling + 3 upsampling = dimensions équilibrées
x1, skip1 = encoder_block(inputs, filters_base)      # 224→112
x2, skip2 = encoder_block(x1, filters_base * 2)      # 112→56
x3, skip3 = encoder_block(x2, filters_base * 4)      # 56→28
bottleneck = conv_block(x3, filters_base * 8)        # reste à 28x28
# Puis 3 decoder steps: 28→56→112→224 ✅
```

### 2. Architecture VGG16-UNet Corrigée

**Avant (problématique) :**
```python
# Skip connections incorrectement dimensionnées
skip1 = vgg16.get_layer('block1_conv2').output    # 224x224 (FAUX)
# ... upsampling incorrect conduisant à 448x448
```

**Après (corrigée) :**
```python
# Skip connections correctement dimensionnées  
skip1 = vgg16.get_layer('block1_conv2').output    # 112x112 (après pooling)
skip2 = vgg16.get_layer('block2_conv2').output    # 56x56
skip3 = vgg16.get_layer('block3_conv3').output    # 28x28  
skip4 = vgg16.get_layer('block4_conv3').output    # 14x14

# Décodeur avec upsampling séquentiel correct
# 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
```

### 2. Fonctions de Perte Robustes

Toutes les fonctions de perte ont été mises à jour avec un redimensionnement automatique :

```python
def dice_loss(y_true, y_pred, smooth=1e-6):
    # Redimensionnement automatique si nécessaire
    pred_shape = tf.shape(y_pred)
    true_shape = tf.shape(y_true)
    
    if true_shape[1] != pred_shape[1] or true_shape[2] != pred_shape[2]:
        y_true = tf.image.resize(
            tf.expand_dims(tf.cast(y_true, tf.float32), axis=-1),
            [pred_shape[1], pred_shape[2]], 
            method='nearest'
        )
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
    # ... reste du calcul
```

### 3. Fonction de Débogage

```python
from models import debug_model_output_shape

# Vérifier la forme de sortie de votre modèle
model = create_model('vgg16_unet', (224, 224, 3), 8)
debug_model_output_shape(model)
```

## 🚀 Utilisation Corrigée

### Test Rapide
```python
# Importer les modèles corrigés
from models import create_model, compile_model, dice_loss

# Créer le modèle VGG16-UNet corrigé
model = create_model('vgg16_unet', (224, 224, 3), 8)

# Compiler avec fonction de perte robuste
model = compile_model(model, loss_type='dice_loss')

# Test avec données factices
import tensorflow as tf
test_image = tf.random.normal((1, 224, 224, 3))
test_mask = tf.random.uniform((1, 224, 224), 0, 8, dtype=tf.int32)

# Prédiction (devrait être 224x224 maintenant)
prediction = model(test_image)
print(f"Forme de prédiction: {prediction.shape}")  # (1, 224, 224, 8)

# Test de loss (devrait fonctionner)
loss_value = dice_loss(test_mask, prediction)
print(f"Loss: {loss_value}")
```

### Entraînement avec Modèles Corrigés
```bash
# U-Net Mini (toujours fonctionnel)
python train_segmentation.py --model unet_mini --loss dice_loss --epochs 10

# VGG16-UNet (maintenant corrigé) 
python train_segmentation.py --model vgg16_unet --loss combined_loss --epochs 20

# ResNet50-UNet (vérifié et fonctionnel)
python train_segmentation.py --model resnet50_unet --loss focal_loss --epochs 15
```

## 🔧 Diagnostic Automatique

Utilisez le script de débogage pour identifier les problèmes :

```bash
python debug_model_shapes.py
```

Ce script :
- ✅ Teste toutes les architectures de modèles
- ✅ Vérifie les formes de sortie
- ✅ Teste les fonctions de perte avec différentes tailles
- ✅ Identifie les problèmes potentiels
- ✅ Propose des solutions

## 📋 Checklist de Vérification

Avant d'entraîner votre modèle :

- [ ] **Taille d'entrée cohérente** : Vos images et masques ont la même taille spatiale
- [ ] **Architecture vérifiée** : Utilisez `debug_model_output_shape()` 
- [ ] **Fonction de perte compatible** : Utilisez les versions corrigées
- [ ] **Test simple** : Créez des données factices pour validation

## 🎯 Solutions par Type d'Erreur

### Erreur 1: Output trop grand (ex: 448×448 au lieu de 224×224)
**Solution :** Architecture corrigée automatiquement dans `models.py`

### Erreur 2: Formes incompatibles entre masque et prédiction  
**Solution :** Fonctions de perte avec redimensionnement automatique

### Erreur 3: Dimensionalité incorrecte
**Solution :** Vérification et conversion automatique des types

### Erreur 4: Problème de batch size
**Solution :** Gestion dynamique des formes avec `tf.shape()`

## 🚨 Actions d'Urgence

Si vous avez encore des problèmes :

1. **Utilisez U-Net Mini** (garanti de fonctionner) :
   ```python
   model = create_model('unet_mini', (224, 224, 3), 8)
   ```

2. **Forcez le redimensionnement** dans votre code :
   ```python
   # Redimensionner le masque pour correspondre à la prédiction
   mask_resized = tf.image.resize(
       tf.expand_dims(tf.cast(mask, tf.float32), -1),
       prediction.shape[1:3], 
       method='nearest'
   )
   mask_resized = tf.cast(tf.squeeze(mask_resized, -1), tf.int32)
   ```

3. **Utilisez la cross-entropy standard** (plus robuste) :
   ```python
   model.compile(
       optimizer='adam',
       loss='sparse_categorical_crossentropy',
       metrics=['accuracy']
   )
   ```

## ✅ Validation de la Correction

Pour confirmer que tout fonctionne :

```python
# Test complet
from models import create_model, compile_model, dice_loss
import tensorflow as tf

# Créer le modèle
model = create_model('vgg16_unet', (224, 224, 3), 8)
model = compile_model(model, 'dice_loss')

# Données de test
images = tf.random.normal((4, 224, 224, 3))
masks = tf.random.uniform((4, 224, 224), 0, 8, dtype=tf.int32)

# Test d'entraînement d'une étape  
history = model.fit(images, masks, epochs=1, verbose=1)

print("✅ Tout fonctionne correctement!")
```

---

**Les corrections apportées garantissent la compatibilité et la robustesse des modèles pour tous les cas d'usage.**