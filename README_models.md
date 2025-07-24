# Modèles de Segmentation Sémantique

Ce repository implémente deux modèles recommandés pour la segmentation sémantique, ainsi que plusieurs fonctions de perte avancées pour optimiser les performances.

## 📋 Table des Matières

- [Modèles Implémentés](#modèles-implémentés)
- [Fonctions de Perte](#fonctions-de-perte)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Exemples d'Entraînement](#exemples-dentraînement)
- [Structure des Fichiers](#structure-des-fichiers)
- [Performances Attendues](#performances-attendues)

## 🏗️ Modèles Implémentés

### 1. U-Net Mini (Modèle de Base)

**Caractéristiques :**
- Architecture U-Net simplifiée sans pré-entraînement
- ~1.7M paramètres
- Temps d'entraînement rapide
- Consommation mémoire réduite

**Avantages :**
- Idéal pour le prototypage rapide
- Peu de ressources GPU requises
- Simple à comprendre et modifier

**Usage recommandé :**
- Tests et validation rapides
- Environnements avec ressources limitées
- Baseline pour comparaisons

### 2. VGG16-UNet (Modèle Avancé)

**Caractéristiques :**
- Backbone VGG16 pré-entraîné sur ImageNet
- ~21M paramètres
- Architecture U-Net avec skip connections
- Transfert learning efficace

**Avantages :**
- Performances supérieures attendues
- Bénéficie du transfert learning
- Encodeur robuste et éprouvé

**Usage recommandé :**
- Production et performances optimales
- Datasets de taille moyenne à grande
- Quand la précision est prioritaire

### 3. ResNet50-UNet (Bonus)

**Caractéristiques :**
- Backbone ResNet50 avec connexions résiduelles
- ~35M paramètres
- Architecture moderne et performante

## 🎯 Fonctions de Perte

### 1. Cross-Entropy Standard
```python
loss = 'sparse_categorical_crossentropy'
```
- Fonction de perte baseline
- Simple et stable
- Bon point de départ

### 2. Dice Loss
```python
loss = dice_loss
```
- Optimisée pour la segmentation
- Gère bien l'imbalance de classes
- Focus sur le chevauchement des régions

### 3. Focal Loss
```python
loss = focal_loss
```
- Excellent pour les classes déséquilibrées
- Réduit l'importance des exemples faciles
- Paramètres `alpha` et `gamma` ajustables

### 4. Combined Loss (Dice + Cross-Entropy)
```python
loss = combined_loss
```
- Combine les avantages de Dice et Cross-Entropy
- Poids ajustables entre les deux composantes
- Souvent la meilleure option

### 5. Balanced Cross-Entropy
```python
loss = balanced_cross_entropy
```
- Pondère les classes selon leur fréquence
- Poids prédéfinis pour Cityscapes
- Améliore la détection des classes rares

## 🚀 Installation

1. **Cloner le repository :**
```bash
git clone <repository-url>
cd segmentation-models
```

2. **Installer les dépendances :**
```bash
pip install -r requirements.txt
```

3. **Vérifier l'installation :**
```python
import tensorflow as tf
from models import create_model
print("Installation réussie!")
```

## 💻 Utilisation

### Création d'un Modèle

```python
from models import create_model, compile_model

# U-Net Mini
model_mini = create_model(
    model_type='unet_mini',
    input_shape=(224, 224, 3),
    num_classes=8,
    filters_base=32
)

# VGG16-UNet
model_vgg = create_model(
    model_type='vgg16_unet',
    input_shape=(224, 224, 3),
    num_classes=8,
    freeze_encoder=False  # Permettre l'entraînement du backbone
)

# Compilation avec différentes fonctions de perte
model_mini = compile_model(model_mini, loss_type='dice_loss')
model_vgg = compile_model(model_vgg, loss_type='combined_loss')
```

### Entraînement avec le Script

```bash
# Entraînement rapide avec U-Net Mini
python train_segmentation.py \
    --model unet_mini \
    --loss dice_loss \
    --epochs 20 \
    --batch_size 32 \
    --augment

# Entraînement performance avec VGG16-UNet
python train_segmentation.py \
    --model vgg16_unet \
    --loss combined_loss \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --augment
```

## 📊 Exemples d'Entraînement

### 1. Validation Rapide (U-Net Mini)
```bash
python train_segmentation.py \
    --model unet_mini \
    --loss cross_entropy \
    --epochs 10 \
    --batch_size 32 \
    --image_size 224
```

### 2. Performance Optimale (VGG16-UNet)
```bash
python train_segmentation.py \
    --model vgg16_unet \
    --loss combined_loss \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --image_size 224 \
    --augment \
    --validation_split 0.2
```

### 3. Transfer Learning (Backbone Gelé)
```bash
python train_segmentation.py \
    --model vgg16_unet \
    --loss focal_loss \
    --epochs 30 \
    --freeze_encoder \
    --learning_rate 1e-3 \
    --batch_size 24
```

### 4. Haute Résolution (ResNet50)
```bash
python train_segmentation.py \
    --model resnet50_unet \
    --loss balanced_cross_entropy \
    --epochs 40 \
    --batch_size 8 \
    --image_size 256 \
    --learning_rate 2e-5
```

## 📁 Structure des Fichiers

```
├── models.py                 # Architectures et fonctions de perte
├── train_segmentation.py     # Script d'entraînement principal
├── requirements.txt          # Dépendances Python
├── README_models.md          # Documentation (ce fichier)
└── models/                   # Dossier pour sauvegarder les modèles
    ├── best_*.h5            # Meilleurs modèles
    ├── final_*.h5           # Modèles finaux
    └── history_*.png        # Graphiques d'entraînement
```

## 🎯 Performances Attendues

### Temps d'Entraînement (par époque)

| Modèle | Batch Size | Temps/Époque | GPU Mémoire |
|--------|------------|--------------|-------------|
| U-Net Mini | 32 | ~2-3 min | ~4 GB |
| VGG16-UNet | 16 | ~5-7 min | ~8 GB |
| ResNet50-UNet | 8 | ~8-12 min | ~12 GB |

### Métriques de Performance

| Modèle | Mean IoU | Accuracy | Paramètres |
|--------|----------|----------|------------|
| U-Net Mini | ~0.65-0.70 | ~0.85-0.90 | 1.7M |
| VGG16-UNet | ~0.75-0.80 | ~0.90-0.93 | 21M |
| ResNet50-UNet | ~0.78-0.83 | ~0.91-0.94 | 35M |

*Note: Performances indicatives sur Cityscapes, peuvent varier selon le dataset*

## ⚙️ Paramètres Recommandés

### Pour Débutants
```bash
--model unet_mini --loss dice_loss --epochs 20 --batch_size 32
```

### Pour Production
```bash
--model vgg16_unet --loss combined_loss --epochs 50 --batch_size 16 --augment
```

### Pour Recherche
```bash
--model resnet50_unet --loss focal_loss --epochs 100 --batch_size 8 --image_size 256
```

## 🔧 Personnalisation

### Modifier les Fonctions de Perte

```python
def custom_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred)
    return 0.7 * dice + 0.3 * focal

# Utilisation
model = compile_model(model, loss_type=custom_loss)
```

### Ajuster l'Architecture

```python
# U-Net avec plus de filtres
model = unet_mini(filters_base=64)  # Plus de capacité

# VGG16 avec backbone gelé
model = vgg16_unet(freeze_encoder=True)  # Transfer learning strict
```

## 📈 Suivi de l'Entraînement

Le script génère automatiquement :
- **Checkpoints** : Sauvegarde du meilleur modèle
- **Graphiques** : Courbes de loss, accuracy, IoU
- **Logs** : Métriques détaillées par époque

## 🤝 Contribution

1. Tester les modèles sur votre dataset
2. Expérimenter avec les fonctions de perte
3. Optimiser les hyperparamètres
4. Partager vos résultats et améliorations

## 📝 Licence

Ce code est fourni à des fins éducatives et de recherche. Veuillez respecter les licences des datasets utilisés.

---

**Note**: Ce README accompagne l'implémentation des modèles recommandés pour réduire le temps de traitement et améliorer les performances en segmentation sémantique.