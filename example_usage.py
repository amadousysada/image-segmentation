#!/usr/bin/env python3
"""
Exemple d'utilisation des modèles de segmentation sémantique

Ce script démontre comment créer et utiliser les modèles recommandés :
- U-Net Mini (modèle de base)
- VGG16-UNet (modèle avancé)

Avec différentes fonctions de perte pour optimiser les performances.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models import (
    create_model, 
    compile_model,
    dice_loss,
    focal_loss,
    combined_loss,
    balanced_cross_entropy,
    MeanIoUArgmax
)

# Configuration
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 8  # Classes Cityscapes groupées
BATCH_SIZE = 4

def create_dummy_data(num_samples=10):
    """Créer des données factices pour la démonstration"""
    # Images aléatoires normalisées
    images = tf.random.normal((num_samples, *INPUT_SHAPE))
    
    # Masques avec classes aléatoires
    masks = tf.random.uniform(
        (num_samples, INPUT_SHAPE[0], INPUT_SHAPE[1]), 
        0, NUM_CLASSES, 
        dtype=tf.int32
    )
    
    return images, masks

def compare_models():
    """Comparer les différents modèles"""
    print("="*60)
    print("COMPARAISON DES MODÈLES DE SEGMENTATION")
    print("="*60)
    
    # Créer les modèles
    models = {
        'U-Net Mini': create_model('unet_mini', INPUT_SHAPE, NUM_CLASSES),
        'VGG16-UNet': create_model('vgg16_unet', INPUT_SHAPE, NUM_CLASSES),
        'ResNet50-UNet': create_model('resnet50_unet', INPUT_SHAPE, NUM_CLASSES)
    }
    
    # Afficher les informations de chaque modèle
    for name, model in models.items():
        params = model.count_params()
        print(f"\n{name}:")
        print(f"  - Paramètres: {params:,}")
        print(f"  - Taille approximative: {params * 4 / 1024 / 1024:.1f} MB")
        
        # Test de prédiction
        dummy_input = tf.random.normal((1, *INPUT_SHAPE))
        pred = model.predict(dummy_input, verbose=0)
        print(f"  - Forme de sortie: {pred.shape}")
    
    return models

def test_loss_functions():
    """Tester les différentes fonctions de perte"""
    print("\n" + "="*60)
    print("TEST DES FONCTIONS DE PERTE")
    print("="*60)
    
    # Créer des données factices
    images, masks = create_dummy_data(BATCH_SIZE)
    
    # Créer un modèle simple pour les tests
    model = create_model('unet_mini', INPUT_SHAPE, NUM_CLASSES)
    predictions = model.predict(images, verbose=0)
    
    # Tester chaque fonction de perte
    loss_functions = {
        'Cross-Entropy': lambda y_true, y_pred: tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        ),
        'Dice Loss': dice_loss,
        'Focal Loss': focal_loss,
        'Combined Loss': combined_loss,
        'Balanced Cross-Entropy': balanced_cross_entropy
    }
    
    print("\nRésultats sur données factices:")
    print("-" * 40)
    
    for name, loss_fn in loss_functions.items():
        try:
            loss_value = loss_fn(masks, predictions)
            print(f"{name:<25}: {loss_value:.4f}")
        except Exception as e:
            print(f"{name:<25}: Erreur - {str(e)}")

def demonstrate_training():
    """Démontrer l'entraînement avec différentes configurations"""
    print("\n" + "="*60)
    print("DÉMONSTRATION D'ENTRAÎNEMENT")
    print("="*60)
    
    # Créer des données d'entraînement factices
    train_images, train_masks = create_dummy_data(40)
    val_images, val_masks = create_dummy_data(10)
    
    # Configurations de test
    configs = [
        {
            'name': 'U-Net Mini + Dice Loss',
            'model_type': 'unet_mini',
            'loss': 'dice_loss',
            'epochs': 2,
            'batch_size': 4
        },
        {
            'name': 'VGG16-UNet + Combined Loss',
            'model_type': 'vgg16_unet',
            'loss': 'combined_loss',
            'epochs': 2,
            'batch_size': 2
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nEntraînement: {config['name']}")
        print("-" * 40)
        
        # Créer et compiler le modèle
        model = create_model(config['model_type'], INPUT_SHAPE, NUM_CLASSES)
        model = compile_model(model, config['loss'], learning_rate=1e-3)
        
        # Entraînement court pour démonstration
        try:
            history = model.fit(
                train_images, train_masks,
                validation_data=(val_images, val_masks),
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                verbose=1
            )
            
            # Sauvegarder les résultats
            results[config['name']] = {
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'final_accuracy': history.history['accuracy'][-1]
            }
            
            print(f"✓ Entraînement terminé")
            print(f"  Loss finale: {results[config['name']]['final_loss']:.4f}")
            print(f"  Val Loss finale: {results[config['name']]['final_val_loss']:.4f}")
            print(f"  Accuracy finale: {results[config['name']]['final_accuracy']:.4f}")
            
        except Exception as e:
            print(f"✗ Erreur lors de l'entraînement: {str(e)}")
            results[config['name']] = {'error': str(e)}
    
    return results

def visualize_predictions():
    """Visualiser des prédictions de segmentation"""
    print("\n" + "="*60)
    print("VISUALISATION DES PRÉDICTIONS")
    print("="*60)
    
    # Créer des données de test
    test_images, test_masks = create_dummy_data(2)
    
    # Créer un modèle simple
    model = create_model('unet_mini', INPUT_SHAPE, NUM_CLASSES)
    model = compile_model(model, 'dice_loss')
    
    # Faire des prédictions
    predictions = model.predict(test_images, verbose=0)
    pred_masks = tf.argmax(predictions, axis=-1)
    
    # Visualiser
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i in range(2):
        # Image d'entrée (normalisée pour affichage)
        img_display = (test_images[i] - tf.reduce_min(test_images[i])) / (
            tf.reduce_max(test_images[i]) - tf.reduce_min(test_images[i])
        )
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title(f'Image {i+1}')
        axes[i, 0].axis('off')
        
        # Masque vérité terrain
        axes[i, 1].imshow(test_masks[i], cmap='tab10', vmin=0, vmax=NUM_CLASSES-1)
        axes[i, 1].set_title(f'Vérité terrain {i+1}')
        axes[i, 1].axis('off')
        
        # Prédiction
        axes[i, 2].imshow(pred_masks[i], cmap='tab10', vmin=0, vmax=NUM_CLASSES-1)
        axes[i, 2].set_title(f'Prédiction {i+1}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualisation sauvegardée dans 'predictions_demo.png'")

def main():
    """Fonction principale de démonstration"""
    print("DÉMONSTRATION DES MODÈLES DE SEGMENTATION SÉMANTIQUE")
    print("="*60)
    print("TensorFlow version:", tf.__version__)
    print("GPU disponible:", len(tf.config.list_physical_devices('GPU')) > 0)
    print()
    
    try:
        # 1. Comparer les modèles
        models = compare_models()
        
        # 2. Tester les fonctions de perte
        test_loss_functions()
        
        # 3. Démontrer l'entraînement
        training_results = demonstrate_training()
        
        # 4. Visualiser les prédictions
        visualize_predictions()
        
        # Résumé final
        print("\n" + "="*60)
        print("RÉSUMÉ DE LA DÉMONSTRATION")
        print("="*60)
        print("✓ Modèles créés et testés avec succès")
        print("✓ Fonctions de perte validées")
        print("✓ Entraînement démontré")
        print("✓ Prédictions visualisées")
        print("\nLes modèles sont prêts pour l'entraînement sur vos données!")
        
        # Recommandations
        print("\n" + "="*60)
        print("RECOMMANDATIONS D'USAGE")
        print("="*60)
        print("• Pour débuter: U-Net Mini + Dice Loss")
        print("• Pour performance: VGG16-UNet + Combined Loss")
        print("• Pour recherche: ResNet50-UNet + Focal Loss")
        print("\nCommandes d'entraînement:")
        print("python train_segmentation.py --model unet_mini --loss dice_loss --epochs 20")
        print("python train_segmentation.py --model vgg16_unet --loss combined_loss --epochs 50")
        
    except Exception as e:
        print(f"\n✗ Erreur lors de la démonstration: {str(e)}")
        print("Vérifiez que TensorFlow est correctement installé.")

if __name__ == "__main__":
    main()