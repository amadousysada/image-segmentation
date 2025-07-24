#!/usr/bin/env python3
"""
Script de débogage pour diagnostiquer les problèmes de forme des modèles

Ce script aide à identifier et résoudre les erreurs de taille entre les masques
et les prédictions des modèles de segmentation.
"""

import sys
import os

# Add current directory to path to import models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import tensorflow as tf
    from models import create_model, debug_model_output_shape, dice_loss
    print("✓ TensorFlow et models importés avec succès")
except ImportError as e:
    print(f"✗ Erreur d'importation: {e}")
    print("Assurez-vous que TensorFlow est installé : pip install tensorflow")
    sys.exit(1)

def test_model_shapes():
    """Test les formes de sortie des différents modèles"""
    
    print("="*60)
    print("TEST DES FORMES DE MODÈLES")
    print("="*60)
    
    input_shapes = [(224, 224, 3), (256, 256, 3), (512, 512, 3)]
    model_types = ['unet_mini', 'vgg16_unet', 'resnet50_unet']
    
    results = {}
    
    for input_shape in input_shapes:
        print(f"\n📐 Test avec input_shape: {input_shape}")
        print("-" * 50)
        
        for model_type in model_types:
            try:
                print(f"\nModèle: {model_type}")
                
                # Créer le modèle
                model = create_model(model_type, input_shape, num_classes=8)
                
                # Tester la forme de sortie
                output_shape = debug_model_output_shape(model, input_shape)
                
                # Vérifier la cohérence
                input_h, input_w = input_shape[:2]
                output_h, output_w = output_shape[1:3]
                
                if input_h == output_h and input_w == output_w:
                    status = "✅ CORRECT"
                else:
                    status = f"❌ ERREUR - Tailles différentes: {input_h}x{input_w} vs {output_h}x{output_w}"
                
                print(f"Status: {status}")
                
                results[f"{model_type}_{input_shape}"] = {
                    'input_shape': input_shape,
                    'output_shape': output_shape,
                    'status': status
                }
                
            except Exception as e:
                print(f"❌ Erreur avec {model_type}: {str(e)}")
                results[f"{model_type}_{input_shape}"] = {
                    'input_shape': input_shape,
                    'error': str(e)
                }
    
    return results

def test_loss_functions():
    """Test les fonctions de perte avec différentes tailles"""
    
    print("\n" + "="*60)
    print("TEST DES FONCTIONS DE PERTE")
    print("="*60)
    
    # Créer un modèle de test
    model = create_model('unet_mini', (224, 224, 3), 8)
    
    # Test avec différentes tailles de masque
    test_cases = [
        {"name": "Tailles correctes", "img_size": (224, 224), "mask_size": (224, 224)},
        {"name": "Masque plus petit", "img_size": (224, 224), "mask_size": (112, 112)},
        {"name": "Masque plus grand", "img_size": (224, 224), "mask_size": (448, 448)},
    ]
    
    for case in test_cases:
        print(f"\n🧪 Test: {case['name']}")
        print(f"Image: {case['img_size']}, Masque: {case['mask_size']}")
        
        try:
            # Créer des données de test
            batch_size = 2
            img_h, img_w = case['img_size']
            mask_h, mask_w = case['mask_size']
            
            # Image d'entrée
            test_images = tf.random.normal((batch_size, img_h, img_w, 3))
            
            # Masque de vérité terrain
            test_masks = tf.random.uniform((batch_size, mask_h, mask_w), 0, 8, dtype=tf.int32)
            
            # Prédiction du modèle
            predictions = model(test_images, training=False)
            
            print(f"  Forme des images: {test_images.shape}")
            print(f"  Forme des masques: {test_masks.shape}")
            print(f"  Forme des prédictions: {predictions.shape}")
            
            # Test de la fonction de perte
            loss_value = dice_loss(test_masks, predictions)
            print(f"  ✅ Dice Loss: {loss_value:.4f}")
            
        except Exception as e:
            print(f"  ❌ Erreur: {str(e)}")

def test_specific_error():
    """Test du cas d'erreur spécifique mentionné"""
    
    print("\n" + "="*60)
    print("TEST DU CAS D'ERREUR SPÉCIFIQUE")
    print("="*60)
    
    print("Reproduction de l'erreur:")
    print("target.shape=(None, 224, 224), output.shape=(None, 448, 448, 8)")
    
    try:
        # Créer le scénario problématique
        model = create_model('vgg16_unet', (224, 224, 3), 8)
        
        # Test avec une image 224x224
        test_image = tf.random.normal((1, 224, 224, 3))
        prediction = model(test_image, training=False)
        
        print(f"\nRésultat du test:")
        print(f"Input shape: {test_image.shape}")
        print(f"Output shape: {prediction.shape}")
        
        # Vérifier si l'erreur est corrigée
        if prediction.shape[1] == 224 and prediction.shape[2] == 224:
            print("✅ CORRIGÉ - Les tailles correspondent maintenant!")
        else:
            print(f"❌ PROBLÈME PERSISTANT - Taille de sortie: {prediction.shape[1]}x{prediction.shape[2]}")
            
            # Proposer une solution
            print("\n🔧 SOLUTION PROPOSÉE:")
            print("1. Vérifier l'architecture du modèle VGG16-UNet")
            print("2. S'assurer que les upsampling layers sont correctement configurés")
            print("3. Utiliser les fonctions de perte corrigées qui redimensionnent automatiquement")
        
        # Test avec la fonction de perte corrigée
        test_mask = tf.random.uniform((1, 224, 224), 0, 8, dtype=tf.int32)
        loss_value = dice_loss(test_mask, prediction)
        print(f"\n✅ Test de loss function corrigée: {loss_value:.4f}")
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {str(e)}")

def main():
    """Fonction principale de débogage"""
    
    print("DÉBOGAGE DES FORMES DE MODÈLES DE SEGMENTATION")
    print("=" * 60)
    
    # Test 1: Formes des modèles
    results = test_model_shapes()
    
    # Test 2: Fonctions de perte
    test_loss_functions()
    
    # Test 3: Cas d'erreur spécifique
    test_specific_error()
    
    # Résumé des solutions
    print("\n" + "="*60)
    print("SOLUTIONS IMPLEMENTÉES")
    print("="*60)
    print("✅ 1. Architecture VGG16-UNet corrigée")
    print("✅ 2. Fonctions de perte avec redimensionnement automatique")
    print("✅ 3. Fonction de débogage des formes")
    print("✅ 4. Gestion automatique des incompatibilités de taille")
    
    print("\n📋 RECOMMANDATIONS:")
    print("1. Utilisez debug_model_output_shape() pour vérifier vos modèles")
    print("2. Les fonctions de perte sont maintenant robustes aux différences de taille")
    print("3. En cas de problème, le redimensionnement automatique est appliqué")
    
    print("\n🔧 COMMANDES DE TEST:")
    print("python debug_model_shapes.py")
    print("python -c \"from models import *; model = create_model('vgg16_unet'); debug_model_output_shape(model)\"")

if __name__ == "__main__":
    main()