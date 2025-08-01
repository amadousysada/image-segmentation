#!/usr/bin/env python3
"""
Test rapide des corrections de dimensions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import tensorflow as tf
    from models import create_model, compile_model, dice_loss
    print("✓ Imports réussis")
except ImportError as e:
    print(f"✗ Erreur d'import: {e}")
    print("Note: TensorFlow n'est peut-être pas installé")
    sys.exit(1)

def test_model_outputs():
    """Test que tous les modèles produisent la bonne taille de sortie"""
    
    print("="*60)
    print("TEST DES CORRECTIONS DE DIMENSIONS")
    print("="*60)
    
    input_shape = (224, 224, 3)
    num_classes = 8
    
    models_to_test = [
        'unet_mini',
        'unet_mini_deep', 
        'vgg16_unet',
        'resnet50_unet'
    ]
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n🧪 Test de {model_name}")
        print("-" * 40)
        
        try:
            # Créer le modèle
            model = create_model(model_name, input_shape, num_classes)
            print(f"✓ Modèle créé: {model.count_params():,} paramètres")
            
            # Test de prédiction
            test_input = tf.random.normal((1, *input_shape))
            prediction = model(test_input, training=False)
            
            print(f"Input shape: {test_input.shape}")
            print(f"Output shape: {prediction.shape}")
            
            # Vérification
            expected_h, expected_w = input_shape[:2]
            actual_h, actual_w = prediction.shape[1:3]
            
            if actual_h == expected_h and actual_w == expected_w:
                status = "✅ CORRECT"
                results[model_name] = "SUCCESS"
            else:
                status = f"❌ ERREUR - {actual_h}x{actual_w} au lieu de {expected_h}x{expected_w}"
                results[model_name] = "FAIL"
            
            print(f"Status: {status}")
            
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")
            results[model_name] = f"ERROR: {str(e)}"
    
    return results

def test_loss_functions():
    """Test que les fonctions de perte fonctionnent avec les nouvelles dimensions"""
    
    print("\n" + "="*60)
    print("TEST DES FONCTIONS DE PERTE")
    print("="*60)
    
    # Utiliser U-Net Mini (corrigé)
    model = create_model('unet_mini', (224, 224, 3), 8)
    
    # Données de test
    test_images = tf.random.normal((2, 224, 224, 3))
    test_masks = tf.random.uniform((2, 224, 224), 0, 8, dtype=tf.int32)
    
    print(f"Images: {test_images.shape}")
    print(f"Masques: {test_masks.shape}")
    
    # Prédiction
    predictions = model(test_images, training=False)
    print(f"Prédictions: {predictions.shape}")
    
    # Test des fonctions de perte
    loss_functions = [
        ('Dice Loss', dice_loss),
        ('Cross-Entropy', lambda y_true, y_pred: tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        ))
    ]
    
    print("\nTest des fonctions de perte:")
    print("-" * 30)
    
    for name, loss_fn in loss_functions:
        try:
            loss_value = loss_fn(test_masks, predictions)
            print(f"✓ {name}: {loss_value:.4f}")
        except Exception as e:
            print(f"❌ {name}: Erreur - {str(e)}")

def test_training_step():
    """Test d'une étape d'entraînement complète"""
    
    print("\n" + "="*60)
    print("TEST D'ENTRAÎNEMENT")
    print("="*60)
    
    try:
        # Créer et compiler le modèle
        model = create_model('unet_mini', (224, 224, 3), 8)
        model = compile_model(model, 'dice_loss', learning_rate=1e-3)
        
        print("✓ Modèle compilé")
        
        # Données factices
        train_images = tf.random.normal((4, 224, 224, 3))
        train_masks = tf.random.uniform((4, 224, 224), 0, 8, dtype=tf.int32)
        
        print(f"✓ Données créées: {train_images.shape}, {train_masks.shape}")
        
        # Une étape d'entraînement
        history = model.fit(
            train_images, train_masks,
            epochs=1,
            batch_size=2,
            verbose=1
        )
        
        print("✅ Entraînement réussi!")
        print(f"Loss finale: {history.history['loss'][0]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur d'entraînement: {str(e)}")
        return False

def main():
    """Test principal"""
    
    print("TEST DES CORRECTIONS DE SEGMENTATION")
    print("=" * 60)
    
    # Test 1: Dimensions des modèles
    model_results = test_model_outputs()
    
    # Test 2: Fonctions de perte
    test_loss_functions()
    
    # Test 3: Entraînement
    training_success = test_training_step()
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)
    
    print("\n📊 Résultats des modèles:")
    for model_name, result in model_results.items():
        icon = "✅" if result == "SUCCESS" else "❌"
        print(f"  {icon} {model_name}: {result}")
    
    print(f"\n🏋️ Entraînement: {'✅ Réussi' if training_success else '❌ Échec'}")
    
    # Vérification globale
    all_models_ok = all(result == "SUCCESS" for result in model_results.values())
    
    if all_models_ok and training_success:
        print("\n🎉 TOUTES LES CORRECTIONS FONCTIONNENT!")
        print("Vous pouvez maintenant utiliser les modèles sans erreur de dimension.")
    else:
        print("\n⚠️ Certains problèmes persistent.")
        print("Vérifiez les erreurs ci-dessus.")
    
    # Commandes d'usage
    print("\n" + "="*60)
    print("COMMANDES D'USAGE")
    print("="*60)
    print("# Test rapide:")
    print("python test_fixes.py")
    print("\n# Entraînement U-Net Mini (corrigé):")
    print("python train_segmentation.py --model unet_mini --loss dice_loss --epochs 5")
    print("\n# Entraînement VGG16-UNet (corrigé):")
    print("python train_segmentation.py --model vgg16_unet --loss combined_loss --epochs 10")

if __name__ == "__main__":
    main()