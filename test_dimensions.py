#!/usr/bin/env python3
"""
Script pour tracer les dimensions exactes dans U-Net Mini
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    print("✓ TensorFlow importé")
except ImportError as e:
    print(f"✗ Erreur: {e}")
    sys.exit(1)

def conv_block(x, filters, kernel_size=3, activation='relu', batch_norm=True):
    """Basic convolutional block with optional batch normalization"""
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x

def encoder_block(x, filters, pool_size=2):
    """Encoder block for U-Net: conv + conv + maxpool"""
    print(f"  Encoder input shape: {x.shape}")
    skip = conv_block(x, filters)
    print(f"  After first conv: {skip.shape}")
    skip = conv_block(skip, filters)
    print(f"  After second conv (skip): {skip.shape}")
    x = layers.MaxPooling2D(pool_size)(skip)
    print(f"  After MaxPool: {x.shape}")
    return x, skip

def decoder_block(x, skip, filters):
    """Decoder block for U-Net: upsample + concat + conv + conv"""
    print(f"  Decoder input shape: {x.shape}")
    print(f"  Skip connection shape: {skip.shape}")
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
    print(f"  After Conv2DTranspose: {x.shape}")
    x = layers.Concatenate()([x, skip])
    print(f"  After Concatenate: {x.shape}")
    x = conv_block(x, filters)
    x = conv_block(x, filters)
    print(f"  After conv blocks: {x.shape}")
    return x

def trace_unet_mini_dimensions():
    """Trace les dimensions à travers U-Net Mini"""
    
    print("="*60)
    print("TRACE DES DIMENSIONS - U-NET MINI")
    print("="*60)
    
    input_shape = (224, 224, 3)
    filters_base = 32
    
    inputs = layers.Input(input_shape)
    print(f"Input shape: {inputs.shape}")
    
    print("\n🔽 ENCODER")
    print("-" * 30)
    
    print("Encoder Block 1:")
    x1, skip1 = encoder_block(inputs, filters_base)
    
    print("\nEncoder Block 2:")
    x2, skip2 = encoder_block(x1, filters_base * 2)
    
    print("\nEncoder Block 3:")
    x3, skip3 = encoder_block(x2, filters_base * 4)
    
    print("\nEncoder Block 4:")
    x4, skip4 = encoder_block(x3, filters_base * 8)
    
    print("\n🔄 BOTTLENECK")
    print("-" * 30)
    print(f"Before bottleneck: {x4.shape}")
    bottleneck = conv_block(x4, filters_base * 16)
    print(f"After first bottleneck conv: {bottleneck.shape}")
    bottleneck = conv_block(bottleneck, filters_base * 16)
    print(f"After second bottleneck conv: {bottleneck.shape}")
    
    print("\n🔼 DECODER")
    print("-" * 30)
    
    print("Decoder Block 1:")
    d1 = decoder_block(bottleneck, skip4, filters_base * 8)
    
    print("\nDecoder Block 2:")
    d2 = decoder_block(d1, skip3, filters_base * 4)
    
    print("\nDecoder Block 3:")
    d3 = decoder_block(d2, skip2, filters_base * 2)
    
    print("\nDecoder Block 4:")
    d4 = decoder_block(d3, skip1, filters_base)
    
    print("\n🎯 FINAL LAYERS")
    print("-" * 30)
    print(f"Before final upsampling: {d4.shape}")
    
    # Final upsampling
    d4_final = layers.Conv2DTranspose(filters_base, 2, strides=2, padding='same')(d4)
    print(f"After final Conv2DTranspose: {d4_final.shape}")
    
    # Final classification
    outputs = layers.Conv2D(8, 1, activation='softmax', name='segmentation_output')(d4_final)
    print(f"Final output shape: {outputs.shape}")
    
    # Créer le modèle pour test
    model = Model(inputs, outputs, name='UNet_Mini_Debug')
    
    print(f"\n🧪 TEST AVEC DONNÉES RÉELLES")
    print("-" * 30)
    
    # Test avec des données réelles
    test_input = tf.random.normal((1, 224, 224, 3))
    test_output = model(test_input, training=False)
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {test_output.shape}")
    
    # Vérification
    if test_output.shape[1] == 224 and test_output.shape[2] == 224:
        print("✅ DIMENSIONS CORRECTES!")
    else:
        print(f"❌ PROBLÈME - Sortie: {test_output.shape[1]}x{test_output.shape[2]} au lieu de 224x224")
        
        # Identifier où est le problème
        expected_sizes = [224, 112, 56, 28, 14, 7, 14, 28, 56, 112, 224]
        actual_height = test_output.shape[1]
        
        if actual_height == 448:
            print("🔍 DIAGNOSTIC: Upsampling en trop - la sortie est 2x trop grande")
            print("💡 SOLUTION: Retirer un layer d'upsampling ou ajuster les strides")
        elif actual_height == 112:
            print("🔍 DIAGNOSTIC: Upsampling manquant - la sortie est 2x trop petite")
            print("💡 SOLUTION: Ajouter un layer d'upsampling ou ajuster les strides")
    
    return model

def compare_with_working_architecture():
    """Compare avec une architecture qui fonctionne garantie"""
    
    print("\n" + "="*60)
    print("ARCHITECTURE ALTERNATIVE SIMPLE")
    print("="*60)
    
    def simple_unet_mini(input_shape=(224, 224, 3), num_classes=8):
        """Version simplifiée garantie de fonctionner"""
        inputs = layers.Input(input_shape)
        
        # Encoder simple
        c1 = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
        c1 = layers.Conv2D(32, 3, padding='same', activation='relu')(c1)
        p1 = layers.MaxPooling2D(2)(c1)  # 112x112
        
        c2 = layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
        c2 = layers.Conv2D(64, 3, padding='same', activation='relu')(c2)
        p2 = layers.MaxPooling2D(2)(c2)  # 56x56
        
        c3 = layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
        c3 = layers.Conv2D(128, 3, padding='same', activation='relu')(c3)
        p3 = layers.MaxPooling2D(2)(c3)  # 28x28
        
        # Bottleneck
        c4 = layers.Conv2D(256, 3, padding='same', activation='relu')(p3)
        c4 = layers.Conv2D(256, 3, padding='same', activation='relu')(c4)  # 28x28
        
        # Decoder simple
        u5 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c4)  # 56x56
        u5 = layers.Concatenate()([u5, c3])
        c5 = layers.Conv2D(128, 3, padding='same', activation='relu')(u5)
        c5 = layers.Conv2D(128, 3, padding='same', activation='relu')(c5)
        
        u6 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c5)  # 112x112
        u6 = layers.Concatenate()([u6, c2])
        c6 = layers.Conv2D(64, 3, padding='same', activation='relu')(u6)
        c6 = layers.Conv2D(64, 3, padding='same', activation='relu')(c6)
        
        u7 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c6)  # 224x224
        u7 = layers.Concatenate()([u7, c1])
        c7 = layers.Conv2D(32, 3, padding='same', activation='relu')(u7)
        c7 = layers.Conv2D(32, 3, padding='same', activation='relu')(c7)
        
        outputs = layers.Conv2D(num_classes, 1, activation='softmax')(c7)
        
        return Model(inputs, outputs, name='Simple_UNet_Mini')
    
    # Test de l'architecture alternative
    simple_model = simple_unet_mini()
    test_input = tf.random.normal((1, 224, 224, 3))
    test_output = simple_model(test_input, training=False)
    
    print(f"Architecture simple - Input: {test_input.shape}")
    print(f"Architecture simple - Output: {test_output.shape}")
    
    if test_output.shape[1] == 224 and test_output.shape[2] == 224:
        print("✅ L'architecture simple fonctionne correctement!")
        return simple_model
    else:
        print("❌ Même l'architecture simple a un problème")
        return None

def main():
    print("DIAGNOSTIC DES DIMENSIONS U-NET MINI")
    print("=" * 60)
    
    # Test 1: Tracer l'architecture actuelle
    try:
        current_model = trace_unet_mini_dimensions()
    except Exception as e:
        print(f"❌ Erreur dans l'architecture actuelle: {e}")
        current_model = None
    
    # Test 2: Architecture alternative
    try:
        simple_model = compare_with_working_architecture()
    except Exception as e:
        print(f"❌ Erreur dans l'architecture simple: {e}")
        simple_model = None
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DU DIAGNOSTIC")
    print("="*60)
    
    if current_model is not None:
        test_out = current_model(tf.random.normal((1, 224, 224, 3)), training=False)
        print(f"Architecture actuelle: {test_out.shape}")
    
    if simple_model is not None:
        test_out = simple_model(tf.random.normal((1, 224, 224, 3)), training=False)
        print(f"Architecture simple: {test_out.shape}")

if __name__ == "__main__":
    main()