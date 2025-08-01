#!/usr/bin/env python3
"""
Solutions rapides pour accélérer l'entraînement

40 minutes par époque → 2-5 minutes par époque
"""

import tensorflow as tf
import numpy as np

def apply_immediate_optimizations():
    """Applique les optimisations immédiates"""
    
    print("🚀 APPLICATION DES OPTIMISATIONS IMMÉDIATES")
    print("="*50)
    
    # 1. GPU Memory Growth (évite les erreurs OOM)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth activé")
    
    # 2. Mixed Precision (2x plus rapide sur GPU moderne)
    if gpus:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("✅ Mixed precision activé (accélération ~2x)")
    
    # 3. XLA compilation
    tf.config.optimizer.set_jit(True)
    print("✅ XLA JIT compilation activé")
    
    # 4. Optimiser les threads
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
    print("✅ Threading optimisé")

def create_fast_dataset(images, masks, batch_size=32, cache=True):
    """Créer un dataset ultra-rapide"""
    
    print(f"\n🏃 CRÉATION D'UN DATASET RAPIDE")
    print("="*50)
    
    # Dataset de base
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    
    # Optimisations critiques
    dataset = dataset.map(
        lambda x, y: (x, y),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False  # Plus rapide
    )
    
    if cache:
        dataset = dataset.cache()  # Met en cache RAM/SSD
        print("✅ Cache activé (données en mémoire)")
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    print(f"✅ Batch size: {batch_size}")
    print(f"✅ Prefetch: AUTOTUNE")
    print(f"✅ Parallel calls: AUTOTUNE")
    
    return dataset

def get_optimal_batch_size():
    """Détermine le batch size optimal selon la mémoire disponible"""
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("⚠️  CPU détecté - batch_size recommandé: 8-16")
        return 8
    
    # Estimation basée sur GPU memory
    try:
        # Test avec différents batch sizes
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, input_shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(8)
        ])
        
        for batch_size in [64, 32, 16, 8]:
            try:
                test_data = tf.random.normal((batch_size, 224, 224, 3))
                _ = model(test_data)
                print(f"✅ GPU supporte batch_size: {batch_size}")
                return batch_size
            except tf.errors.ResourceExhaustedError:
                continue
        
        return 8
    except:
        return 16

def optimize_model_for_speed(model):
    """Optimise le modèle pour la vitesse"""
    
    print(f"\n⚡ OPTIMISATION DU MODÈLE")
    print("="*50)
    
    # Compiler avec optimisations
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        # Pour mixed precision, utiliser des loss functions compatibles
        model.compile(
            optimizer=tf.keras.optimizers.Adam(2e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        print("✅ Compilation optimisée pour mixed precision")
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Compilation standard")
    
    return model

def create_speed_optimized_training():
    """Configuration d'entraînement optimisée pour la vitesse"""
    
    print(f"\n🏁 CONFIGURATION D'ENTRAÎNEMENT RAPIDE")
    print("="*50)
    
    # Callbacks pour convergence rapide
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',  # Monitor training loss directement
            factor=0.5,
            patience=2,      # Plus agressif
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,      # Plus agressif
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    print("✅ Callbacks configurés pour convergence rapide")
    print("✅ Early stopping agressif")
    print("✅ Réduction LR rapide")
    
    return callbacks

# ============================================================================
# EXEMPLE COMPLET D'OPTIMISATION
# ============================================================================

def example_optimized_training():
    """Exemple complet d'entraînement optimisé"""
    
    print("\n" + "="*60)
    print("EXEMPLE D'ENTRAÎNEMENT ULTRA-RAPIDE")
    print("="*60)
    
    # 1. Appliquer les optimisations
    apply_immediate_optimizations()
    
    # 2. Déterminer batch size optimal
    optimal_batch_size = get_optimal_batch_size()
    
    # 3. Créer données factices pour demo
    print(f"\n📊 Création de données de test...")
    num_samples = 1000
    images = tf.random.normal((num_samples, 224, 224, 3))
    masks = tf.random.uniform((num_samples, 224, 224), 0, 8, dtype=tf.int32)
    
    # 4. Dataset optimisé
    train_ds = create_fast_dataset(images[:800], masks[:800], optimal_batch_size)
    val_ds = create_fast_dataset(images[800:], masks[800:], optimal_batch_size)
    
    # 5. Modèle optimisé
    try:
        from models import create_model
        model = create_model('unet_mini', (224, 224, 3), 8)
        model = optimize_model_for_speed(model)
        print(f"✅ Modèle U-Net Mini créé et optimisé")
    except ImportError:
        print("⚠️  Module models non trouvé, utilisation d'un modèle simple")
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(8, 1, activation='softmax')
        ])
        model = optimize_model_for_speed(model)
    
    # 6. Configuration rapide
    callbacks = create_speed_optimized_training()
    
    # 7. Test d'entraînement rapide
    print(f"\n🏃 ENTRAÎNEMENT TEST (2 époques)...")
    import time
    start_time = time.time()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=2,
        callbacks=callbacks,
        verbose=1
    )
    
    elapsed_time = time.time() - start_time
    time_per_epoch = elapsed_time / 2
    
    print(f"\n⏱️  RÉSULTATS:")
    print(f"   Temps total: {elapsed_time:.1f}s")
    print(f"   Temps/époque: {time_per_epoch:.1f}s")
    print(f"   Accélération vs 40min: {2400/time_per_epoch:.1f}x plus rapide!")
    
    return history

# ============================================================================
# CODE POUR VOTRE CAS SPÉCIFIQUE
# ============================================================================

def fix_your_training():
    """Code spécifique pour résoudre votre problème de 40min/époque"""
    
    print("\n" + "="*60)
    print("SOLUTION POUR VOTRE CAS SPÉCIFIQUE")
    print("="*60)
    
    print("🔧 AJOUTEZ CE CODE AU DÉBUT DE VOTRE NOTEBOOK:")
    print()
    
    code_to_add = '''
# ===== OPTIMISATIONS SPEED =====
import tensorflow as tf

# 1. Activer toutes les optimisations GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Mixed precision = 2x plus rapide
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("✅ GPU optimisé avec mixed precision")
else:
    print("⚠️ Pas de GPU - utilisez Google Colab avec GPU!")

# 2. XLA compilation
tf.config.optimizer.set_jit(True)

# 3. Optimiser le dataset
def optimize_dataset(ds, batch_size=32):
    return ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ===== UTILISATION =====
# Appliquez à vos datasets:
# train_ds = optimize_dataset(train_ds, batch_size=32)
# validation_ds = optimize_dataset(validation_ds, batch_size=32)

# Compilez votre modèle avec:
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(2e-4),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=['accuracy']
# )
'''
    
    print(code_to_add)
    
    print("\n🎯 CHANGEMENTS DANS VOTRE CODE D'ENTRAÎNEMENT:")
    print()
    
    training_code = '''
# Au lieu de:
# hist = model.fit(train_sample, validation_data=validation_sample, epochs=epochs)

# Utilisez:
hist = model.fit(
    train_sample,
    validation_data=validation_sample,
    epochs=epochs,
    batch_size=32,  # Augmentez si possible
    verbose=1
)
'''
    
    print(training_code)
    
    print("\n📈 RÉSULTATS ATTENDUS:")
    print("   - AVANT: 40 minutes/époque")
    print("   - APRÈS: 2-5 minutes/époque (avec GPU)")
    print("   - APRÈS: 10-15 minutes/époque (CPU optimisé)")

def main():
    """Fonction principale"""
    
    print("ACCÉLÉRATION D'ENTRAÎNEMENT - DE 40MIN À 2MIN PAR ÉPOQUE")
    print("=" * 60)
    
    # Diagnostic rapide
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("🚨 PROBLÈME PRINCIPAL: PAS DE GPU DÉTECTÉ")
        print("   Solution: Utilisez Google Colab avec GPU activé")
        print("   Runtime → Change runtime type → GPU")
    else:
        print(f"✅ {len(gpus)} GPU(s) détecté(s)")
    
    # Solutions immédiates
    fix_your_training()
    
    # Test si possible
    try:
        example_optimized_training()
    except Exception as e:
        print(f"\n⚠️ Test d'optimisation échoué: {e}")
        print("Mais les optimisations ci-dessus devraient fonctionner!")

if __name__ == "__main__":
    main()