#!/usr/bin/env python3
"""
Script d'optimisation des performances d'entraînement

Diagnostic et solutions pour accélérer l'entraînement des modèles de segmentation.
"""

import time
import tensorflow as tf
import os
import psutil
import sys

def diagnose_system():
    """Diagnostic du système pour identifier les goulots d'étranglement"""
    
    print("="*60)
    print("DIAGNOSTIC DU SYSTÈME")
    print("="*60)
    
    # GPU
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"🖥️  GPU disponibles: {len(gpu_devices)}")
    
    if gpu_devices:
        for i, gpu in enumerate(gpu_devices):
            print(f"   GPU {i}: {gpu.name}")
        
        # Vérifier si GPU est utilisé
        print(f"🔧 GPU configuré: {tf.test.is_gpu_available()}")
        print(f"🏃 GPU actif: {tf.test.is_built_with_cuda()}")
    else:
        print("⚠️  AUCUN GPU DÉTECTÉ - Utilisation CPU uniquement")
        print("💡 Solution: Utiliser Google Colab avec GPU ou installer CUDA")
    
    # Mémoire
    memory = psutil.virtual_memory()
    print(f"\n💾 RAM totale: {memory.total / (1024**3):.1f} GB")
    print(f"💾 RAM disponible: {memory.available / (1024**3):.1f} GB")
    print(f"💾 RAM utilisée: {memory.percent:.1f}%")
    
    # CPU
    print(f"\n🔲 CPU cores: {psutil.cpu_count()}")
    print(f"🔲 CPU usage: {psutil.cpu_percent():.1f}%")
    
    # TensorFlow config
    print(f"\n⚙️  TensorFlow version: {tf.__version__}")
    print(f"⚙️  Mixed precision: {tf.keras.mixed_precision.global_policy().name}")

def optimize_tensorflow():
    """Optimiser la configuration TensorFlow"""
    
    print("\n" + "="*60)
    print("OPTIMISATION TENSORFLOW")
    print("="*60)
    
    # 1. Configuration GPU
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        try:
            # Permettre la croissance mémoire GPU
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Croissance mémoire GPU activée")
        except:
            print("⚠️  Impossible de configurer la croissance mémoire GPU")
    
    # 2. Mixed Precision (accélération sur GPU moderne)
    if gpu_devices:
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("✅ Mixed precision activée (float16)")
        except:
            print("⚠️  Mixed precision non supportée")
    
    # 3. XLA (compilation optimisée)
    try:
        tf.config.optimizer.set_jit(True)
        print("✅ XLA JIT compilation activée")
    except:
        print("⚠️  XLA non disponible")
    
    # 4. Parallel processing
    tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available
    tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available
    print("✅ Parallélisme optimisé")

def create_optimized_dataset(image_paths, mask_paths, image_size=224, batch_size=16):
    """Créer un dataset optimisé pour les performances"""
    
    print("\n" + "="*60)
    print("OPTIMISATION DU DATASET")
    print("="*60)
    
    def parse_function(image_path, mask_path):
        """Fonction de parsing optimisée"""
        # Lecture des fichiers
        image = tf.io.read_file(image_path)
        mask = tf.io.read_file(mask_path)
        
        # Décodage
        image = tf.image.decode_png(image, channels=3)
        mask = tf.image.decode_png(mask, channels=1)
        
        # Redimensionnement
        image = tf.image.resize(image, [image_size, image_size])
        mask = tf.image.resize(mask, [image_size, image_size], method='nearest')
        
        # Normalisation
        image = tf.cast(image, tf.float32) / 255.0
        mask = tf.cast(mask, tf.int32)
        mask = tf.squeeze(mask, axis=-1)
        
        return image, mask
    
    # Créer le dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
    # Optimisations
    dataset = dataset.map(
        parse_function, 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False  # Plus rapide mais non déterministe
    )
    
    dataset = dataset.cache()  # Met en cache en mémoire
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    print(f"✅ Dataset optimisé créé:")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Parallélisme: AUTOTUNE")
    print(f"   - Cache: Activé")
    print(f"   - Prefetch: AUTOTUNE")
    
    return dataset

def benchmark_model_speed(model, input_shape=(224, 224, 3), batch_sizes=[4, 8, 16, 32]):
    """Benchmark la vitesse du modèle avec différents batch sizes"""
    
    print("\n" + "="*60)
    print("BENCHMARK DE VITESSE")
    print("="*60)
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n🧪 Test batch_size = {batch_size}")
        
        # Créer données de test
        test_data = tf.random.normal((batch_size, *input_shape))
        
        # Warmup
        for _ in range(3):
            _ = model(test_data, training=False)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            _ = model(test_data, training=False)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        samples_per_sec = batch_size / avg_time
        
        results[batch_size] = {
            'time_per_batch': avg_time,
            'samples_per_sec': samples_per_sec
        }
        
        print(f"   Temps/batch: {avg_time:.3f}s")
        print(f"   Échantillons/sec: {samples_per_sec:.1f}")
    
    # Recommandation
    best_batch = max(results.keys(), key=lambda x: results[x]['samples_per_sec'])
    print(f"\n💡 RECOMMANDATION: Batch size optimal = {best_batch}")
    print(f"   Performance: {results[best_batch]['samples_per_sec']:.1f} échantillons/sec")
    
    return results

def create_fast_training_setup():
    """Créer une configuration d'entraînement rapide"""
    
    print("\n" + "="*60)
    print("CONFIGURATION D'ENTRAÎNEMENT RAPIDE")
    print("="*60)
    
    # Callbacks optimisés
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=3,  # Plus agressif
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,  # Plus agressif
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Optimiseur rapide
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=2e-4,  # Plus élevé pour convergence rapide
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    print("✅ Configuration créée:")
    print("   - Callbacks agressifs pour convergence rapide")
    print("   - Learning rate optimisé")
    print("   - Adam optimizer configuré")
    
    return callbacks, optimizer

def estimate_training_time(num_samples, batch_size, epochs, samples_per_sec):
    """Estimer le temps d'entraînement"""
    
    batches_per_epoch = num_samples // batch_size
    total_batches = batches_per_epoch * epochs
    estimated_time = total_batches / (samples_per_sec / batch_size)
    
    print(f"\n⏱️  ESTIMATION DE TEMPS:")
    print(f"   Échantillons: {num_samples}")
    print(f"   Batch size: {batch_size}")
    print(f"   Époques: {epochs}")
    print(f"   Batches/époque: {batches_per_epoch}")
    print(f"   Temps estimé: {estimated_time/60:.1f} minutes")
    
    return estimated_time

def main():
    """Diagnostic et optimisation principale"""
    
    print("OPTIMISATION DES PERFORMANCES D'ENTRAÎNEMENT")
    print("=" * 60)
    
    # 1. Diagnostic système
    diagnose_system()
    
    # 2. Optimisation TensorFlow
    optimize_tensorflow()
    
    # 3. Test avec un modèle simple
    try:
        from models import create_model
        
        print(f"\n🧪 Test avec U-Net Mini...")
        model = create_model('unet_mini', (224, 224, 3), 8)
        
        # Benchmark
        results = benchmark_model_speed(model)
        
        # Configuration optimisée
        callbacks, optimizer = create_fast_training_setup()
        
        # Estimation pour votre cas
        print(f"\n📊 ANALYSE DE VOTRE CAS:")
        print(f"   Modèle: ~2M paramètres")
        print(f"   Temps actuel: 40 min/époque")
        
        # Si 40min pour 1 époque, calculer le throughput actuel
        # Supposons ~2000 échantillons par époque (estimation)
        current_samples_per_min = 2000 / 40
        print(f"   Throughput actuel: ~{current_samples_per_min:.1f} échantillons/minute")
        
        if len(tf.config.list_physical_devices('GPU')) == 0:
            print(f"\n⚠️  PROBLÈME PRINCIPAL: PAS DE GPU")
            print(f"   🚀 Avec GPU: ~2-5 minutes/époque attendues")
            print(f"   🐌 Avec CPU: 30-60 minutes/époque (normal)")
    
    except ImportError:
        print("⚠️  Module models non trouvé")
    
    # Recommandations
    print(f"\n" + "="*60)
    print("RECOMMANDATIONS PRIORITAIRES")
    print("="*60)
    
    if len(tf.config.list_physical_devices('GPU')) == 0:
        print("🎯 1. UTILISER UN GPU (priorité absolue)")
        print("   - Google Colab avec GPU T4/V100")
        print("   - Kaggle Notebooks avec GPU")
        print("   - AWS/GCP avec instances GPU")
        
    print("🎯 2. OPTIMISER LE BATCH SIZE")
    print("   - Tester batch_size=16, 32, 64")
    print("   - Plus grand = plus rapide (si mémoire suffisante)")
    
    print("🎯 3. RÉDUIRE LA TAILLE DES DONNÉES")
    print("   - Images 224x224 au lieu de 512x512")
    print("   - Sous-échantillonnage du dataset")
    
    print("🎯 4. UTILISER MIXED PRECISION")
    print("   - tf.keras.mixed_precision.set_global_policy('mixed_float16')")
    
    print("🎯 5. OPTIMISER LE PIPELINE DE DONNÉES")
    print("   - .cache(), .prefetch(), num_parallel_calls=AUTOTUNE")

if __name__ == "__main__":
    main()