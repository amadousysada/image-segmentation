#!/usr/bin/env python3
"""
Script d'entraînement pour la segmentation sémantique avec différents modèles et fonctions de perte

Usage:
    python train_segmentation.py --model unet_mini --loss dice_loss --epochs 20
    python train_segmentation.py --model vgg16_unet --loss combined_loss --epochs 50
"""

import os
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

from models import (
    create_model, 
    compile_model, 
    MeanIoUArgmax,
    dice_loss,
    focal_loss,
    combined_loss,
    balanced_cross_entropy
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train semantic segmentation models')
    
    parser.add_argument('--model', type=str, default='unet_mini',
                       choices=['unet_mini', 'unet_mini_deep', 'vgg16_unet', 'resnet50_unet'],
                       help='Model architecture to use')
    
    parser.add_argument('--loss', type=str, default='cross_entropy',
                       choices=['cross_entropy', 'dice_loss', 'focal_loss', 
                               'combined_loss', 'balanced_cross_entropy'],
                       help='Loss function to use')
    
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
    
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size (square images)')
    
    parser.add_argument('--freeze_encoder', action='store_true',
                       help='Freeze encoder weights for pretrained models')
    
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/data/',
                       help='Path to the dataset')
    
    parser.add_argument('--save_dir', type=str, default='./models',
                       help='Directory to save trained models')
    
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    parser.add_argument('--augment', action='store_true',
                       help='Use data augmentation')
    
    return parser.parse_args()


def load_dataset_paths(data_path):
    """Load dataset paths for images and masks"""
    image_dir = os.path.join(data_path, "leftImg8bit")
    mask_dir = os.path.join(data_path, "gtFine")
    
    # Class mapping for Cityscapes
    CLASS_GROUPS = {
        "flat":        ["road", "sidewalk", "parking", "rail track"],
        "human":       ["person", "rider"],
        "vehicle":     ["car", "truck", "bus", "on rails", "motorcycle", "bicycle", "caravan", "trailer"],
        "construction":["building", "wall", "fence", "guard rail", "bridge", "tunnel"],
        "object":      ["pole", "pole group", "traffic sign", "traffic light"],
        "nature":      ["vegetation", "terrain"],
        "sky":         ["sky"],
        "void":        ["unlabeled", "ego vehicle", "ground", "rectification border", "out of roi", "dynamic", "static"]
    }
    
    ordered_groups = list(CLASS_GROUPS.keys())
    
    LABEL_ID_TO_NAME = {
        0: "unlabeled", 1: "ego vehicle", 2: "rectification border", 3: "out of roi",
        4: "static", 5: "dynamic", 6: "ground", 7: "road", 8: "sidewalk", 9: "parking",
        10: "rail track", 11: "building", 12: "wall", 13: "fence", 14: "guard rail",
        15: "bridge", 16: "tunnel", 17: "pole", 18: "pole group", 19: "traffic light",
        20: "traffic sign", 21: "vegetation", 22: "terrain", 23: "sky", 24: "person",
        25: "rider", 26: "car", 27: "truck", 28: "bus", 29: "caravan", 30: "trailer",
        31: "on rails", 32: "motorcycle", 33: "bicycle",
    }
    
    NAME_TO_LABEL_ID = {v: k for k, v in sorted(LABEL_ID_TO_NAME.items())}
    
    CLASS_MAP = {}
    for group_idx, group_name in enumerate(ordered_groups):
        for class_name in CLASS_GROUPS[group_name]:
            cid = NAME_TO_LABEL_ID.get(class_name, -1)
            CLASS_MAP[cid] = group_idx
    
    mapping = [7] * 34   # initialized to 7 (void)
    for orig_id, new_id in CLASS_MAP.items():
        mapping[orig_id] = new_id
    
    return image_dir, mask_dir, mapping


def augment_data(image, mask):
    """Data augmentation function"""
    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask[..., tf.newaxis])[..., 0]
    
    # Random contrast
    image = tf.image.random_contrast(image, 0.8, 1.2)
    
    # Random brightness
    image = tf.image.random_brightness(image, 0.1)
    
    # Random saturation
    image = tf.image.random_saturation(image, 0.8, 1.2)
    
    return image, mask


def build_dataset(img_paths, mask_paths, image_size, mapping, batch_size=32, augment=False, shuffle=False):
    """Build TensorFlow dataset from image and mask paths"""
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    
    def _load_and_preprocess(img_path, mask_path):
        # Load and decode images
        img = tf.image.decode_png(tf.io.read_file(img_path), channels=3)
        mask = tf.image.decode_png(tf.io.read_file(mask_path), channels=1)
        
        # Resize images
        img = tf.image.resize(img, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR)
        mask = tf.image.resize(mask, [image_size, image_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        # Normalize image to [0, 1]
        img = tf.cast(img, tf.float32) / 255.0
        
        # Convert mask to int32 and apply class mapping
        mask = tf.cast(mask, tf.int32)
        mask = tf.gather(tf.constant(mapping, dtype=tf.int32), mask)
        mask = tf.squeeze(mask, axis=-1)  # Remove channel dimension
        
        return img, mask
    
    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    
    if augment:
        ds = ds.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def make_dataset(image_dir, mask_dir, mapping, data_type="train", image_size=224, 
                batch_size=32, validation_split=0.0, augment=False, shuffle=False):
    """Create dataset from directory structure"""
    images_path = []
    masks_path = []
    
    # Collect all image and mask paths
    data_dir = os.path.join(image_dir, data_type)
    mask_data_dir = os.path.join(mask_dir, data_type)
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    for city in os.listdir(data_dir):
        city_img_path = os.path.join(data_dir, city)
        city_mask_path = os.path.join(mask_data_dir, city)
        
        if not os.path.isdir(city_img_path):
            continue
            
        for image_file in os.listdir(city_img_path):
            if image_file.endswith("_leftImg8bit.png"):
                base_name = image_file.replace("_leftImg8bit.png", "")
                mask_file = base_name + "_gtFine_labelIds.png"
                
                img_path = os.path.join(city_img_path, image_file)
                mask_path = os.path.join(city_mask_path, mask_file)
                
                if os.path.exists(mask_path):
                    images_path.append(img_path)
                    masks_path.append(mask_path)
    
    print(f"Found {len(images_path)} images for {data_type} set")
    
    # Split into train/validation if needed
    if data_type == "train" and validation_split > 0:
        n = len(images_path)
        split_idx = int((1 - validation_split) * n)
        
        train_imgs, val_imgs = images_path[:split_idx], images_path[split_idx:]
        train_masks, val_masks = masks_path[:split_idx], masks_path[split_idx:]
        
        print(f"Training set: {len(train_imgs)} images")
        print(f"Validation set: {len(val_imgs)} images")
        
        train_ds = build_dataset(train_imgs, train_masks, image_size, mapping, 
                               batch_size, augment=augment, shuffle=shuffle)
        val_ds = build_dataset(val_imgs, val_masks, image_size, mapping, batch_size)
        
        return train_ds, val_ds
    else:
        return build_dataset(images_path, masks_path, image_size, mapping, 
                           batch_size, shuffle=shuffle, augment=augment)


def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # Mean IoU
    iou_key = 'mean_io_u_argmax'
    val_iou_key = 'val_mean_io_u_argmax'
    if iou_key in history.history:
        axes[1, 0].plot(history.history[iou_key], label='Training IoU')
        if val_iou_key in history.history:
            axes[1, 0].plot(history.history[val_iou_key], label='Validation IoU')
        axes[1, 0].set_title('Mean IoU')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].legend()
    
    # Learning rate
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()


def main():
    """Main training function"""
    args = parse_arguments()
    
    print("="*60)
    print("SEMANTIC SEGMENTATION TRAINING")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Loss function: {args.loss}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    print("="*60)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset paths and mapping
    print("Loading dataset...")
    image_dir, mask_dir, mapping = load_dataset_paths(args.data_path)
    
    # Create datasets
    train_ds, val_ds = make_dataset(
        image_dir, mask_dir, mapping,
        data_type="train",
        image_size=args.image_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        augment=args.augment,
        shuffle=True
    )
    
    # Test dataset (optional)
    try:
        test_ds = make_dataset(
            image_dir, mask_dir, mapping,
            data_type="val",  # Using val as test for demonstration
            image_size=args.image_size,
            batch_size=args.batch_size
        )
        print("Test dataset loaded successfully")
    except:
        test_ds = None
        print("Test dataset not available")
    
    # Create model
    print(f"Creating {args.model} model...")
    model = create_model(
        model_type=args.model,
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=8,
        freeze_encoder=args.freeze_encoder
    )
    
    # Compile model
    model = compile_model(
        model,
        loss_type=args.loss,
        learning_rate=args.learning_rate
    )
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Define callbacks
    model_name = f"{args.model}_{args.loss}_{args.image_size}"
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(args.save_dir, f"best_{model_name}.h5"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Start training
    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, f"final_{model_name}.h5")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Plot training history
    history_plot_path = os.path.join(args.save_dir, f"history_{model_name}.png")
    plot_training_history(history, history_plot_path)
    
    # Evaluate on test set if available
    if test_ds is not None:
        print("\nEvaluating on test set...")
        test_results = model.evaluate(test_ds, verbose=1)
        print(f"Test results: {dict(zip(model.metrics_names, test_results))}")
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()