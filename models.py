import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K


def debug_model_output_shape(model, input_shape=(224, 224, 3)):
    """
    Debug function to check model output shape
    
    Args:
        model: Keras model to debug
        input_shape: Input shape to test
    
    Returns:
        Output shape information
    """
    test_input = tf.random.normal((1, *input_shape))
    output = model(test_input, training=False)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected mask shape for loss: {test_input.shape[:-1]} (without channels)")
    
    return output.shape


class MeanIoUArgmax(tf.keras.metrics.MeanIoU):
    """Custom MeanIoU metric that applies argmax to predictions"""
    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_pred : (batch, H, W, num_classes) → take the winning class
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice Loss for semantic segmentation
    
    Args:
        y_true: Ground truth masks (batch_size, H, W)
        y_pred: Predicted logits (batch_size, H, W, num_classes)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice loss value
    """
    # Ensure y_pred has softmax applied
    if y_pred.shape[-1] > 1:
        y_pred = tf.nn.softmax(y_pred, axis=-1)
    
    # Resize y_true to match y_pred spatial dimensions if needed
    pred_shape = tf.shape(y_pred)
    true_shape = tf.shape(y_true)
    
    if true_shape[1] != pred_shape[1] or true_shape[2] != pred_shape[2]:
        y_true = tf.image.resize(
            tf.expand_dims(tf.cast(y_true, tf.float32), axis=-1),
            [pred_shape[1], pred_shape[2]],
            method='nearest'
        )
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
    
    # Convert ground truth to one-hot encoding
    num_classes = tf.shape(y_pred)[-1]
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
    
    # Flatten tensors
    y_true_flat = tf.reshape(y_true_one_hot, [-1, num_classes])
    y_pred_flat = tf.reshape(y_pred, [-1, num_classes])
    
    # Calculate Dice coefficient for each class
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
    union = tf.reduce_sum(y_true_flat, axis=0) + tf.reduce_sum(y_pred_flat, axis=0)
    
    dice_coeff = (2. * intersection + smooth) / (union + smooth)
    
    # Return 1 - mean Dice coefficient as loss
    return 1 - tf.reduce_mean(dice_coeff)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss for addressing class imbalance
    
    Args:
        y_true: Ground truth masks (batch_size, H, W)
        y_pred: Predicted logits (batch_size, H, W, num_classes)
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
    
    Returns:
        Focal loss value
    """
    # Ensure y_pred has softmax applied
    if y_pred.shape[-1] > 1:
        y_pred = tf.nn.softmax(y_pred, axis=-1)
    
    # Resize y_true to match y_pred spatial dimensions if needed
    pred_shape = tf.shape(y_pred)
    true_shape = tf.shape(y_true)
    
    if true_shape[1] != pred_shape[1] or true_shape[2] != pred_shape[2]:
        y_true = tf.image.resize(
            tf.expand_dims(tf.cast(y_true, tf.float32), axis=-1),
            [pred_shape[1], pred_shape[2]],
            method='nearest'
        )
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
    
    # Convert ground truth to one-hot encoding
    num_classes = tf.shape(y_pred)[-1]
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
    
    # Calculate cross entropy
    ce_loss = -y_true_one_hot * tf.math.log(tf.clip_by_value(y_pred, 1e-8, 1.0))
    
    # Calculate focal weight
    p_t = y_true_one_hot * y_pred
    alpha_t = y_true_one_hot * alpha
    focal_weight = alpha_t * tf.pow(1 - p_t, gamma)
    
    # Apply focal weight
    focal_loss = focal_weight * ce_loss
    
    return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))


def combined_loss(y_true, y_pred, dice_weight=0.5, ce_weight=0.5):
    """
    Combined Dice + Cross-Entropy Loss
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted logits
        dice_weight: Weight for dice loss
        ce_weight: Weight for cross-entropy loss
    
    Returns:
        Combined loss value
    """
    dice = dice_loss(y_true, y_pred)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    ce = tf.reduce_mean(ce)
    
    return dice_weight * dice + ce_weight * ce


def balanced_cross_entropy(y_true, y_pred, class_weights=None):
    """
    Balanced Cross-Entropy Loss with class weights
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted logits
        class_weights: Optional class weights tensor
    
    Returns:
        Weighted cross-entropy loss
    """
    if class_weights is None:
        # Default weights for Cityscapes (8 classes)
        class_weights = tf.constant([0.5, 2.0, 2.0, 1.0, 1.5, 3.0, 1.0, 0.1])
    
    # Ensure y_pred has softmax applied
    if y_pred.shape[-1] > 1:
        y_pred = tf.nn.softmax(y_pred, axis=-1)
    
    # Resize y_true to match y_pred spatial dimensions if needed
    pred_shape = tf.shape(y_pred)
    true_shape = tf.shape(y_true)
    
    if true_shape[1] != pred_shape[1] or true_shape[2] != pred_shape[2]:
        y_true = tf.image.resize(
            tf.expand_dims(tf.cast(y_true, tf.float32), axis=-1),
            [pred_shape[1], pred_shape[2]],
            method='nearest'
        )
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
    
    # Convert ground truth to one-hot
    num_classes = tf.shape(y_pred)[-1]
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
    
    # Calculate weighted cross entropy
    ce_loss = -y_true_one_hot * tf.math.log(tf.clip_by_value(y_pred, 1e-8, 1.0))
    weighted_ce = ce_loss * class_weights
    
    return tf.reduce_mean(tf.reduce_sum(weighted_ce, axis=-1))


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

def conv_block(x, filters, kernel_size=3, activation='relu', batch_norm=True):
    """Basic convolutional block with optional batch normalization"""
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def encoder_block(x, filters, pool_size=2):
    """Encoder block for U-Net: conv + conv + maxpool"""
    skip = conv_block(x, filters)
    skip = conv_block(skip, filters)
    x = layers.MaxPooling2D(pool_size)(skip)
    return x, skip


def decoder_block(x, skip, filters):
    """Decoder block for U-Net: upsample + concat + conv + conv"""
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    x = conv_block(x, filters)
    return x


def unet_mini(input_shape=(224, 224, 3), num_classes=8, filters_base=32):
    """
    Mini U-Net model (non-pretrained baseline)
    
    Args:
        input_shape: Input image shape
        num_classes: Number of segmentation classes
        filters_base: Base number of filters
    
    Returns:
        Keras Model
    """
    inputs = layers.Input(input_shape)
    
    # Encoder
    x1, skip1 = encoder_block(inputs, filters_base)      # 112x112
    x2, skip2 = encoder_block(x1, filters_base * 2)      # 56x56
    x3, skip3 = encoder_block(x2, filters_base * 4)      # 28x28
    x4, skip4 = encoder_block(x3, filters_base * 8)      # 14x14
    
    # Bottleneck
    bottleneck = conv_block(x4, filters_base * 16)       # 7x7
    bottleneck = conv_block(bottleneck, filters_base * 16)
    
    # Decoder
    d1 = decoder_block(bottleneck, skip4, filters_base * 8)  # 14x14
    d2 = decoder_block(d1, skip3, filters_base * 4)          # 28x28
    d3 = decoder_block(d2, skip2, filters_base * 2)          # 56x56
    d4 = decoder_block(d3, skip1, filters_base)              # 112x112
    
    # Final upsampling and classification
    d4 = layers.Conv2DTranspose(filters_base, 2, strides=2, padding='same')(d4)  # 224x224
    outputs = layers.Conv2D(num_classes, 1, activation='softmax', name='segmentation_output')(d4)
    
    model = Model(inputs, outputs, name='UNet_Mini')
    return model


def vgg16_unet(input_shape=(224, 224, 3), num_classes=8, freeze_encoder=False):
    """
    U-Net with VGG16 pretrained backbone
    
    Args:
        input_shape: Input image shape
        num_classes: Number of segmentation classes
        freeze_encoder: Whether to freeze the encoder weights
    
    Returns:
        Keras Model
    """
    # Load pretrained VGG16 as encoder
    vgg16 = tf.keras.applications.VGG16(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False
    )
    
    if freeze_encoder:
        vgg16.trainable = False
    
    # Extract skip connections from VGG16
    # Note: VGG16 applies max pooling, so sizes are:
    skip1 = vgg16.get_layer('block1_conv2').output    # 112x112, 64 (after pool)
    skip2 = vgg16.get_layer('block2_conv2').output    # 56x56, 128 (after pool)
    skip3 = vgg16.get_layer('block3_conv3').output    # 28x28, 256 (after pool)
    skip4 = vgg16.get_layer('block4_conv3').output    # 14x14, 512 (after pool)
    
    # Bottleneck (center of U-Net)
    bottleneck = vgg16.get_layer('block5_conv3').output  # 7x7, 512 (after pool)
    x = conv_block(bottleneck, 1024)
    x = conv_block(x, 1024)
    
    # Decoder with skip connections
    # Upsample and concatenate with skip4 (7x7 -> 14x14)
    x = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(x)  # 14x14
    x = layers.Concatenate()([x, skip4])
    x = conv_block(x, 512)
    x = conv_block(x, 256)
    
    # Upsample and concatenate with skip3 (14x14 -> 28x28)
    x = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(x)  # 28x28
    x = layers.Concatenate()([x, skip3])
    x = conv_block(x, 256)
    x = conv_block(x, 128)
    
    # Upsample and concatenate with skip2 (28x28 -> 56x56)
    x = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(x)  # 56x56
    x = layers.Concatenate()([x, skip2])
    x = conv_block(x, 128)
    x = conv_block(x, 64)
    
    # Upsample and concatenate with skip1 (56x56 -> 112x112)
    x = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x)   # 112x112
    x = layers.Concatenate()([x, skip1])
    x = conv_block(x, 64)
    x = conv_block(x, 32)
    
    # Final upsampling to original input size (112x112 -> 224x224)
    x = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(x)   # 224x224
    x = conv_block(x, 32)
    
    # Final classification layer
    outputs = layers.Conv2D(num_classes, 1, activation='softmax', name='segmentation_output')(x)
    
    model = Model(vgg16.input, outputs, name='VGG16_UNet')
    return model


def resnet50_unet(input_shape=(224, 224, 3), num_classes=8, freeze_encoder=False):
    """
    U-Net with ResNet50 pretrained backbone
    
    Args:
        input_shape: Input image shape
        num_classes: Number of segmentation classes
        freeze_encoder: Whether to freeze the encoder weights
    
    Returns:
        Keras Model
    """
    # Load pretrained ResNet50 as encoder
    resnet50 = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False
    )
    
    if freeze_encoder:
        resnet50.trainable = False
    
    # Extract skip connections from ResNet50
    skip1 = resnet50.get_layer('conv1_relu').output           # 112x112, 64
    skip2 = resnet50.get_layer('conv2_block3_out').output     # 56x56, 256
    skip3 = resnet50.get_layer('conv3_block4_out').output     # 28x28, 512
    skip4 = resnet50.get_layer('conv4_block6_out').output     # 14x14, 1024
    
    # Bottleneck
    bottleneck = resnet50.get_layer('conv5_block3_out').output  # 7x7, 2048
    x = conv_block(bottleneck, 2048)
    
    # Decoder with skip connections
    # Upsample and concatenate with skip4
    x = layers.Conv2DTranspose(1024, 2, strides=2, padding='same')(x)  # 14x14
    x = layers.Concatenate()([x, skip4])
    x = conv_block(x, 1024)
    x = conv_block(x, 512)
    
    # Upsample and concatenate with skip3
    x = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(x)   # 28x28
    x = layers.Concatenate()([x, skip3])
    x = conv_block(x, 512)
    x = conv_block(x, 256)
    
    # Upsample and concatenate with skip2
    x = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(x)   # 56x56
    x = layers.Concatenate()([x, skip2])
    x = conv_block(x, 256)
    x = conv_block(x, 128)
    
    # Upsample and concatenate with skip1
    x = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(x)   # 112x112
    x = layers.Concatenate()([x, skip1])
    x = conv_block(x, 128)
    x = conv_block(x, 64)
    
    # Final upsampling to original size
    x = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x)    # 224x224
    x = conv_block(x, 64)
    
    # Final classification layer
    outputs = layers.Conv2D(num_classes, 1, activation='softmax', name='segmentation_output')(x)
    
    model = Model(resnet50.input, outputs, name='ResNet50_UNet')
    return model


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model(model_type='unet_mini', input_shape=(224, 224, 3), num_classes=8, **kwargs):
    """
    Factory function to create different model architectures
    
    Args:
        model_type: Type of model ('unet_mini', 'vgg16_unet', 'resnet50_unet')
        input_shape: Input image shape
        num_classes: Number of segmentation classes
        **kwargs: Additional model-specific arguments
    
    Returns:
        Keras Model
    """
    if model_type == 'unet_mini':
        return unet_mini(input_shape, num_classes, **kwargs)
    elif model_type == 'vgg16_unet':
        return vgg16_unet(input_shape, num_classes, **kwargs)
    elif model_type == 'resnet50_unet':
        return resnet50_unet(input_shape, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_loss_function(loss_type='cross_entropy'):
    """
    Get loss function by name
    
    Args:
        loss_type: Type of loss function
    
    Returns:
        Loss function
    """
    if loss_type == 'cross_entropy':
        return 'sparse_categorical_crossentropy'
    elif loss_type == 'dice_loss':
        return dice_loss
    elif loss_type == 'focal_loss':
        return focal_loss
    elif loss_type == 'combined_loss':
        return combined_loss
    elif loss_type == 'balanced_cross_entropy':
        return balanced_cross_entropy
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compile_model(model, loss_type='cross_entropy', learning_rate=1e-4, metrics=None):
    """
    Compile model with specified loss and metrics
    
    Args:
        model: Keras model to compile
        loss_type: Type of loss function
        learning_rate: Learning rate for optimizer
        metrics: List of metrics to track
    
    Returns:
        Compiled model
    """
    if metrics is None:
        metrics = [MeanIoUArgmax(num_classes=8), 'accuracy']
    
    loss_fn = get_loss_function(loss_type)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=metrics
    )
    
    return model