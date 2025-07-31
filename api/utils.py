import mlflow
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)

class MeanIoUArgmax(tf.keras.metrics.MeanIoU):
    """Custom MeanIoU metric that applies argmax to predictions"""
    def __init__(self, num_classes, name="mean_io_u_argmax", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_pred : (batch, H, W, num_classes) → take the winning class
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


class Model:
    instance = None
    initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super(Model, cls).__new__(cls)
        return cls.instance

    def __init__(self, model=None):
        if not self.initialized:
            self.initialized = True
            self.model: mlflow.pyfunc.PyFuncModel = model

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

def preprocess_image(file_bytes: bytes):
    img = tf.image.decode_png(file_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)  # shape → (1, 224, 224, 3)

    return img

def postprocess_mask(mask_logits: np.ndarray):
    """
    mask_logits: np.ndarray shape (1, H, W, C)
    Retourne les octets d'un PNG du mask (H, W, 1) en uint8.
    """
    # argmax → (1, H, W)
    mask_indices = np.argmax(mask_logits, axis=-1)
    # squeeze batch → (H, W)
    mask_indices = mask_indices[0]
    
    # Palette de couleurs optimisée pour 8 classes (0-7)
    # Chaque classe aura une valeur bien distincte en niveaux de gris
    color_map = np.array([
        0,    # Classe 0: Noir (background généralement)
        36,   # Classe 1: Gris très foncé
        73,   # Classe 2: Gris foncé
        109,  # Classe 3: Gris moyen-foncé
        146,  # Classe 4: Gris moyen
        182,  # Classe 5: Gris moyen-clair
        219,  # Classe 6: Gris clair
        255   # Classe 7: Blanc
    ])
    
    # Appliquer la palette de couleurs
    if mask_indices.max() < len(color_map):
        mask_normalized = color_map[mask_indices]
    else:
        # Fallback: normalisation linéaire si plus de 8 classes détectées
        mask_normalized = (mask_indices / mask_indices.max() * 255).astype(np.uint8)
    
    # Convertir en uint8 si ce n'est pas déjà fait
    mask_normalized = mask_normalized.astype(np.uint8)
    
    # ajout du canal → (H, W, 1)
    mask_normalized = mask_normalized[..., np.newaxis]
    # encode en PNG
    encoded = tf.io.encode_png(mask_normalized).numpy()
    return encoded

def postprocess_mask_color(mask_logits: np.ndarray):
    """
    Alternative en couleurs pour une meilleure visualisation des 8 classes.
    mask_logits: np.ndarray shape (1, H, W, C)
    Retourne les octets d'un PNG du mask (H, W, 3) en couleurs RGB.
    """
    # argmax → (1, H, W)
    mask_indices = np.argmax(mask_logits, axis=-1)
    # squeeze batch → (H, W)
    mask_indices = mask_indices[0]
    
    # Palette de couleurs RGB distinctes pour 8 classes
    color_palette = np.array([
        [0, 0, 0],       # Classe 0: Noir (background)
        [255, 0, 0],     # Classe 1: Rouge
        [0, 255, 0],     # Classe 2: Vert
        [0, 0, 255],     # Classe 3: Bleu
        [255, 255, 0],   # Classe 4: Jaune
        [255, 0, 255],   # Classe 5: Magenta
        [0, 255, 255],   # Classe 6: Cyan
        [255, 255, 255]  # Classe 7: Blanc
    ], dtype=np.uint8)
    
    # Créer l'image couleur
    h, w = mask_indices.shape
    mask_color = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(min(len(color_palette), mask_indices.max() + 1)):
        mask_color[mask_indices == class_id] = color_palette[class_id]
    
    # encode en PNG
    encoded = tf.io.encode_png(mask_color).numpy()
    return encoded
