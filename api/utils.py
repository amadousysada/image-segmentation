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
    Retourne les octets d’un PNG du mask (H, W, 1) en uint8.
    """
    # argmax → (1, H, W)
    mask_indices = np.argmax(mask_logits, axis=-1)
    # squeeze batch → (H, W)
    mask_indices = mask_indices[0]
    # cast en uint8 → (H, W)
    mask_uint8 = mask_indices.astype(np.uint8)
    # ajout du canal → (H, W, 1)
    mask_uint8 = mask_uint8[..., np.newaxis]
    # encode en PNG
    encoded = tf.io.encode_png(mask_uint8).numpy()
    return encoded
