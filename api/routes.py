import logging
from typing import Dict, Any, Coroutine

import mlflow
from fastapi import APIRouter
from fastapi import UploadFile, HTTPException, File
from fastapi.responses import Response

from pydantic import BaseModel
import numpy as np

import tensorflow as tf

from starlette.responses import StreamingResponse, Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()

IMG_SIZE = (224, 224)

class Model():
    instance = None

    def __new__(cls):
        if not cls.instance:
            cls.instance = super(Model, cls).__new__(cls)
        return cls.instance

    def __init__(self, model: mlflow.pyfunc.PyFuncModel = None):
        self.model: mlflow.pyfunc.PyFuncModel = model

    def get_model(self) -> mlflow.pyfunc.PyFuncModel:
        return self.model

    def set_model(self, model: mlflow.pyfunc.PyFuncModel):
        self.model = model

class PredictRequest(BaseModel):
    picture: UploadFile

def preprocess_image(file_bytes: bytes):
    print(file_bytes)
    print(type(file_bytes))
    img = tf.image.decode_png(tf.io.read_file(file_bytes), channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0

    return img

def postprocess_mask(mask_logits: np.ndarray):
    """
    Transforme les logits en mask uint8 et encode en PNG.
    mask_logits: (1, H, W, C) float32 probabilities
    Retourne un Tensor de bytes contenant l’image PNG.
    """
    mask = tf.argmax(mask_logits, axis=-1, output_type=tf.uint8)  # (1, H, W)
    mask = tf.squeeze(mask, axis=0)  # (H, W)
    mask = tf.expand_dims(mask, axis=-1)  # (H, W, 1)

    png = tf.io.encode_png(mask)  # dtype=uint8, shape=() scalar string tensor
    return png

@router.get("/health-check/", summary="Healthcheck")
def root() -> Dict[str, str]:
    return {"status": "ok"}

@router.post("/predict/")
async def predict(
    *,
    picture: UploadFile = File(...),
) -> Response:
    """Return image segment (mask)."""
    if picture.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Image must be JPEG or PNG")

    model = Model().get_model()

    data = await picture.read()
    x = preprocess_image(data)
    mask_img = model.predict(x)
    png_bytes = postprocess_mask(mask_img).numpy()
    logger.info(f"Predicted: {mask_img}")

    return Response(content=png_bytes, media_type="image/png")