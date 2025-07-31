from __future__ import annotations

import asyncio
import logging
import time

import mlflow
import os, tempfile, zipfile

from contextlib import asynccontextmanager
from fastapi import FastAPI
from routes import router, Model
from settings import get_settings
import tensorflow as tf
import keras

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conf = get_settings()

MLFLOW_TRACKING_URI = conf.MLFLOW_TRACKING_URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

async def load_model():
    client = mlflow.MlflowClient()
    model_uri = f"runs:/{conf.RUN_ID}/model-artifact"
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            #local_dir = mlflow.artifacts.download_artifacts(run_id=conf.RUN_ID, artifact_path="model-artifact")
            model_path = client.download_artifacts(
                conf.RUN_ID,
                "model-artifact",
                dst_path=temp_dir
            )
            keras_model_path = os.path.join(model_path, "data", "model.keras")
            if not os.path.exists(keras_model_path):
                logger.info("Looking else where")
                # Essayer d'autres chemins possibles
                keras_model_path = os.path.join(model_path, "model.keras")
                if not os.path.exists(keras_model_path):
                    # Chercher le fichier .keras dans le répertoire
                    for root, dirs, files in os.walk(model_path):
                        for file in files:
                            if file.endswith('.keras'):
                                keras_model_path = os.path.join(root, file)
                                break
            if os.path.isdir(keras_model_path):
                logger.info("yes")
                logger.info(os.path.join(keras_model_path, "data", "model.keras"))
            logger.info(f"Chargement du modèle depuis: {keras_model_path}")

            # Charger le modèle avec Keras 3.x
            model = keras.saving.load_model(keras_model_path, compile=False)
            #model = tf.keras.models.load_model(f"{local_dir}/data/model.keras", compile=False)
            logger.info("Modèle chargé, summary :")
            model.summary()
            #pyfunc_model = mlflow.pyfunc.load_model(model_uri)
            Model().set_model(model)
            logger.info("Loaded model from %s", model_uri)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed loading model %s: %s", model_uri, exc)
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await load_model()
    except Exception as exc:
        logger.error("Failed loading model during startup: %s", exc)
        raise
    yield
    logger.info("Application shutdown complete.")

app = FastAPI(title="Segmentation API", version="1.0.0", lifespan=lifespan)

# Include all routes
app.include_router(router=router)