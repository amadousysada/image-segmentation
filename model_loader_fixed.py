import os
import tempfile
import logging
import mlflow
import keras
from pathlib import Path

logger = logging.getLogger(__name__)

async def load_model():
    client = mlflow.MlflowClient()
    model_uri = f"runs:/{conf.RUN_ID}/model-artifact"
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download artifacts
            model_path = client.download_artifacts(
                conf.RUN_ID,
                "model-artifact",
                dst_path=temp_dir
            )
            
            logger.info(f"Downloaded artifacts to: {model_path}")
            
            # Debug: List all files in the downloaded directory
            logger.info("Contents of downloaded directory:")
            for root, dirs, files in os.walk(model_path):
                level = root.replace(model_path, '').count(os.sep)
                indent = ' ' * 2 * level
                logger.info(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    logger.info(f"{subindent}{file}")
            
            # Try multiple possible paths for the Keras model
            possible_paths = [
                os.path.join(model_path, "data", "model.keras"),
                os.path.join(model_path, "model.keras"),
                os.path.join(model_path, "model", "model.keras"),
                os.path.join(model_path, "artifacts", "model.keras"),
                os.path.join(model_path, "artifacts", "data", "model.keras"),
            ]
            
            keras_model_path = None
            
            # Check each possible path
            for path in possible_paths:
                logger.info(f"Checking path: {path}")
                if os.path.exists(path) and os.path.isfile(path):
                    keras_model_path = path
                    logger.info(f"Found model at: {keras_model_path}")
                    break
            
            # If not found in expected locations, search recursively
            if keras_model_path is None:
                logger.info("Model not found in expected locations, searching recursively...")
                for root, dirs, files in os.walk(model_path):
                    for file in files:
                        if file.endswith('.keras'):
                            potential_path = os.path.join(root, file)
                            logger.info(f"Found .keras file: {potential_path}")
                            # Verify it's actually a file and not a directory
                            if os.path.isfile(potential_path):
                                keras_model_path = potential_path
                                logger.info(f"Using Keras model: {keras_model_path}")
                                break
                    if keras_model_path:
                        break
            
            if keras_model_path is None:
                raise FileNotFoundError(f"No .keras model file found in {model_path}")
            
            # Verify the file exists and is readable
            if not os.path.exists(keras_model_path):
                raise FileNotFoundError(f"Keras model file does not exist: {keras_model_path}")
            
            if not os.path.isfile(keras_model_path):
                raise ValueError(f"Path is not a file: {keras_model_path}")
            
            # Check file size to ensure it's not empty
            file_size = os.path.getsize(keras_model_path)
            logger.info(f"Model file size: {file_size} bytes")
            
            if file_size == 0:
                raise ValueError(f"Model file is empty: {keras_model_path}")
            
            logger.info(f"Loading model from: {keras_model_path}")
            
            # Load the model with Keras 3.x
            model = keras.saving.load_model(keras_model_path, compile=False)
            logger.info("Model loaded successfully!")
            model.summary()
            
            # Set the model (assuming Model() is defined elsewhere)
            Model().set_model(model)
            logger.info("Loaded model from %s", model_uri)
            
    except Exception as exc:
        logger.error("Failed loading model %s: %s", model_uri, exc)
        # Additional debugging information
        logger.error(f"Exception type: {type(exc).__name__}")
        if hasattr(exc, '__traceback__'):
            import traceback
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
        raise