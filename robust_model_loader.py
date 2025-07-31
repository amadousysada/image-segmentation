import os
import tempfile
import logging
import time
import mlflow
import keras
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

async def load_model_robust():
    """
    Robust model loader that handles intermittent MLflow issues
    """
    client = mlflow.MlflowClient()
    model_uri = f"runs:/{conf.RUN_ID}/model-artifact"
    
    # Try multiple strategies with retries
    strategies = [
        ("mlflow_keras_direct", load_with_mlflow_keras),
        ("mlflow_pyfunc", load_with_mlflow_pyfunc),
        ("artifact_download_robust", load_with_artifact_download_robust),
        ("artifact_download_simple", load_with_artifact_download_simple),
    ]
    
    last_exception = None
    
    for strategy_name, strategy_func in strategies:
        logger.info(f"Trying strategy: {strategy_name}")
        
        for attempt in range(3):  # 3 retries per strategy
            try:
                logger.info(f"Strategy {strategy_name}, attempt {attempt + 1}")
                model = await strategy_func(client, model_uri)
                
                if model is not None:
                    logger.info(f"Successfully loaded model using strategy: {strategy_name}")
                    Model().set_model(model)
                    return model
                    
            except Exception as exc:
                logger.warning(f"Strategy {strategy_name} attempt {attempt + 1} failed: {exc}")
                last_exception = exc
                
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(2 ** attempt)  # Exponential backoff
    
    # All strategies failed
    logger.error("All loading strategies failed")
    raise last_exception or Exception("All model loading strategies failed")


async def load_with_mlflow_keras(client, model_uri):
    """Strategy 1: Direct MLflow Keras loading"""
    try:
        model = mlflow.keras.load_model(model_uri)
        logger.info("Model loaded with mlflow.keras.load_model")
        return model
    except Exception as e:
        logger.debug(f"mlflow.keras.load_model failed: {e}")
        raise


async def load_with_mlflow_pyfunc(client, model_uri):
    """Strategy 2: MLflow PyFunc loading"""
    try:
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)
        
        # Try to extract underlying Keras model
        if hasattr(pyfunc_model, '_model_impl'):
            if hasattr(pyfunc_model._model_impl, 'keras_model'):
                model = pyfunc_model._model_impl.keras_model
                logger.info("Extracted Keras model from PyFunc wrapper")
                return model
            elif hasattr(pyfunc_model._model_impl, '_model'):
                model = pyfunc_model._model_impl._model
                logger.info("Extracted model from PyFunc _model attribute")
                return model
        
        # Use PyFunc model directly if no Keras model found
        logger.info("Using PyFunc model directly")
        return pyfunc_model
        
    except Exception as e:
        logger.debug(f"mlflow.pyfunc.load_model failed: {e}")
        raise


async def load_with_artifact_download_robust(client, model_uri):
    """Strategy 3: Robust artifact download with comprehensive search"""
    run_id = conf.RUN_ID
    
    # Try different artifact paths
    artifact_paths = ["model-artifact", "model", "artifacts"]
    
    for artifact_path in artifact_paths:
        try:
            logger.info(f"Trying artifact path: {artifact_path}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download artifacts
                try:
                    model_path = client.download_artifacts(run_id, artifact_path, dst_path=temp_dir)
                except Exception as download_exc:
                    logger.debug(f"Failed to download {artifact_path}: {download_exc}")
                    continue
                
                logger.info(f"Downloaded artifacts to: {model_path}")
                
                # Debug: List all contents
                logger.info("=== Downloaded contents ===")
                all_files = []
                for root, dirs, files in os.walk(model_path):
                    level = root.replace(model_path, '').count(os.sep)
                    indent = '  ' * level
                    rel_path = os.path.relpath(root, model_path) if root != model_path else '.'
                    logger.info(f"{indent}{rel_path}/")
                    subindent = '  ' * (level + 1)
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        all_files.append(file_path)
                        logger.info(f"{subindent}{file} ({file_size} bytes)")
                
                # Search for model files comprehensively
                model_file = find_model_file(model_path, all_files)
                
                if model_file:
                    logger.info(f"Found model file: {model_file}")
                    model = load_keras_model_safely(model_file)
                    if model:
                        return model
                else:
                    logger.warning(f"No model file found in {artifact_path}")
                    
        except Exception as exc:
            logger.debug(f"Artifact path {artifact_path} failed: {exc}")
            continue
    
    raise Exception("No valid model found in any artifact path")


async def load_with_artifact_download_simple(client, model_uri):
    """Strategy 4: Simple artifact download (fallback)"""
    run_id = conf.RUN_ID
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Try to download the entire run artifacts
        try:
            model_path = client.download_artifacts(run_id, "", dst_path=temp_dir)
            logger.info(f"Downloaded all run artifacts to: {model_path}")
            
            # Search for any model file in the entire run
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file.endswith(('.keras', '.h5', '.pb')):
                        full_path = os.path.join(root, file)
                        logger.info(f"Found potential model: {full_path}")
                        
                        if file.endswith('.keras'):
                            model = load_keras_model_safely(full_path)
                            if model:
                                return model
            
            raise Exception("No model files found in run artifacts")
            
        except Exception as exc:
            logger.debug(f"Simple download failed: {exc}")
            raise


def find_model_file(base_path, all_files):
    """Find the best model file from available files"""
    
    # Priority order for different file types
    model_patterns = [
        ('.keras', ['model.keras', 'saved_model.keras']),
        ('.h5', ['model.h5', 'weights.h5']),
        ('.pb', ['saved_model.pb']),
    ]
    
    for extension, preferred_names in model_patterns:
        # First, look for preferred names
        for name in preferred_names:
            for file_path in all_files:
                if file_path.endswith(name):
                    return file_path
        
        # Then look for any file with the extension
        for file_path in all_files:
            if file_path.endswith(extension):
                return file_path
    
    return None


def load_keras_model_safely(model_path):
    """Safely load a Keras model with error handling"""
    try:
        # Verify file exists and has content
        if not os.path.exists(model_path):
            logger.error(f"Model file does not exist: {model_path}")
            return None
        
        file_size = os.path.getsize(model_path)
        if file_size == 0:
            logger.error(f"Model file is empty: {model_path}")
            return None
        
        logger.info(f"Loading Keras model from: {model_path} (size: {file_size} bytes)")
        
        # Try different loading methods
        loading_methods = [
            lambda: keras.saving.load_model(model_path, compile=False),
            lambda: keras.models.load_model(model_path, compile=False),
            lambda: keras.saving.load_model(model_path, compile=True),
        ]
        
        for i, load_method in enumerate(loading_methods):
            try:
                model = load_method()
                logger.info(f"Successfully loaded model using method {i + 1}")
                
                # Verify model is valid
                if hasattr(model, 'summary'):
                    model.summary()
                
                return model
                
            except Exception as load_exc:
                logger.debug(f"Loading method {i + 1} failed: {load_exc}")
                continue
        
        logger.error(f"All loading methods failed for: {model_path}")
        return None
        
    except Exception as exc:
        logger.error(f"Error in load_keras_model_safely: {exc}")
        return None


# Enhanced debugging function
def debug_mlflow_run():
    """Debug the MLflow run to understand artifact structure"""
    client = mlflow.MlflowClient()
    run_id = conf.RUN_ID
    
    try:
        # Get run info
        run = client.get_run(run_id)
        logger.info(f"Run status: {run.info.status}")
        logger.info(f"Run lifecycle_stage: {run.info.lifecycle_stage}")
        
        # List all artifacts at root level
        logger.info("=== Root level artifacts ===")
        try:
            artifacts = client.list_artifacts(run_id)
            for artifact in artifacts:
                logger.info(f"  {artifact.path} ({'dir' if artifact.is_dir else 'file'})")
                
                # If it's a directory, list its contents too
                if artifact.is_dir:
                    try:
                        sub_artifacts = client.list_artifacts(run_id, artifact.path)
                        for sub_artifact in sub_artifacts:
                            logger.info(f"    {sub_artifact.path} ({'dir' if sub_artifact.is_dir else 'file'})")
                    except Exception as sub_exc:
                        logger.debug(f"Couldn't list contents of {artifact.path}: {sub_exc}")
                        
        except Exception as list_exc:
            logger.error(f"Couldn't list artifacts: {list_exc}")
        
    except Exception as exc:
        logger.error(f"Debug failed: {exc}")


# Usage example
async def load_model():
    """Main entry point - replace your current load_model with this"""
    try:
        # Optional: Debug the run first
        # debug_mlflow_run()
        
        # Load model with robust strategy
        model = await load_model_robust()
        return model
        
    except Exception as exc:
        logger.error("All model loading attempts failed")
        # Run debug to help troubleshoot
        debug_mlflow_run()
        raise