import logging
from typing import Dict

from fastapi import APIRouter
from fastapi import UploadFile, HTTPException, File

import tensorflow as tf

from starlette.responses import Response

from utils import preprocess_image, postprocess_mask, Model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health-check/", summary="Healthcheck")
def root() -> Dict[str, str]:
    return {"status": "ok"}

@router.post("/segment/")
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
    logger.info(f"Predicted: {mask_img}")
    png_bytes = postprocess_mask(mask_img)
    logger.info(f"PostProcessed {mask_img}")
    return Response(content=png_bytes, media_type="image/png")