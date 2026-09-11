# uvicorn app.main:app --reload
# Tìm biến app bên trong file app/main.py
# Tự động reload lại server mỗi khi sửa code và lưu lại.

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.ai.pipeline.tire_pipeline import TireInspectionPipeline

from app.api.v1.inspection import (
    router as inspection_router,
)

from app.api.v1.health import (
    router as health_router,
)

from app.core.config import (
    APP_NAME,
    DETECTION_MODEL_PATH,
    OCR_DICTIONARY_PATH,
    OCR_MODEL_DIR,
    OUTPUT_DIR,
    SEGMENTATION_MODEL_PATH,
    USE_GPU,
    INSPECTION_IMAGE_DIR,
)

from app.core.exceptions import APIError

from app.core.exception_handlers import (
    api_error_handler,
    generic_error_handler,
    validation_error_handler,
)

from app.storage import LocalImageStorage

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Application lifecycle.

    Load AI models once when FastAPI starts.
    """

    print("=" * 60)
    print("Initializing DRC AI Tire Inspection...")
    print("=" * 60)
    
    pipeline = TireInspectionPipeline(
        segmentation_model_path=SEGMENTATION_MODEL_PATH,
        detection_model_path=DETECTION_MODEL_PATH,
        ocr_model_dir=OCR_MODEL_DIR,
        ocr_dictionary_path=OCR_DICTIONARY_PATH,
        output_dir=OUTPUT_DIR,
        use_gpu=USE_GPU,
    )

    app.state.inspection_pipeline = pipeline

    image_storage = LocalImageStorage(
        root_dir=INSPECTION_IMAGE_DIR
    )
    
    app.state.image_storage = image_storage

    print("=" * 60)
    print("AI Pipeline ready.")
    print("=" * 60)

    yield

    print("=" * 60)
    print("Shutting down DRC AI...")
    print("=" * 60)


app = FastAPI(
    title=APP_NAME,
    description=(
        "AI-powered tire inspection service "
        "using YOLO11 and PaddleOCR."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.add_exception_handler(
    APIError,
    api_error_handler,
)

app.add_exception_handler(
    RequestValidationError,
    validation_error_handler,
)

app.add_exception_handler(
    Exception,
    generic_error_handler,
)


app.include_router(
    inspection_router,
    prefix="/api/v1",
)

app.include_router(
    health_router,
    prefix="/api/v1",
)