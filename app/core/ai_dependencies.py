from app.ai.pipeline.tire_pipeline import TireInspectionPipeline
from app.core.config import (
    DETECTION_MODEL_PATH,
    INSPECTION_IMAGE_DIR,
    OCR_DICTIONARY_PATH,
    OCR_MODEL_DIR,
    OUTPUT_DIR,
    SEGMENTATION_MODEL_PATH,
    USE_GPU,
)
from app.storage import LocalImageStorage


def create_inspection_pipeline() -> TireInspectionPipeline:
    return TireInspectionPipeline(
        segmentation_model_path=SEGMENTATION_MODEL_PATH,
        detection_model_path=DETECTION_MODEL_PATH,
        ocr_model_dir=OCR_MODEL_DIR,
        ocr_dictionary_path=OCR_DICTIONARY_PATH,
        output_dir=OUTPUT_DIR,
        use_gpu=USE_GPU,
    )


def create_image_storage() -> LocalImageStorage:
    return LocalImageStorage(
        root_dir=INSPECTION_IMAGE_DIR,
    )