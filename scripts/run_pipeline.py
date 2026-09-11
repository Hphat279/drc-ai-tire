import argparse
from pathlib import Path

from app.ai.pipeline.tire_pipeline import (
    TireInspectionPipeline,
)


# ==========================================================
# PROJECT ROOT
# ==========================================================

BASE_DIR = Path(
    __file__
).resolve().parents[1]


# ==========================================================
# MODEL PATHS
# ==========================================================

SEGMENTATION_MODEL = (
    BASE_DIR
    / "ml_models"
    / "segmentation"
    / "yolo_seg.pt"
)

DETECTION_MODEL = (
    BASE_DIR
    / "ml_models"
    / "detection"
    / "yolo_detect.pt"
)

OCR_MODEL_DIR = (
    BASE_DIR
    / "ml_models"
    / "ocr"
)

OCR_DICTIONARY = (
    OCR_MODEL_DIR
    / "tire_dict.txt"
)


# ==========================================================
# OUTPUT
# ==========================================================

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "output"
)


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "DRC AI Tire Inspection Pipeline"
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input tire image.",
    )

    args = parser.parse_args()

    image_path = Path(
        args.image
    )

    if not image_path.is_absolute():
        image_path = (
            BASE_DIR
            / image_path
        )

    # ======================================================
    # VALIDATE MODEL FILES
    # ======================================================

    required_files = [
        SEGMENTATION_MODEL,
        DETECTION_MODEL,
        OCR_MODEL_DIR,
        OCR_DICTIONARY,
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required model/file not found: "
                f"{file_path}"
            )

    # ======================================================
    # CREATE PIPELINE
    # ======================================================

    pipeline = TireInspectionPipeline(

        segmentation_model_path=(
            SEGMENTATION_MODEL
        ),

        detection_model_path=(
            DETECTION_MODEL
        ),

        ocr_model_dir=(
            OCR_MODEL_DIR
        ),

        ocr_dictionary_path=(
            OCR_DICTIONARY
        ),

        output_dir=(
            OUTPUT_DIR
        ),

        # Local machine:
        use_gpu=False,
    )

    # ======================================================
    # RUN
    # ======================================================

    result = pipeline.run(
        image_path
    )

    # ======================================================
    # PRINT FINAL RESULT
    # ======================================================

    print("\n")
    print("=" * 60)
    print("FINAL OCR RESULT")
    print("=" * 60)

    print(
        f"Pipeline status: {result['status']}"
    )

    print(
        f"Detection status: "
        f"{result['detections']['status']}"
    )

    for key in [
        "size",
        "pattern",
        "brand",
    ]:

        item = result["ocr"][key]

        text = item["text"]
        confidence = item["confidence"]
        status = item["status"]

        print(
            f"{key.capitalize():10s}: "
            f"{text} "
            f"({confidence:.4f}) "
            f"[{status}]"
        )

    print("=" * 60)

    print(
        "Processing time: "
        f"{result['processing_time_ms']} ms"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
