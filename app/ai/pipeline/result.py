from pathlib import Path


REQUIRED_CLASSES = [
    "brand",
    "size",
    "pattern",
]


def finalize_ocr_results(
    extraction_results: dict,
    ocr_results: dict,
) -> dict:
    """
    Combine extraction metadata with OCR result.

    Extraction is the source of truth for:
        - direct
        - reconstructed
        - not_detected

    OCR contributes:
        - text
        - confidence
        - ocr_failed
    """

    final_results = {}

    for class_name in REQUIRED_CLASSES:
        extraction_item = (
            extraction_results[class_name]
        )

        extraction_status = (
            extraction_item["status"]
        )

        ocr_item = ocr_results.get(
            class_name,
            {},
        )

        text = ocr_item.get("text")
        confidence = float(
            ocr_item.get(
                "confidence",
                0.0,
            )
        )

        # ----------------------------------------------
        # No detection / no extraction
        # ----------------------------------------------

        if extraction_status == "not_detected":
            final_results[class_name] = {
                "text": None,
                "confidence": 0.0,
                "status": "not_detected",
            }

            continue

        # ----------------------------------------------
        # Detection exists but OCR failed
        # ----------------------------------------------

        if (
            text is None
            or not str(text).strip()
        ):
            final_results[class_name] = {
                "text": None,
                "confidence": confidence,
                "status": "ocr_failed",
            }

            continue

        # ----------------------------------------------
        # OCR succeeded
        #
        # Keep extraction status:
        #   direct
        #   reconstructed
        # ----------------------------------------------

        final_results[class_name] = {
            "text": text,
            "confidence": confidence,
            "status": extraction_status,
        }

    return final_results


def get_overall_status(
    ocr_results: dict,
) -> str:
    """
    Determine overall inspection status.

    success:
        All required fields were successfully extracted
        and recognized.

    partial:
        At least one field is:
            - not_detected
            - ocr_failed
    """

    statuses = [
        ocr_results[class_name]["status"]
        for class_name in REQUIRED_CLASSES
    ]

    if all(
        status in {
            "direct",
            "reconstructed",
        }
        for status in statuses
    ):
        return "success"

    return "partial"


def build_extraction_result(
    extraction_results: dict,
) -> dict:
    """
    Convert internal extraction objects into
    JSON-safe metadata.

    Crop images are intentionally excluded.
    """

    result = {}

    for class_name in REQUIRED_CLASSES:
        item = extraction_results[
            class_name
        ]

        result[class_name] = {
            "status": item["status"],
            "source_detections": (
                item["source_detections"]
            ),
        }

    return result


def build_pipeline_result(
    image_path: Path,
    detections: list[dict],
    extraction_results: dict,
    ocr_results: dict,
    processing_time_ms: float,
) -> dict:
    """
    Build the final standardized pipeline result.
    """

    final_ocr_results = finalize_ocr_results(
        extraction_results=extraction_results,
        ocr_results=ocr_results,
    )

    return {
        "image": str(image_path),

        "status": get_overall_status(
            final_ocr_results
        ),

        "segmentation": {
            "status": "success",
        },

        "detections": {
            "status": (
                "success"
                if detections
                else "not_detected"
            ),
            "items": detections,
        },

        "extraction": build_extraction_result(
            extraction_results
        ),

        "ocr": final_ocr_results,

        "processing_time_ms": round(
            processing_time_ms,
            2,
        ),
    }