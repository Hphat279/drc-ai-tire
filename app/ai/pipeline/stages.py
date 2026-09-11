import cv2

from app.ai.vision.extraction import (
    unwrap_tire,
    extract_text_regions,
)

from app.ai.ocr.preprocessing import sharpen_clahe


def run_segmentation(
    segmenter,
    image_path,
):
    """
    Run YOLO11 segmentation.

    Returns:
        extracted_tire
    """

    extracted_tire, _ = segmenter.segment(
        image_path=image_path,
        confidence=0.15,
    )

    return extracted_tire


def run_unwrap(
    extracted_tire,
):
    """
    Unwrap segmented tire into a flat representation.
    """

    return unwrap_tire(
        extracted_tire
    )


def run_preprocessing(
    unwrapped_img,
):
    """
    Convert unwrapped image to grayscale
    and apply sharpening + CLAHE.
    """

    gray_unwrapped = cv2.cvtColor(
        unwrapped_img,
        cv2.COLOR_BGR2GRAY,
    )

    return sharpen_clahe(
        gray_unwrapped
    )


def run_detection(
    detector,
    processed_img,
):
    """
    Run YOLO11 text-region detection.

    Returns:
        detect_input
        detections
        detection_vis
    """

    detect_input = cv2.cvtColor(
        processed_img,
        cv2.COLOR_GRAY2BGR,
    )

    _, detections = detector.detect(
        image=detect_input,
        image_size=1280,
        confidence=0.25,
    )

    if detections:
        detection_vis = detector.visualize(
            image=detect_input,
            detections=detections,
        )
    else:
        detection_vis = detect_input.copy()

    return (
        detect_input,
        detections,
        detection_vis,
    )


def run_extraction(
    detect_input,
    detections,
):
    """
    Extract brand, size and pattern regions.

    Extraction owns the metadata:
        - direct
        - reconstructed
        - not_detected

    Returns:
        extraction_results
        crop_results
    """

    extraction_results = extract_text_regions(
        detect_input=detect_input,
        detections=detections,
    )

    crop_results = {
        class_name: item["crop"]
        for class_name, item
        in extraction_results.items()
    }

    return (
        extraction_results,
        crop_results,
    )


def run_ocr(
    ocr,
    crop_results,
    extraction_results,
):
    """
    Run PaddleOCR.

    Extraction status is NOT inferred here.
    Extraction remains the source of truth.
    """

    return ocr.read_all(
        crop_results
    )