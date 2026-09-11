from app.ai.vision.extraction.crop import crop_box
from app.ai.vision.extraction.seam import (
    reconstruct_wrapped_text,
)


TEXT_CLASSES = {
    "brand": 0.05,
    "size": 0.05,
    "pattern": 0.03,
}


def extract_class(
    image,
    detections: list[dict],
    pad_y_ratio: float,
) -> dict:
    """
    Extract một class và trả metadata ngay tại extraction layer.

    Status:
        direct
        reconstructed
        not_detected
    """

    if not detections:
        return {
            "crop": None,
            "status": "not_detected",
            "source_detections": 0,
        }

    reconstructed = reconstruct_wrapped_text(
        image=image,
        detections=detections,
        pad_y_ratio=pad_y_ratio,
    )

    if reconstructed is not None:
        return {
            "crop": reconstructed,
            "status": "reconstructed",
            "source_detections": len(detections),
        }

    if len(detections) == 1:
        crop = crop_box(
            image=image,
            bbox=detections[0]["bbox"],
            pad_y_ratio=pad_y_ratio,
        )

        if crop is None:
            return {
                "crop": None,
                "status": "not_detected",
                "source_detections": len(detections),
            }

        return {
            "crop": crop,
            "status": "direct",
            "source_detections": 1,
        }

    # Giữ behavior cũ:
    # detection cuối cùng được sử dụng.
    crop = crop_box(
        image=image,
        bbox=detections[-1]["bbox"],
        pad_y_ratio=pad_y_ratio,
    )

    if crop is None:
        return {
            "crop": None,
            "status": "not_detected",
            "source_detections": len(detections),
        }

    return {
        "crop": crop,
        "status": "direct",
        "source_detections": len(detections),
    }


def extract_text_regions(
    detect_input,
    detections: list[dict],
) -> dict:
    """Extract brand, size và pattern cùng metadata."""

    results = {}

    for class_name, pad_y_ratio in TEXT_CLASSES.items():
        class_detections = [
            detection
            for detection in detections
            if detection["class_name"] == class_name
        ]

        results[class_name] = extract_class(
            image=detect_input,
            detections=class_detections,
            pad_y_ratio=pad_y_ratio,
        )

    return results