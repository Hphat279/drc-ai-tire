import numpy as np


EDGE_RATIO = 0.05


def box_area(detection: dict) -> int:
    """Tính diện tích bounding box."""

    x1, y1, x2, y2 = detection["bbox"]

    return max(0, x2 - x1) * max(0, y2 - y1)


def is_left_edge(
    detection: dict,
    image_width: int,
) -> bool:
    """Kiểm tra detection có nằm sát mép trái."""

    x1, _, _, _ = detection["bbox"]
    edge_margin = int(image_width * EDGE_RATIO)

    return x1 <= edge_margin


def is_right_edge(
    detection: dict,
    image_width: int,
) -> bool:
    """Kiểm tra detection có nằm sát mép phải."""

    _, _, x2, _ = detection["bbox"]
    edge_margin = int(image_width * EDGE_RATIO)

    return x2 >= image_width - edge_margin


def reconstruct_wrapped_text(
    image: np.ndarray,
    detections: list[dict],
    pad_y_ratio: float,
) -> np.ndarray | None:
    """
    Ghép text bị cắt qua seam.

    Thứ tự ghép:
        RIGHT + LEFT
    """

    if len(detections) < 2:
        return None

    image_width = image.shape[1]

    left_detections = [
        detection
        for detection in detections
        if is_left_edge(
            detection,
            image_width,
        )
    ]

    right_detections = [
        detection
        for detection in detections
        if is_right_edge(
            detection,
            image_width,
        )
    ]

    if not left_detections or not right_detections:
        return None

    left_detection = max(
        left_detections,
        key=box_area,
    )

    right_detection = max(
        right_detections,
        key=box_area,
    )

    left_bbox = left_detection["bbox"]
    right_bbox = right_detection["bbox"]

    y1 = min(
        left_bbox[1],
        right_bbox[1],
    )

    y2 = max(
        left_bbox[3],
        right_bbox[3],
    )

    pad_y = int(
        (y2 - y1) * pad_y_ratio
    )

    y1 = max(0, y1 - pad_y)
    y2 = min(image.shape[0], y2 + pad_y)

    right_crop = image[
        y1:y2,
        max(0, right_bbox[0]):min(
            image_width,
            right_bbox[2],
        ),
    ]

    left_crop = image[
        y1:y2,
        max(0, left_bbox[0]):min(
            image_width,
            left_bbox[2],
        ),
    ]

    if (
        right_crop.size == 0
        or left_crop.size == 0
    ):
        return None

    return np.concatenate(
        [
            right_crop,
            left_crop,
        ],
        axis=1,
    )