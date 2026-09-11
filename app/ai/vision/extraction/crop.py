import numpy as np


def crop_box(
    image: np.ndarray,
    bbox: list[int],
    pad_y_ratio: float,
) -> np.ndarray | None:
    """Crop một bounding box với padding theo chiều dọc."""

    x1, y1, x2, y2 = bbox

    pad_y = int(
        (y2 - y1) * pad_y_ratio
    )

    x1p = max(0, x1)
    y1p = max(0, y1 - pad_y)
    x2p = min(image.shape[1], x2)
    y2p = min(image.shape[0], y2 + pad_y)

    crop = image[
        y1p:y2p,
        x1p:x2p,
    ]

    if crop.size == 0:
        return None

    return crop