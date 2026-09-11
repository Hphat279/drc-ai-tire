import cv2
import numpy as np


def sharpen_clahe(gray: np.ndarray) -> np.ndarray:
    """
    Làm sắc nét ảnh thang độ xám và tăng cường độ tương phản cục bộ
    sử dụng CLAHE.
    """

    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    sharp = cv2.filter2D(
        gray,
        -1,
        kernel,
    )

    clahe = cv2.createCLAHE(
        clipLimit=4.0,
        tileGridSize=(8, 4),
    )

    return clahe.apply(sharp)


def prepare_ocr_crop(
    crop_img: np.ndarray,
    target_height: int = 48,
    max_width: int = 320,
) -> np.ndarray:
    """
    Thay đổi kích thước vùng cắt OCR đã phát hiện theo 
    cấu hình được sử dụng trong quá trình huấn luyện/kiểm thử.
    """

    if crop_img is None:
        raise ValueError("OCR crop is None.")

    h, w = crop_img.shape[:2]

    if h == 0 or w == 0:
        raise ValueError("OCR crop has invalid dimensions.")

    scale = target_height / h
    new_width = int(w * scale)

    new_width = min(
        new_width,
        max_width,
    )

    return cv2.resize(
        crop_img,
        (new_width, target_height),
        interpolation=cv2.INTER_CUBIC,
    )