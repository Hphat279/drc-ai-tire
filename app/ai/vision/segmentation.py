from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class TireSegmenter:
    """Phân đoạn lốp xe dựa trên YOLO11-Seg."""

    def __init__(self, model_path: str | Path):
        self.model = YOLO(str(model_path))

    def predict(
        self,
        image_path: str | Path,
        confidence: float = 0.15,
    ):
        results = self.model.predict(
            source=str(image_path),
            conf=confidence,
            verbose=False,
        )

        if not results:
            raise RuntimeError("YOLO segmentation returned no result.")

        result = results[0]

        if result.masks is None:
            raise RuntimeError("No tire mask detected.")

        return result

    @staticmethod
    def get_first_mask(result) -> np.ndarray:
        """Trích xuất mặt nạ phân đoạn đầu tiên."""

        mask_binary = result.masks.data[0].cpu().numpy()

        orig_h, orig_w = result.orig_shape

        if mask_binary.shape[:2] != (orig_h, orig_w):
            mask_binary = cv2.resize(
                mask_binary,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

        return mask_binary > 0.5

    @staticmethod
    def apply_mask(
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Chỉ giữ lại vùng lốp xe đã được phân đoạn."""

        extracted_tire = np.zeros_like(image)
        extracted_tire[mask] = image[mask]

        return extracted_tire

    def segment(
        self,
        image_path: str | Path,
        confidence: float = 0.15,
    ) -> tuple[np.ndarray, object]:

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Cannot read image: {image_path}"
            )

        result = self.predict(
            image_path=image_path,
            confidence=confidence,
        )

        mask = self.get_first_mask(result)

        extracted_tire = self.apply_mask(
            image=image,
            mask=mask,
        )

        return extracted_tire, result