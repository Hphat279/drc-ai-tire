from pathlib import Path

import cv2
from ultralytics import YOLO


CLASS_NAMES = {
    0: "size",
    1: "pattern",
    2: "brand",
}

COLORS = {
    0: (255, 80, 80),
    1: (80, 80, 255),
    2: (80, 200, 80),
}


class TireDetector:
    """Tìm vị trí bằng YOLO11 dành cho các thông số kỹ thuật của lốp xe."""

    def __init__(self, model_path: str | Path):
        self.model = YOLO(str(model_path))

    def predict(
        self,
        image,
        image_size: int = 1280,
        confidence: float = 0.25,
    ):
        results = self.model.predict(
            source=image,
            imgsz=image_size,
            conf=confidence,
            verbose=False,
        )

        if not results:
            raise RuntimeError(
                "YOLO detection returned no result."
            )

        return results[0]

    @staticmethod
    def parse_results(results) -> list[dict]:
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if cls_id not in CLASS_NAMES:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0],
            )

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES[cls_id],
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return detections

    def detect(
        self,
        image,
        image_size: int = 1280,
        confidence: float = 0.25,
    ) -> tuple[object, list[dict]]:

        results = self.predict(
            image=image,
            image_size=image_size,
            confidence=confidence,
        )

        detections = self.parse_results(results)

        return results, detections

    @staticmethod
    def visualize(
        image,
        detections: list[dict],
    ):
        vis_img = image.copy()

        for detection in detections:
            cls_id = detection["class_id"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]

            x1, y1, x2, y2 = detection["bbox"]

            color = COLORS[cls_id]

            cv2.rectangle(
                vis_img,
                (x1, y1),
                (x2, y2),
                color,
                3,
            )

            cv2.putText(
                vis_img,
                f"{class_name} {confidence:.0%}",
                (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        return vis_img