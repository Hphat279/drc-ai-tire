from __future__ import annotations

from app.models import InspectionRun


class InspectionResultMapper:
    """
    Maps InspectionRun ORM objects into the API response structure.

    This class contains no database operations and no business logic.
    """

    CLASS_ID_MAPPING = {
        "size": 0,
        "pattern": 1,
        "brand": 2,
    }

    FIELD_NAMES = ("brand", "size", "pattern")

    @classmethod
    def to_response(
        cls,
        inspection: InspectionRun,
    ) -> dict:
        """
        Convert an InspectionRun ORM object into the inspection response.
        """

        detection_items = [
            cls._map_detection(detection)
            for detection in inspection.detections
        ]

        fields = {
            field.field_name: field
            for field in inspection.fields
        }

        return {
            "inspection_id": inspection.id,
            "image": inspection.image_path,
            "status": inspection.status,
            "segmentation": {
                "status": inspection.segmentation_status,
            },
            "extraction": {
                "brand": cls._map_extraction_field(
                    fields.get("brand")
                ),
                "size": cls._map_extraction_field(
                    fields.get("size")
                ),
                "pattern": cls._map_extraction_field(
                    fields.get("pattern")
                ),
            },
            "detections": {
                "status": inspection.detection_status,
                "items": detection_items,
            },
            "ocr": {
                "brand": cls._map_ocr_field(
                    fields.get("brand")
                ),
                "size": cls._map_ocr_field(
                    fields.get("size")
                ),
                "pattern": cls._map_ocr_field(
                    fields.get("pattern")
                ),
            },
            "processing_time_ms": (
                inspection.processing_time_ms or 0.0
            ),
        }

    @classmethod
    def _map_detection(cls, detection) -> dict:
        """
        Map a persisted detection into the API detection structure.
        """

        return {
            "class_id": cls.CLASS_ID_MAPPING.get(
                detection.class_name,
                -1,
            ),
            "class_name": detection.class_name,
            "confidence": detection.confidence,
            "bbox": [
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2,
            ],
        }

    @staticmethod
    def _map_extraction_field(field) -> dict:
        """
        Map an InspectionField extraction state.
        """

        if field is None:
            return {
                "status": "not_detected",
                "source_detections": 0,
            }

        return {
            "status": field.extraction_status,
            "source_detections": field.source_detections,
        }

    @staticmethod
    def _map_ocr_field(field) -> dict:
        """
        Map an InspectionField OCR state.
        """

        if field is None:
            return {
                "text": None,
                "confidence": 0.0,
                "status": "not_detected",
            }

        return {
            "text": field.text,
            "confidence": field.confidence,
            "status": field.ocr_status,
        }