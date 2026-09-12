from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.repositories.inspection_repository import InspectionRepository

from app.core.inspection_status import (
    InspectionStageStatus,
    InspectionStatus,
)

from app.core.pipeline_errors import PipelineStageError

class InspectionService:
    def __init__(
        self,
        db: Session,
        pipeline=None,
        image_store=None,
    ):
        self.pipeline = pipeline
        self.repository = InspectionRepository(db)
        self.image_store = image_store

    # =========================================================
    # CREATE INSPECTION
    # =========================================================

    def inspect(
        self,
        image_path: Path,
    ) -> dict:
        started_at = datetime.utcnow()


        # =========================================================
        # 1. Save image permanently
        # =========================================================

        storage_key = self.image_store.save(
            image_path
        )

        # =========================================================
        # 2. Create inspection run as PENDING
        # =========================================================

        inspection = self.repository.create_run(
            image_path=storage_key,
            status=InspectionStatus.PENDING,
            segmentation_status=InspectionStageStatus.PENDING,
            detection_status=InspectionStageStatus.PENDING,
            processing_time_ms=None,
        )

        inspection_id = inspection.id

        # IMPORTANT
        # Lưu lại kết quả kiểm tra trước khi chạy quy trình xử lý AI.
        self.repository.commit()

        try:
            # =========================
            # 3. Mark inspection as PROCESSING
            # =========================

            self.repository.update_run_status(
                inspection,
                status=InspectionStatus.PROCESSING,
                segmentation_status=InspectionStageStatus.PROCESSING,
                started_at=started_at,
            )

            self.repository.commit()

            # =====================================================
            # 4. Run AI pipeline
            # =====================================================

            result = self.pipeline.run(
                image_path
            )

            # =========================
            # 5. Save detections
            # =========================

            detections = result.get(
                "detections",
                {},
            ).get(
                "items",
                [],
            )

            for detection in detections:
                bbox = detection["bbox"]

                self.repository.add_detection(
                    inspection_id=inspection_id,
                    class_name=detection["class_name"],
                    confidence=detection["confidence"],
                    x1=bbox[0],
                    y1=bbox[1],
                    x2=bbox[2],
                    y2=bbox[3],
                )

            # =========================
            # 6. Save extracted fields
            # =========================

            extraction = result.get(
                "extraction",
                {}
            )

            ocr = result.get(
                "ocr",
                {}
            )

            for field_name in (
                "size",
                "pattern",
                "brand",
            ):
                extraction_field = extraction.get(
                    field_name,
                    {},
                )

                ocr_field = ocr.get(
                    field_name,
                    {},
                )

                self.repository.add_field(
                    inspection_id=inspection_id,
                    field_name=field_name,
                    text=ocr_field.get("text"),
                    confidence=ocr_field.get(
                        "confidence",
                        0.0,
                    ),
                    extraction_status=(
                        extraction_field.get(
                            "status",
                            "not_detected",
                        )
                    ),
                    ocr_status=ocr_field.get(
                        "status",
                        "not_detected",
                    ),
                    source_detections=(
                        extraction_field.get(
                            "source_detections",
                            0,
                        )
                    ),
                )

            # =========================
            # 7. Finalize inspection
            # =========================

            completed_at = datetime.utcnow()

            self.repository.update_run_status(
                inspection,
                status=result["status"],
                segmentation_status=(
                    result["segmentation"]["status"]
                ),
                detection_status=(
                    result["detections"]["status"]
                ),
                completed_at=completed_at,
            )

            # Preserve pipeline processing time.
            inspection.processing_time_ms = result.get(
                "processing_time_ms"
            )

            self.repository.commit()

            # =========================
            # 8. Return result
            # =========================

            result["inspection_id"] = inspection_id
            result["image"] = storage_key

            return result

        except PipelineStageError as exc:

            self.repository.rollback()

            inspection = self.repository.get_run(
                inspection_id
            )

            if inspection is None:
                raise

            self.repository.update_run_status(
                inspection,
                status=InspectionStatus.FAILED,
                error_code=exc.code,
                error_message=exc.message,
                failed_stage=exc.stage,
                completed_at=datetime.utcnow(),
            )

            self.repository.commit()

            return self.get_inspection(
                inspection_id
            )

        except Exception:
            # =====================================================
            # Unexpected failure
            # =====================================================

            self.repository.rollback()

            inspection = self.repository.get_run(
                inspection_id
            )

            if inspection is None:
                raise

            self.repository.update_run_status(
                inspection,
                status=InspectionStatus.FAILED,
                error_code="INSPECTION_FAILED",
                error_message="Inspection processing failed.",
                failed_stage="pipeline",
                completed_at=datetime.utcnow(),
            )

            self.repository.commit()

            return self.get_inspection(
                inspection_id
            )

    # =========================================================
    # READ INSPECTION
    # =========================================================

    def get_inspection(
        self,
        inspection_id: int,
    ) -> dict | None:

        inspection = self.repository.get_run(
            inspection_id
        )

        if inspection is None:
            return None

        # =====================================================
        # Detection mapping
        # =====================================================

        class_id_mapping = {
            "size": 0,
            "pattern": 1,
            "brand": 2,
        }

        detection_items = []

        for detection in inspection.detections:
            detection_items.append(
                {
                    "class_id": class_id_mapping.get(
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
            )

        # =====================================================
        # Field mapping
        # =====================================================

        fields = {
            field.field_name: field
            for field in inspection.fields
        }

        def build_extraction_item(
            field_name: str,
        ) -> dict:

            field = fields.get(field_name)

            if field is None:
                return {
                    "status": "not_detected",
                    "source_detections": 0,
                }

            return {
                "status":field.extraction_status,
                "source_detections": (
                    field.source_detections
                ),
            }

        def build_ocr_item(
            field_name: str,
        ) -> dict:

            field = fields.get(field_name)

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

        # =====================================================
        # Build API response
        # =====================================================

        return {
            "inspection_id": inspection_id,
            "image": inspection.image_path,
            "status": inspection.status,

            "segmentation": {
                "status": inspection.segmentation_status,
            },

            "extraction": {
                "brand": build_extraction_item(
                    "brand"
                ),
                "size": build_extraction_item(
                    "size"
                ),
                "pattern": build_extraction_item(
                    "pattern"
                ),
            },

            "detections": {
                "status": inspection.detection_status,
                "items": detection_items,
            },

            "ocr": {
                "brand": build_ocr_item(
                    "brand"
                ),
                "size": build_ocr_item(
                    "size"
                ),
                "pattern": build_ocr_item(
                    "pattern"
                ),
            },

            "processing_time_ms": (
                inspection.processing_time_ms
                or 0.0
            ),
        }