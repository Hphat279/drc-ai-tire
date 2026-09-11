from sqlalchemy.orm import Session

from app.models import (
    InspectionDetection,
    InspectionField,
    InspectionRun,
)


class InspectionRepository:
    def __init__(self, db: Session):
        self.db = db

    # =========================================================
    # CREATE
    # =========================================================
    
    def create_run(
        self,
        image_path: str,
        status: str,
        segmentation_status: str,
        detection_status: str,
        processing_time_ms: float | None,
    ) -> InspectionRun:
        inspection = InspectionRun(
            image_path=image_path,
            status=status,
            segmentation_status=segmentation_status,
            detection_status=detection_status,
            processing_time_ms=processing_time_ms,
        )

        self.db.add(inspection)
        self.db.flush()

        return inspection

    def add_detection(
        self,
        inspection_id: int,
        class_name: str,
        confidence: float,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> InspectionDetection:
        detection = InspectionDetection(
            inspection_id=inspection_id,
            class_name=class_name,
            confidence=confidence,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

        self.db.add(detection)

        return detection

    def add_field(
        self,
        inspection_id: int,
        field_name: str,
        text: str | None,
        confidence: float,
        extraction_status: str,
        ocr_status: str,
        source_detections: int,
    ) -> InspectionField:
        field = InspectionField(
            inspection_id=inspection_id,
            field_name=field_name,
            text=text,
            confidence=confidence,
            extraction_status=extraction_status,
            ocr_status=ocr_status,
            source_detections=source_detections,
        )

        self.db.add(field)

        return field

    # =========================================================
    # READ
    # =========================================================

    def get_run(
        self,
        inspection_id: int,
    ) -> InspectionRun | None:
        return (
            self.db.query(InspectionRun)
            .filter(
                InspectionRun.id == inspection_id
            )
            .first()
        )
        
    # =========================================================
    # TRANSACTION
    # =========================================================
    
    def commit(self) -> None:
        self.db.commit()

    def rollback(self) -> None:
        self.db.rollback()

    def refresh(
        self,
        inspection: InspectionRun
    ) -> InspectionRun:
        self.db.refresh(inspection)
        
        return inspection