from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import update

from app.models import (
    InspectionDetection,
    InspectionField,
    InspectionRun,
)


class InspectionRepository:
    def __init__(self, db: Session):
        self.db = db

    # =========================================================
    # CREATE / UPDATE
    # =========================================================

    def create_run(
        self,
        image_path: str,
        status: str,
        segmentation_status: str,
        detection_status: str,
        processing_time_ms: float | None,
        error_code: str | None = None,
        error_message: str | None = None,
        failed_stage: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> InspectionRun:
        inspection = InspectionRun(
            image_path=image_path,
            status=status,
            segmentation_status=segmentation_status,
            detection_status=detection_status,
            processing_time_ms=processing_time_ms,
            error_code=error_code,
            error_message=error_message,
            failed_stage=failed_stage,
            started_at=started_at,
            completed_at=completed_at,
        )

        self.db.add(inspection)
        self.db.flush()

        return inspection

    def update_run_status(
        self,
        inspection: InspectionRun,
        *,
        status: str,
        segmentation_status: str | None = None,
        detection_status: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        failed_stage: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> InspectionRun:
        inspection.status = status

        if segmentation_status is not None:
            inspection.segmentation_status = segmentation_status

        if detection_status is not None:
            inspection.detection_status = detection_status

        if error_code is not None:
            inspection.error_code = error_code

        if error_message is not None:
            inspection.error_message = error_message

        if failed_stage is not None:
            inspection.failed_stage = failed_stage

        if started_at is not None:
            inspection.started_at = started_at

        if completed_at is not None:
            inspection.completed_at = completed_at

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

    def claim_pending_run(
        self,
        inspection_id: int,
        started_at: datetime,
    ) -> bool:
        """
        Thực hiện thao tác nguyên tử để nhận một lượt kiểm tra đang ở trạng thái chờ (pending) về xử lý.

        - Trả về True nếu người gọi chuyển đổi thành công trạng thái 
        của lượt kiểm tra từ chờ sang đang xử lý. 
        - Trả về False nếu lượt kiểm tra không còn ở trạng thái chờ
        (nghĩa là đã được nhận hoặc đã hoàn tất).
        """

        result = self.db.execute(
            update(InspectionRun)
            .where(
                InspectionRun.id == inspection_id,
                InspectionRun.status == "pending",
            )
            .values(
                status="processing",
                segmentation_status="processing",
                started_at=started_at,
            )
        )

        self.commit()

        return result.rowcount == 1
    
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