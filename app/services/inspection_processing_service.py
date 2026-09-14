from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.core.inspection_status import (
    InspectionStageStatus,
    InspectionStatus,
)
from app.core.pipeline_errors import PipelineStageError
from app.repositories.inspection_repository import (
    InspectionRepository,
)
from app.services.inspection_result_mapper import (
    InspectionResultMapper,
)


class InspectionProcessingService:
    """
    Handles the execution and persistence of an inspection run.

    Responsibilities:
    - transition lifecycle state
    - execute AI pipeline
    - persist detections
    - persist OCR/extraction fields
    - persist success/partial/failed state
    """

    FIELD_NAMES = ("size", "pattern", "brand")

    def __init__(
        self,
        repository: InspectionRepository,
        pipeline,
        image_store,
    ):
        self.repository = repository
        self.pipeline = pipeline
        self.image_store = image_store

    def process(
        self,
        inspection_id: int,
    ) -> dict | None:
        """
        Process an existing inspection run.

        The inspection image is resolved from its persisted
        storage key before executing the AI pipeline.
        """

        inspection = self.repository.get_run(inspection_id)

        if inspection is None:
            return None

        try:
            image_path = self.image_store.resolve(
                inspection.image_path
            )

            return self._process_run(
                inspection_id=inspection_id,
                image_path=image_path,
            )

        except PipelineStageError as exc:
            return self._handle_pipeline_error(
                inspection_id=inspection_id,
                error=exc,
            )

        except Exception:
            return self._handle_unexpected_error(
                inspection_id=inspection_id,
            )

    def process_new_run(
        self,
        *,
        inspection_id: int,
        image_path: Path,
        started_at: datetime,
        storage_key: str,
    ) -> dict:
        """
        Process an inspection immediately after creating it.

        This is used by the legacy synchronous inspection path.
        """

        try:
            return self._process_run(
                inspection_id=inspection_id,
                image_path=image_path,
                started_at=started_at,
                storage_key=storage_key,
            )

        except PipelineStageError as exc:
            return self._handle_pipeline_error(
                inspection_id=inspection_id,
                error=exc,
            )

        except Exception:
            return self._handle_unexpected_error(
                inspection_id=inspection_id,
            )

    def _process_run(
        self,
        *,
        inspection_id: int,
        image_path: Path,
        started_at: datetime | None = None,
        storage_key: str | None = None,
    ) -> dict:
        """
        Execute the complete AI processing flow.
        """

        inspection = self.repository.get_run(
            inspection_id
        )

        if inspection is None:
            raise ValueError(
                f"Inspection not found: {inspection_id}"
            )

        if started_at is None:
            started_at = datetime.now(UTC)

        if storage_key is None:
            storage_key = inspection.image_path

        self._mark_processing(
            inspection=inspection,
            started_at=started_at,
        )

        result = self.pipeline.run(image_path)

        self._persist_detections(
            inspection_id=inspection_id,
            result=result,
        )

        self._persist_fields(
            inspection_id=inspection_id,
            result=result,
        )

        self._mark_completed(
            inspection=inspection,
            result=result,
        )

        response = InspectionResultMapper.to_response(
            inspection
        )

        # Keep the raw pipeline result fields that existing
        # callers/tests may rely on.
        response.update(result)

        response["inspection_id"] = inspection_id
        response["image"] = storage_key

        return response

    def _mark_processing(
        self,
        *,
        inspection,
        started_at: datetime,
    ) -> None:
        """
        Transition inspection into processing state.
        """

        self.repository.update_run_status(
            inspection,
            status=InspectionStatus.PROCESSING,
            segmentation_status=(
                InspectionStageStatus.PROCESSING
            ),
            started_at=started_at,
        )

        self.repository.commit()

    def _persist_detections(
        self,
        *,
        inspection_id: int,
        result: dict,
    ) -> None:
        """
        Persist YOLO detection results.
        """

        detections = (
            result
            .get("detections", {})
            .get("items", [])
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

    def _persist_fields(
        self,
        *,
        inspection_id: int,
        result: dict,
    ) -> None:
        """
        Persist extraction and OCR results.
        """

        extraction = result.get(
            "extraction",
            {},
        )

        ocr = result.get(
            "ocr",
            {},
        )

        for field_name in self.FIELD_NAMES:
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
                extraction_status=extraction_field.get(
                    "status",
                    "not_detected",
                ),
                ocr_status=ocr_field.get(
                    "status",
                    "not_detected",
                ),
                source_detections=extraction_field.get(
                    "source_detections",
                    0,
                ),
            )

    def _mark_completed(
        self,
        *,
        inspection,
        result: dict,
    ) -> None:
        """
        Persist final inspection state.
        """

        completed_at = datetime.now(UTC)

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

        inspection.processing_time_ms = result.get(
            "processing_time_ms"
        )

        self.repository.commit()

    def _handle_pipeline_error(
        self,
        *,
        inspection_id: int,
        error: PipelineStageError,
    ) -> dict:
        """
        Persist a known AI pipeline failure.
        """

        self.repository.rollback()

        inspection = self.repository.get_run(
            inspection_id
        )

        if inspection is None:
            raise ValueError(
                f"Inspection not found: {inspection_id}"
            )

        self.repository.update_run_status(
            inspection,
            status=InspectionStatus.FAILED,
            error_code=error.code,
            error_message=error.message,
            failed_stage=error.stage,
            completed_at=datetime.now(UTC),
        )

        self.repository.commit()

        return InspectionResultMapper.to_response(
            inspection
        )

    def _handle_unexpected_error(
        self,
        *,
        inspection_id: int,
    ) -> dict:
        """
        Persist an unexpected inspection failure.

        The internal exception is intentionally not exposed
        through the API response.
        """

        self.repository.rollback()

        inspection = self.repository.get_run(
            inspection_id
        )

        if inspection is None:
            raise ValueError(
                f"Inspection not found: {inspection_id}"
            )

        self.repository.update_run_status(
            inspection,
            status=InspectionStatus.FAILED,
            error_code="INSPECTION_FAILED",
            error_message="Inspection processing failed.",
            failed_stage="pipeline",
            completed_at=datetime.now(UTC),
        )

        self.repository.commit()

        return InspectionResultMapper.to_response(
            inspection
        )