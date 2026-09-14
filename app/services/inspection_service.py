from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.inspection_status import (
    InspectionStageStatus,
    InspectionStatus,
)
from app.repositories.inspection_repository import (
    InspectionRepository,
)
from app.services.inspection_processing_service import (
    InspectionProcessingService,
)
from app.services.inspection_result_mapper import (
    InspectionResultMapper,
)


class InspectionService:
    """
    Application service for tire inspection.

    Responsibilities:
    - create inspection runs
    - coordinate synchronous inspection
    - coordinate asynchronous inspection
    - retrieve inspection results

    Heavy processing and response mapping are delegated to
    dedicated services.
    """

    def __init__(
        self,
        db: Session,
        pipeline=None,
        image_store=None,
    ):
        self.pipeline = pipeline
        self.image_store = image_store

        self.repository = InspectionRepository(db)

        self.processing_service = None

        if (
            pipeline is not None
            and image_store is not None
        ):
            self.processing_service = (
                InspectionProcessingService(
                    repository=self.repository,
                    pipeline=pipeline,
                    image_store=image_store,
                )
            )

    # ------------------------------------------------------------------
    # CREATE
    # ------------------------------------------------------------------

    def create_pending_inspection(
        self,
        image_path: Path,
    ) -> dict:
        """
        Save the uploaded image and create a pending inspection run.

        This method does NOT execute the AI pipeline.
        """

        self._require_image_store()

        storage_key = self.image_store.save(
            image_path
        )

        inspection = self.repository.create_run(
            image_path=storage_key,
            status=InspectionStatus.PENDING,
            segmentation_status=(
                InspectionStageStatus.PENDING
            ),
            detection_status=(
                InspectionStageStatus.PENDING
            ),
            processing_time_ms=None,
        )

        self.repository.commit()

        return {
            "inspection_id": inspection.id,
            "status": inspection.status,
        }

    # ------------------------------------------------------------------
    # SYNCHRONOUS
    # ------------------------------------------------------------------

    def inspect(
        self,
        image_path: Path,
    ) -> dict:
        """
        Execute an inspection synchronously.

        This method is retained for backward compatibility
        and existing synchronous tests.
        """

        self._require_image_store()
        self._require_processing_service()

        started_at = datetime.now(UTC)

        storage_key = self.image_store.save(
            image_path
        )

        inspection = self.repository.create_run(
            image_path=storage_key,
            status=InspectionStatus.PENDING,
            segmentation_status=(
                InspectionStageStatus.PENDING
            ),
            detection_status=(
                InspectionStageStatus.PENDING
            ),
            processing_time_ms=None,
        )

        inspection_id = inspection.id

        self.repository.commit()

        return self.processing_service.process_new_run(
            inspection_id=inspection_id,
            image_path=image_path,
            started_at=started_at,
            storage_key=storage_key,
        )

    # ------------------------------------------------------------------
    # ENQUEUE FAILURE
    # ------------------------------------------------------------------
    
    def mark_enqueue_failed(
        self,
        *,
        inspection_id: int,
        error_message: str,
    ) -> dict | None:
        inspection = self.repository.get_run(inspection_id)

        if inspection is None:
            return None

        self.repository.update_run_status(
            inspection,
            status=InspectionStatus.FAILED,
            error_code="TASK_ENQUEUE_FAILED",
            error_message=error_message,
            failed_stage="queue",
            completed_at=datetime.now(UTC),
        )
        self.repository.commit()

        return InspectionResultMapper.to_response(inspection)
    
    # ------------------------------------------------------------------
    # ASYNCHRONOUS
    # ------------------------------------------------------------------

    def process_inspection(
        self,
        inspection_id: int,
    ) -> dict | None:
        """
        Process an existing pending inspection.

        The worker calls this method after receiving the
        inspection ID from Celery.
        """

        self._require_processing_service()

        return self.processing_service.process(
            inspection_id
        )

    # ------------------------------------------------------------------
    # READ
    # ------------------------------------------------------------------

    def get_inspection(
        self,
        inspection_id: int,
    ) -> dict | None:
        """
        Retrieve an inspection from PostgreSQL.
        """

        inspection = self.repository.get_run(
            inspection_id
        )

        if inspection is None:
            return None

        return InspectionResultMapper.to_response(
            inspection
        )

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def _require_image_store(self) -> None:
        if self.image_store is None:
            raise RuntimeError(
                "Image storage is not configured."
            )

    def _require_processing_service(self) -> None:
        if self.processing_service is None:
            raise RuntimeError(
                "Inspection processing service "
                "is not configured."
            )