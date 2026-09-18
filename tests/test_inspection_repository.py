from app.repositories.inspection_repository import (
    InspectionRepository,
)

from datetime import datetime, UTC

def test_create_run(db_session):
    repository = InspectionRepository(db_session)

    inspection = repository.create_run(
        image_path="inspections/2026/09/test.jpg",
        status="success",
        segmentation_status="success",
        detection_status="success",
        processing_time_ms=1856.53,
    )

    repository.commit()

    assert inspection.id is not None
    assert inspection.image_path == (
        "inspections/2026/09/test.jpg"
    )
    assert inspection.status == "success"
    assert inspection.segmentation_status == "success"
    assert inspection.detection_status == "success"
    assert inspection.processing_time_ms == 1856.53


def test_add_detection(db_session):
    repository = InspectionRepository(db_session)

    inspection = repository.create_run(
        image_path="inspections/2026/09/test.jpg",
        status="success",
        segmentation_status="success",
        detection_status="success",
        processing_time_ms=1000.0,
    )

    detection = repository.add_detection(
        inspection_id=inspection.id,
        class_name="brand",
        confidence=0.9971,
        x1=0,
        y1=95,
        x2=234,
        y2=172,
    )

    repository.commit()

    assert detection.id is not None
    assert detection.inspection_id == inspection.id
    assert detection.class_name == "brand"
    assert detection.confidence == 0.9971
    assert detection.x1 == 0
    assert detection.y1 == 95
    assert detection.x2 == 234
    assert detection.y2 == 172


def test_add_field(db_session):
    repository = InspectionRepository(db_session)

    inspection = repository.create_run(
        image_path="inspections/2026/09/test.jpg",
        status="success",
        segmentation_status="success",
        detection_status="success",
        processing_time_ms=1000.0,
    )

    field = repository.add_field(
        inspection_id=inspection.id,
        field_name="brand",
        text="DPLUS",
        confidence=0.9971,
        extraction_status="reconstructed",
        ocr_status="reconstructed",
        source_detections=2,
    )

    repository.commit()

    assert field.id is not None
    assert field.inspection_id == inspection.id
    assert field.field_name == "brand"
    assert field.text == "DPLUS"
    assert field.confidence == 0.9971
    assert field.extraction_status == "reconstructed"
    assert field.ocr_status == "reconstructed"
    assert field.source_detections == 2


def test_get_run(db_session):
    repository = InspectionRepository(db_session)

    inspection = repository.create_run(
        image_path="inspections/2026/09/test.jpg",
        status="success",
        segmentation_status="success",
        detection_status="success",
        processing_time_ms=1856.53,
    )

    repository.commit()

    result = repository.get_run(
        inspection_id=inspection.id
    )

    assert result is not None
    assert result.id == inspection.id
    assert result.image_path == (
        "inspections/2026/09/test.jpg"
    )
    assert result.status == "success"


def test_get_run_not_found(db_session):
    repository = InspectionRepository(db_session)

    result = repository.get_run(
        inspection_id=999999
    )

    assert result is None


def test_create_run_pending(db_session):
    repository = InspectionRepository(db_session)

    inspection = repository.create_run(
        image_path="inspections/2026/09/test.jpg",
        status="pending",
        segmentation_status="pending",
        detection_status="pending",
        processing_time_ms=None,
    )

    repository.commit()

    assert inspection.id is not None
    assert inspection.status == "pending"
    assert inspection.segmentation_status == "pending"
    assert inspection.detection_status == "pending"
    assert inspection.error_code is None
    assert inspection.error_message is None
    assert inspection.failed_stage is None
    assert inspection.started_at is None
    assert inspection.completed_at is None


def test_update_run_status_processing(db_session):
    repository = InspectionRepository(db_session)

    inspection = repository.create_run(
        image_path="inspections/2026/09/test.jpg",
        status="pending",
        segmentation_status="pending",
        detection_status="pending",
        processing_time_ms=None,
    )
    repository.commit()

    started_at = datetime.utcnow()

    repository.update_run_status(
        inspection,
        status="processing",
        segmentation_status="processing",
        detection_status="pending",
        started_at=started_at,
    )
    repository.commit()

    refreshed = repository.get_run(inspection.id)

    assert refreshed is not None
    assert refreshed.status == "processing"
    assert refreshed.segmentation_status == "processing"
    assert refreshed.detection_status == "pending"
    assert refreshed.started_at == started_at
    assert refreshed.completed_at is None


def test_update_run_status_failed(db_session):
    repository = InspectionRepository(db_session)

    inspection = repository.create_run(
        image_path="inspections/2026/09/test.jpg",
        status="pending",
        segmentation_status="pending",
        detection_status="pending",
        processing_time_ms=None,
    )
    repository.commit()

    started_at = datetime.utcnow()
    completed_at = datetime.utcnow()

    repository.update_run_status(
        inspection,
        status="failed",
        segmentation_status="failed",
        detection_status="skipped",
        error_code="SEGMENTATION_FAILED",
        error_message="Segmentation stage failed.",
        failed_stage="segmentation",
        started_at=started_at,
        completed_at=completed_at,
    )
    repository.commit()

    refreshed = repository.get_run(inspection.id)

    assert refreshed is not None
    assert refreshed.status == "failed"
    assert refreshed.segmentation_status == "failed"
    assert refreshed.detection_status == "skipped"
    assert refreshed.error_code == "SEGMENTATION_FAILED"
    assert refreshed.error_message == "Segmentation stage failed."
    assert refreshed.failed_stage == "segmentation"
    assert refreshed.started_at == started_at
    assert refreshed.completed_at == completed_at


def test_update_run_status_success(db_session):
    repository = InspectionRepository(db_session)

    inspection = repository.create_run(
        image_path="inspections/2026/09/test.jpg",
        status="pending",
        segmentation_status="pending",
        detection_status="pending",
        processing_time_ms=None,
    )
    repository.commit()

    started_at = datetime.utcnow()
    completed_at = datetime.utcnow()

    repository.update_run_status(
        inspection,
        status="success",
        segmentation_status="success",
        detection_status="success",
        started_at=started_at,
        completed_at=completed_at,
    )
    repository.commit()

    refreshed = repository.get_run(inspection.id)

    assert refreshed is not None
    assert refreshed.status == "success"
    assert refreshed.segmentation_status == "success"
    assert refreshed.detection_status == "success"
    assert refreshed.started_at == started_at
    assert refreshed.completed_at == completed_at
    
def test_claim_pending_run_success(db_session):
    repository = InspectionRepository(db_session)

    inspection = repository.create_run(
        image_path="test.jpg",
        status="pending",
        segmentation_status="pending",
        detection_status="pending",
        processing_time_ms=None,
    )

    started_at = datetime.now(UTC)

    claimed = repository.claim_pending_run(
        inspection_id=inspection.id,
        started_at=started_at,
    )

    assert claimed is True

    db_session.refresh(inspection)

    assert inspection.status == "processing"
    assert inspection.segmentation_status == "processing"
    assert inspection.started_at == started_at.replace(tzinfo=None)

def test_claim_pending_run_returns_false_when_already_claimed(db_session):
    repository = InspectionRepository(db_session)

    inspection = repository.create_run(
        image_path="test.jpg",
        status="pending",
        segmentation_status="pending",
        detection_status="pending",
        processing_time_ms=None,
    )

    first_started_at = datetime.now(UTC)

    first_claimed = repository.claim_pending_run(
        inspection_id=inspection.id,
        started_at=first_started_at,
    )

    second_started_at = datetime.now(UTC)

    second_claimed = repository.claim_pending_run(
        inspection_id=inspection.id,
        started_at=second_started_at,
    )

    assert first_claimed is True
    assert second_claimed is False

    db_session.refresh(inspection)

    assert inspection.status == "processing"
    assert inspection.segmentation_status == "processing"
    assert inspection.started_at == first_started_at.replace(tzinfo=None)
    
    