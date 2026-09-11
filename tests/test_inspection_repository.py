from app.repositories.inspection_repository import (
    InspectionRepository,
)


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