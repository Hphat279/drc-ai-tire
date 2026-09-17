from pathlib import Path
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.main import app
from app.db.database import get_db
from app.repositories.inspection_repository import InspectionRepository
from app.services.inspection_service import InspectionService

from app.core.inspection_status import (
    DetectionStatus,
    ExtractionStatus,
    InspectionStatus,
    InspectionStageStatus,
    OCRStatus,
)

import pytest
from pydantic import ValidationError

from app.schemas.inspection import (
    DetectionResult,
    ExtractionItem,
    InspectionResponse,
    OCRItem,
    SegmentationResult,
)

class FakePipeline:
    def run(self, image_path: Path) -> dict:
        return {
            "status": "success",
            "segmentation": {
                "status": "success",
            },
            "extraction": {
                "size": {
                    "status": "direct",
                    "source_detections": 1,
                },
                "pattern": {
                    "status": "direct",
                    "source_detections": 1,
                },
                "brand": {
                    "status": "reconstructed",
                    "source_detections": 2,
                },
            },
            "detections": {
                "status": "success",
                "items": [
                    {
                        "class_id": 0,
                        "class_name": "size",
                        "confidence": 0.9927,
                        "bbox": [100, 200, 300, 250],
                    },
                    {
                        "class_id": 1,
                        "class_name": "pattern",
                        "confidence": 0.9991,
                        "bbox": [400, 200, 500, 250],
                    },
                    {
                        "class_id": 2,
                        "class_name": "brand",
                        "confidence": 0.9971,
                        "bbox": [0, 100, 234, 172],
                    },
                    {
                        "class_id": 3,
                        "class_name": "brand",
                        "confidence": 0.6427,
                        "bbox": [4490, 102, 4536, 169],
                    },
                ],
            },
            "ocr": {
                "size": {
                    "text": "100/90-14",
                    "confidence": 0.9927,
                    "status": "direct",
                },
                "pattern": {
                    "text": "121",
                    "confidence": 0.9991,
                    "status": "direct",
                },
                "brand": {
                    "text": "DPLUS",
                    "confidence": 0.9971,
                    "status": "reconstructed",
                },
            },
            "processing_time_ms": 1856.53,
            "unexpected": "must_not_appear",
        }


class FakeImageStorage:
    def save(self, image_path: Path) -> str:
        return "inspections/2026/09/test.jpg"


def override_get_db(db_session):
    def _get_db():
        yield db_session

    return _get_db


def setup_app(db_session):
    app.dependency_overrides[get_db] = override_get_db(
        db_session
    )

    app.state.inspection_pipeline = FakePipeline()
    app.state.image_storage = FakeImageStorage()


def teardown_app():
    app.dependency_overrides.clear()

    if hasattr(app.state, "inspection_pipeline"):
        del app.state.inspection_pipeline

    if hasattr(app.state, "image_storage"):
        del app.state.image_storage


def test_post_inspection_returns_pending(
    db_session,
    monkeypatch,
):
    setup_app(db_session)

    try:
        queued = {}

        def fake_delay(inspection_id):
            queued["inspection_id"] = inspection_id

        monkeypatch.setattr(
            "app.api.v1.inspection.process_inspection.delay",
            fake_delay,
        )

        client = TestClient(app)

        response = client.post(
            "/api/v1/inspection",
            files={
                "image": (
                    "test.jpg",
                    b"fake-image-data",
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 202

        data = response.json()

        assert data["inspection_id"] is not None
        assert data["status"] == "pending"

        assert queued["inspection_id"] == (
            data["inspection_id"]
        )

        repository = InspectionRepository(db_session)

        inspection = repository.get_run(
            data["inspection_id"]
        )

        assert inspection is not None
        assert inspection.status == "pending"
        assert inspection.segmentation_status == "pending"
        assert inspection.detection_status == "pending"

    finally:
        teardown_app()

def test_post_inspection_missing_image_returns_validation_error(
    db_session,
):
    setup_app(db_session)

    try:
        client = TestClient(app)

        response = client.post(
            "/api/v1/inspection",
        )

        assert response.status_code == 422

        data = response.json()

        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert data["error"]["message"] == (
            "Request validation failed."
        )
        assert isinstance(
            data["error"]["details"],
            list,
        )
        assert data["error"]["details"]

    finally:
        teardown_app()

def test_unexpected_exception_returns_internal_server_error(
    db_session,
    monkeypatch,
):
    setup_app(db_session)

    try:
        def fake_create_pending_inspection(*args, **kwargs):
            raise RuntimeError(
                "SECRET_INTERNAL_DATABASE_ERROR"
            )

        monkeypatch.setattr(
            InspectionService,
            "create_pending_inspection",
            fake_create_pending_inspection,
        )

        client = TestClient(
            app,
            raise_server_exceptions=False,
        )

        response = client.post(
            "/api/v1/inspection",
            files={
                "image": (
                    "test.jpg",
                    b"fake-image-data",
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 500

        data = response.json()

        assert data == {
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": (
                    "An internal server error occurred."
                ),
                "details": None,
            }
        }

        assert (
            "SECRET_INTERNAL_DATABASE_ERROR"
            not in response.text
        )

    finally:
        teardown_app()

def test_post_inspection_enqueue_failure(
    db_session,
    monkeypatch,
):
    setup_app(db_session)

    try:
        def fake_delay(inspection_id):
            raise RuntimeError("Redis unavailable")

        monkeypatch.setattr(
            "app.api.v1.inspection.process_inspection.delay",
            fake_delay,
        )

        client = TestClient(app)

        response = client.post(
            "/api/v1/inspection",
            files={
                "image": (
                    "test.jpg",
                    b"fake-image-data",
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 503

        data = response.json()

        assert data["error"]["code"] == (
            "TASK_ENQUEUE_FAILED"
        )

        assert data["error"]["message"] == (
            "Inspection task could not be queued."
        )

        repository = InspectionRepository(db_session)

        inspection = repository.get_run(
            data["error"]["details"]["inspection_id"]
        )

        assert inspection is not None
        assert inspection.status == "failed"
        assert inspection.error_code == (
            "TASK_ENQUEUE_FAILED"
        )
        assert inspection.failed_stage == "queue"
        assert inspection.error_message == (
            "Failed to enqueue inspection task."
        )

    finally:
        teardown_app()

def test_inspection_detection_persistence_failure(
    db_session,
    monkeypatch,
):
    setup_app(db_session)

    try:
        repository = InspectionRepository(db_session)

        inspection = repository.create_run(
            image_path="inspections/2026/09/test.jpg",
            status="pending",
            segmentation_status="pending",
            detection_status="pending",
            processing_time_ms=None,
        )
        repository.commit()

        def fake_add_detection(*args, **kwargs):
            raise RuntimeError(
                "Detection persistence failed"
            )

        monkeypatch.setattr(
            InspectionRepository,
            "add_detection",
            fake_add_detection,
        )

        service = InspectionService(
            db=db_session,
            pipeline=FakePipeline(),
            image_store=FakeImageStorage(),
        )

        result = service.process_inspection(
            inspection.id
        )

        assert result is not None
        assert result["status"] == "failed"

        inspection = repository.get_run(
            inspection.id
        )

        assert inspection is not None
        assert inspection.status == "failed"
        assert inspection.error_code == (
            "INSPECTION_FAILED"
        )
        assert inspection.failed_stage == "pipeline"
        assert inspection.error_message == (
            "Inspection processing failed."
        )

    finally:
        teardown_app()

def test_inspection_field_persistence_failure(
    db_session,
    monkeypatch,
):
    setup_app(db_session)

    try:
        repository = InspectionRepository(db_session)

        inspection = repository.create_run(
            image_path="inspections/2026/09/test.jpg",
            status="pending",
            segmentation_status="pending",
            detection_status="pending",
            processing_time_ms=None,
        )
        repository.commit()

        def fake_add_field(*args, **kwargs):
            raise RuntimeError(
                "Field persistence failed"
            )

        monkeypatch.setattr(
            InspectionRepository,
            "add_field",
            fake_add_field,
        )

        service = InspectionService(
            db=db_session,
            pipeline=FakePipeline(),
            image_store=FakeImageStorage(),
        )

        result = service.process_inspection(
            inspection.id
        )

        assert result is not None
        assert result["status"] == "failed"

        inspection = repository.get_run(
            inspection.id
        )

        assert inspection is not None
        assert inspection.status == "failed"
        assert inspection.error_code == (
            "INSPECTION_FAILED"
        )
        assert inspection.failed_stage == "pipeline"
        assert inspection.error_message == (
            "Inspection processing failed."
        )

        detections = inspection.detections

        assert detections == []

    finally:
        teardown_app()

def test_inspection_commit_failure(
    db_session,
    monkeypatch,
):
    setup_app(db_session)

    try:
        repository = InspectionRepository(db_session)

        inspection = repository.create_run(
            image_path="inspections/2026/09/test.jpg",
            status="pending",
            segmentation_status="pending",
            detection_status="pending",
            processing_time_ms=None,
        )
        repository.commit()

        original_commit = InspectionRepository.commit
        commit_calls = {"count": 0}

        def fake_commit(self):
            commit_calls["count"] += 1

            if commit_calls["count"] == 2:
                raise RuntimeError(
                    "Database commit failed"
                )

            original_commit(self)

        monkeypatch.setattr(
            InspectionRepository,
            "commit",
            fake_commit,
        )

        service = InspectionService(
            db=db_session,
            pipeline=FakePipeline(),
            image_store=FakeImageStorage(),
        )

        result = service.processing_service.process_new_run(
            inspection_id=inspection.id,
            image_path=Path("test.jpg"),
            started_at=datetime.now(UTC),
            storage_key="inspections/2026/09/test.jpg",
        )

        assert result is not None
        assert result["status"] == "failed"

        inspection = repository.get_run(
            inspection.id
        )

        assert inspection is not None
        assert inspection.status == "failed"
        assert inspection.error_code == (
            "INSPECTION_FAILED"
        )
        assert inspection.failed_stage == "pipeline"
        assert inspection.error_message == (
            "Inspection processing failed."
        )

    finally:
        teardown_app()

def test_inspection_response_uses_canonical_mapper(
    db_session,
):
    setup_app(db_session)

    try:
        repository = InspectionRepository(db_session)

        inspection = repository.create_run(
            image_path="inspections/2026/09/test.jpg",
            status="pending",
            segmentation_status="pending",
            detection_status="pending",
            processing_time_ms=None,
        )
        repository.commit()

        service = InspectionService(
            db=db_session,
            pipeline=FakePipeline(),
            image_store=FakeImageStorage(),
        )

        result = service.processing_service.process_new_run(
            inspection_id=inspection.id,
            image_path=Path("test.jpg"),
            started_at=datetime.now(UTC),
            storage_key="inspections/2026/09/test.jpg",
        )

        assert result is not None

        assert result["inspection_id"] == inspection.id
        assert result["image"] == (
            "inspections/2026/09/test.jpg"
        )
        assert result["status"] == "success"

        assert "unexpected" not in result

        assert result["detections"]["status"] == "success"
        assert len(result["detections"]["items"]) == 4

        assert result["extraction"]["brand"] == {
            "status": "reconstructed",
            "source_detections": 2,
        }

        assert result["ocr"]["brand"] == {
            "text": "DPLUS",
            "confidence": 0.9971,
            "status": "reconstructed",
        }

        assert result["processing_time_ms"] == 1856.53

    finally:
        teardown_app()
        
def test_post_inspection_unsupported_format(db_session):
    setup_app(db_session)

    try:
        client = TestClient(app)

        response = client.post(
            "/api/v1/inspection",
            files={
                "image": (
                    "test.txt",
                    b"fake-image-data",
                    "text/plain",
                )
            },
        )

        assert response.status_code == 400

        data = response.json()

        assert data["error"]["code"] == (
            "UNSUPPORTED_IMAGE_FORMAT"
        )

        assert data["error"]["message"] == (
            "Unsupported image format."
        )

        assert data["error"]["details"] == {
            "allowed_formats": [
                "jpg",
                "jpeg",
                "png",
                "webp",
            ]
        }

    finally:
        teardown_app()


def test_post_inspection_empty_image(db_session):
    setup_app(db_session)

    try:
        client = TestClient(app)

        response = client.post(
            "/api/v1/inspection",
            files={
                "image": (
                    "test.jpg",
                    b"",
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 400

        data = response.json()

        assert data["error"]["code"] == "EMPTY_IMAGE"

        assert data["error"]["details"] == (
            "Uploaded image is empty."
        )

    finally:
        teardown_app()


def test_get_inspection_pending(db_session):
    setup_app(db_session)

    try:
        repository = InspectionRepository(db_session)

        inspection = repository.create_run(
            image_path=(
                "inspections/2026/09/test.jpg"
            ),
            status="pending",
            segmentation_status="pending",
            detection_status="pending",
            processing_time_ms=None,
        )

        repository.commit()

        client = TestClient(app)

        response = client.get(
            f"/api/v1/inspection/{inspection.id}"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["inspection_id"] == inspection.id
        assert data["image"] == (
            "inspections/2026/09/test.jpg"
        )
        assert data["status"] == "pending"

        assert data["segmentation"]["status"] == (
            "pending"
        )

        assert data["detections"]["status"] == (
            "pending"
        )

        assert data["extraction"]["brand"] == {
            "status": "not_detected",
            "source_detections": 0,
        }

        assert data["extraction"]["size"] == {
            "status": "not_detected",
            "source_detections": 0,
        }

        assert data["extraction"]["pattern"] == {
            "status": "not_detected",
            "source_detections": 0,
        }

        assert data["ocr"]["brand"] == {
            "text": None,
            "confidence": 0.0,
            "status": "not_detected",
        }

        assert data["ocr"]["size"] == {
            "text": None,
            "confidence": 0.0,
            "status": "not_detected",
        }

        assert data["ocr"]["pattern"] == {
            "text": None,
            "confidence": 0.0,
            "status": "not_detected",
        }

        assert data["processing_time_ms"] == 0.0

    finally:
        teardown_app()


def test_get_inspection_success(db_session):
    setup_app(db_session)

    try:
        repository = InspectionRepository(db_session)

        inspection = repository.create_run(
            image_path=(
                "inspections/2026/09/test.jpg"
            ),
            status="success",
            segmentation_status="success",
            detection_status="success",
            processing_time_ms=1856.53,
        )

        repository.add_detection(
            inspection_id=inspection.id,
            class_name="size",
            confidence=0.9927,
            x1=100,
            y1=200,
            x2=300,
            y2=250,
        )

        repository.add_detection(
            inspection_id=inspection.id,
            class_name="pattern",
            confidence=0.9991,
            x1=400,
            y1=200,
            x2=500,
            y2=250,
        )

        repository.add_detection(
            inspection_id=inspection.id,
            class_name="brand",
            confidence=0.9971,
            x1=0,
            y1=100,
            x2=234,
            y2=172,
        )

        repository.add_field(
            inspection_id=inspection.id,
            field_name="size",
            text="100/90-14",
            confidence=0.9927,
            extraction_status="direct",
            ocr_status="direct",
            source_detections=1,
        )

        repository.add_field(
            inspection_id=inspection.id,
            field_name="pattern",
            text="121",
            confidence=0.9991,
            extraction_status="direct",
            ocr_status="direct",
            source_detections=1,
        )

        repository.add_field(
            inspection_id=inspection.id,
            field_name="brand",
            text="DPLUS",
            confidence=0.9971,
            extraction_status="reconstructed",
            ocr_status="reconstructed",
            source_detections=2,
        )

        repository.commit()

        client = TestClient(app)

        response = client.get(
            f"/api/v1/inspection/{inspection.id}"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["inspection_id"] == inspection.id
        assert data["image"] == (
            "inspections/2026/09/test.jpg"
        )
        assert data["status"] == "success"

        assert data["segmentation"]["status"] == (
            "success"
        )

        assert data["detections"]["status"] == (
            "success"
        )

        assert len(data["detections"]["items"]) == 3

        assert data["extraction"]["brand"] == {
            "status": "reconstructed",
            "source_detections": 2,
        }

        assert data["extraction"]["size"] == {
            "status": "direct",
            "source_detections": 1,
        }

        assert data["extraction"]["pattern"] == {
            "status": "direct",
            "source_detections": 1,
        }

        assert data["ocr"]["brand"] == {
            "text": "DPLUS",
            "confidence": 0.9971,
            "status": "reconstructed",
        }

        assert data["ocr"]["size"] == {
            "text": "100/90-14",
            "confidence": 0.9927,
            "status": "direct",
        }

        assert data["ocr"]["pattern"] == {
            "text": "121",
            "confidence": 0.9991,
            "status": "direct",
        }

        assert data["processing_time_ms"] == 1856.53

    finally:
        teardown_app()


def test_get_inspection_not_found(db_session):
    setup_app(db_session)

    try:
        client = TestClient(app)

        response = client.get(
            "/api/v1/inspection/999999"
        )

        assert response.status_code == 404

        data = response.json()

        assert data["error"]["code"] == (
            "INSPECTION_NOT_FOUND"
        )

        assert data["error"]["message"] == (
            "Inspection not found."
        )

        assert data["error"]["details"] == {
            "inspection_id": 999999
        }

    finally:
        teardown_app()
        
def test_inspection_status_contract():
    assert InspectionStatus.PENDING == "pending"
    assert InspectionStatus.PROCESSING == "processing"
    assert InspectionStatus.SUCCESS == "success"
    assert InspectionStatus.PARTIAL == "partial"
    assert InspectionStatus.FAILED == "failed"


def test_inspection_stage_status_contract():
    assert InspectionStageStatus.PENDING == "pending"
    assert InspectionStageStatus.PROCESSING == "processing"
    assert InspectionStageStatus.SUCCESS == "success"
    assert InspectionStageStatus.FAILED == "failed"
    assert InspectionStageStatus.SKIPPED == "skipped"


def test_extraction_status_contract():
    assert ExtractionStatus.DIRECT == "direct"
    assert ExtractionStatus.RECONSTRUCTED == "reconstructed"
    assert ExtractionStatus.NOT_DETECTED == "not_detected"


def test_ocr_status_contract():
    assert OCRStatus.DIRECT == "direct"
    assert OCRStatus.RECONSTRUCTED == "reconstructed"
    assert OCRStatus.NOT_DETECTED == "not_detected"
    assert OCRStatus.OCR_FAILED == "ocr_failed"


def test_detection_status_contract():
    assert DetectionStatus.SUCCESS == "success"
    assert DetectionStatus.NOT_DETECTED == "not_detected"
    
def test_segmentation_status_rejects_invalid_value():
    with pytest.raises(ValidationError):
        SegmentationResult(status="banana")


def test_extraction_status_rejects_invalid_value():
    with pytest.raises(ValidationError):
        ExtractionItem(
            status="banana",
            source_detections=1,
        )


def test_detection_status_rejects_invalid_value():
    with pytest.raises(ValidationError):
        DetectionResult(
            status="banana",
            items=[],
        )


def test_ocr_status_rejects_invalid_value():
    with pytest.raises(ValidationError):
        OCRItem(
            text="DPLUS",
            confidence=0.99,
            status="banana",
        )