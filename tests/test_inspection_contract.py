from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.db.database import get_db
from app.repositories.inspection_repository import InspectionRepository
from app.services.inspection_processing_service import InspectionProcessingService
from app.services.inspection_service import InspectionService


API_PATH = "/api/v1/inspection"


def _setup_app(db_session):
    """Use the same application state pattern as the existing inspection API tests."""
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

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
            }

    class FakeImageStorage:
        def save(self, image_path: Path) -> str:
            return "inspections/2026/09/test.jpg"

        def resolve(self, storage_key: str) -> Path:
            return Path("test.jpg")

    app.state.inspection_pipeline = FakePipeline()
    app.state.image_storage = FakeImageStorage()


def _teardown_app():
    app.dependency_overrides.clear()

    for state_name in ("inspection_pipeline", "image_storage"):
        if hasattr(app.state, state_name):
            delattr(app.state, state_name)


def _canonical_success_payload(inspection_id: int = 1) -> dict:
    return {
        "inspection_id": inspection_id,
        "image": "inspections/2026/09/test.jpg",
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
    }


def test_openapi_exposes_inspection_response_contract():
    """The public OpenAPI document must expose the actual /api/v1 inspection routes."""
    schema = app.openapi()

    assert API_PATH in schema["paths"]
    assert f"{API_PATH}/{{inspection_id}}" in schema["paths"]

    post_operation = schema["paths"][API_PATH]["post"]
    get_operation = schema["paths"][f"{API_PATH}/{{inspection_id}}"]["get"]

    assert (
        post_operation["responses"]["202"]["content"]["application/json"]["schema"][
            "$ref"
        ]
        == "#/components/schemas/PendingInspectionResponse"
    )

    assert (
        get_operation["responses"]["200"]["content"]["application/json"]["schema"][
            "$ref"
        ]
        == "#/components/schemas/InspectionResponse"
    )

    assert "422" in post_operation["responses"]
    assert "422" in get_operation["responses"]


def test_inspection_response_model_accepts_canonical_success_payload():
    """The canonical persisted response must validate against InspectionResponse."""
    from app.schemas.inspection import InspectionResponse

    payload = _canonical_success_payload()

    result = InspectionResponse.model_validate(payload)

    assert result.inspection_id == 1
    assert result.status.value == "success"
    assert result.extraction.brand.source_detections == 2
    assert result.ocr.brand.text == "DPLUS"
    assert result.processing_time_ms == 1856.53


def test_post_then_worker_then_get_preserves_lifecycle(
    db_session,
    monkeypatch,
):
    """
    Verify the real API lifecycle:
    POST -> pending DB row -> simulated worker -> GET completed result.
    """
    _setup_app(db_session)

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
            API_PATH,
            files={
                "image": (
                    "test.jpg",
                    b"fake-image-data",
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 202

        pending = response.json()
        inspection_id = pending["inspection_id"]

        assert pending == {
            "inspection_id": inspection_id,
            "status": "pending",
        }
        assert queued["inspection_id"] == inspection_id

        repository = InspectionRepository(db_session)
        inspection = repository.get_run(inspection_id)

        assert inspection is not None
        assert inspection.status == "pending"
        assert inspection.segmentation_status == "pending"
        assert inspection.detection_status == "pending"

        service = InspectionService(
            db=db_session,
            pipeline=app.state.inspection_pipeline,
            image_store=app.state.image_storage,
        )

        result = service.processing_service.process_new_run(
            inspection_id=inspection_id,
            image_path=Path("test.jpg"),
            started_at=None,
            storage_key="inspections/2026/09/test.jpg",
        )

        assert result is not None
        assert result["status"] == "success"

        get_response = client.get(
            f"{API_PATH}/{inspection_id}"
        )

        assert get_response.status_code == 200

        completed = get_response.json()

        assert completed["inspection_id"] == inspection_id
        assert completed["status"] == "success"
        assert completed["segmentation"]["status"] == "success"
        assert completed["detections"]["status"] == "success"
        assert completed["ocr"]["brand"]["text"] == "DPLUS"

    finally:
        _teardown_app()


def test_get_inspection_unexpected_exception_returns_500(
    db_session,
    monkeypatch,
):
    """GET must use the global internal-error response contract."""
    _setup_app(db_session)

    try:
        def explode(*args, **kwargs):
            raise RuntimeError("SECRET_INTERNAL_ERROR")

        monkeypatch.setattr(
            InspectionService,
            "get_inspection",
            explode,
        )

        client = TestClient(
            app,
            raise_server_exceptions=False,
        )

        response = client.get(
            f"{API_PATH}/1"
        )

        assert response.status_code == 500

        assert response.json() == {
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An internal server error occurred.",
                "details": None,
            }
        }

        assert "SECRET_INTERNAL_ERROR" not in response.text

    finally:
        _teardown_app()


def test_post_inspection_cleans_temporary_upload(
    db_session,
    monkeypatch,
    tmp_path,
):
    """Temporary API uploads must be removed after the request completes."""
    _setup_app(db_session)

    try:
        import app.api.v1.inspection as inspection_api

        monkeypatch.setattr(
            inspection_api,
            "API_INPUT_DIR",
            tmp_path,
        )

        queued = {}

        def fake_delay(inspection_id):
            queued["inspection_id"] = inspection_id

        monkeypatch.setattr(
            "app.api.v1.inspection.process_inspection.delay",
            fake_delay,
        )

        client = TestClient(app)

        response = client.post(
            API_PATH,
            files={
                "image": (
                    "cleanup.jpg",
                    b"fake-image-data",
                    "image/jpeg",
                )
            },
        )

        # POST is asynchronous: the contract is 202 Accepted.
        assert response.status_code == 202

        data = response.json()

        assert data["status"] == "pending"
        assert data["inspection_id"] == queued["inspection_id"]

        # The endpoint must clean up the temporary upload.
        assert list(tmp_path.iterdir()) == []

    finally:
        _teardown_app()