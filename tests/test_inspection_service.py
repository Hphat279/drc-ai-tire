from pathlib import Path

from app.services.inspection_service import InspectionService

from app.models import InspectionRun

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
                        "class_name": "size",
                        "confidence": 0.9927,
                        "bbox": [100, 200, 300, 250],
                    },
                    {
                        "class_name": "pattern",
                        "confidence": 0.9991,
                        "bbox": [400, 200, 500, 250],
                    },
                    {
                        "class_name": "brand",
                        "confidence": 0.9971,
                        "bbox": [0, 100, 234, 172],
                    },
                    {
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


class FakeImageStore:
    def save(self, image_path: Path) -> str:
        return "inspections/2026/09/test.jpg"


def test_inspect_success(db_session, tmp_path):
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"fake-image")

    service = InspectionService(
        db=db_session,
        pipeline=FakePipeline(),
        image_store=FakeImageStore(),
    )

    result = service.inspect(image_path)

    assert result["inspection_id"] is not None
    assert result["image"] == "inspections/2026/09/test.jpg"
    assert result["status"] == "success"
    assert result["segmentation"]["status"] == "success"
    assert result["detections"]["status"] == "success"
    assert result["processing_time_ms"] == 1856.53

    inspection = service.repository.get_run(
        result["inspection_id"]
    )

    assert inspection is not None
    assert inspection.image_path == "inspections/2026/09/test.jpg"
    assert inspection.status == "success"
    assert inspection.segmentation_status == "success"
    assert inspection.detection_status == "success"

    assert len(inspection.detections) == 4
    assert len(inspection.fields) == 3


def test_inspect_saves_detections(db_session, tmp_path):
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"fake-image")

    service = InspectionService(
        db=db_session,
        pipeline=FakePipeline(),
        image_store=FakeImageStore(),
    )

    result = service.inspect(image_path)

    inspection = service.repository.get_run(
        result["inspection_id"]
    )

    assert inspection is not None

    detections = inspection.detections

    assert len(detections) == 4

    assert detections[0].class_name == "size"
    assert detections[0].confidence == 0.9927
    assert detections[0].x1 == 100
    assert detections[0].y1 == 200
    assert detections[0].x2 == 300
    assert detections[0].y2 == 250


def test_inspect_saves_fields(db_session, tmp_path):
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"fake-image")

    service = InspectionService(
        db=db_session,
        pipeline=FakePipeline(),
        image_store=FakeImageStore(),
    )

    result = service.inspect(image_path)

    inspection = service.repository.get_run(
        result["inspection_id"]
    )

    assert inspection is not None

    fields = {
        field.field_name: field
        for field in inspection.fields
    }

    assert set(fields.keys()) == {
        "size",
        "pattern",
        "brand",
    }

    assert fields["size"].text == "100/90-14"
    assert fields["size"].confidence == 0.9927
    assert fields["size"].extraction_status == "direct"
    assert fields["size"].ocr_status == "direct"
    assert fields["size"].source_detections == 1

    assert fields["pattern"].text == "121"
    assert fields["pattern"].confidence == 0.9991
    assert fields["pattern"].extraction_status == "direct"
    assert fields["pattern"].ocr_status == "direct"
    assert fields["pattern"].source_detections == 1

    assert fields["brand"].text == "DPLUS"
    assert fields["brand"].confidence == 0.9971
    assert fields["brand"].extraction_status == "reconstructed"
    assert fields["brand"].ocr_status == "reconstructed"
    assert fields["brand"].source_detections == 2


def test_inspect_persists_failure_on_pipeline_error(db_session, tmp_path):
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"fake-image")

    class FailingPipeline:
        def run(self, image_path: Path) -> dict:
            raise RuntimeError("pipeline failed")

    service = InspectionService(
        db=db_session,
        pipeline=FailingPipeline(),
        image_store=FakeImageStore(),
    )

    result = service.inspect(image_path)

    assert result["status"] == "failed"

    inspection = (
        db_session.query(InspectionRun)
        .filter_by(id=result["inspection_id"])
        .one()
    )

    assert inspection.status == "failed"
    assert inspection.error_code == "INSPECTION_FAILED"
    assert inspection.error_message == "Inspection processing failed."
    assert inspection.failed_stage == "pipeline"
    assert inspection.started_at is not None
    assert inspection.completed_at is not None