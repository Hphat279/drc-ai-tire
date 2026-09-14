import pytest
from pydantic import ValidationError

from app.schemas.inspection import (
    DetectionItem,
    ExtractionItem,
    InspectionResponse,
    OCRItem,
    PendingInspectionResponse,
)


def test_detection_item_valid():
    item = DetectionItem(
        class_id=0,
        class_name="size",
        confidence=0.9927,
        bbox=[100, 200, 300, 250],
    )

    assert item.confidence == 0.9927
    assert item.bbox == [100, 200, 300, 250]


@pytest.mark.parametrize("confidence", [-0.01, 1.01, 2.0])
def test_detection_confidence_must_be_between_zero_and_one(confidence):
    with pytest.raises(ValidationError):
        DetectionItem(
            class_id=0,
            class_name="size",
            confidence=confidence,
            bbox=[100, 200, 300, 250],
        )


@pytest.mark.parametrize(
    "bbox",
    [
        [100, 200],
        [100, 200, 300],
        [100, 200, 300, 250, 400],
    ],
)
def test_detection_bbox_must_contain_exactly_four_values(bbox):
    with pytest.raises(ValidationError):
        DetectionItem(
            class_id=0,
            class_name="size",
            confidence=0.99,
            bbox=bbox,
        )


def test_detection_class_id_cannot_be_negative():
    with pytest.raises(ValidationError):
        DetectionItem(
            class_id=-1,
            class_name="size",
            confidence=0.99,
            bbox=[100, 200, 300, 250],
        )


def test_extraction_source_detections_cannot_be_negative():
    with pytest.raises(ValidationError):
        ExtractionItem(
            status="direct",
            source_detections=-1,
        )


def test_ocr_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        OCRItem(
            text="DPLUS",
            confidence=1.5,
            status="direct",
        )


def test_inspection_id_must_be_positive():
    with pytest.raises(ValidationError):
        InspectionResponse(
            inspection_id=0,
            image="inspections/2026/09/test.jpg",
            status="success",
            segmentation={"status": "success"},
            extraction={
                "brand": {
                    "status": "reconstructed",
                    "source_detections": 2,
                },
                "size": {
                    "status": "direct",
                    "source_detections": 1,
                },
                "pattern": {
                    "status": "direct",
                    "source_detections": 1,
                },
            },
            detections={
                "status": "success",
                "items": [
                    {
                        "class_id": 0,
                        "class_name": "size",
                        "confidence": 0.99,
                        "bbox": [100, 200, 300, 250],
                    }
                ],
            },
            ocr={
                "brand": {
                    "text": "DPLUS",
                    "confidence": 0.99,
                    "status": "reconstructed",
                },
                "size": {
                    "text": "100/90-14",
                    "confidence": 0.99,
                    "status": "direct",
                },
                "pattern": {
                    "text": "121",
                    "confidence": 0.99,
                    "status": "direct",
                },
            },
            processing_time_ms=100.0,
        )


def test_processing_time_cannot_be_negative():
    with pytest.raises(ValidationError):
        InspectionResponse(
            inspection_id=1,
            image="inspections/2026/09/test.jpg",
            status="success",
            segmentation={"status": "success"},
            extraction={
                "brand": {
                    "status": "reconstructed",
                    "source_detections": 2,
                },
                "size": {
                    "status": "direct",
                    "source_detections": 1,
                },
                "pattern": {
                    "status": "direct",
                    "source_detections": 1,
                },
            },
            detections={
                "status": "success",
                "items": [],
            },
            ocr={
                "brand": {
                    "text": "DPLUS",
                    "confidence": 0.99,
                    "status": "reconstructed",
                },
                "size": {
                    "text": "100/90-14",
                    "confidence": 0.99,
                    "status": "direct",
                },
                "pattern": {
                    "text": "121",
                    "confidence": 0.99,
                    "status": "direct",
                },
            },
            processing_time_ms=-1.0,
        )
        
def test_pending_inspection_response():
    response = PendingInspectionResponse(
        inspection_id=123,
        status="pending",
    )

    assert response.inspection_id == 123
    assert response.status == "pending"