# Pydantic validate JSON output - Định nghĩa contract của API

from pydantic import BaseModel, Field

class SegmentationResult(BaseModel):
    status: str


class ExtractionItem(BaseModel):
    status: str
    source_detections: int


class ExtractionResult(BaseModel):
    brand: ExtractionItem
    size: ExtractionItem
    pattern: ExtractionItem


class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: list[int]


class DetectionResult(BaseModel):
    status: str
    items: list[DetectionItem]


class OCRItem(BaseModel):
    text: str | None
    confidence: float
    status: str


class OCRResult(BaseModel):
    brand: OCRItem
    size: OCRItem
    pattern: OCRItem


class InspectionResponse(BaseModel):
    inspection_id: int
    
    image: str
    status: str

    segmentation: SegmentationResult

    extraction: ExtractionResult

    detections: DetectionResult

    ocr: OCRResult

    processing_time_ms: float = Field(
        ge=0
    )