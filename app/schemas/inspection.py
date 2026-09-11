# Pydantic validate JSON output - Định nghĩa contract của API

from pydantic import BaseModel, Field

class SegmentationResult(BaseModel):
    status: str = Field(min_length=1)


class ExtractionItem(BaseModel):
    status: str = Field(min_length=1)
    source_detections: int = Field(ge=0)


class ExtractionResult(BaseModel):
    brand: ExtractionItem
    size: ExtractionItem
    pattern: ExtractionItem


class DetectionItem(BaseModel):
    class_id: int = Field(ge=0) # >=0
    class_name: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1) # >=0, <=1
    bbox: list[int] = Field(min_length=4, max_length=4)


class DetectionResult(BaseModel):
    status: str = Field(min_length=1)
    items: list[DetectionItem]


class OCRItem(BaseModel):
    text: str | None
    confidence: float = Field(ge=0, le=1)
    status: str = Field(min_length=1)


class OCRResult(BaseModel):
    brand: OCRItem
    size: OCRItem
    pattern: OCRItem


class InspectionResponse(BaseModel):
    inspection_id: int = Field(ge=1)
    
    image: str = Field(min_length=1)
    status: str = Field(min_length=1)

    segmentation: SegmentationResult

    extraction: ExtractionResult

    detections: DetectionResult

    ocr: OCRResult

    processing_time_ms: float = Field(ge=0)