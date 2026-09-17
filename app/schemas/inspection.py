# Pydantic validate JSON output - Định nghĩa contract của API

from pydantic import BaseModel, Field
from typing import Literal

from app.core.inspection_status import (
    DetectionStatus,
    ExtractionStatus,
    InspectionStageStatus,
    InspectionStatus,
    OCRStatus,
)

class SegmentationResult(BaseModel):
    status: InspectionStageStatus


class ExtractionItem(BaseModel):
    status: ExtractionStatus
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
    status: DetectionStatus
    items: list[DetectionItem]


class OCRItem(BaseModel):
    text: str | None
    confidence: float = Field(ge=0, le=1)
    status: OCRStatus


class OCRResult(BaseModel):
    brand: OCRItem
    size: OCRItem
    pattern: OCRItem


class InspectionResponse(BaseModel):
    inspection_id: int = Field(ge=1)
    
    image: str = Field(min_length=1)
    status: InspectionStatus

    segmentation: SegmentationResult

    extraction: ExtractionResult

    detections: DetectionResult

    ocr: OCRResult

    processing_time_ms: float = Field(ge=0)
    
class PendingInspectionResponse(BaseModel):
    inspection_id: int = Field(ge=1)
    status :Literal["pending"]= "pending"