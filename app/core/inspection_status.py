from enum import StrEnum


class InspectionStatus(StrEnum):
    PENDING = "pending"         # Record đã tạo nhưng chưa bắt đầu xử lý
    PROCESSING = "processing"   # AI pipeline đang chạy
    SUCCESS = "success"         # Đọc đủ 3 field
    PARTIAL = "partial"         # Pipeline chạy xong nhưng thiếu/failed một hoặc nhiều field
    FAILED = "failed"           # Pipeline/system failure khiến inspection không hoàn tất


class InspectionStageStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

class ExtractionStatus(StrEnum):
    DIRECT = "direct"
    RECONSTRUCTED = "reconstructed"
    NOT_DETECTED = "not_detected"


class OCRStatus(StrEnum):
    DIRECT = "direct"
    RECONSTRUCTED = "reconstructed"
    NOT_DETECTED = "not_detected"
    OCR_FAILED = "ocr_failed"


class DetectionStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    NOT_DETECTED = "not_detected"