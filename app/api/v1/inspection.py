from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from fastapi import (
    APIRouter, 
    Depends, 
    File,
    Request,
    UploadFile,
    status,
)   

from app.core.config import API_INPUT_DIR
from app.core.exceptions import APIError
from app.schemas.inspection import (
    InspectionResponse,
    PendingInspectionResponse,
)

from app.db.database import get_db

from app.services.inspection_service import InspectionService

from app.workers.inspection_tasks import process_inspection


router = APIRouter(
    prefix="/inspection",
    tags=["Inspection"],
)


ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

# =============================================================
# POST /api/v1/inspection
# =============================================================

@router.post(
    "",
    response_model=PendingInspectionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def inspect_tire(
    request : Request,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """"
    Kiểm tra hình ảnh lốp xe.
    Input:
        multipart/form-data
        image=<file>

    Output:
        Kết quả kiểm tra bằng AI ở định dạng có cấu trúc.
    """
    
    # =========================================================
    # 1. Validate filename
    # =========================================================
    
    if not image.filename:
        raise APIError(
            status_code=400,
            code="IMAGE_FILENAME_REQUIRED",
            message="Image filename is required.",
            details=None,
        )
    
    # =========================================================
    # 2. Validate extension
    # =========================================================
    
    extension = Path(
        image.filename
    ).suffix.lower()
    
    if extension not in ALLOWED_EXTENSIONS:
        raise APIError(
            status_code=400,
            code="UNSUPPORTED_IMAGE_FORMAT",
            message=(
                "Unsupported image format."
            ),
            details={
              "allowed_formats": [
                    "jpg",
                    "jpeg",
                    "png",
                    "webp",
                ],
            },
        )
    
    # =========================================================
    # 3. Request-scoped service
    # =========================================================
    
    service =InspectionService(
        db=db,
        image_store=request.app.state.image_storage,
    )
    
    # =========================================================
    # 4. Create temporary upload directory
    # =========================================================
    
    API_INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    save_filename = (
        f"{uuid4().hex}{extension}"
    )

    image_path = (
        API_INPUT_DIR
        / save_filename
    )
    
    try:
        # =====================================================
        # 5. Read upload
        # =====================================================
        
        content = await image.read()
        
        if not content:
            raise APIError(
                status_code=400,
                code="EMPTY_IMAGE",
                message="Uploaded image is empty.",
                details="Uploaded image is empty.",
            )
        
        # =====================================================
        # 6. Save temporary image
        # =====================================================

        image_path.write_bytes(
            content
        )
        
        # =====================================================
        # 7. Run inspection
        # =====================================================
        
        result = service.create_pending_inspection(
            image_path=image_path
        )
        try:
            process_inspection.delay(
                result["inspection_id"]
            )
        except Exception:
            service.mark_enqueue_failed(
                inspection_id=result["inspection_id"],
                error_message="Failed to enqueue inspection task.",
            )
            raise APIError(
                status_code=503,
                code="TASK_ENQUEUE_FAILED",
                message="Inspection task could not be queued.",
                details={
                    "inspection_id": result["inspection_id"],
                },
            )
        
        return result
    
    finally:
        # =====================================================
        # 8. Remove temporary upload
        # =====================================================
        
        if image_path.exists():
            image_path.unlink()
        
# =============================================================
# GET /api/v1/inspection/{inspection_id}
# =============================================================

@router.get(
    "/{inspection_id}",
    response_model=InspectionResponse,
)

def get_inspection(
    inspection_id: int,
    db: Session = Depends(get_db),
):
    """
    Lấy lại kết quả inspection từ PostgreSQL.
    """
    
    # =========================================================
    # 1. Request-scoped service
    # =========================================================

    service = InspectionService(
        db=db,
    )
    
    # =========================================================
    # 2. Query inspection
    # =========================================================
    
    result = service.get_inspection(
        inspection_id=inspection_id
    )
    
    # =========================================================
    # 3. Not found
    # =========================================================
    
    if result is None:
        raise APIError(
            status_code=404,
            code="INSPECTION_NOT_FOUND",
            message="Inspection not found.",
            details={
                "inspection_id": inspection_id,
            },
        )
    
    # =========================================================
    # 4. Return result
    # =========================================================
    
    return result 