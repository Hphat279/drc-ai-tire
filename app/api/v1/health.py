from fastapi import APIRouter, Request


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get("")
async def health_check(
    request: Request,
):
    """
    Check whether the API and AI pipeline are ready.
    """

    pipeline_ready = hasattr(
        request.app.state,
        "inspection_pipeline",
    )

    service_ready = hasattr(
        request.app.state,
        "inspection_service",
    )

    if pipeline_ready and service_ready:
        return {
            "status": "ok",
            "service": "DRC AI Tire Inspection API",
            "pipeline": "ready",
        }

    return {
        "status": "starting",
        "service": "DRC AI Tire Inspection API",
        "pipeline": "not_ready",
    }