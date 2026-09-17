import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import APIError


logger = logging.getLogger(__name__)


async def api_error_handler(
    request: Request,
    exc: APIError,
):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
):
    
    errors = []
    
    for error in exc.errors():
        normalized_error = dict(error)
        
        ctx = normalized_error.get("ctx")
        
        if isinstance(ctx, dict):
            normalized_ctx = dict(ctx)
            
            if "error" in normalized_ctx:
                normalized_ctx["error"] = str(
                    normalized_ctx["error"]
                )
                
            normalized_error["ctx"] = normalized_ctx
            
        errors.append(normalized_error)
        
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed.",
                "details": errors,
            }
        },
    )


async def generic_error_handler(
    request: Request,
    exc: Exception,
):
    logger.exception(
        "Unhandled exception: %s",
        exc,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An internal server error occurred.",
                "details": None,
            }
        },
    )