from app.core.ai_dependencies import create_inspection_pipeline
from app.core.exceptions import TransientInfrastructureError
from app.core.storage_dependencies import create_image_storage
from app.db.database import SessionLocal
from app.services.inspection_service import InspectionService
from app.workers.celery_app import celery_app


# Worker-process singletons.
# The AI models are initialized once per worker process,
# not once per task.
_pipeline = None
_image_storage = None


def get_inspection_pipeline():
    global _pipeline

    if _pipeline is None:
        _pipeline = create_inspection_pipeline()

    return _pipeline


def get_image_storage():
    global _image_storage

    if _image_storage is None:
        _image_storage = create_image_storage()

    return _image_storage


@celery_app.task(
    bind=True,
    name="inspection.process_inspection",
    max_retries=2,
    acks_late=True,
    track_started=True,
)
def process_inspection(
    self,
    inspection_id: int,
) -> dict | None:
    """
    Process an existing inspection asynchronously.

    Celery receives only the inspection ID. The worker
    loads the persisted inspection and resolves its image
    through the configured image storage.

    Transient infrastructure errors are retried up to
    two times with exponential backoff.

    Business and AI pipeline errors are handled by the
    inspection processing service and are not retried here.
    """

    db = SessionLocal()

    try:
        service = InspectionService(
            db=db,
            pipeline=get_inspection_pipeline(),
            image_store=get_image_storage(),
        )

        return service.process_inspection(
            inspection_id
        )

    except TransientInfrastructureError as exc:
        raise self.retry(
            exc=exc,
            countdown=5 * (2 ** self.request.retries),
        )

    finally:
        db.close()