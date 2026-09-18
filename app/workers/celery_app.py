from celery import Celery

from app.core.config import REDIS_URL


celery_app = Celery(
    "drc_ai_tire",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.workers.inspection_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    timezone="Asia/Ho_Chi_Minh",
    enable_utc=False,
    
    task_track_started=True,
    
    # Reliability
    task_acks_late=True,
    task_acks_on_failure_or_timeout=True,
    worker_prefetch_multiplier=1,

    # Prevent result backend from growing indefinitely.
    result_expires=3600,

    # Do not keep retrying a task forever.
    task_default_max_retries=2,

    # Reject tasks that were not acknowledged when a worker dies.
    task_reject_on_worker_lost=True,
)