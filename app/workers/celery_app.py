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
)