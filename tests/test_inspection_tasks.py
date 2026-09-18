from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import TransientInfrastructureError
from app.workers import inspection_tasks


@pytest.fixture
def task():
    return inspection_tasks.process_inspection


@pytest.fixture
def fake_db():
    return MagicMock()


@pytest.fixture
def fake_service():
    return MagicMock()


def configure_task_request(task, retries=0):
    task.request.retries = retries


def test_process_inspection_success(
    task,
    fake_db,
    fake_service,
):
    expected = {
        "inspection_id": 1,
        "status": "success",
    }

    fake_service.process_inspection.return_value = expected

    with (
        patch(
            "app.workers.inspection_tasks.SessionLocal",
            return_value=fake_db,
        ),
        patch(
            "app.workers.inspection_tasks.InspectionService",
            return_value=fake_service,
        ),
        patch(
            "app.workers.inspection_tasks.get_inspection_pipeline",
            return_value=MagicMock(),
        ),
        patch(
            "app.workers.inspection_tasks.get_image_storage",
            return_value=MagicMock(),
        ),
    ):
        result = task.run(1)

    assert result == expected
    fake_service.process_inspection.assert_called_once_with(1)
    fake_db.close.assert_called_once()


def test_transient_error_triggers_retry(
    task,
    fake_db,
    fake_service,
):
    error = TransientInfrastructureError(
        "Temporary infrastructure failure."
    )

    fake_service.process_inspection.side_effect = error

    retry_mock = MagicMock(
        side_effect=RuntimeError("retry-called")
    )

    configure_task_request(task, retries=0)

    with (
        patch(
            "app.workers.inspection_tasks.SessionLocal",
            return_value=fake_db,
        ),
        patch(
            "app.workers.inspection_tasks.InspectionService",
            return_value=fake_service,
        ),
        patch(
            "app.workers.inspection_tasks.get_inspection_pipeline",
            return_value=MagicMock(),
        ),
        patch(
            "app.workers.inspection_tasks.get_image_storage",
            return_value=MagicMock(),
        ),
        patch.object(task, "retry", retry_mock),
        pytest.raises(RuntimeError, match="retry-called"),
    ):
        task.run(1)

    retry_mock.assert_called_once_with(
        exc=error,
        countdown=5,
    )
    fake_db.close.assert_called_once()


def test_transient_error_second_retry_uses_10_seconds(
    task,
    fake_db,
    fake_service,
):
    error = TransientInfrastructureError(
        "Temporary infrastructure failure."
    )

    fake_service.process_inspection.side_effect = error

    retry_mock = MagicMock(
        side_effect=RuntimeError("retry-called")
    )

    configure_task_request(task, retries=1)

    with (
        patch(
            "app.workers.inspection_tasks.SessionLocal",
            return_value=fake_db,
        ),
        patch(
            "app.workers.inspection_tasks.InspectionService",
            return_value=fake_service,
        ),
        patch(
            "app.workers.inspection_tasks.get_inspection_pipeline",
            return_value=MagicMock(),
        ),
        patch(
            "app.workers.inspection_tasks.get_image_storage",
            return_value=MagicMock(),
        ),
        patch.object(task, "retry", retry_mock),
        pytest.raises(RuntimeError, match="retry-called"),
    ):
        task.run(1)

    retry_mock.assert_called_once_with(
        exc=error,
        countdown=10,
    )
    fake_db.close.assert_called_once()


def test_business_error_is_not_retried(
    task,
    fake_db,
    fake_service,
):
    error = ValueError("Invalid inspection state.")

    fake_service.process_inspection.side_effect = error

    retry_mock = MagicMock()

    configure_task_request(task, retries=0)

    with (
        patch(
            "app.workers.inspection_tasks.SessionLocal",
            return_value=fake_db,
        ),
        patch(
            "app.workers.inspection_tasks.InspectionService",
            return_value=fake_service,
        ),
        patch(
            "app.workers.inspection_tasks.get_inspection_pipeline",
            return_value=MagicMock(),
        ),
        patch(
            "app.workers.inspection_tasks.get_image_storage",
            return_value=MagicMock(),
        ),
        patch.object(task, "retry", retry_mock),
        pytest.raises(ValueError, match="Invalid inspection state."),
    ):
        task.run(1)

    retry_mock.assert_not_called()
    fake_db.close.assert_called_once()


def test_pipeline_stage_error_is_not_retried(
    task,
    fake_db,
    fake_service,
):
    from app.core.pipeline_errors import PipelineStageError

    error = PipelineStageError(
        stage="ocr",
        code="OCR_FAILED",
        message="OCR failed.",
    )

    fake_service.process_inspection.side_effect = error

    retry_mock = MagicMock()

    configure_task_request(task, retries=0)

    with (
        patch(
            "app.workers.inspection_tasks.SessionLocal",
            return_value=fake_db,
        ),
        patch(
            "app.workers.inspection_tasks.InspectionService",
            return_value=fake_service,
        ),
        patch(
            "app.workers.inspection_tasks.get_inspection_pipeline",
            return_value=MagicMock(),
        ),
        patch(
            "app.workers.inspection_tasks.get_image_storage",
            return_value=MagicMock(),
        ),
        patch.object(task, "retry", retry_mock),
        pytest.raises(PipelineStageError),
    ):
        task.run(1)

    retry_mock.assert_not_called()
    fake_db.close.assert_called_once()


def test_transient_error_always_closes_db(
    task,
    fake_db,
    fake_service,
):
    error = TransientInfrastructureError(
        "Temporary infrastructure failure."
    )

    fake_service.process_inspection.side_effect = error

    retry_mock = MagicMock(
        side_effect=RuntimeError("retry-called")
    )

    configure_task_request(task, retries=0)

    with (
        patch(
            "app.workers.inspection_tasks.SessionLocal",
            return_value=fake_db,
        ),
        patch(
            "app.workers.inspection_tasks.InspectionService",
            return_value=fake_service,
        ),
        patch(
            "app.workers.inspection_tasks.get_inspection_pipeline",
            return_value=MagicMock(),
        ),
        patch(
            "app.workers.inspection_tasks.get_image_storage",
            return_value=MagicMock(),
        ),
        patch.object(task, "retry", retry_mock),
        pytest.raises(RuntimeError, match="retry-called"),
    ):
        task.run(1)

    fake_db.close.assert_called_once()
