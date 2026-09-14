from app.core.config import INSPECTION_IMAGE_DIR
from app.storage import LocalImageStorage


def create_image_storage() -> LocalImageStorage:
    return LocalImageStorage(root_dir=INSPECTION_IMAGE_DIR)