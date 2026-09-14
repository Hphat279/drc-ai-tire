# from config import settings
# print(settings.database_url)
# print(settings.secret_key)

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

APP_NAME = os.environ["APP_NAME"]

DATABASE_URL = os.environ["DATABASE_URL"]

SEGMENTATION_MODEL_PATH = Path(
    os.environ["SEGMENTATION_MODEL_PATH"]
)

DETECTION_MODEL_PATH = Path(
    os.environ["DETECTION_MODEL_PATH"]
)

OCR_MODEL_DIR = Path(
    os.environ["OCR_MODEL_DIR"]
)

OCR_DICTIONARY_PATH = Path(
    os.environ["OCR_DICTIONARY_PATH"]
)

OUTPUT_DIR = Path(
    os.environ["OUTPUT_DIR"]
)

API_INPUT_DIR = Path(
    os.environ["API_INPUT_DIR"]
)

INSPECTION_IMAGE_DIR = Path(
    os.environ["INSPECTION_IMAGE_DIR"]
)

USE_GPU = (
    os.environ["USE_GPU"].lower() == "true"
)

REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0",
)