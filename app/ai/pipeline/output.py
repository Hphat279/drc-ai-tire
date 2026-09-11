import json
from pathlib import Path

import cv2


def validate_output_dirs(
    output_dir: Path,
):
    """
    Validate required output directories.
    """

    required_dirs = [
        output_dir / "segmentation",
        output_dir / "extraction",
        output_dir / "preprocessing",
        output_dir / "detection",
        output_dir / "ocr",
    ]

    for directory in required_dirs:
        if not directory.exists():
            raise FileNotFoundError(
                "Required output directory "
                f"does not exist: {directory}"
            )


def save_image(
    image,
    output_path: Path,
    error_message: str,
):
    """
    Save an image to disk.
    """

    if not cv2.imwrite(
        str(output_path),
        image,
    ):
        raise RuntimeError(
            error_message
        )


def save_ocr_crops(
    output_dir: Path,
    image_name: str,
    extraction_results: dict,
):
    """
    Save crops produced by extraction
    for OCR inspection/debugging.
    """

    for class_name in [
        "brand",
        "size",
        "pattern",
    ]:
        crop = extraction_results[
            class_name
        ]["crop"]

        if crop is None:
            continue

        output_path = (
            output_dir
            / "ocr"
            / f"{image_name}_{class_name}.png"
        )

        save_image(
            image=crop,
            output_path=output_path,
            error_message=(
                "Failed to save OCR crop: "
                f"{output_path}"
            ),
        )


def save_result_json(
    output_dir: Path,
    image_name: str,
    result: dict,
):
    """
    Save standardized pipeline result as JSON.
    """

    json_output = (
        output_dir
        / "ocr"
        / f"{image_name}_result.json"
    )

    with open(
        json_output,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"      Saved: {json_output}"
    )