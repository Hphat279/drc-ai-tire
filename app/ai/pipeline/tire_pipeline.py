from pathlib import Path
import time

from app.ai.vision.detection import TireDetector
from app.ai.vision.segmentation import TireSegmenter
from app.ai.ocr.paddle_ocr import TireOCR

from app.ai.pipeline.stages import (
    run_segmentation,
    run_unwrap,
    run_preprocessing,
    run_detection,
    run_extraction,
    run_ocr,
)

from app.ai.pipeline.output import (
    validate_output_dirs,
    save_image,
    save_ocr_crops,
    save_result_json,
)

from app.ai.pipeline.result import build_pipeline_result


class TireInspectionPipeline:
    """
    Main orchestrator for the AI tire inspection pipeline.

    This class is responsible only for:
        1. Initializing AI components.
        2. Executing pipeline stages in order.
        3. Passing data between stages.
        4. Returning the final standardized result.

    Detailed business logic belongs to the corresponding
    stage modules.
    """

    def __init__(
        self,
        segmentation_model_path: str | Path,
        detection_model_path: str | Path,
        ocr_model_dir: str | Path,
        ocr_dictionary_path: str | Path,
        output_dir: str | Path,
        use_gpu: bool = False,
    ):
        self.segmenter = TireSegmenter(
            segmentation_model_path
        )

        self.detector = TireDetector(
            detection_model_path
        )

        self.ocr = TireOCR(
            model_dir=ocr_model_dir,
            dictionary_path=ocr_dictionary_path,
            use_gpu=use_gpu,
        )

        self.output_dir = Path(output_dir)

        validate_output_dirs(self.output_dir)

    def run(
        self,
        image_path: str | Path,
    ) -> dict:
        """
        Execute the complete AI inspection pipeline.

        Pipeline:

            1. YOLO11 Segmentation
            2. Tire Unwrapping
            3. Sharpen + CLAHE
            4. YOLO11 Detection
            5. Text Extraction
            6. PaddleOCR
            7. Result Generation
        """

        start_time = time.perf_counter()

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image_name = image_path.stem

        # ==================================================
        # 1. YOLO11 SEGMENTATION
        # ==================================================

        print("\n[1/6] YOLO11-Segmentation...")

        extracted_tire = run_segmentation(
            segmenter=self.segmenter,
            image_path=image_path,
        )

        segmentation_output = (
            self.output_dir
            / "segmentation"
            / f"{image_name}_segmented.png"
        )

        save_image(
            image=extracted_tire,
            output_path=segmentation_output,
            error_message="Failed to save segmentation output.",
        )

        print(
            f"      Saved: {segmentation_output}"
        )

        # ==================================================
        # 2. UNWRAP
        # ==================================================

        print("[2/6] Unwrapping tire...")

        unwrapped_img = run_unwrap(
            extracted_tire
        )

        if unwrapped_img is None:
            raise RuntimeError(
                "Failed to unwrap tire."
            )

        extraction_output = (
            self.output_dir
            / "extraction"
            / f"{image_name}_unwrapped.png"
        )

        save_image(
            image=unwrapped_img,
            output_path=extraction_output,
            error_message="Failed to save extraction output.",
        )

        print(
            f"      Saved: {extraction_output}"
        )

        # ==================================================
        # 3. SHARPEN + CLAHE
        # ==================================================

        print("[3/6] Sharpen + CLAHE...")

        processed_img = run_preprocessing(
            unwrapped_img
        )

        preprocessing_output = (
            self.output_dir
            / "preprocessing"
            / f"{image_name}_enhanced.png"
        )

        save_image(
            image=processed_img,
            output_path=preprocessing_output,
            error_message="Failed to save preprocessing output.",
        )

        print(
            f"      Saved: {preprocessing_output}"
        )

        # ==================================================
        # 4. YOLO11 DETECTION
        # ==================================================

        print("[4/6] YOLO11 Detection...")

        detect_input, detections, detection_vis = (
            run_detection(
                detector=self.detector,
                processed_img=processed_img,
            )
        )

        detection_output = (
            self.output_dir
            / "detection"
            / f"{image_name}_detected.png"
        )

        save_image(
            image=detection_vis,
            output_path=detection_output,
            error_message="Failed to save detection output.",
        )

        print(
            f"      Saved: {detection_output}"
        )

        # ==================================================
        # 5. TEXT EXTRACTION
        # ==================================================

        print("[5/6] Text extraction...")

        extraction_results, crop_results = (
            run_extraction(
                detect_input=detect_input,
                detections=detections,
            )
        )

        save_ocr_crops(
            output_dir=self.output_dir,
            image_name=image_name,
            extraction_results=extraction_results,
        )

        # ==================================================
        # 6. PADDLEOCR
        # ==================================================

        print("[6/6] PaddleOCR...")

        ocr_results = run_ocr(
            ocr=self.ocr,
            crop_results=crop_results,
            extraction_results=extraction_results,
        )

        # ==================================================
        # FINAL RESULT
        # ==================================================

        processing_time_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        result = build_pipeline_result(
            image_path=image_path,
            detections=detections,
            extraction_results=extraction_results,
            ocr_results=ocr_results,
            processing_time_ms=processing_time_ms,
        )

        save_result_json(
            output_dir=self.output_dir,
            image_name=image_name,
            result=result,
        )

        return result