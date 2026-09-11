from pathlib import Path

import cv2
from paddleocr import PaddleOCR

from app.ai.ocr.preprocessing import prepare_ocr_crop


class TireOCR:
    """Mô hình nhận diện PaddleOCR tùy chỉnh dành cho văn bản trên lốp xe."""

    def __init__(
        self,
        model_dir: str | Path,
        dictionary_path: str | Path,
        use_gpu: bool = False,
    ):
        self.model_dir = str(model_dir)
        self.dictionary_path = str(dictionary_path)

        self.ocr = PaddleOCR(
            rec_model_dir=self.model_dir,
            rec_char_dict_path=self.dictionary_path,
            use_angle_cls=False,
            use_gpu=use_gpu,
        )

    def read_crop(
        self,
        crop_img,
    ) -> tuple[str | None, float]:
        """
        Chạy OCR trên một crop.

        Returns:
            tuple:
                text: Kết quả OCR hoặc None nếu không nhận diện được.
                confidence: Confidence của OCR, 0.0 nếu không có kết quả.
        """

        if crop_img is None:
            return None, 0.0

        h, w = crop_img.shape[:2]

        if h == 0 or w == 0:
            return None, 0.0

        crop_img = prepare_ocr_crop(crop_img)

        result = self.ocr.ocr(
            crop_img,
            det=False,
            cls=False,
        )

        if not result:
            return None, 0.0

        if not result[0]:
            return None, 0.0

        text = result[0][0][0]
        confidence = float(result[0][0][1])

        return text, confidence

    def read_all(
        self,
        crop_results: dict,
    ) -> dict:
        """
        Chạy OCR cho brand, size và pattern.

        Status ở đây chỉ phản ánh trạng thái OCR:
            - success: Có crop và OCR nhận diện được text.
            - not_detected: Không có crop để OCR.
            - ocr_failed: Có crop nhưng OCR không trả về text.

        Trạng thái direct/reconstructed được xác định ở pipeline,
        vì extraction.py hiện tại chỉ trả crop ảnh.
        """

        final_result = {}

        for key in [
            "size",
            "pattern",
            "brand",
        ]:
            crop = crop_results.get(key)

            if crop is None:
                final_result[key] = {
                    "text": None,
                    "confidence": 0.0,
                    "status": "not_detected",
                }
                continue

            text, confidence = self.read_crop(crop)

            if text is None:
                status = "ocr_failed"
            else:
                status = "success"

            final_result[key] = {
                "text": text,
                "confidence": confidence,
                "status": status,
            }

        return final_result