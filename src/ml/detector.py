from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np 
import torch
from ultralytics import YOLO

from .ocr import OCRResult, TrOCREngine


MODEL_PATH = Path("src/models/best.pt")
CONF_THRESHOLD = 0.25
IOU_NMS = 0.45
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DetectionResult:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    ocr: OCRResult


@dataclass
class DetectorOutput:
    success: bool
    detection: Optional[DetectionResult]
    error: Optional[str] = None


class PlateDetector:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        *,
        conf_threshold: float = CONF_THRESHOLD,
        iou_nms: float = IOU_NMS,
        device: str = DEVICE,
        cut_left_px: int = 0,          
        cut_left_ratio: float = 0.0) -> None:
        
        if not model_path.exists():
            raise FileNotFoundError(str(model_path))

        self.model = YOLO(str(model_path))
        self.model.to(device)

        self.conf_threshold = conf_threshold
        self.iou_nms = iou_nms
        self.device = device

        self.ocr_engine = TrOCREngine(model_name="microsoft/trocr-base-printed")

        self.cut_left_px = max(0, int(cut_left_px))
        self.cut_left_ratio = float(cut_left_ratio)

    @staticmethod
    def _clip_bbox(x1, y1, x2, y2, w, h):
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        return x1, y1, x2, y2

    def _trim_eu_flag(self, crop_bgr: np.ndarray) -> tuple[np.ndarray, int]:
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr, 0

        _, w = crop_bgr.shape[:2]
        cut = self.cut_left_px

        if self.cut_left_ratio > 0.0:
            cut = max(cut, int(round(w * self.cut_left_ratio)))

        cut = min(cut, max(0, w - 5))  
        if cut <= 0:
            return crop_bgr, 0

        return crop_bgr[:, cut:].copy(), cut

    def detect_and_ocr(self, image_bgr: np.ndarray) -> DetectorOutput:
        if image_bgr is None or image_bgr.size == 0:
            return DetectorOutput(False, None, "Empty image")

        h, w = image_bgr.shape[:2]

        results = self.model.predict(
            source=image_bgr,
            conf=self.conf_threshold,
            iou=self.iou_nms,
            device=self.device,
            verbose=False,
        )

        if not results or len(results[0].boxes) == 0:
            return DetectorOutput(False, None, "No plate detected")

        boxes = results[0].boxes
        best_idx = int(torch.argmax(boxes.conf))
        box = boxes[best_idx]

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
        conf = float(box.conf[0].cpu().item())

        x1, y1, x2, y2 = self._clip_bbox(x1, y1, x2, y2, w, h)

        if x2 <= x1 or y2 <= y1:
            return DetectorOutput(False, None, "Invalid bbox")

        crop = image_bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return DetectorOutput(False, None, "Empty crop")

        crop_no_flag, cut = self._trim_eu_flag(crop)

        ocr_result = self.ocr_engine.read_plate(crop_no_flag, preprocess=False)

        x1_adj = x1 + cut
        x1_adj = min(x1_adj, x2 - 1)  

        det = DetectionResult(
            bbox_xyxy=(x1_adj, y1, x2, y2),
            confidence=conf,
            ocr=ocr_result,
        )

        return DetectorOutput(True, det)

