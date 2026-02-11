from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.ml.detector import DetectorOutput, PlateDetector


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.detector = PlateDetector(cut_left_ratio=0.1)
    try:
        yield
    finally:
        pass


app = FastAPI(title="License Plate Recognition API", lifespan=lifespan)


class DetectionResponse(BaseModel):
    success: bool
    text: Optional[str] = None


def _to_response(det_out: DetectorOutput) -> DetectionResponse:
    if not det_out.success or det_out.detection is None:
        return DetectionResponse(
            success=False,
            text=None,
        )

    det = det_out.detection
    ocr = det.ocr

    return DetectionResponse(success=True, text=ocr.best_text)


@app.post("/detect", response_model=DetectionResponse)
async def detect_plate(image: UploadFile = File(...)) -> DetectionResponse:
    if image.content_type is not None and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    np_buf = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    detector: PlateDetector = app.state.detector
    out = detector.detect_and_ocr(img_bgr)

    return _to_response(out)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

