from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from src.database.db import ensure_schema, plate_exists
from src.ml.detector import DetectorOutput, PlateDetector


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.detector = PlateDetector(cut_left_ratio=0.1)
    ensure_schema()
    try:
        yield
    finally:
        pass


app = FastAPI(title="License Plate Recognition API", lifespan=lifespan)


class DetectionResponse(BaseModel):
    success: bool
    normalized_text: Optional[str] = None
    found: Optional[bool] = None


def _to_response(det_out: DetectorOutput, *, found: Optional[bool] = None, normalized_text: Optional[str] = None) -> DetectionResponse:
    if not det_out.success or det_out.detection is None:
        return DetectionResponse(
            success=False,
            normalized_text=None,
            found=None,
        )

    return DetectionResponse(success=True, normalized_text=normalized_text, found=found)


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
    out: DetectorOutput = await run_in_threadpool(detector.detect_and_ocr, img_bgr)

    found = None
    normalized_text = None
    if out.success and out.detection is not None:
        norm = out.detection.ocr.best_text_norm
        normalized_text = norm
        found = await run_in_threadpool(plate_exists, norm) if norm else False

    return _to_response(out, found=found, normalized_text=normalized_text)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

