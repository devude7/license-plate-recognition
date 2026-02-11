from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

_PL_ALLOWED = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def normalize_plate_text(text: str) -> str:
    text = (text or "").upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


@dataclass
class OCRCandidate:
    text: str
    text_norm: str
    confidence: float


@dataclass
class OCRResult:
    best_text: str
    best_text_norm: str
    best_confidence: float
    candidates: list[OCRCandidate]


def _pil_from_any(img: np.ndarray) -> Image.Image:
    if img is None or img.size == 0:
        raise ValueError("Empty image")

    if img.ndim == 2: 
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = img[:, :, ::-1]
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    return Image.fromarray(img.astype(np.uint8), mode="RGB")


def _filter_allowlist(text: str, allow: str = _PL_ALLOWED) -> str:
    text = (text or "").upper()
    return "".join(c for c in text if c in allow)


class TrOCREngine:
    def __init__(self, *, model_name: str = "microsoft/trocr-base-printed", device: Optional[str] = None,) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.processor = TrOCRProcessor.from_pretrained(model_name, use_fast=False)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def read_plate(
        self,
        img_bgr_or_gray: np.ndarray,
        *,
        preprocess: bool = True,
        num_beams: int = 4,
        max_new_tokens: int = 16,
    ) -> OCRResult:
        if img_bgr_or_gray is None or img_bgr_or_gray.size == 0:
            return OCRResult("", "", 0.0, [])

        img = img_bgr_or_gray

        if preprocess:
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        pil = _pil_from_any(img)

        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)

        out = self.model.generate(
            **inputs,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )

        text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
        text_norm = normalize_plate_text(_filter_allowlist(text))

        conf = 0.0
        if getattr(out, "scores", None):
            probs: list[float] = []
            for step_logits in out.scores:
                p = torch.softmax(step_logits[0], dim=-1)
                probs.append(float(torch.max(p).item()))
            conf = float(sum(probs) / len(probs)) if probs else 0.0

        if not text_norm:
            return OCRResult("", "", 0.0, [])

        cand = OCRCandidate(text=text, text_norm=text_norm, confidence=conf)
        return OCRResult(
            best_text=cand.text,
            best_text_norm=cand.text_norm,
            best_confidence=cand.confidence,
            candidates=[cand],
        )