"""Pure helpers for image handling, JSON parsing, and box overlays.

Kept independent of the API client so they're easy to unit-test and reuse.
"""
from __future__ import annotations

import io
import json
import re
from typing import Any

from PIL import Image, ImageDraw, ImageFont

MAX_BYTES = 1_500_000  # 1.5 MB ceiling before resize


def resize_if_needed(image_bytes: bytes, mime: str) -> tuple[bytes, str]:
    """Resize image if it exceeds MAX_BYTES.

    Returns (possibly-new-bytes, possibly-new-mime). Re-encodes to JPEG when
    shrinking PNGs to bound size; preserves format otherwise.
    On unreadable input, returns the original bytes/mime unchanged.
    """
    if len(image_bytes) <= MAX_BYTES:
        return image_bytes, mime

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except (Image.UnidentifiedImageError, OSError, ValueError):
        return image_bytes, mime

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    quality = 85
    max_dim = 1600
    while max_dim >= 400:
        resized = img.copy()
        resized.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= MAX_BYTES:
            return data, "image/jpeg"
        max_dim -= 200

    # Last-ditch: hardest thumbnail AND lowest acceptable quality
    final = img.copy()
    final.thumbnail((400, 400))
    buf = io.BytesIO()
    final.save(buf, format="JPEG", quality=60, optimize=True)
    return buf.getvalue(), "image/jpeg"


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_PARSE_INPUT_CAP = 200_000  # avoid pathological regex on multi-MB outputs


def parse_detections(text: str) -> list[dict[str, Any]] | None:
    """Try to extract a list of {box_2d, label} dicts from a model response.

    Tolerates markdown fences, leading prose, and trailing commentary. Returns
    None on any failure — never raises.
    """
    if not text:
        return None
    candidate = text.strip()[:_PARSE_INPUT_CAP]
    # Strip markdown code fences if present
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)

    parsed: Any = None
    # Direct parse only if it looks like a JSON array — avoids accepting
    # bare literals like "true" or a number.
    if candidate.startswith("["):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
    if parsed is None:
        match = _JSON_ARRAY_RE.search(candidate)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed, list):
        return None
    valid: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        box = item.get("box_2d") or item.get("box")
        label = item.get("label")
        if not (isinstance(box, list) and len(box) == 4 and isinstance(label, str)):
            continue
        try:
            box_floats = [float(v) for v in box]
        except (TypeError, ValueError):
            continue
        valid.append({"box_2d": box_floats, "label": label})
    return valid or None


def draw_boxes(image_bytes: bytes, detections: list[dict[str, Any]]) -> Image.Image:
    """Overlay bounding boxes + labels on the image. Returns a PIL Image."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None

    palette = [
        (239, 68, 68), (59, 130, 246), (16, 185, 129),
        (245, 158, 11), (139, 92, 246), (236, 72, 153),
        (20, 184, 166), (251, 113, 133), (132, 204, 22),
        (99, 102, 241),
    ]

    for i, det in enumerate(detections):
        y_min, x_min, y_max, x_max = det["box_2d"]
        # Clamp to [0, 1000] then normalise to pixel coords
        x_min_c = max(0.0, min(1000.0, x_min))
        x_max_c = max(0.0, min(1000.0, x_max))
        y_min_c = max(0.0, min(1000.0, y_min))
        y_max_c = max(0.0, min(1000.0, y_max))
        # Sort to tolerate swapped coordinates from the model
        left, right = sorted([x_min_c / 1000.0 * w, x_max_c / 1000.0 * w])
        top, bottom = sorted([y_min_c / 1000.0 * h, y_max_c / 1000.0 * h])
        if right - left < 1 or bottom - top < 1:
            continue  # degenerate — skip silently
        color = palette[i % len(palette)]
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        label = det["label"]
        # Label background
        if font is not None:
            try:
                bbox = draw.textbbox((left, max(top - 16, 0)), label, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((left, max(top - 16, 0)), label, fill=(255, 255, 255), font=font)
            except Exception:
                draw.text((left + 2, top + 2), label, fill=color)
    return img


def split_thinking(text: str) -> tuple[str | None, str]:
    """Split a response into (thoughts, answer) if a <thinking> block is present."""
    if not text:
        return None, ""
    match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None, text
    thoughts = match.group(1).strip()
    answer = (text[: match.start()] + text[match.end():]).strip()
    return thoughts, answer
