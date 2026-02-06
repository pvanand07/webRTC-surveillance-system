"""
Neat, Apple-style drawing for detection bounding boxes on live stream.
Uses system TTF for Apple-like typography when available.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

# Pastel palette (BGR)
COLOR_TRACKED = (200, 235, 190)   # pastel mint – tracked object
COLOR_DEFAULT = (230, 215, 210)   # pastel lavender – other detections
COLOR_LABEL_BG = (52, 50, 58)    # soft dark pill background
COLOR_LABEL_TEXT = (248, 248, 250)
LINE_THICKNESS = 2
LABEL_PADDING = (6, 5)  # vertical, horizontal
CORNER_RADIUS = 6  # pixels for rounded corners
LABEL_FONT_SIZE = 13  # for TTF

# Apple-like / clean system fonts (order of preference)
_FONT_PATHS = [
    "/System/Library/Fonts/SFNSMono.ttf",            # macOS SF Mono
    "/System/Library/Fonts/Helvetica.ttc",           # macOS Helvetica
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",                  # Windows Segoe UI
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
]

_label_font = None


def _get_label_font():
    """Load first available TTF for label text (Apple-like when on macOS)."""
    global _label_font
    if _label_font is not None:
        return _label_font
    try:
        from PIL import ImageFont
        for path in _FONT_PATHS:
            if Path(path).exists():
                _label_font = ImageFont.truetype(path, LABEL_FONT_SIZE)
                return _label_font
    except Exception:
        pass
    _label_font = False  # no TTF found
    return _label_font


def _draw_text_pil(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color_bgr: Tuple[int, int, int],
) -> None:
    """Draw text using PIL TTF when available, else OpenCV fallback."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = _get_label_font()
        if font is False:
            # Fallback to OpenCV
            cv2.putText(
                img, text, (x, y + LABEL_FONT_SIZE),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_bgr, 1, cv2.LINE_AA,
            )
            return
        # Render text to a small RGBA image
        pil_font = font
        # Get bounding box of text
        bbox = pil_font.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = 2
        w, h = tw + pad * 2, th + pad * 2
        pil_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pil_img)
        # Draw text (white); PIL uses RGB
        draw.text((pad - bbox[0], pad - bbox[1]), text, font=pil_font, fill=(color_bgr[2], color_bgr[1], color_bgr[0], 255))
        arr = np.array(pil_img)
        # Overlay onto frame using alpha
        y1 = max(0, y)
        y2 = min(img.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(img.shape[1], x + w)
        sy, sx = y2 - y1, x2 - x1
        if sx <= 0 or sy <= 0:
            return
        roi = img[y1:y2, x1:x2].astype(np.float32)
        ah, aw = arr.shape[0], arr.shape[1]
        use_h, use_w = min(sy, ah), min(sx, aw)
        alpha = arr[:use_h, :use_w, 3:4].astype(np.float32) / 255.0
        text_rgb = arr[:use_h, :use_w, :3]
        roi_slice = roi[:use_h, :use_w]
        for c in range(3):
            roi_slice[:, :, c] = roi_slice[:, :, c] * (1 - alpha[:, :, 0]) + text_rgb[:, :, 2 - c] * alpha[:, :, 0]
        img[y1:y2, x1:x2] = np.clip(roi, 0, 255).astype(np.uint8)
    except Exception:
        cv2.putText(
            img, text, (x, y + LABEL_FONT_SIZE),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_bgr, 1, cv2.LINE_AA,
        )


def _draw_rounded_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a rounded rectangle (Apple-style) using lines and arcs."""
    h, w = img.shape[:2]
    r = min(CORNER_RADIUS, (x2 - x1) // 2, (y2 - y1) // 2)
    if r <= 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        return
    # Top, bottom, left, right segments (inset by r)
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    # Corners
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)


def draw_detections(
    frame: np.ndarray,
    boxes_xyxy: List[List[float]],
    track_ids: Optional[List[int]] = None,
    class_names: Optional[List[str]] = None,
    confidences: Optional[List[float]] = None,
    event_track_id: Optional[int] = None,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on frame with a clean, Apple-like style.
    Uses rounded rectangles, soft colors, and a small label above each box.
    """
    out = frame.copy()
    if not boxes_xyxy:
        return out

    n = len(boxes_xyxy)
    track_ids = track_ids or [None] * n
    class_names = class_names or ["object"] * n
    confidences = confidences or [0.0] * n

    for i in range(n):
        x1, y1, x2, y2 = boxes_xyxy[i]
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        tid = track_ids[i] if i < len(track_ids) else None
        cls_name = class_names[i] if i < len(class_names) else "object"
        conf = confidences[i] if i < len(confidences) else 0.0

        is_tracked = event_track_id is not None and tid == event_track_id
        color = COLOR_TRACKED if is_tracked else COLOR_DEFAULT

        _draw_rounded_rect(out, x1, y1, x2, y2, color, LINE_THICKNESS)

        # Label above the box (pill-style background)
        label = f"{cls_name}"
        if conf > 0:
            label += f" {conf:.0%}"
        if is_tracked and tid is not None:
            label += f" #{tid}"

        pad_v, pad_h = LABEL_PADDING
        # Estimate label size (PIL bbox or fallback)
        font = _get_label_font()
        if font:
            try:
                bbox = font.getbbox(label)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                tw, th = len(label) * 7, LABEL_FONT_SIZE + 2
        else:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        lx1 = max(0, x1)
        ly1 = max(0, y1 - th - pad_v * 2 - 2)
        lx2 = min(frame.shape[1], lx1 + tw + pad_h * 2)
        ly2 = max(0, y1 - 2)

        if ly1 < ly2 and lx1 < lx2:
            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), COLOR_LABEL_BG, -1, cv2.LINE_AA)
            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), color, 1, cv2.LINE_AA)
            _draw_text_pil(out, label, lx1 + pad_h, ly1 + pad_v, COLOR_LABEL_TEXT)

    return out
