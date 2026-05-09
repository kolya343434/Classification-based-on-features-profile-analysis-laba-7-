"""
segmentation.py — сегментация символов из строки изображения.

Использует проекцию на горизонтальную ось (вертикальный профиль) для
нахождения границ символов.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Данные о сегменте
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    char_img: np.ndarray   # вырезанный символ (серый, 0=фон 255=символ)
    x_start:  int
    x_end:    int
    y_start:  int
    y_end:    int


# ---------------------------------------------------------------------------
# Рендеринг строки
# ---------------------------------------------------------------------------

def render_text(
    text: str,
    font_size: int,
    font_path: str | Path | None = None,
    padding: int = 10,
) -> np.ndarray:
    """
    Нарисовать строку text чёрным шрифтом на белом фоне.
    Возвращает uint8 grayscale ndarray.
    """
    try:
        font = ImageFont.truetype(str(font_path), font_size) if font_path else \
               ImageFont.load_default(size=font_size)
    except (IOError, TypeError):
        font = ImageFont.load_default()

    # Замер размера текста
    dummy = Image.new("L", (1, 1), 255)
    draw  = ImageDraw.Draw(dummy)
    bbox  = draw.textbbox((0, 0), text, font=font)
    tw    = bbox[2] - bbox[0] + padding * 2
    th    = bbox[3] - bbox[1] + padding * 2

    img  = Image.new("L", (tw, th), 255)
    draw = ImageDraw.Draw(img)
    draw.text((padding - bbox[0], padding - bbox[1]), text, fill=0, font=font)
    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Бинаризация
# ---------------------------------------------------------------------------

def binarize(gray: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Порог: пиксели < threshold считаются символом (→ 255), остальные — фоном (→ 0).
    """
    fg = (gray < threshold).astype(np.uint8) * 255
    return fg


# ---------------------------------------------------------------------------
# Сегментация по вертикальной проекции
# ---------------------------------------------------------------------------

def segment_chars(binary: np.ndarray, min_width: int = 2) -> list[Segment]:
    """
    Разбить строку на символы по вертикальной проекции.

    binary : 2D uint8, 0=фон, 255=символ.
    """
    h, w = binary.shape
    # Вертикальная проекция (сумма по строкам)
    v_proj = (binary > 0).sum(axis=0)

    segments: list[Segment] = []
    in_char = False
    x_start = 0

    for x in range(w):
        if v_proj[x] > 0 and not in_char:
            in_char = True
            x_start = x
        elif v_proj[x] == 0 and in_char:
            in_char = False
            x_end = x
            if x_end - x_start >= min_width:
                # Горизонтальная проекция для tight bounding box
                col = binary[:, x_start:x_end]
                h_proj = (col > 0).sum(axis=1)
                ys = np.where(h_proj > 0)[0]
                if len(ys) == 0:
                    continue
                y0, y1 = int(ys[0]), int(ys[-1]) + 1
                segments.append(Segment(
                    char_img=col[y0:y1, :],
                    x_start=x_start,
                    x_end=x_end,
                    y_start=y0,
                    y_end=y1,
                ))

    # Закрыть последний сегмент
    if in_char:
        x_end = w
        if x_end - x_start >= min_width:
            col = binary[:, x_start:x_end]
            h_proj = (col > 0).sum(axis=1)
            ys = np.where(h_proj > 0)[0]
            if len(ys) > 0:
                y0, y1 = int(ys[0]), int(ys[-1]) + 1
                segments.append(Segment(
                    char_img=col[y0:y1, :],
                    x_start=x_start,
                    x_end=x_end,
                    y_start=y0,
                    y_end=y1,
                ))

    return segments


# ---------------------------------------------------------------------------
# Визуализация сегментации
# ---------------------------------------------------------------------------

def draw_segmentation(gray: np.ndarray, segments: list[Segment],
                       labels: list[str] | None = None) -> Image.Image:
    """
    Нарисовать красные bounding-box и лучшую гипотезу над каждым символом.
    """
    img = Image.fromarray(gray).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, seg in enumerate(segments):
        draw.rectangle(
            [seg.x_start, seg.y_start, seg.x_end - 1, seg.y_end - 1],
            outline=(255, 0, 0), width=1,
        )
        if labels and i < len(labels):
            draw.text((seg.x_start, max(0, seg.y_start - 12)),
                      labels[i], fill=(255, 0, 0))
    return img
