"""
features.py — вычисление нормализованных признаков символа.

Признаковый вектор: f = [mass, cx, cy, Ixx, Iyy]
"""
from __future__ import annotations

import numpy as np


def extract_features(symbol_img: np.ndarray) -> np.ndarray:
    """
    Вычислить признаковый вектор для бинарного изображения символа.

    Параметры
    ----------
    symbol_img : np.ndarray
        2D массив uint8 (0 = фон, 255 = символ), bounding-box символа.

    Возвращает
    ----------
    np.ndarray shape (5,) : [mass, cx, cy, Ixx, Iyy]

    Формулы
    -------
    Пусть fg(y,x) = 1 если пиксель принадлежит символу, иначе 0.
    h, w — высота и ширина bounding-box.

    mass = sum(fg) / (w * h)

    cx   = mean_x_of_fg / (w - 1)        # нормировано в [0,1]
    cy   = mean_y_of_fg / (h - 1)

    ȳ    = mean y координата fg-пикселей
    x̄    = mean x координата fg-пикселей
    Ixx  = sum((y - ȳ)² * fg) / (mass * (h-1)²)
    Iyy  = sum((x - x̄)² * fg) / (mass * (w-1)²)
    """
    fg = (symbol_img > 127).astype(np.float64)
    h, w = fg.shape

    total = fg.sum()
    if total == 0:
        return np.zeros(5, dtype=np.float64)

    # ── масса ──────────────────────────────────────────────────────────────
    mass = total / (w * h)

    # ── центр тяжести ──────────────────────────────────────────────────────
    ys, xs = np.mgrid[0:h, 0:w]
    x_mean = float(np.sum(xs * fg) / total)
    y_mean = float(np.sum(ys * fg) / total)

    cx = x_mean / max(w - 1, 1)
    cy = y_mean / max(h - 1, 1)

    # ── осевые моменты инерции ─────────────────────────────────────────────
    denom_x = mass * max(h - 1, 1) ** 2
    denom_y = mass * max(w - 1, 1) ** 2

    Ixx = float(np.sum((ys - y_mean) ** 2 * fg) / denom_x)
    Iyy = float(np.sum((xs - x_mean) ** 2 * fg) / denom_y)

    return np.array([mass, cx, cy, Ixx, Iyy], dtype=np.float64)


def euclidean_distance(f1: np.ndarray, f2: np.ndarray) -> float:
    """Евклидово расстояние между двумя признаковыми векторами."""
    return float(np.sqrt(np.sum((f1 - f2) ** 2)))


def similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    """
    Мера близости: sim = 1 / (1 + d)
    При d=0 → sim=1.0,  при d→∞ → sim→0.
    """
    return 1.0 / (1.0 + euclidean_distance(f1, f2))
