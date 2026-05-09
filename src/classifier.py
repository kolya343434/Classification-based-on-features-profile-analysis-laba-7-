"""
classifier.py — классификация символов на основе признаков (и профилей).
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from features import extract_features, similarity
from profiles import profile_similarity
from segmentation import Segment, render_text, binarize, segment_chars


# ---------------------------------------------------------------------------
# Алфавит: эталонные изображения
# ---------------------------------------------------------------------------

class AlphabetEntry(NamedTuple):
    char:     str
    features: np.ndarray
    binary:   np.ndarray   # бинарное изображение (для профилей)


def build_alphabet(
    chars: str,
    font_size: int,
    font_path: str | Path | None = None,
) -> list[AlphabetEntry]:
    """
    Сгенерировать эталонное изображение для каждого символа алфавита
    и извлечь признаки.
    """
    entries: list[AlphabetEntry] = []
    for ch in chars:
        gray   = render_text(ch, font_size, font_path, padding=4)
        binary = binarize(gray)
        segs   = segment_chars(binary)
        if segs:
            img = segs[0].char_img
        else:
            img = binary
        feats = extract_features(img)
        entries.append(AlphabetEntry(char=ch, features=feats, binary=img))
    return entries


# ---------------------------------------------------------------------------
# Гипотезы для одного символа
# ---------------------------------------------------------------------------

Hypothesis = tuple[str, float]   # (символ, мера близости)


def classify_segment(
    seg_img:  np.ndarray,
    alphabet: list[AlphabetEntry],
    *,
    use_profiles: bool = False,
    alpha:        float = 0.3,
    profile_levels: int = 8,
) -> list[Hypothesis]:
    """
    Для изображения сегмента вернуть список гипотез,
    отсортированных по убыванию меры близости.

    Параметры
    ----------
    use_profiles : объединять признаки с профилями (метод магистров)
    alpha        : вес профильной составляющей (0..1)
    """
    f_seg = extract_features(seg_img)
    hyps: list[Hypothesis] = []

    for entry in alphabet:
        sim_f = similarity(f_seg, entry.features)

        if use_profiles:
            sim_p = profile_similarity(seg_img, entry.binary, levels=profile_levels)
            sim_final = (1 - alpha) * sim_f + alpha * sim_p
        else:
            sim_final = sim_f

        hyps.append((entry.char, round(sim_final, 4)))

    hyps.sort(key=lambda h: h[1], reverse=True)
    return hyps


# ---------------------------------------------------------------------------
# Классификация всей строки
# ---------------------------------------------------------------------------

def classify_line(
    segments: list[Segment],
    alphabet: list[AlphabetEntry],
    **kwargs,
) -> list[list[Hypothesis]]:
    """Вернуть список гипотез для каждого сегмента строки."""
    return [classify_segment(seg.char_img, alphabet, **kwargs) for seg in segments]


# ---------------------------------------------------------------------------
# Метрики качества
# ---------------------------------------------------------------------------

def recognition_metrics(
    hypotheses: list[list[Hypothesis]],
    ground_truth: str,
) -> dict:
    """
    Сравнить лучшие гипотезы с эталонной строкой.

    Возвращает словарь:
        recognized   : str   — строка из лучших гипотез
        errors       : int   — число ошибок
        accuracy_pct : float — доля верно распознанных (%)
    """
    best = [hyps[0][0] for hyps in hypotheses]
    recognized = "".join(best)

    n    = min(len(recognized), len(ground_truth))
    errs = sum(1 for a, b in zip(recognized, ground_truth) if a != b)
    errs += abs(len(recognized) - len(ground_truth))

    total    = max(len(ground_truth), len(recognized), 1)
    accuracy = round((total - errs) / total * 100, 1)

    return {
        "recognized":   recognized,
        "ground_truth": ground_truth,
        "errors":       errs,
        "accuracy_pct": accuracy,
    }
