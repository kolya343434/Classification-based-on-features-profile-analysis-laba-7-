"""
profiles.py — профили символа и метрика Левенштейна.

Профиль = проекция символа (количество fg-пикселей по каждой строке/столбцу),
нормированная и квантованная в строку символов.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Построение профилей
# ---------------------------------------------------------------------------

def build_profiles(binary: np.ndarray, levels: int = 8) -> tuple[str, str]:
    """
    Построить горизонтальный и вертикальный профили символа.

    binary : 2D uint8 (0=фон, 255=символ)
    levels : число уровней квантования (используются символы '0'..'levels-1')

    Возвращает
    ----------
    (h_profile, v_profile) — строки из символов '0'..'7'
    """
    fg = (binary > 0).astype(np.float64)

    # Горизонтальный: сумма по столбцам → вектор длиной w
    h_proj = fg.sum(axis=0)
    # Вертикальный: сумма по строкам → вектор длиной h
    v_proj = fg.sum(axis=1)

    def quantize(proj: np.ndarray) -> str:
        if proj.max() == 0:
            return "0" * len(proj)
        norm = proj / proj.max()
        idx  = np.clip((norm * levels).astype(int), 0, levels - 1)
        return "".join(str(i) for i in idx)

    return quantize(h_proj), quantize(v_proj)


# ---------------------------------------------------------------------------
# Расстояние Левенштейна
# ---------------------------------------------------------------------------

def levenshtein(s1: str, s2: str) -> int:
    """Расстояние редактирования (вставка/удаление/замена, цена = 1)."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def profile_similarity(
    binary1: np.ndarray,
    binary2: np.ndarray,
    levels: int = 8,
) -> float:
    """
    Сходство двух символов по профилям через метрику Левенштейна.

    sim_profiles = 1 - (lev_h + lev_v) / (len_h + len_v)

    Нормировка: делим на суммарную максимально возможную длину
    (max(len(h1),len(h2)) + max(len(v1),len(v2))).
    Результат в [0, 1].
    """
    h1, v1 = build_profiles(binary1, levels)
    h2, v2 = build_profiles(binary2, levels)

    max_h = max(len(h1), len(h2))
    max_v = max(len(v1), len(v2))

    lev_h = levenshtein(h1, h2)
    lev_v = levenshtein(v1, v2)

    denom = max_h + max_v
    if denom == 0:
        return 1.0
    return 1.0 - (lev_h + lev_v) / denom
