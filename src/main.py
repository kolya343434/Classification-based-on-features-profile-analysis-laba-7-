"""
main.py — CLI для лаб. работы №7.

Примеры:
    python src/main.py --text "приветмир" --font-size 52
    python src/main.py --text "приветмир" --font-size 52 --variation-font-size 56 --use-profiles
    python src/main.py --text "приветмир" --font-size 52 --save-images --hypotheses-out outputs/hypotheses.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent))

from classifier import build_alphabet, classify_line, recognition_metrics
from segmentation import render_text, binarize, segment_chars, draw_segmentation


ALPHABET_RU = "абвгдежзийклмнопрстуфхцчшщъыьэюя"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_recognition(
    text:          str,
    font_size:     int,
    alphabet_str:  str,
    font_path:     str | None,
    use_profiles:  bool,
    alpha:         float,
    label:         str,
    save_dir:      Path | None,
) -> dict:
    """Прогнать полный цикл распознавания и вернуть результаты."""
    # 1. Рендер строки
    gray   = render_text(text, font_size, font_path, padding=10)
    binary = binarize(gray)

    # 2. Сегментация
    segments = segment_chars(binary)

    # 3. Алфавит — эталоны с тем же шрифтом
    alphabet = build_alphabet(alphabet_str, font_size, font_path)

    # 4. Классификация
    hypotheses = classify_line(segments, alphabet,
                                use_profiles=use_profiles, alpha=alpha)

    # 5. Метрики
    gt      = "".join(ch for ch in text if ch in alphabet_str)
    metrics = recognition_metrics(hypotheses, gt)

    result = {
        "label":       label,
        "font_size":   font_size,
        "text":        text,
        "segments":    len(segments),
        "hypotheses":  hypotheses,
        **metrics,
    }

    # 6. Сохранение изображений
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        best_labels = [h[0][0] for h in hypotheses]
        vis = draw_segmentation(gray, segments, best_labels)
        vis.save(save_dir / f"{label}_segmentation.png")

        from PIL import Image
        import numpy as np
        Image.fromarray(gray).save(save_dir / f"{label}_render.png")

    return result


def _format_hypotheses(hypotheses: list, ground_truth: str) -> str:
    """Форматировать гипотезы в стиле задания."""
    lines = []
    for i, hyps in enumerate(hypotheses, 1):
        hyps_str = ", ".join(f'("{h[0]}", {h[1]:.2f})' for h in hyps[:10])
        lines.append(f"{i}: [{hyps_str}]")
    lines.append("")
    best = "".join(h[0][0] for h in hypotheses)
    lines.append(f"Лучшие гипотезы: {best}")
    lines.append(f"Исходная строка: {ground_truth}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Лаб. работа №7 — классификация символов на основе признаков."
    )
    parser.add_argument("--text",               type=str,   required=True,  help="Распознаваемая строка.")
    parser.add_argument("--alphabet",           type=str,   default=ALPHABET_RU, help="Строка алфавита.")
    parser.add_argument("--font-size",          type=int,   default=52,     help="Базовый размер шрифта.")
    parser.add_argument("--variation-font-size",type=int,   default=None,   help="Размер шрифта для эксперимента.")
    parser.add_argument("--font-path",          type=str,   default=None,   help="Путь к .ttf файлу.")
    parser.add_argument("--use-profiles",       action="store_true",        help="Дополнить меру сравнением профилей (Левенштейн).")
    parser.add_argument("--alpha",              type=float, default=0.3,    help="Вес профилей в итоговой мере (0..1).")
    parser.add_argument("--save-images",        action="store_true",        help="Сохранить визуализации в assets/.")
    parser.add_argument("--hypotheses-out",     type=str,   default=None,   help="Файл для записи гипотез.")
    parser.add_argument("--json-out",           type=str,   default=None,   help="JSON с полными результатами.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    save_dir  = repo_root / "assets" if args.save_images else None

    # ── Базовый эксперимент ────────────────────────────────────────────────
    base = _run_recognition(
        text=args.text,
        font_size=args.font_size,
        alphabet_str=args.alphabet,
        font_path=args.font_path,
        use_profiles=args.use_profiles,
        alpha=args.alpha,
        label="base",
        save_dir=save_dir,
    )

    # ── Эксперимент с другим размером шрифта ──────────────────────────────
    variation = None
    if args.variation_font_size:
        variation = _run_recognition(
            text=args.text,
            font_size=args.variation_font_size,
            alphabet_str=args.alphabet,
            font_path=args.font_path,
            use_profiles=args.use_profiles,
            alpha=args.alpha,
            label="variation",
            save_dir=save_dir,
        )

    # ── Вывод гипотез ─────────────────────────────────────────────────────
    gt = "".join(ch for ch in args.text if ch in args.alphabet)
    hyp_text = _format_hypotheses(base["hypotheses"], gt)

    print("=" * 60)
    print(f"БАЗОВЫЙ ЗАПУСК (шрифт {args.font_size}pt)")
    print("=" * 60)
    print(hyp_text)
    print(f"\nОшибок: {base['errors']}  |  Точность: {base['accuracy_pct']}%")

    if variation:
        gt_var = "".join(ch for ch in args.text if ch in args.alphabet)
        hyp_var = _format_hypotheses(variation["hypotheses"], gt_var)
        print("\n" + "=" * 60)
        print(f"ЭКСПЕРИМЕНТ (шрифт {args.variation_font_size}pt)")
        print("=" * 60)
        print(hyp_var)
        print(f"\nОшибок: {variation['errors']}  |  Точность: {variation['accuracy_pct']}%")

        print("\n" + "=" * 60)
        print("СРАВНЕНИЕ")
        print("=" * 60)
        print(f"Базовый  ({args.font_size}pt):    {base['recognized']:<20}  точность {base['accuracy_pct']}%")
        print(f"Вариация ({args.variation_font_size}pt):    {variation['recognized']:<20}  точность {variation['accuracy_pct']}%")

    # ── Сохранение файлов ─────────────────────────────────────────────────
    if args.hypotheses_out:
        out_path = Path(args.hypotheses_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"=== Базовый ({args.font_size}pt) ===\n")
            f.write(hyp_text + "\n")
            f.write(f"Ошибок: {base['errors']}  Точность: {base['accuracy_pct']}%\n")
            if variation:
                f.write(f"\n=== Вариация ({args.variation_font_size}pt) ===\n")
                f.write(_format_hypotheses(variation["hypotheses"], gt) + "\n")
                f.write(f"Ошибок: {variation['errors']}  Точность: {variation['accuracy_pct']}%\n")
        print(f"\n[гипотезы → {out_path}]")

    if args.json_out:
        data = {"base": {k: v for k, v in base.items() if k != "hypotheses"},
                "base_hypotheses": base["hypotheses"]}
        if variation:
            data["variation"] = {k: v for k, v in variation.items() if k != "hypotheses"}
            data["variation_hypotheses"] = variation["hypotheses"]
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[json    → {out_path}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
