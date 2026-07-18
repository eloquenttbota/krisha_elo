"""Инициализация окружения для ноутбука-презентации.

Идея: вместо блока `import ...` на экране показывается результат —
короткое подтверждение, что всё готово к работе. Сам импорт неизбежно
происходит внутри функции, а имена модулей возвращаются наружу.
"""
from __future__ import annotations


def init_libraries():
    """Импортирует и настраивает всё необходимое для анализа и ML.

    Возвращает: (np, pd, plt, sns, colors)
    """
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from src.viz_utils import card

    warnings.filterwarnings("ignore")

    # ─── Единая палитра для всех графиков ───────────────────────────────
    colors = {
        "primary": "#2F80ED",   # синий   — основные бары, гистограммы, scatter
        "pos":     "#27AE60",   # зелёный — положительная корреляция / успех
        "neg":     "#EB5757",   # красный — отрицательная корреляция / выбросы
        "accent":  "#F2994A",   # оранжевый — акценты, вторичные бары
        "neutral": "#8E9BAE",   # серый   — нейтральные линии
        "light":   "#A8C7F0",   # светло-синий — boxplot facecolor
        "cmap_div":  "RdYlBu_r",
        "cmap_miss": ["#2F80ED", "#FFFFFF"],
    }
    colors["palette"] = [colors["primary"], colors["pos"], colors["neg"],
                          colors["accent"], colors["neutral"]]

    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "axes.facecolor":    "#F7F9FC",
        "figure.facecolor":  "white",
        "axes.grid":         True,
        "grid.color":        "#E2E8F0",
        "grid.linestyle":    "--",
        "grid.alpha":        0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    "#CBD5E0",
        "legend.framealpha": 0.8,
        "figure.dpi":        100,
    })
    sns.set_theme(style="whitegrid", palette=colors["palette"],
                  font="DejaVu Sans", font_scale=1.05)

    pd.options.display.float_format = "{:.2f}".format

    card(
        "Окружение готово к работе",
        rows=[
            ("NumPy / Pandas", f"{np.__version__} / {pd.__version__}"),
            ("Matplotlib / Seaborn", f"{plt.matplotlib.__version__} / {sns.__version__}"),
            ("Стиль графиков", "единая цветовая палитра для всей презентации"),
        ],
        accent=colors["primary"],
    )

    return np, pd, plt, sns, colors
