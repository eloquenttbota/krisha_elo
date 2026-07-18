"""Отбор признаков: слабая корреляция с таргетом, мультиколлинеарность, утечка."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.viz_utils import card


def plot_target_correlation(df: pd.DataFrame, target: str, colors: dict) -> pd.Series:
    corr_target = (
        df.select_dtypes(include="number")
        .corr(method="spearman")[target]
        .drop(target)
        .sort_values(key=abs, ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 9))
    bar_colors = [colors["pos"] if x > 0 else colors["neg"] for x in corr_target]
    ax.barh(range(len(corr_target)), corr_target.values, color=bar_colors, alpha=0.85)
    ax.set_yticks(range(len(corr_target)))
    ax.set_yticklabels(corr_target.index, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"Spearman корреляция с {target}")
    ax.set_title("Влияние признаков на цену за м²\n(зелёный = дороже, красный = дешевле)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
    return corr_target


#: "Базовые" признаки, понятные любому нетехническому человеку — площадь,
#: число комнат и т.д. При прочих равных мультиколлинеарность решается в их
#: пользу, а не в пользу производных/составных признаков (area_x_year и т.п.),
#: иначе итоговая модель рискует остаться вовсе без площади квартиры —
#: главного фактора цены, который аудитория ожидает увидеть в списке.
CORE_FEATURES = {"area", "room_count", "ceiling_height", "distance_to_center",
                  "floor", "floor_count"}


def select_features(df: pd.DataFrame, target: str = "price_per_m2",
                     corr_thr: float = 0.02, multi_thr: float = 0.90,
                     colors: dict = None) -> pd.DataFrame:
    df = df.copy()
    corr_target = plot_target_correlation(df, target, colors or {"pos": "#27AE60", "neg": "#EB5757"})

    # Шаг 1 — слабый сигнал (базовые признаки не трогаем, даже если их
    # корреляция с таргетом мала — они всё равно несут смысл для аудитории)
    weak = [c for c in corr_target[corr_target.abs() < corr_thr].index if c not in CORE_FEATURES]
    df = df.drop(columns=[c for c in weak if c in df.columns])

    # Шаг 2 — мультиколлинеарность: из пары сильно скоррелированных
    # признаков оставляем тот, что сильнее связан с таргетом; если один из
    # пары — базовый признак, оставляем его, пока он не отстаёт критически
    num_df = df.select_dtypes(include="number").drop(columns=[target], errors="ignore")
    corr_raw = num_df.corr(method="spearman").abs()
    corr_arr = corr_raw.to_numpy(copy=True)
    np.fill_diagonal(corr_arr, 0)
    corr_mx = pd.DataFrame(corr_arr, index=corr_raw.index, columns=corr_raw.columns)
    upper = corr_mx.where(np.triu(np.ones(corr_mx.shape, dtype=bool), k=1))

    to_drop = set()
    dropped_pairs = []
    pairs = [(r, c, upper.loc[r, c]) for r in upper.index for c in upper.columns
             if upper.loc[r, c] > multi_thr]
    pairs.sort(key=lambda x: x[2], reverse=True)
    for r, c, v in pairs:
        if r in to_drop or c in to_drop:
            continue
        r_core, c_core = r in CORE_FEATURES, c in CORE_FEATURES
        if r_core and not c_core:
            drop = c
        elif c_core and not r_core:
            drop = r
        else:
            r1, r2 = abs(corr_target.get(r, 0)), abs(corr_target.get(c, 0))
            drop = r if r1 <= r2 else c
        to_drop.add(drop)
        dropped_pairs.append(f"{r} × {c} (|r|={v:.2f}) → убран {drop!r}")
    df = df.drop(columns=[c for c in to_drop if c in df.columns])

    # Шаг 3 — проверка утечки таргета
    assert "price" not in df.columns, "УТЕЧКА: price всё ещё в данных!"

    card(
        "Отбор признаков завершён",
        rows=[
            (f"Убрано слабых (|r| < {corr_thr})", str(len(weak))),
            (f"Убрано из-за мультиколлинеарности (|r| > {multi_thr})", str(len(to_drop))),
            ("Проверка утечки таргета", "OK — price отсутствует"),
            ("Финальный набор признаков", str(df.shape[1] - 1)),
        ],
        note="; ".join(dropped_pairs) if dropped_pairs else "Сильно скоррелированных пар не найдено.",
        accent="#27AE60",
    )
    return df
