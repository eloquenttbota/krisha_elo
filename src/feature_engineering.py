"""Генерация признаков: расстояние до центра, кодирование, взаимодействия."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from IPython.display import display

from src.geo import haversine_km, CENTER_LAT, CENTER_LON
from src.viz_utils import card


def add_distance_to_center(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance_to_center"] = haversine_km(df["lat"], df["lon"], CENTER_LAT, CENTER_LON)
    card(
        "Признак «расстояние до центра» создан",
        rows=[("Формула", "гаверсинус (реальное расстояние по сфере Земли, км)"),
              ("Центр", f"{CENTER_LAT}, {CENTER_LON} (район Байтерека)"),
              ("Медиана", f"{df['distance_to_center'].median():.1f} км")],
        accent="#F2994A",
    )
    return df


def get_year_category(year: float) -> int:
    if year < 1970:
        return 1
    elif year < 1980:
        return 2
    elif year < 1990:
        return 3
    elif year < 2000:
        return 4
    elif year < 2010:
        return 5
    elif year < 2020:
        return 6
    return 7


def add_year_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year_category"] = df["construction_year"].apply(get_year_category)
    df = df.drop(columns=["construction_year"])
    return df


def add_price_per_m2(df: pd.DataFrame, low_q: float = 0.01, high_q: float = 0.99) -> pd.DataFrame:
    df = df.copy()
    df["price_per_m2"] = df["price"] / df["area"]

    lo, hi = df["price_per_m2"].quantile([low_q, high_q])
    before = len(df)
    df = df[df["price_per_m2"].between(lo, hi)].reset_index(drop=True)
    df = df.drop(columns=["price"])

    card(
        "Целевая переменная: цена за м²",
        rows=[
            (f"Обрезка по перцентилям {low_q:.0%}–{high_q:.0%}", f"{lo:,.0f} — {hi:,.0f} тг/м²".replace(",", " ")),
            ("Удалено выбросов", f"{before - len(df)} строк ({(before - len(df)) / before * 100:.1f}%)"),
            ("Осталось", f"{len(df):,}".replace(",", " ")),
        ],
        note="Дальше модель предсказывает именно цену за м², а не полную цену — "
             "так она не путается между большими и маленькими квартирами. "
             "Полная цена = цена/м² × площадь.",
        accent="#2F80ED",
    )
    return df


#: Пояснение к каждому новому признаку: что он означает и на какой вывод
#: из EDA (раздел 4, гипотезы H1–H6) он опирается.
DERIVED_FEATURE_INFO = {
    "is_high_ceiling":            ("Высокий потолок (≥3 м) — да/нет", "премиальность жилья"),
    "premium_factor":             ("Высокий потолок И самый дорогой район", "H2+H5: престижный район и премиум-жильё усиливают друг друга"),
    "economy_old_panel":          ("Старый (до 1990-х) панельный дом", "H3: эпоха постройки → цена"),
    "log_distance":               ("Логарифм расстояния до центра", "H4: расстояние до центра"),
    "area_per_room":              ("Площадь на одну комнату (просторность)", "H1: эффект масштаба"),
    "floor_bin":                  ("Категория этажа: первый/низкий/средний/высокий", "нелинейный эффект этажа"),
    "area_bin":                   ("Категория площади: маленькая/средняя/большая/очень большая", "H1: нелинейный эффект площади"),
    "floor_ratio":                ("Этаж ÷ этажность дома — позиция в здании", "нелинейный эффект этажа"),
    "floor_ratio_x_ceiling":      ("Позиция этажа × высокий потолок", "взаимодействие премиальности"),
    "is_top_floor":                ("Последний этаж", "известный эффект рынка недвижимости — дешевле"),
    "is_ground_floor":            ("Первый этаж", "известный эффект рынка недвижимости — дешевле"),
    "log_area":                   ("Логарифм площади", "H1: сглаживает нелинейный эффект масштаба"),
    "area_squared":               ("Площадь в квадрате", "нелинейный эффект площади"),
    "area_x_rooms":               ("Площадь × число комнат", "взаимодействие просторности"),
    "area_x_year":                ("Площадь × эпоха постройки", "взаимодействие H1 и H3"),
    "rooms_x_year":               ("Комнаты × эпоха постройки", "взаимодействие H1 и H3"),
    "distance_x_premium_district": ("Расстояние до центра × престижный район", "взаимодействие H4 и H2/H5"),
}


def add_derived_features(df: pd.DataFrame, colors: dict = None) -> pd.DataFrame:
    """Признаки, построенные вручную на основе выводов EDA о рынке недвижимости."""
    df = df.copy()

    df["is_high_ceiling"] = (df["ceiling_height"] >= 3.0).astype(int)

    premium_district = df.groupby("district")["price_per_m2"].median().idxmax()
    is_premium = (df["district"] == premium_district).astype(int)
    df["premium_factor"] = df["is_high_ceiling"] * is_premium

    df["economy_old_panel"] = ((df["year_category"] <= 3) & (df["house_type"] == "панельный")).astype(int)

    df["log_distance"] = np.log1p(df["distance_to_center"])
    df["area_per_room"] = df["area"] / df["room_count"]

    df["floor_bin"] = pd.cut(df["floor"], bins=[0, 1, 5, 15, 100],
                              labels=[0, 1, 2, 3], include_lowest=True).astype(int)
    df["area_bin"] = pd.cut(df["area"], bins=[0, 45, 70, 100, 9999],
                             labels=[0, 1, 2, 3], include_lowest=True).astype(int)

    df["floor_ratio"] = df["floor"] / df["floor_count"]
    df["floor_ratio_x_ceiling"] = df["floor_ratio"] * df["is_high_ceiling"]
    df["is_top_floor"] = (df["floor"] == df["floor_count"]).astype(int)
    df["is_ground_floor"] = (df["floor"] == 1).astype(int)

    df["log_area"] = np.log1p(df["area"])
    df["area_squared"] = df["area"] ** 2
    df["area_x_rooms"] = df["area"] * df["room_count"]
    df["area_x_year"] = df["area"] * df["year_category"]
    df["rooms_x_year"] = df["room_count"] * df["year_category"]
    df["distance_x_premium_district"] = df["distance_to_center"] * is_premium

    new_features = list(DERIVED_FEATURE_INFO.keys())

    card(
        "Инженерные признаки построены",
        rows=[
            ("Самый дорогой район (по медиане цены/м²)", premium_district),
            ("Новых признаков", str(len(new_features))),
        ],
        note="premium_factor и distance_x_premium_district используют этот район — "
             "он определён из данных, а не задан вручную.",
        accent="#764ba2",
    )

    info_table = pd.DataFrame([
        {"Признак": f, "Что означает": desc, "На чём основан": hyp}
        for f, (desc, hyp) in DERIVED_FEATURE_INFO.items()
    ])
    display(info_table)

    corr = (
        df[new_features + ["price_per_m2"]]
        .corr(method="spearman")["price_per_m2"]
        .drop("price_per_m2")
        .sort_values()
    )
    colors = colors or {"pos": "#27AE60", "neg": "#EB5757"}
    fig, ax = plt.subplots(figsize=(9, 7))
    bar_colors = [colors["pos"] if x > 0 else colors["neg"] for x in corr]
    corr.plot(kind="barh", ax=ax, color=bar_colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman корреляция с ценой/м²")
    ax.set_title("Насколько новые признаки связаны с ценой")
    plt.tight_layout()
    plt.show()

    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, OneHotEncoder]:
    df = df.copy()
    ohe_cols = ["district", "house_type"]
    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore", dtype=int)
    encoded = encoder.fit_transform(df[ohe_cols])
    encoded_cols = encoder.get_feature_names_out(ohe_cols)
    df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

    drop_after = ["district", "house_type", "lat", "lon"]
    df_final = pd.concat([df.drop(columns=[c for c in drop_after if c in df.columns]), df_encoded], axis=1)

    card(
        "Категориальные признаки закодированы (One-Hot)",
        rows=[
            ("Район → колонок", str(len([c for c in encoded_cols if c.startswith("district")]))),
            ("Тип дома → колонок", str(len([c for c in encoded_cols if c.startswith("house_type")]))),
            ("owner_direct", "уже бинарный, кодирование не требуется"),
        ],
        note="drop='first' убирает одну базовую категорию в каждой группе, чтобы избежать "
             "полной коллинеарности (dummy variable trap).",
        accent="#2F80ED",
    )
    return df_final, encoder
