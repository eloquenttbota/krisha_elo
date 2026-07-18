"""Очистка и восстановление пропусков в сырых данных krisha.kz."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.viz_utils import card


# ─── Нормализация сырых текстовых полей ──────────────────────────────────

def parse_ceiling_height(series: pd.Series) -> pd.Series:
    """'3 м' / '2.75 м' -> 3.0 / 2.75"""
    return series.astype(str).str.extract(r"([\d.]+)")[0].astype(float)


def clean_district(series: pd.Series) -> pd.Series:
    """'Астана, Есильский р-н' -> 'Есильский р-н'; голое 'Астана' -> NaN (район неизвестен)."""
    cleaned = series.str.replace("Астана, ", "", regex=False).str.strip()
    return cleaned.replace("Астана", np.nan)


def owner_is_direct(series: pd.Series) -> pd.Series:
    """Эвристика: в объявлениях от собственника поле owner часто буквально
    содержит слово 'хозяин'. Остальное (имена агентов, 'продавец' и т.п.)
    считаем посредником/агентством — это соответствует вопросу, который
    бот задаёт пользователю ('Хозяин' / 'Агентство')."""
    return series.astype(str).str.lower().str.contains("хозя").astype(int)


def fill_floor_from_name(df: pd.DataFrame) -> pd.DataFrame:
    """В поле name есть паттерн вида '3/9 этаж' — используем его, чтобы
    восстановить floor/floor_count там, где они не распарсились при парсинге."""
    df = df.copy()
    pattern = df["name"].astype(str).str.extract(r"(\d+)\s*/\s*(\d+)\s*этаж")
    before_floor = df["floor"].isnull().sum()
    before_count = df["floor_count"].isnull().sum()

    need_floor = df["floor"].isnull() & pattern[0].notnull()
    df.loc[need_floor, "floor"] = pattern.loc[need_floor, 0].astype(float)

    need_count = df["floor_count"].isnull() & pattern[1].notnull()
    df.loc[need_count, "floor_count"] = pattern.loc[need_count, 1].astype(float)

    card(
        "Пропуски в этаже восстановлены из текста объявления",
        rows=[
            ("floor: было пропусков", str(before_floor)),
            ("floor: осталось", str(df["floor"].isnull().sum())),
            ("floor_count: было пропусков", str(before_count)),
            ("floor_count: осталось", str(df["floor_count"].isnull().sum())),
        ],
        note="Название объявления вида «3-комнатная квартира · 85 м² · 3/9 этаж» "
             "содержит этаж и этажность дома — регулярным выражением достаём их оттуда.",
    )
    return df


def prepare_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит сырые текстовые поля к рабочему виду: числовая высота потолков,
    очищенный район, признак прямого владельца."""
    df = df.copy()
    df["ceiling_height"] = parse_ceiling_height(df["ceiling_height"])
    df["district"] = clean_district(df["district"])
    df["owner_direct"] = owner_is_direct(df["owner"])
    df["in_complex"] = df["complex_name"].notnull().astype(int)
    df = fill_floor_from_name(df)

    card(
        "Сырые поля приведены к рабочему виду",
        rows=[
            ("ceiling_height", "текст → число (метры)"),
            ("district", "убран префикс «Астана, »"),
            ("owner_direct", "эвристика: 1 = «хозяин» в тексте, 0 = агент/посредник"),
            ("in_complex", "1 = квартира в жилом комплексе (есть complex_name)"),
        ],
        accent="#F2994A",
    )
    return df


# ─── Дубликаты ────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    after_url = len(df)

    key_features = ["price", "area", "room_count", "floor", "floor_count", "lat", "lon"]
    key_features = [c for c in key_features if c in df.columns]
    df = df.drop_duplicates(subset=key_features).reset_index(drop=True)
    after_key = len(df)

    card(
        "Дубликаты удалены",
        rows=[
            ("Было объявлений", f"{before:,}".replace(",", " ")),
            ("Дубликаты по ссылке (url)", str(before - after_url)),
            ("Дубликаты по характеристикам", str(after_url - after_key)),
            ("Осталось", f"{after_key:,}".replace(",", " ")),
        ],
        accent="#EB5757",
    )
    return df


# ─── Выбросы и логические несоответствия ──────────────────────────────────

DEFAULT_CLEANING_CONFIG = {
    "price":             (4_500_000, 250_000_000),
    "construction_year": (1960, 2028),
    "ceiling_height":    (2.3, 4.2),
    "area":              (15, 300),
    "room_count":        (1, 7),
    "floor":             (1, 40),
    "floor_count":       (1, 40),
}


def clean_real_estate_data(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Удаляет строки с физически невозможными или экстремальными значениями."""
    config = config or DEFAULT_CLEANING_CONFIG
    df = df.copy()
    log_rows = []

    if "area" in df.columns and "room_count" in df.columns:
        n0 = len(df)
        bad_area_mask = (df["room_count"] >= 2) & (df["area"] < 20)
        df = df[~bad_area_mask]
        log_rows.append(("Комнат ≥2, но площадь <20 м²", str(n0 - len(df))))

    for column, (lo, hi) in config.items():
        if column in df.columns:
            n0 = len(df)
            df = df[(df[column].between(lo, hi)) | (df[column].isnull())]
            log_rows.append((f"{column} вне [{lo:,.0f}, {hi:,.0f}]".replace(",", " "), str(n0 - len(df))))

    if "floor" in df.columns and "floor_count" in df.columns:
        n0 = len(df)
        df = df[(df["floor"] <= df["floor_count"]) | df["floor"].isnull() | df["floor_count"].isnull()]
        log_rows.append(("Этаж выше этажности дома", str(n0 - len(df))))

    df = df.reset_index(drop=True)
    card(
        "Выбросы и логические ошибки отфильтрованы",
        rows=log_rows + [("Итого осталось", f"{len(df):,}".replace(",", " "))],
        accent="#EB5757",
        note="Границы подобраны по реальному рынку Астаны, а не по формальному правилу IQR — "
             "иначе мы бы потеряли дорогие, но настоящие премиальные квартиры.",
    )
    return df


# ─── Пропуски как признак (до импутации) ──────────────────────────────────

def add_missing_indicators(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    rows = []
    for col in cols:
        if col in df.columns and df[col].isnull().any():
            flag = col.strip().replace(" ", "_") + "_was_missing"
            df[flag] = df[col].isnull().astype(int)
            rows.append((flag, f"{df[flag].sum()} строк"))
    card(
        "Флаги пропусков созданы (до импутации)",
        rows=rows if rows else [("Пропусков не найдено", "-")],
        note="Модель сможет учитывать сам факт отсутствия данных как отдельный сигнал.",
    )
    return df


# ─── Импутация: район/тип дома — по геолокации, потолки/этажность — по эпохе ──

def _nearest_indices(known_features: np.ndarray, missing_features: np.ndarray, k: int) -> np.ndarray:
    scaler = StandardScaler().fit(known_features)
    nn = NearestNeighbors(n_neighbors=k).fit(scaler.transform(known_features))
    _, idx = nn.kneighbors(scaler.transform(missing_features))
    return idx


def impute_district_by_location(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_mask = df["district"].isnull()
    if missing_mask.sum() == 0:
        card("Район: пропусков нет", accent="#27AE60")
        return df

    known = df.loc[~missing_mask]
    idx = _nearest_indices(known[["lat", "lon"]].values, df.loc[missing_mask, ["lat", "lon"]].values, k=1)
    df.loc[missing_mask, "district"] = known["district"].values[idx[:, 0]]

    card(
        "Район восстановлен по ближайшему соседу (k=1, координаты)",
        rows=[("Было пропусков", str(missing_mask.sum())),
              ("Осталось пропусков", str(df["district"].isnull().sum()))],
        accent="#2F80ED",
    )
    return df


def _impute_by_house_type_and_decade(df: pd.DataFrame, target_col: str, label: str) -> pd.DataFrame:
    """Заполняет пропуски медианой в группе (тип дома × десятилетие постройки).

    Высота потолков и этажность дома определяются эпохой и материалом
    постройки — а не тем, какие соседние дома стоят рядом (в одном районе
    вполне могут соседствовать старая панельная пятиэтажка и новый монолитный
    небоскрёб с совсем другими потолками и этажностью). Поэтому вместо
    геолокационного k-NN группируем по (house_type, десятилетие).
    """
    df = df.copy()
    missing_mask = df[target_col].isnull()
    if missing_mask.sum() == 0:
        card(f"{label}: пропусков нет", accent="#27AE60")
        return df

    decade = (df["construction_year"] // 10 * 10)
    known_mask = ~missing_mask
    group_median = df.loc[known_mask].groupby([df.loc[known_mask, "house_type"], decade.loc[known_mask]])[target_col].median()
    overall_median = df.loc[known_mask, target_col].median()

    keys = list(zip(df.loc[missing_mask, "house_type"], decade.loc[missing_mask]))
    fill_values = [group_median.get(k, overall_median) for k in keys]
    df.loc[missing_mask, target_col] = fill_values

    card(
        f"{label} восстановлена — медиана по (тип дома × десятилетие постройки)",
        rows=[("Было пропусков", str(missing_mask.sum())),
              ("Осталось пропусков", str(df[target_col].isnull().sum()))],
        note="Например, для старых панельных домов 1980-х берём медиану среди "
             "других известных панельных домов той же эпохи — не соседей по координатам.",
        accent="#2F80ED",
    )
    return df


def impute_ceiling_height_by_type(df: pd.DataFrame) -> pd.DataFrame:
    return _impute_by_house_type_and_decade(df, "ceiling_height", "Высота потолков")


def impute_house_type_knn(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    df = df.copy()
    missing_mask = df["house_type"].isnull()
    if missing_mask.sum() == 0:
        card("Тип дома: пропусков нет", accent="#27AE60")
        return df

    known = df.loc[~missing_mask].copy()
    feat_cols = ["lat", "lon", "construction_year"]
    idx = _nearest_indices(known[feat_cols].values, df.loc[missing_mask, feat_cols].values, k=k)

    neighbor_types = known["house_type"].values[idx]
    modes = pd.DataFrame(neighbor_types).mode(axis=1)[0].values
    df.loc[missing_mask, "house_type"] = modes

    card(
        f"Тип дома восстановлен (k-NN, k={k}, мода по координатам + году постройки)",
        rows=[("Было пропусков", str(missing_mask.sum())),
              ("Осталось пропусков", str(df["house_type"].isnull().sum()))],
        accent="#2F80ED",
    )
    return df


def impute_floor_count_by_type(df: pd.DataFrame) -> pd.DataFrame:
    df = _impute_by_house_type_and_decade(df, "floor_count", "Этажность дома")
    df["floor_count"] = np.round(df["floor_count"]).astype(int)
    return df


def finalize_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Убирает строки без этажа (не восстановить), текстовые служебные колонки."""
    df = df.copy()
    n0 = len(df)
    df = df.dropna(subset=["floor"]).reset_index(drop=True)

    drop_cols = ["name", "address", "information", "url", "complex_name", "bathroom_info", "owner"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    card(
        "Очистка завершена",
        rows=[
            ("Удалено строк без этажа", str(n0 - len(df))),
            ("Итоговый размер", f"{df.shape[0]:,} строк × {df.shape[1]} колонок".replace(",", " ")),
            ("Оставшиеся пропуски", str(int(df.isnull().sum().sum()))),
        ],
        accent="#27AE60",
    )
    display(df.head())
    return df
