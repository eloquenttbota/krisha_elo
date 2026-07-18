"""Загрузка сырых данных, полученных парсингом krisha.kz."""
from __future__ import annotations

import pandas as pd
from IPython.display import display

from src.viz_utils import card


def load_data(path: str, logo_path: str = "krisha.kz.png") -> pd.DataFrame:
    """Читает CSV с результатом парсинга и показывает краткую сводку."""
    df = pd.read_csv(path)

    card(
        "Данные загружены",
        rows=[
            ("Источник", "krisha.kz — веб-парсинг объявлений о продаже квартир в Астане"),
            ("Файл", f"{path} (локальная директория проекта, без скачивания)"),
            ("Объявлений", f"{len(df):,}".replace(",", " ")),
            ("Колонок", str(df.shape[1])),
            ("Диапазон цен", f"{df['price'].min():,.0f} — {df['price'].max():,.0f} тг".replace(",", " ")),
        ],
        logo_path=logo_path,
    )
    display(df.head())
    return df
