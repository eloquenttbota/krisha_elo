"""Строит вектор признаков для одного объявления — зеркало пайплайна
feature engineering из ноутбука (src/feature_engineering.py), но без
восстановления пропусков: пользователь бота вводит все значения явно,
поэтому все *_was_missing флаги всегда равны 0.
"""
import numpy as np
import pandas as pd

# Категории, реально присутствующие в обучающих данных (после clean_district).
DISTRICTS = [
    "Есильский р-н",
    "Нура р-н",
    "Алматы р-н",
    "Сарыарка р-н",
    "Сарайшык р-н",
    "р-н Байконур",
]

HOUSE_TYPES = [
    "монолитный",
    "кирпичный",
    "панельный",
    "иной",
]

# OneHotEncoder(drop="first") в ноутбуке отбрасывает первую по алфавиту
# категорию как базовую — у district это "Алматы р-н", у house_type "иной".
DISTRICT_OHE = [d for d in DISTRICTS if d != "Алматы р-н"]
HOUSE_TYPE_OHE = [h for h in HOUSE_TYPES if h != "иной"]

# Самый дорогой район по медианной цене/м² (определено в EDA ноутбука).
PREMIUM_DISTRICT = "Есильский р-н"


def get_year_category(year: int) -> int:
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
    else:
        return 7


def build_features(
    area: float,
    room_count: int,
    floor: int,
    floor_count: int,
    construction_year: int,
    ceiling_height: float,
    distance_to_center: float,
    district: str,
    house_type: str,
    owner: str,
    in_complex: bool,
    feature_names: list[str],
) -> pd.DataFrame:
    year_category = get_year_category(construction_year)
    is_high_ceiling = int(ceiling_height >= 3.0)
    owner_direct = int(owner == "Хозяин")

    premium_factor = is_high_ceiling * int(district == PREMIUM_DISTRICT)
    economy_old_panel = int(year_category <= 3 and house_type == "панельный")

    floor_bin = pd.cut([floor], bins=[0, 1, 5, 15, 100], labels=[0, 1, 2, 3], include_lowest=True)[0]
    floor_ratio = floor / floor_count if floor_count else 0.0

    district_cols = {f"district_{d}": int(district == d) for d in DISTRICT_OHE}
    house_type_cols = {f"house_type_{h}": int(house_type == h) for h in HOUSE_TYPE_OHE}

    row = {
        "ceiling_height": ceiling_height,
        "area": area,
        "room_count": room_count,
        "floor": floor,
        "floor_count": floor_count,
        "owner_direct": owner_direct,
        "in_complex": int(in_complex),
        "ceiling_height_was_missing": 0,
        "house_type_was_missing": 0,
        "floor_count_was_missing": 0,
        "distance_to_center": distance_to_center,
        "year_category": year_category,
        "premium_factor": premium_factor,
        "economy_old_panel": economy_old_panel,
        "floor_bin": int(floor_bin),
        "floor_ratio": floor_ratio,
        "is_top_floor": int(floor == floor_count),
        "is_ground_floor": int(floor == 1),
        **district_cols,
        **house_type_cols,
    }

    df = pd.DataFrame([row])
    # Оставляем только те признаки, которые использовала модель, в её порядке
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]
