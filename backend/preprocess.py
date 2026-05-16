import numpy as np
import pandas as pd


DISTRICTS = [
    "Есильский р-н",
    "Нура р-н",
    "Сарайшык р-н",
    "Сарыарка р-н",
    "р-н Байконур",
]

HOUSE_TYPES = [
    "монолитный дом",
    "панельный дом",
]


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
    feature_names: list[str],
) -> pd.DataFrame:
    year_category = get_year_category(construction_year)
    is_high_ceiling = int(ceiling_height >= 3.0)
    owner_enc = int(owner == "Хозяин")

    district_cols = {f"district_{d}": int(district == d) for d in DISTRICTS}
    house_type_cols = {f"house_type_{h}": int(house_type == h) for h in HOUSE_TYPES}

    floor_bin = pd.cut([floor], bins=[0, 1, 5, 15, 100], labels=[0, 1, 2, 3], include_lowest=True)[0]
    area_bin = pd.cut([area], bins=[0, 45, 70, 100, 9999], labels=[0, 1, 2, 3], include_lowest=True)[0]

    row = {
        "area": area,
        "room_count": room_count,
        "floor ": floor,
        "floor_count": floor_count,
        "distance_to_center": distance_to_center,
        "owner_enc": owner_enc,
        "year_category": year_category,
        "is_high_ceiling": is_high_ceiling,
        **district_cols,
        **house_type_cols,
        "premium_factor": is_high_ceiling * district_cols.get("district_Есильский р-н", 0),
        "economy_old_panel": int(year_category <= 3) * house_type_cols.get("house_type_панельный дом", 0),
        "log_distance": np.log1p(distance_to_center),
        "area_per_room": area / room_count if room_count > 0 else area,
        "floor_bin": int(floor_bin),
        "area_bin": int(area_bin),
        "floor_ratio_x_ceiling": (floor / floor_count) * is_high_ceiling if floor_count > 0 else 0,
        "area_x_year": area * year_category,
        "rooms_x_year": room_count * year_category,
        "floor_ratio": floor / floor_count if floor_count > 0 else 0,
        "is_top_floor": int(floor == floor_count),
        "is_ground_floor": int(floor == 1),
        "log_area": np.log1p(area),
        "area_squared": area ** 2,
        "area_x_rooms": area * room_count,
        "distance_x_district_esilsky": distance_to_center * district_cols.get("district_Есильский р-н", 0),
    }

    df = pd.DataFrame([row])
    # Оставляем только те признаки, которые использовала модель
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]
