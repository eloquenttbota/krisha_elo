import numpy as np
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from preprocess import build_features, DISTRICTS, HOUSE_TYPES

BASE = Path(__file__).parent.parent
MODEL_PATH = BASE / "dt_model.pkl"
FEATURES_PATH = BASE / "feature_names.pkl"

app = FastAPI(title="Krisha Price Predictor")

model = joblib.load(MODEL_PATH)
feature_names: list[str] = joblib.load(FEATURES_PATH)


class ApartmentInput(BaseModel):
    area: float = Field(..., gt=0, description="Площадь, м²")
    room_count: int = Field(..., ge=1, le=10)
    floor: int = Field(..., ge=1)
    floor_count: int = Field(..., ge=1)
    construction_year: int = Field(..., ge=1950, le=2025)
    ceiling_height: float = Field(default=2.7, gt=1.5, le=6.0)
    distance_to_center: float = Field(..., gt=0, description="Расстояние до центра, км")
    district: str
    house_type: str
    owner: str = Field(..., pattern="^(Хозяин|Агентство)$")


class PredictionResponse(BaseModel):
    price_per_m2: int
    total_price: int


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/options")
def options():
    return {"districts": DISTRICTS, "house_types": HOUSE_TYPES}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: ApartmentInput):
    if data.floor > data.floor_count:
        raise HTTPException(status_code=400, detail="Этаж не может быть больше этажности дома")

    df = build_features(
        area=data.area,
        room_count=data.room_count,
        floor=data.floor,
        floor_count=data.floor_count,
        construction_year=data.construction_year,
        ceiling_height=data.ceiling_height,
        distance_to_center=data.distance_to_center,
        district=data.district,
        house_type=data.house_type,
        owner=data.owner,
        feature_names=feature_names,
    )

    log_pred = model.predict(df)[0]
    price_per_m2 = int(np.expm1(log_pred))
    total_price = int(price_per_m2 * data.area)

    return PredictionResponse(price_per_m2=price_per_m2, total_price=total_price)
