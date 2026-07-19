"""Обучение, подбор гиперпараметров и оценка моделей.

Ключевой принцип: тестовая выборка (X_test/y_test) создаётся один раз и
никогда не участвует в кросс-валидации или подборе гиперпараметров —
только в финальной оценке. Это и есть защита от утечки данных.

Единственная метрика — MAPE (средняя ошибка в процентах от реальной цены).
Она одна понятна любому нетехническому человеку без пояснений, поэтому
именно её мы и оптимизируем на кросс-валидации, и показываем в финале —
R², MAE и другие метрики в тенге сознательно не используются.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split, KFold, GridSearchCV, RandomizedSearchCV, learning_curve,
)
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.viz_utils import card

RANDOM_STATE = 42


# ─── Метрика для подбора гиперпараметров: MAPE в реальной шкале (%) ───────
# y хранится как log1p(цена/м²); чтобы кросс-валидация оптимизировала
# понятную величину, а не абстрактный логарифм, ошибка считается после
# обратного преобразования np.expm1.

def _neg_mape_real_scale(y_true_log, y_pred_log):
    return -mean_absolute_percentage_error(np.expm1(y_true_log), np.expm1(y_pred_log)) * 100


MAPE_SCORER = make_scorer(_neg_mape_real_scale, greater_is_better=True)


def split_data(df: pd.DataFrame, target: str = "price_per_m2",
                test_size: float = 0.2, random_state: int = RANDOM_STATE):
    obj_cols = df.select_dtypes("object").columns.tolist()
    assert not obj_cols, f"В данных остались текстовые колонки: {obj_cols}"

    y = np.log1p(df[target])
    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    card(
        "Train / Test разделение",
        rows=[
            ("Train", f"{len(X_train):,} объектов".replace(",", " ")),
            ("Test (отложенная выборка)", f"{len(X_test):,} объектов ({test_size:.0%})".replace(",", " ")),
            ("Признаков", str(X_train.shape[1])),
            ("Цель", "log(цена за м²) — для симметричности распределения"),
        ],
        note="Test-выборка «запечатана»: она не используется ни в кросс-валидации, "
             "ни в подборе гиперпараметров — только один раз, в самом конце, для честной оценки.",
        accent="#2F80ED",
    )
    return X_train, X_test, y_train, y_test


# ─── Подбор гиперпараметров (кросс-валидация только на train) ─────────────

def tune_decision_tree(X_train, y_train, cv: int = 5, random_state: int = RANDOM_STATE,
                        colors: dict = None):
    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    param_grid = {
        "max_depth": list(range(3, 16)),
        "min_samples_leaf": [5, 10, 20, 40],
    }
    search = GridSearchCV(
        DecisionTreeRegressor(random_state=random_state),
        param_grid, cv=kfold, scoring=MAPE_SCORER, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    _plot_validation_curve_dt(search, colors or {})
    _report_search(search, "Decision Tree")
    return search.best_estimator_, search.best_params_


def tune_random_forest(X_train, y_train, cv: int = 5, random_state: int = RANDOM_STATE,
                        n_iter: int = 15, colors: dict = None):
    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    param_dist = {
        "n_estimators": [100, 150, 200, 300],
        "max_depth": [8, 12, 16, 20, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", 0.6],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=random_state, n_jobs=-1),
        param_dist, n_iter=n_iter, cv=kfold, scoring=MAPE_SCORER,
        random_state=random_state, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    _plot_param_vs_score(search, "n_estimators", "Random Forest", colors or {})
    _report_search(search, "Random Forest")
    return search.best_estimator_, search.best_params_


def tune_xgboost(X_train, y_train, cv: int = 5, random_state: int = RANDOM_STATE,
                  n_iter: int = 20, colors: dict = None):
    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    search = RandomizedSearchCV(
        XGBRegressor(random_state=random_state, n_jobs=-1, verbosity=0),
        param_dist, n_iter=n_iter, cv=kfold, scoring=MAPE_SCORER,
        random_state=random_state, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    _plot_param_vs_score(search, "n_estimators", "XGBoost", colors or {})
    _report_search(search, "XGBoost")
    return search.best_estimator_, search.best_params_


def _report_search(search, name: str) -> None:
    card(
        f"{name}: лучшие гиперпараметры (5-fold CV на train)",
        rows=[(k, str(v)) for k, v in search.best_params_.items()]
        + [("CV MAPE", f"{-search.best_score_:.1f}%")],
        accent="#27AE60",
    )


def _plot_validation_curve_dt(search, colors) -> None:
    res = pd.DataFrame(search.cv_results_)
    res["mape"] = -res["mean_test_score"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for leaf in sorted(res["param_min_samples_leaf"].unique()):
        sub = res[res["param_min_samples_leaf"] == leaf].sort_values("param_max_depth")
        ax.plot(sub["param_max_depth"], sub["mape"], marker="o", markersize=4,
                 label=f"min_samples_leaf={leaf}")
    ax.axvline(search.best_params_["max_depth"], color=colors.get("neg", "#EB5757"),
                linestyle="--", label="выбранная глубина")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("CV MAPE, %")
    ax.set_title("Validation curve — Decision Tree\n(рост MAPE справа = переобучение)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def _plot_param_vs_score(search, param_name: str, model_name: str, colors) -> None:
    res = pd.DataFrame(search.cv_results_)
    res["mape"] = -res["mean_test_score"]
    col = f"param_{param_name}"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(res[col].astype(str), res["mape"], color=colors.get("primary", "#2F80ED"), alpha=0.7)
    best_val = str(search.best_params_[param_name])
    best_mape = -search.best_score_
    ax.scatter([best_val], [best_mape], color=colors.get("neg", "#EB5757"), s=120,
               zorder=5, label="лучшая комбинация")
    ax.set_xlabel(param_name)
    ax.set_ylabel("CV MAPE, %")
    ax.set_title(f"Поиск гиперпараметров — {model_name}")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ─── Learning curve: диагностика недо-/переобучения (все модели вместе) ────

def plot_learning_curves_grid(models: dict, X_train, y_train, colors: dict,
                               cv: int = 5, random_state: int = RANDOM_STATE) -> None:
    """Learning curve для каждой модели — в один ряд, в одном масштабе по Y,
    чтобы модели можно было честно сравнить взглядом, а не по памяти."""
    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    curves = {}
    for name, model in models.items():
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=kfold, scoring=MAPE_SCORER,
            train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1, random_state=random_state,
        )
        curves[name] = (train_sizes, -train_scores.mean(axis=1), -val_scores.mean(axis=1))

    all_vals = np.concatenate([np.concatenate([tm, vm]) for _, tm, vm in curves.values()])
    ylim = (0, all_vals.max() * 1.15)

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, (sizes, train_mape, val_mape)) in zip(axes, curves.items()):
        ax.plot(sizes, train_mape, marker="o", color=colors["primary"], label="Train MAPE")
        ax.plot(sizes, val_mape, marker="o", color=colors["neg"], label="CV MAPE")
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Размер обучающей выборки")
        ax.set_ylim(*ylim)
    axes[0].set_ylabel("MAPE, %")
    axes[0].legend(fontsize=9)
    plt.suptitle("Learning curves — одинаковый масштаб для честного сравнения моделей", y=1.03)
    plt.tight_layout()
    plt.show()

    rows = []
    for name, (sizes, train_mape, val_mape) in curves.items():
        gap = val_mape[-1] - train_mape[-1]
        verdict = "переобучается (большой разрыв train/CV)" if gap > 0.3 * train_mape[-1] else "обобщает адекватно"
        rows.append((name, verdict))
    card("Диагностика по learning curve", rows=rows, accent=colors["accent"])


# ─── Итоговая оценка на отложенном test ────────────────────────────────────

def evaluate_model(model, X_test, y_test, name: str, colors: dict) -> dict:
    pred_log = model.predict(X_test)
    y_real, pred_real = np.expm1(y_test), np.expm1(pred_log)

    metrics = {
        "name": name,
        "mape": mean_absolute_percentage_error(y_real, pred_real) * 100,
    }

    card(
        f"{name}: итоговая метрика на test",
        rows=[("MAPE", f"{metrics['mape']:.1f}%")],
        note="MAPE — средняя ошибка модели в процентах от реальной цены: "
             "«в среднем модель промахивается на N% от того, сколько квартира стоит на самом деле». ",
        accent=colors["primary"],
    )
    return metrics


# ─── Как на самом деле выглядят предсказания деревьев (не как в линейной регрессии) ─

def plot_predictions_grid(models: dict, X_test, y_test, colors: dict) -> None:
    y_real = np.expm1(y_test)
    preds = {name: np.expm1(model.predict(X_test)) for name, model in models.items()}

    all_vals = np.concatenate([y_real.values] + list(preds.values()))
    bins = np.linspace(all_vals.min(), all_vals.max(), 60)

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, pred_real) in zip(axes, preds.items()):
        ax.hist(y_real, bins=bins, color=colors["neutral"], alpha=0.5, label="Реальная цена")
        ax.hist(pred_real, bins=bins, color=colors["primary"], alpha=0.6, label="Предсказание модели")
        n_unique = len(np.unique(np.round(pred_real, 0)))
        pct_unique = n_unique / len(pred_real) * 100
        ax.set_title(f"{name}\n{n_unique:,} уникальных предсказаний из {len(pred_real):,} ({pct_unique:.0f}%)"
                     .replace(",", " "), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Цена/м²")
    axes[0].set_ylabel("Количество объектов")
    axes[0].legend(fontsize=9)
    plt.suptitle("Реальная цена (серый) vs предсказания модели (синий)", y=1.05)
    plt.tight_layout()
    plt.show()


# ─── Важность признаков (все модели вместе, одинаковый масштаб) ────────────

def plot_feature_importance_grid(models: dict, feature_names, colors: dict, top_n: int = 10) -> dict:
    importances = {
        name: pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        for name, model in models.items()
    }
    max_val = max(imp.head(top_n).max() for imp in importances.values())

    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5.5))
    axes = np.atleast_1d(axes)
    for ax, (name, imp) in zip(axes, importances.items()):
        imp.head(top_n).plot(kind="barh", ax=ax, color=colors["primary"])
        ax.invert_yaxis()
        ax.set_xlim(0, max_val * 1.1)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Feature importance")
    plt.suptitle(f"Топ-{top_n} важных признаков — одинаковый масштаб для сравнения", y=1.03)
    plt.tight_layout()
    plt.show()
    return importances


# ─── Сравнение моделей и выбор лучшей ──────────────────────────────────────

def compare_models(results: dict, colors: dict) -> str:
    table = pd.DataFrame(results).T[["mape"]]
    table.columns = ["MAPE %"]
    display_table = table.round(1)

    fig, ax = plt.subplots(figsize=(7, 5))
    display_table["MAPE %"].plot(kind="bar", ax=ax, color=colors["palette"][: len(table)])
    ax.set_title("MAPE по моделям (%, чем меньше — тем лучше)")
    ax.set_ylabel("MAPE, %")
    ax.tick_params(axis="x", rotation=0)
    for i, v in enumerate(display_table["MAPE %"]):
        ax.text(i, v + 0.1, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.show()

    best_name = table["MAPE %"].astype(float).idxmin()

    card(
        f"Лучшая модель: {best_name}",
        rows=[(idx, f"MAPE {row['MAPE %']:.1f}%") for idx, row in display_table.iterrows()],
        note=f"Модель «{best_name}» выбрана как продакшн-модель для бота — "
             "у неё наименьшая средняя ошибка в процентах (MAPE) на отложенном test. "
             "Это же число верно и для цены за м², и для полной стоимости квартиры.",
        accent=colors["pos"],
    )
    return best_name


def save_production_model(model, feature_names: list[str], model_name: str,
                           model_path: str = "model.pkl",
                           features_path: str = "feature_names.pkl") -> None:
    import joblib
    joblib.dump(model, model_path)
    joblib.dump(list(feature_names), features_path)
    card(
        "Модель сохранена для продакшна (используется ботом)",
        rows=[
            ("Алгоритм", model_name),
            ("Файл модели", model_path),
            ("Файл признаков", features_path),
            ("Признаков", str(len(feature_names))),
        ],
        accent="#27AE60",
    )
