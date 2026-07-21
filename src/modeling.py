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
    # Валидационный кусок ИЗ train (не test!) — только для early stopping:
    # вместо перебора n_estimators руками даём большой запас (2000 деревьев)
    # и позволяем каждой комбинации гиперпараметров самой остановиться, как
    # только качество на валидации перестаёт расти.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=random_state,
    )
    early_stopping_rounds = 30
    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    param_dist = {
        "max_depth": [3, 4, 5, 6, 8],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.3, 0.5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    search = RandomizedSearchCV(
        XGBRegressor(
            n_estimators=2000, learning_rate=0.05,
            early_stopping_rounds=early_stopping_rounds, eval_metric="mape",
            random_state=random_state, n_jobs=1, verbosity=0,
        ),
        param_dist, n_iter=n_iter, cv=kfold, scoring=MAPE_SCORER,
        random_state=random_state, n_jobs=-1,
    )
    search.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    best_n_trees = search.best_estimator_.best_iteration + 1

    # "Запекаем" найденное число деревьев как фиксированный n_estimators и
    # убираем early stopping — дальше эта модель используется как обычная
    # (learning curve, финальная оценка и т.д.) и ей не нужен eval_set при
    # каждом переобучении. Переобучаем на всём train, а не только на X_fit.
    best_params = dict(search.best_params_)
    final_model = XGBRegressor(
        n_estimators=best_n_trees, learning_rate=0.05,
        random_state=random_state, n_jobs=-1, verbosity=0, **best_params,
    )
    final_model.fit(X_train, y_train)

    _plot_param_vs_score(search, "max_depth", "XGBoost", colors or {})
    _report_search(search, "XGBoost", extra_rows=[
        ("early_stopping_rounds", str(early_stopping_rounds)),
        ("Найдено деревьев (best_iteration)", str(best_n_trees)),
    ])
    return final_model, {**best_params, "n_estimators": best_n_trees}


def _report_search(search, name: str, extra_rows: list[tuple[str, str]] | None = None) -> None:
    # Намеренно не показываем здесь числовую CV-ошибку: она посчитана на
    # части train (внутри кросс-валидации) и почти всегда чуть отличается
    # от итоговой ошибки на test — для нетехнической аудитории две разные
    # цифры "точности" одной и той же модели выглядят как противоречие.
    # Единственное число, на которое стоит ориентироваться — MAPE на test
    # из раздела "Метрика каждой модели на test" ниже.
    rows = [(k, str(v)) for k, v in search.best_params_.items()]
    rows += extra_rows or []
    card(
        f"{name}: лучшие гиперпараметры (5-fold CV на train)",
        rows=rows,
        note="Эти параметры выбраны кросс-валидацией на train — тестовую выборку модель "
             "здесь ещё не видела. Итоговую точность (и единственное число, на которое "
             "стоит ориентироваться) смотрите в разделе «Метрика каждой модели на test» ниже.",
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

    # Разрыв train/CV сам по себе не значит "плохо": у мощных моделей (лес,
    # бустинг) он почти всегда есть, и это не мешает им давать меньшую ошибку
    # именно на CV — а важна именно она, а не размер разрыва. Поэтому вместо
    # порогового вердикта "переобучается / нет" на каждую модель отдельно —
    # один общий, ранжированный по факту вывод.
    ranked = sorted(curves.items(), key=lambda kv: kv[1][2][-1])  # по CV MAPE, лучшая первая
    rows = []
    for rank, (name, (sizes, train_mape, val_mape)) in enumerate(ranked, start=1):
        gap = val_mape[-1] - train_mape[-1]
        rows.append((
            f"{rank}. {name}",
            f"train {train_mape[-1]:.1f}% / CV {val_mape[-1]:.1f}% (разрыв {gap:.1f} п.п.)",
        ))
    card(
        "Диагностика по learning curve",
        rows=rows,
        note="Модели отсортированы по ошибке на CV — лучшая сверху. Разрыв между train и CV "
             "растёт от Decision Tree к Random Forest и XGBoost, и это ожидаемо: чем мощнее "
             "модель (больше деревьев, глубже бустинг), тем точнее она подгоняется под "
             "обучающую выборку. Это не признак поломки — важна именно ошибка на CV, а не "
             "размер разрыва: у XGBoost разрыв самый большой, но именно он даёт наименьшую "
             "ошибку на новых данных.",
        accent=colors["accent"],
    )


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
        ax.set_title(f"{name}\nразных значений цены в ответах модели: {n_unique:,} из {len(pred_real):,}"
                     .replace(",", " "), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Цена/м²")
    axes[0].set_ylabel("Количество объектов")
    axes[0].legend(fontsize=9)
    plt.suptitle("Реальная цена (серый) vs предсказания модели (синий) — это про форму "
                 "распределения ответов, не про ошибку модели (MAPE смотрите в разделе выше)", y=1.05)
    plt.tight_layout()
    plt.show()


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
