"""Разведывательный анализ (EDA) сырых данных krisha.kz, до очистки."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.display import display

from src.cleaning import parse_ceiling_height, clean_district, owner_is_direct
from src.geo import haversine_km, CENTER_LAT, CENTER_LON
from src.viz_utils import card, hypothesis_result


def eda_overview(df: pd.DataFrame, colors: dict) -> None:
    card(
        "Первый взгляд на данные",
        rows=[
            ("Числовых признаков", str(len(df.select_dtypes(include="number").columns))),
            ("Текстовых признаков", str(len(df.select_dtypes(include="object").columns))),
        ],
        accent=colors["primary"],
    )
    display(df.head())
    print("Описательная статистика (числовые):")
    display(df.describe())
    print("Описательная статистика (категориальные):")
    display(df.describe(include="object"))


def eda_missing(df: pd.DataFrame, colors: dict) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    missing_pct = (missing / len(df) * 100).round(2)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    missing_pct.plot(kind="barh", ax=ax, color=colors["neg"])
    ax.set_title("Доля пропусков по колонкам")
    ax.set_xlabel("% объявлений с пропуском")
    for i, (col, pct) in enumerate(missing_pct.items()):
        ax.text(pct + 0.5, i, f"{pct:.0f}%", va="center", fontsize=9, color=colors["neg"])
    ax.set_xlim(0, missing_pct.max() * 1.15)
    plt.tight_layout()
    plt.show()

    card(
        "Вывод",
        note="Больше всего пропусков — в bathroom_info (санузел) и ceiling_height "
             "(высота потолков): продавцы часто не указывают эти детали. "
             "district и house_type восстановимы по соседним объявлениям (геолокации).",
        accent=colors["neg"],
    )


def eda_price(df: pd.DataFrame, colors: dict) -> None:
    """Сырое распределение цены — без обрезки хвоста: EDA показывает данные
    как есть, очистка и удаление аномалий будет отдельным шагом (раздел 5)."""
    price = df["price"].dropna()
    median, mean = price.median(), price.mean()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(price, bins=80, color=colors["primary"], edgecolor="none")
    ax.axvline(median, color=colors["pos"], lw=2, linestyle="--")
    ax.axvline(mean, color=colors["neg"], lw=2, linestyle="--")
    ymax = ax.get_ylim()[1]
    ax.text(median, ymax * 0.95, f" медиана: {median/1e6:.0f} млн тг",
            color=colors["pos"], fontsize=10, fontweight="bold", va="top")
    ax.text(mean, ymax * 0.80, f" среднее: {mean/1e6:.0f} млн тг",
            color=colors["neg"], fontsize=10, fontweight="bold", va="top")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f} млн"))
    ax.set_xlabel("Цена, тенге (сырые данные, без обрезки)")
    ax.set_ylabel("Количество объявлений")
    ax.set_title("Распределение цен на квартиры в Астане — как есть, до очистки")
    plt.tight_layout()
    plt.show()

    q1, q3 = price.quantile(0.25), price.quantile(0.75)
    iqr = q3 - q1
    outliers = price[(price < q1 - 1.5 * iqr) | (price > q3 + 1.5 * iqr)]

    card(
        "Цена: распределение сильно скошено вправо",
        rows=[
            ("Минимум", f"{price.min():,.0f} тг".replace(",", " ")),
            ("Максимум", f"{price.max():,.0f} тг".replace(",", " ")),
            ("Медиана", f"{median:,.0f} тг".replace(",", " ")),
            ("Среднее", f"{mean:,.0f} тг".replace(",", " ")),
            ("Выбросов по формальному IQR", f"{len(outliers):,} ({len(outliers) / len(price) * 100:.1f}%)".replace(",", " ")),
        ],
        note="Среднее заметно выше медианы — значит, небольшое число очень дорогих "
             "квартир (до 1.6 млрд тг) тянет среднее вверх. На графике видны и они — "
             "мы намеренно не обрезаем хвост, чтобы честно показать сырые данные. "
             "Число выбросов по IQR здесь — чисто иллюстративная оценка «на глаз», "
             "по формальному статистическому правилу. Она НЕ используется напрямую "
             "для удаления строк: реальную очистку в разделе 5 мы делаем по границам "
             "реального рынка Астаны (а не по IQR), а хвост цены/м² при подготовке "
             "признаков обрезаем по перцентилям — поэтому итоговое число удалённых "
             "строк в этих разделах будет другим. А для самой модели цена будет "
             "прологарифмирована — так редкие дорогие квартиры не искажают обучение.",
        accent=colors["pos"],
    )


NUMERIC_LABELS = {
    "area": "Площадь, м²",
    "room_count": "Комнат",
    "construction_year": "Год постройки",
    "ceiling_height": "Высота потолков, м",
}


#: Дискретные признаки с небольшим числом уникальных значений — им нужны
#: бины, выровненные по целым числам, иначе гистограмма визуально "теряет"
#: реальный максимум (например, room_count доходит до 13, а не до 12).
DISCRETE_FEATURES = {"room_count"}


def eda_numeric(df: pd.DataFrame, colors: dict) -> None:
    """Полный диапазон значений как в сырых данных — без обрезки хвостов."""
    plot_df = df.copy()
    plot_df["ceiling_height"] = parse_ceiling_height(plot_df["ceiling_height"])
    num_feat = [c for c in NUMERIC_LABELS if c in plot_df.columns]

    fig, axes = plt.subplots(1, len(num_feat), figsize=(4.2 * len(num_feat), 4.5))
    for ax, col in zip(axes, num_feat):
        data = plot_df[col].dropna()
        lo, hi = data.min(), data.max()
        if col in DISCRETE_FEATURES:
            bins = np.arange(lo, hi + 2) - 0.5  # по одному бину на каждое целое значение
        else:
            bins = 35
        ax.hist(data, bins=bins, color=colors["primary"], edgecolor="none")
        median = data.median()
        ax.axvline(median, color=colors["neg"], lw=1.5, linestyle="--")
        ax.set_title(NUMERIC_LABELS[col], fontsize=13, fontweight="bold")
        ax.set_xlabel(f"мин {lo:,.0f} · медиана {median:,.1f} · макс {hi:,.0f}".replace(",", " "),
                      fontsize=9, color=colors["neutral"])
        ax.set_yticks([])
    plt.suptitle("Числовые признаки: полный диапазон сырых данных (min–max)", y=1.03)
    plt.tight_layout()
    plt.show()


def eda_categorical(df: pd.DataFrame, colors: dict) -> None:
    """Все категории без обрезки — их и так немного (не топ-N)."""
    plot_df = df.copy()
    plot_df["district"] = clean_district(plot_df["district"])
    cat_cols = [c for c in ["house_type", "district"] if c in plot_df.columns]

    fig, axes = plt.subplots(1, len(cat_cols), figsize=(7 * len(cat_cols), 5))
    if len(cat_cols) == 1:
        axes = [axes]
    for i, col in enumerate(cat_cols):
        vc = plot_df[col].value_counts(dropna=False)
        vc.plot(kind="barh", ax=axes[i], color=colors["primary"])
        for j, v in enumerate(vc.values):
            axes[i].text(v, j, f" {v:,}".replace(",", " "), va="center", fontsize=9)
        axes[i].set_title(f"{col} — все {len(vc)} категорий")
        axes[i].invert_yaxis()
    plt.tight_layout()
    plt.show()

    plot_df["price_per_sqm"] = plot_df["price"] / plot_df["area"]
    for col in cat_cols:
        print(f"\n{col} — медианная цена/м²:")
        print(plot_df.groupby(col)["price_per_sqm"].median().sort_values(ascending=False).round(0))


#: Широкий санитарный фильтр — только чтобы отсечь явно ошибочные координаты
#: (объекты за пределами Казахстана). Границы отображения ниже подбираются
#: не по этому фильтру, а по фактическому min/max оставшихся точек — так
#: рамка графика плотно облегает реальные данные, и выбросы на краю видно.
SANITY_LAT_RANGE = (50.8, 51.5)
SANITY_LON_RANGE = (71.0, 71.8)


def eda_geo(df: pd.DataFrame, colors: dict) -> None:
    geo = df.dropna(subset=["lat", "lon"])
    astana_mask = geo["lat"].between(*SANITY_LAT_RANGE) & geo["lon"].between(*SANITY_LON_RANGE)
    in_astana = geo.loc[astana_mask]
    n_anomaly = (~astana_mask).sum()

    # Плотная рамка по фактическим границам данных (+ небольшой отступ),
    # а не по округлённому боксу — иначе точки теряются в пустом поле.
    lat_lo, lat_hi = in_astana["lat"].min(), in_astana["lat"].max()
    lon_lo, lon_hi = in_astana["lon"].min(), in_astana["lon"].max()
    lat_pad = (lat_hi - lat_lo) * 0.03
    lon_pad = (lon_hi - lon_lo) * 0.03

    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(
        in_astana["lon"], in_astana["lat"],
        c=in_astana["price"] / in_astana["area"],
        cmap="plasma", alpha=0.45, s=10,
    )
    plt.colorbar(sc, ax=ax, label="тг/м²", shrink=0.8)
    ax.scatter([CENTER_LON], [CENTER_LAT], marker="*", s=550,
               color=colors["pos"], edgecolor="white", linewidth=1.3,
               zorder=5, label="Центр (Байтерек)")
    ax.set_xlim(lon_lo - lon_pad, lon_hi + lon_pad)
    ax.set_ylim(lat_lo - lat_pad, lat_hi + lat_pad)
    ax.set_title("Цена/м² по геолокации — Астана (масштаб по факт. границам данных)")
    ax.set_xlabel("Долгота")
    ax.set_ylabel("Широта")
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.show()

    card(
        "География",
        rows=[
            ("Точек на карте", f"{len(in_astana):,}".replace(",", " ")),
            ("Границы карты (широта)", f"{lat_lo:.3f} — {lat_hi:.3f}"),
            ("Границы карты (долгота)", f"{lon_lo:.3f} — {lon_hi:.3f}"),
            ("Аномальных координат (вне Казахстана, не на карте)", str(n_anomaly)),
        ],
        note="Яркие (жёлтые) точки — дороже, тёмно-фиолетовые — дешевле. "
             "Концентрация ярких точек у центра — первый визуальный намёк на гипотезу H4 ниже. "
             "Точки на самом краю рамки — самые удалённые/нетипичные объявления среди "
             "оставшихся в границах Астаны.",
        accent=colors["accent"],
    )


def eda_correlation(df: pd.DataFrame, colors: dict) -> None:
    plot_df = df.copy()
    plot_df["ceiling_height"] = parse_ceiling_height(plot_df["ceiling_height"])
    num_cols = plot_df.select_dtypes(include="number").columns.tolist()
    corr_df = plot_df[num_cols].copy()
    corr_df["price_per_sqm"] = plot_df["price"] / plot_df["area"]
    corr_matrix = corr_df.corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap=colors["cmap_div"],
                center=0, ax=axes[0], linewidths=0.5)
    axes[0].set_title("Spearman корреляция (числовые признаки)")

    corr_target = corr_matrix["price_per_sqm"].drop("price_per_sqm").sort_values()
    bar_colors = [colors["neg"] if x < 0 else colors["primary"] for x in corr_target]
    corr_target.plot(kind="barh", ax=axes[1], color=bar_colors)
    axes[1].set_title("Корреляция с ценой/м² (Spearman)")
    axes[1].axvline(0, color=colors["neutral"], linewidth=0.8)
    plt.tight_layout()
    plt.show()


def eda_hypotheses(df: pd.DataFrame, colors: dict) -> dict:
    """Проверка гипотез по конкретно этому датасету (не шаблонных)."""
    h = df.copy()
    h["price_per_sqm"] = h["price"] / h["area"]
    h["district_clean"] = clean_district(h["district"])
    h["owner_direct"] = owner_is_direct(h["owner"])
    h["in_complex"] = h["complex_name"].notnull().astype(int)

    results = {}

    # H1: площадь -> цена/м² (эффект масштаба)
    r1, p1 = stats.spearmanr(h["area"], h["price_per_sqm"])
    results["H1"] = hypothesis_result(
        "H1", "Больше площадь квартиры → ниже цена за м² (эффект масштаба)",
        "Spearman r", r1, p1,
    )

    # H2: район -> цена/м² (ANOVA)
    groups = [g["price_per_sqm"].dropna() for _, g in h.dropna(subset=["district_clean"]).groupby("district_clean")]
    f2, p2 = stats.f_oneway(*groups)
    results["H2"] = hypothesis_result(
        "H2", "Район города влияет на цену за м²",
        "F", f2, p2,
    )

    # H3: год постройки -> цена/м²
    yr = h.dropna(subset=["construction_year", "price_per_sqm"])
    r3, p3 = stats.spearmanr(yr["construction_year"], yr["price_per_sqm"])
    results["H3"] = hypothesis_result(
        "H3", "Более новый год постройки → выше цена за м²",
        "Spearman r", r3, p3,
    )

    # H4: расстояние от центра -> цена/м²
    geo = h.dropna(subset=["lat", "lon", "price_per_sqm"]).copy()
    geo["dist_km"] = haversine_km(geo["lat"], geo["lon"], CENTER_LAT, CENTER_LON)
    r4, p4 = stats.spearmanr(geo["dist_km"], geo["price_per_sqm"])
    results["H4"] = hypothesis_result(
        "H4", "Чем дальше от центра города, тем ниже цена за м²",
        "Spearman r", r4, p4,
    )

    # H5 (по этим данным): квартиры в ЖК ценятся выше объектов без указания ЖК
    with_complex = h.loc[h["in_complex"] == 1, "price_per_sqm"].dropna()
    without_complex = h.loc[h["in_complex"] == 0, "price_per_sqm"].dropna()
    u5, p5 = stats.mannwhitneyu(with_complex, without_complex, alternative="two-sided")
    results["H5"] = hypothesis_result(
        "H5", "Квартиры в жилых комплексах ценятся выше, чем объекты без указания ЖК",
        "Mann-Whitney U", u5, p5,
    )

    # H6 (по этим данным): прямые владельцы выставляют цену ниже, чем агентства/посредники
    direct = h.loc[h["owner_direct"] == 1, "price_per_sqm"].dropna()
    agency = h.loc[h["owner_direct"] == 0, "price_per_sqm"].dropna()
    u6, p6 = stats.mannwhitneyu(direct, agency, alternative="two-sided")
    results["H6"] = hypothesis_result(
        "H6", "Собственники («хозяин») указывают цену за м² ниже, чем агентства/посредники",
        "Mann-Whitney U", u6, p6,
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    dist_bins = [0, 2, 4, 6, 8, 10, 15, 100]
    dist_labels = ["0–2", "2–4", "4–6", "6–8", "8–10", "10–15", "15+"]
    geo["dist_band"] = pd.cut(geo["dist_km"], bins=dist_bins, labels=dist_labels)
    by_band = geo.groupby("dist_band", observed=True)["price_per_sqm"].median()
    axes[0].bar(by_band.index.astype(str), by_band.values, color=colors["primary"])
    axes[0].set_xlabel("Расстояние от центра, км")
    axes[0].set_ylabel("Медианная цена/м²")
    axes[0].set_title(f"H4: чем дальше от центра — тем дешевле (r={r4:.3f})")
    axes[0].tick_params(axis="x", rotation=0)

    district_median = h.groupby("district_clean")["price_per_sqm"].median().sort_values()
    (district_median / 1000).plot(kind="barh", ax=axes[1], color=colors["accent"])
    for j, v in enumerate(district_median.values):
        axes[1].text(v / 1000, j, f" {v/1000:,.0f}".replace(",", " "), va="center", fontsize=9)
    axes[1].set_title("H2: медианная цена/м² по районам")
    axes[1].set_xlabel("тыс. тг/м²")

    box_data = [with_complex, without_complex]
    axes[2].boxplot(box_data, patch_artist=True,
                     boxprops=dict(facecolor=colors["light"]), showfliers=False)
    axes[2].set_xticks([1, 2])
    axes[2].set_xticklabels(["В ЖК", "Без ЖК"])
    axes[2].set_title("H5: цена/м² — с ЖК vs без указания ЖК")
    axes[2].set_ylabel("Цена/м²")
    plt.tight_layout()
    plt.show()

    return results


def eda_summary(df: pd.DataFrame, hypothesis_results: dict, colors: dict) -> None:
    n_confirmed = sum(hypothesis_results.values())
    rows = [(tag, "✅ подтверждена" if ok else "❌ не подтверждена")
            for tag, ok in hypothesis_results.items()]
    card(
        "Итоги EDA",
        rows=[
            ("Гипотез подтверждено", f"{n_confirmed} из {len(hypothesis_results)}"),
        ] + rows,
        note="EDA завершён. Дальше — очистка данных и восстановление пропусков "
             "на основе того, что мы здесь увидели.",
        accent=colors["pos"],
    )
