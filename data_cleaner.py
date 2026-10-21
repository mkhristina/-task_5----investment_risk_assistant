import os
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


RAW_PATH = "data/data_raw.csv"
CLEAN_PATH = "data/data_clean.csv"

SOURCE_NUMERIC_COLS = [
    "project_cost",
    "expected_profit",
    "payback_period",
    "team_experience",
    "market_volatility",
    "technology_readiness",
    "competition_level",
    "strategic_fit",
    "regulatory_risk",
]

DERIVED_COLS = [
    "profitability",
    "roi_index",
    "risk_density",
    "capability_score",
]


def calculate_derived_features(df):
    """Пересчитывает производные показатели из очищенных признаков."""
    df = df.copy()

    df["profitability"] = (
        df["expected_profit"] / df["project_cost"] * 100
    ).round(2)

    df["roi_index"] = (
        df["expected_profit"]
        * df["payback_period"]
        / df["project_cost"]
    ).round(2)

    df["risk_density"] = (
        df["market_volatility"]
        + df["regulatory_risk"]
        + df["competition_level"] * 10
    ).div(3).round(2)

    df["capability_score"] = (
        df["team_experience"]
        * df["technology_readiness"]
        * df["strategic_fit"]
    ).round(2)

    return df


def clean_data():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Не найден файл {RAW_PATH}. "
            "Сначала запусти python data_generator.py"
        )

    df = pd.read_csv(RAW_PATH)

    print(f"Загружено строк: {len(df)}")
    print(f"Пропусков до очистки: {int(df.isna().sum().sum())}")

    # Проверяем необходимые столбцы
    missing_cols = set(SOURCE_NUMERIC_COLS + ["risk_class"]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"В исходных данных отсутствуют столбцы: {missing_cols}")

    # 1. Ограничиваем выбросы только в исходных числовых признаках.
    # Целевой класс и risk_score_reference не изменяем.
    for col in SOURCE_NUMERIC_COLS:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = series.clip(lower=lower, upper=upper)

    # 2. Заполняем пропуски методом KNN.
    # Масштабирование нужно, чтобы признаки с большими числами
    # не доминировали при вычислении расстояний.
    imputer_input = df[SOURCE_NUMERIC_COLS].copy()

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(imputer_input)

    imputer = KNNImputer(n_neighbors=5, weights="distance")
    imputed_scaled = imputer.fit_transform(scaled_values)

    restored_values = scaler.inverse_transform(imputed_scaled)

    df[SOURCE_NUMERIC_COLS] = restored_values

    # 3. Возвращаем значения к допустимым диапазонам признаков.
    # Это полезно после обратного преобразования KNN.
    bounds = {
        "project_cost": (1, 100),
        "expected_profit": (0.5, 150),
        "payback_period": (1, 10),
        "team_experience": (0, 20),
        "market_volatility": (0, 100),
        "technology_readiness": (1, 9),
        "competition_level": (1, 10),
        "strategic_fit": (1, 10),
        "regulatory_risk": (0, 100),
    }

    for col, (lower, upper) in bounds.items():
        df[col] = df[col].clip(lower=lower, upper=upper)

    # Признаки, изначально заданные целыми числами, округляем.
    for col in [
        "technology_readiness",
        "competition_level",
        "strategic_fit",
    ]:
        df[col] = df[col].round().astype(int)

    # 4. Пересчитываем производные показатели после очистки.
    df = calculate_derived_features(df)

    # Сохраняем исходный класс риска и справочный показатель риска.
    # Эти столбцы не используются для импутации.
    os.makedirs("data", exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    print(f"Пропусков после очистки: {int(df.isna().sum().sum())}")
    print(f"Очищенные данные сохранены: {CLEAN_PATH}")
    print("\nРаспределение классов после очистки:")
    print(df["risk_class"].value_counts().sort_index())

    return df


if __name__ == "__main__":
    clean_data()
