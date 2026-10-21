import os
import numpy as np
import pandas as pd


def generate_data(n=2000, random_state=42):
    """Генерирует синтетические проекты с обоснованным классом риска,
    пропусками и выбросами для последующей обработки.
    """
    if n < 1:
        raise ValueError("Количество проектов n должно быть положительным.")

    rng = np.random.default_rng(random_state)

    # 1. Генерируем исходные параметры проектов
    df = pd.DataFrame({
        "project_cost": rng.uniform(1, 100, n).round(2),
        "expected_profit": rng.uniform(0.5, 150, n).round(2),
        "payback_period": rng.uniform(1, 10, n).round(2),
        "team_experience": rng.uniform(0, 20, n).round(1),
        "market_volatility": rng.uniform(0, 100, n).round(1),
        "technology_readiness": rng.integers(1, 10, n),
        "competition_level": rng.integers(1, 11, n),
        "strategic_fit": rng.integers(1, 11, n),
        "regulatory_risk": rng.uniform(0, 100, n).round(1),
        "sector": rng.choice(
            ["производство", "ИТ", "Логистика", "Энергетика"], n
        ),
    })

    # 2. Рассчитываем показатели на чистых исходных данных
    df["profitability"] = (
        df["expected_profit"] / df["project_cost"] * 100
    ).round(2)

    df["roi_index"] = (
        df["expected_profit"] * df["payback_period"]
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

    # 3. Строим непрерывный показатель риска.
    # Высокая окупаемость в годах, волатильность, регуляторный риск
    # и конкуренция повышают риск.
    # Прибыльность, опыт команды, готовность технологии
    # и соответствие стратегии снижают риск.
    def normalize(series):
        min_value = series.min()
        max_value = series.max()
        if max_value == min_value:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - min_value) / (max_value - min_value)

    risk_score = (
        0.18 * normalize(df["payback_period"])
        + 0.16 * normalize(df["market_volatility"])
        + 0.14 * normalize(df["regulatory_risk"])
        + 0.10 * normalize(df["competition_level"])
        + 0.12 * (1 - normalize(df["profitability"]))
        + 0.10 * (1 - normalize(df["team_experience"]))
        + 0.10 * (1 - normalize(df["technology_readiness"]))
        + 0.10 * (1 - normalize(df["strategic_fit"]))
    )

    # Добавляем небольшой шум, чтобы данные не были идеально разделимыми.
    risk_score += rng.normal(0, 0.025, n)

    # 4. Назначаем классы по квантилям.
    # Это обеспечивает примерно 60/25/15 в выборке,
    # сохраняя связь класса с параметрами проекта.
    threshold_60 = risk_score.quantile(0.60)
    threshold_85 = risk_score.quantile(0.85)

    df["risk_class"] = np.select(
        [
            risk_score <= threshold_60,
            risk_score <= threshold_85,
        ],
        [0, 1],
        default=2,
    ).astype(int)

    # 5. Сохраняем чистую целевую переменную и эталонные значения
    # до искусственного внесения пропусков и выбросов.
    df["risk_score_reference"] = risk_score.round(4)

    # 6. Добавляем пропуски в исходные признаки (~5% в каждом столбце)
    for col in ["team_experience", "regulatory_risk", "expected_profit"]:
        mask = rng.random(n) < 0.05
        df.loc[mask, col] = np.nan

    # 7. Добавляем выбросы стоимости и прибыли (~3% записей).
    # Целевой класс и эталонный риск при этом не изменяются.
    outlier_idx = rng.choice(
        df.index, size=max(1, int(n * 0.03)), replace=False
    )

    df.loc[outlier_idx, "project_cost"] *= rng.uniform(
        5, 10, len(outlier_idx)
    )
    df.loc[outlier_idx, "expected_profit"] *= rng.uniform(
        0.05, 0.2, len(outlier_idx)
    )

    # 8. Сохраняем результат
    os.makedirs("data", exist_ok=True)
    output_path = "data/data_raw.csv"
    df.to_csv(output_path, index=False)

    print(f"Сгенерировано проектов: {len(df)}")
    print("\nРаспределение классов риска:")
    print(
        df["risk_class"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(1)
        .rename("Доля, %")
    )
    print(f"\nФайл сохранён: {output_path}")

    return df


if __name__ == "__main__":
    generate_data()
