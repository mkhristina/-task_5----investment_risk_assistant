
import os
import itertools

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl


INPUT_PATH = "data/data_clean.csv"
OUTPUT_PATH = "data/data_with_fuzzy.csv"


def create_fuzzy_system(input_names, output_name):
    """Создаёт нечёткую систему с тремя входами и полным набором правил."""
    inputs = {
        name: ctrl.Antecedent(np.arange(0, 101, 1), name)
        for name in input_names
    }
    output = ctrl.Consequent(np.arange(0, 101, 1), output_name)

    for variable in list(inputs.values()) + [output]:
        variable["low"] = fuzz.trimf(variable.universe, [0, 0, 50])
        variable["medium"] = fuzz.trimf(variable.universe, [0, 50, 100])
        variable["high"] = fuzz.trimf(variable.universe, [50, 100, 100])

    labels = ["low", "medium", "high"]
    risk_values = {"low": 20, "medium": 50, "high": 80}
    rules = []

    # 27 комбинаций для трёх входов.
    # Результат зависит от количества и уровня рисковых факторов.
    for combination in itertools.product(labels, repeat=3):
        average = sum(risk_values[label] for label in combination) / 3

        if average < 40:
            result = "low"
        elif average < 65:
            result = "medium"
        else:
            result = "high"

        condition = (
            inputs[input_names[0]][combination[0]]
            & inputs[input_names[1]][combination[1]]
            & inputs[input_names[2]][combination[2]]
        )
        rules.append(ctrl.Rule(condition, output[result]))

    return ctrl.ControlSystem(rules)


def normalize(value, minimum, maximum):
    """Нормализует число до диапазона 0–100."""
    if maximum <= minimum:
        return 50.0
    return float(np.clip(
        (value - minimum) / (maximum - minimum) * 100,
        0,
        100
    ))


def calculate_fuzzy_score(simulation, values, output_name):
    for name, value in values.items():
        simulation.input[name] = float(np.clip(value, 0, 100))

    simulation.compute()
    return float(simulation.output[output_name])


def calculate_indices(row, simulations):
    # Финансовый риск.
    profitability = max(float(row["profitability"]), 0)
    roi = max(float(row["roi_index"]), 0)

    profitability_score = normalize(
        np.log1p(profitability), 0, np.log1p(1000)
    )
    roi_score = normalize(
        np.log1p(roi), 0, np.log1p(100)
    )
    payback_risk = normalize(
        float(row["payback_period"]), 1, 10
    )

    financial_risk = calculate_fuzzy_score(
        simulations["financial"],
        {
            "profitability_risk": 100 - profitability_score,
            "roi_risk": 100 - roi_score,
            "payback_risk": payback_risk,
        },
        "financial_risk",
    )

    # Рыночный риск.
    market_risk = calculate_fuzzy_score(
        simulations["market"],
        {
            "market_volatility": normalize(
                float(row["market_volatility"]), 0, 100
            ),
            "regulatory_risk": normalize(
                float(row["regulatory_risk"]), 0, 100
            ),
            "competition_risk": normalize(
                float(row["competition_level"]), 1, 10
            ),
        },
        "market_risk",
    )

    # Организационный риск.
    organizational_risk = calculate_fuzzy_score(
        simulations["organization"],
        {
            "team_risk": 100 - normalize(
                float(row["team_experience"]), 0, 20
            ),
            "technology_risk": 100 - normalize(
                float(row["technology_readiness"]), 1, 9
            ),
            "strategy_risk": 100 - normalize(
                float(row["strategic_fit"]), 1, 10
            ),
        },
        "organizational_risk",
    )

    # Иерархический агрегатор.
    fuzzy_risk = calculate_fuzzy_score(
        simulations["aggregator"],
        {
            "financial_index": financial_risk,
            "market_index": market_risk,
            "organizational_index": organizational_risk,
        },
        "fuzzy_risk",
    )

    # Небольшая добавка от непрерывного среднего трёх подиндексов
    # помогает сохранить различия между проектами.
    average_risk = (
        0.4 * financial_risk
        + 0.35 * market_risk
        + 0.25 * organizational_risk
    )

    final_risk = 0.4 * fuzzy_risk + 0.6 * average_risk
    final_risk = float(np.clip(final_risk, 0, 100))

    return {
        "financial_index": round(financial_risk, 2),
        "market_index": round(market_risk, 2),
        "organizational_index": round(organizational_risk, 2),
        "fuzzy_risk": round(final_risk, 2),
    }


def run_fuzzy_model():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Не найден файл {INPUT_PATH}. "
            "Сначала запусти: python data_cleaner.py"
        )

    df = pd.read_csv(INPUT_PATH)

    required = [
        "profitability",
        "roi_index",
        "payback_period",
        "market_volatility",
        "regulatory_risk",
        "competition_level",
        "team_experience",
        "technology_readiness",
        "strategic_fit",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"В данных отсутствуют столбцы: {missing}")

    if df[required].isna().any().any():
        raise ValueError(
            "Есть пропуски во входных данных. "
            "Сначала запусти python data_cleaner.py"
        )

    print("Создаём финансовую нечёткую систему...")
    financial_system = create_fuzzy_system(
        ["profitability_risk", "roi_risk", "payback_risk"],
        "financial_risk",
    )

    print("Создаём рыночную нечёткую систему...")
    market_system = create_fuzzy_system(
        ["market_volatility", "regulatory_risk", "competition_risk"],
        "market_risk",
    )

    print("Создаём организационную нечёткую систему...")
    organization_system = create_fuzzy_system(
        ["team_risk", "technology_risk", "strategy_risk"],
        "organizational_risk",
    )

    print("Создаём агрегатор...")
    aggregator_system = create_fuzzy_system(
        ["financial_index", "market_index", "organizational_index"],
        "fuzzy_risk",
    )

    simulations = {
        "financial": ctrl.ControlSystemSimulation(financial_system),
        "market": ctrl.ControlSystemSimulation(market_system),
        "organization": ctrl.ControlSystemSimulation(organization_system),
        "aggregator": ctrl.ControlSystemSimulation(aggregator_system),
    }

    results = []

    print(f"Рассчитываем риск для {len(df)} проектов...")

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        results.append(calculate_indices(row, simulations))

        if i % 500 == 0:
            print(f"Обработано проектов: {i}/{len(df)}")

    result_df = pd.DataFrame(results, index=df.index)
    output_df = pd.concat([df, result_df], axis=1)

    os.makedirs("data", exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print("\nМодель успешно завершила расчёт.")
    print(f"Результат сохранён: {OUTPUT_PATH}")

    print("\nСтатистика индексов:")
    print(
        output_df[
            [
                "financial_index",
                "market_index",
                "organizational_index",
                "fuzzy_risk",
            ]
        ].describe().round(2)
    )

    print("\nПримеры проектов:")
    print(
        output_df[
            [
                "financial_index",
                "market_index",
                "organizational_index",
                "fuzzy_risk",
            ]
        ].head(10).to_string(index=False)
    )

    return output_df


if __name__ == "__main__":
    run_fuzzy_model()
