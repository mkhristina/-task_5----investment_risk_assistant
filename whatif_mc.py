
import os
import joblib
import numpy as np
import pandas as pd


MODEL_PATH = "models/best_model.joblib"
DATA_PATH = "data/data_with_fuzzy.csv"

CLASS_NAMES = {
    0: "Низкий риск",
    1: "Средний риск",
    2: "Высокий риск",
}

NOISE_COLUMNS = [
    "project_cost",
    "expected_profit",
    "payback_period",
    "market_volatility",
    "regulatory_risk",
]

RANDOM_SEED = 42


def load_resources():
    """Загружает модель, скалер, список признаков и данные."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Модель не найдена: {MODEL_PATH}. Сначала запустите ml_model.py."
        )

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Данные не найдены: {DATA_PATH}. Сначала запустите fuzzy_model.py."
        )

    bundle = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    required_keys = ["model", "scaler", "feature_cols"]
    missing_keys = [key for key in required_keys if key not in bundle]
    if missing_keys:
        raise ValueError(
            f"В сохранённом файле модели отсутствуют поля: {missing_keys}"
        )

    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]
    cost_matrix = bundle.get(
        "cost_matrix",
        np.array([
            [0, 1, 2],
            [5, 0, 1],
            [10, 5, 0],
        ]),
    )

    missing_columns = [col for col in feature_cols if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"В данных отсутствуют необходимые признаки: {missing_columns}"
        )

    return model, scaler, df, feature_cols, cost_matrix


def recalculate_derived_features(project):
    """Пересчитывает производные показатели после изменения исходных данных."""

    project = project.copy()

    cost = max(float(project["project_cost"]), 1e-6)
    profit = float(project["expected_profit"])
    payback = max(float(project["payback_period"]), 1e-6)

    if "profitability" in project.index:
        project["profitability"] = profit / cost * 100

    if "roi_index" in project.index:
        project["roi_index"] = profit / (cost / payback)

    if "risk_density" in project.index:
        project["risk_density"] = (
            float(project["market_volatility"])
            + float(project["regulatory_risk"])
            + float(project["competition_level"]) * 10
        ) / 3

    if "capability_score" in project.index:
        project["capability_score"] = (
            float(project["team_experience"])
            * float(project["technology_readiness"])
            * float(project["strategic_fit"])
        )

    return project


def predict_project(
    model,
    scaler,
    project,
    feature_cols,
    cost_matrix,
):
    """Возвращает предсказанный класс, вероятности и ожидаемые потери."""

    row = pd.DataFrame(
        [[float(project[col]) for col in feature_cols]],
        columns=feature_cols,
    )

    scaled_row = scaler.transform(row)
    probabilities = model.predict_proba(scaled_row)[0]

    classes = np.asarray(model.classes_, dtype=int)

    # Ожидаемые потери для каждого возможного предсказанного класса.
    # cost_matrix[истинный_класс, предсказанный_класс].
    expected_costs = []

    for predicted_class in classes:
        total_cost = 0.0
        for true_index, true_class in enumerate(classes):
            total_cost += (
                probabilities[true_index]
                * cost_matrix[true_class, predicted_class]
            )
        expected_costs.append(total_cost)

    predicted_class = int(classes[np.argmin(expected_costs)])

    probability_by_class = {
        int(class_id): float(prob)
        for class_id, prob in zip(classes, probabilities)
    }

    return predicted_class, probability_by_class, float(min(expected_costs))


def what_if_analysis(
    model,
    scaler,
    base_project,
    feature_cols,
    cost_matrix,
    feature_name,
    delta_percent,
):
    """Меняет выбранный параметр на заданный процент и оценивает риск."""

    if feature_name not in base_project.index:
        raise ValueError(f"Неизвестный признак: {feature_name}")

    if feature_name not in feature_cols:
        raise ValueError(
            f"Признак {feature_name} не используется выбранной моделью."
        )

    new_project = base_project.copy()
    new_project[feature_name] = (
        float(new_project[feature_name]) * (1 + delta_percent)
    )

    if feature_name in ["project_cost", "payback_period"]:
        new_project[feature_name] = max(
            float(new_project[feature_name]), 0.01
        )

    new_project = recalculate_derived_features(new_project)

    pred_class, probabilities, expected_cost = predict_project(
        model,
        scaler,
        new_project,
        feature_cols,
        cost_matrix,
    )

    return pred_class, probabilities, expected_cost, new_project


def monte_carlo_simulation(
    model,
    scaler,
    base_project,
    feature_cols,
    cost_matrix,
    n_runs=500,
    noise_level=0.05,
    random_seed=RANDOM_SEED,
):
    """Проводит моделирование риска при случайном изменении параметров."""

    rng = np.random.default_rng(random_seed)
    results = []

    for run_number in range(n_runs):
        noisy_project = base_project.copy()

        for col in NOISE_COLUMNS:
            if col in noisy_project.index:
                noise = rng.normal(0, noise_level)
                noisy_project[col] = float(noisy_project[col]) * (1 + noise)

        # Не допускаем физически некорректных значений.
        noisy_project["project_cost"] = max(
            float(noisy_project["project_cost"]), 0.01
        )
        noisy_project["payback_period"] = max(
            float(noisy_project["payback_period"]), 0.01
        )

        noisy_project = recalculate_derived_features(noisy_project)

        # В этой версии fuzzy_risk остаётся базовым значением.
        # Для строгого пересчёта нужно повторно запустить нечёткую систему.
        pred_class, probabilities, expected_cost = predict_project(
            model,
            scaler,
            noisy_project,
            feature_cols,
            cost_matrix,
        )

        results.append({
            "simulation": run_number + 1,
            "predicted_class": pred_class,
            "prob_low_risk": probabilities.get(0, 0.0),
            "prob_medium_risk": probabilities.get(1, 0.0),
            "prob_high_risk": probabilities.get(2, 0.0),
            "expected_business_cost": expected_cost,
        })

    return pd.DataFrame(results)


def show_prediction(pred_class, probabilities, expected_cost):
    print(f"\nПредсказанный класс: {CLASS_NAMES.get(pred_class, pred_class)}")
    print("Вероятности классов:")

    for class_id in sorted(CLASS_NAMES):
        print(
            f"  {CLASS_NAMES[class_id]}: "
            f"{probabilities.get(class_id, 0.0):.1%}"
        )

    print(f"Ожидаемые бизнес-потери: {expected_cost:.3f}")


def main():
    print("=== Анализ чувствительности и Монте-Карло ===")

    try:
        model, scaler, df, feature_cols, cost_matrix = load_resources()
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"Ошибка загрузки ресурсов: {error}")
        return

    base_project = df.iloc[0].copy()

    print("\nБазовый проект:")
    print(f"ID: {base_project.get('id', base_project.name)}")
    print(f"Стоимость: {base_project['project_cost']:.2f}")
    print(f"Ожидаемая прибыль: {base_project['expected_profit']:.2f}")
    print(f"Срок окупаемости: {base_project['payback_period']:.2f} лет")

    if "risk_class" in base_project.index:
        original_class = int(base_project["risk_class"])
        print(
            "Исходный класс в данных: "
            f"{CLASS_NAMES.get(original_class, original_class)}"
        )

    while True:
        print("\nВыберите действие:")
        print("1. What-if анализ")
        print("2. Монте-Карло (500 симуляций)")
        print("3. Выход")

        choice = input("Ваш выбор: ").strip()

        if choice == "1":
            print("\nДоступные признаки модели:")

            for i, col in enumerate(feature_cols, start=1):
                print(f"{i}. {col}: {float(base_project[col]):.3f}")

            try:
                index = int(input("Номер признака: ")) - 1

                if not 0 <= index < len(feature_cols):
                    print("Такого номера нет в списке.")
                    continue

                feature = feature_cols[index]
                delta = float(
                    input(
                        "Изменение в процентах "
                        "(например, -20 или 10): "
                    )
                ) / 100

                if delta < -1:
                    print("Нельзя уменьшить значение более чем на 100%.")
                    continue

                pred_class, probabilities, expected_cost, new_project = (
                    what_if_analysis(
                        model,
                        scaler,
                        base_project,
                        feature_cols,
                        cost_matrix,
                        feature,
                        delta,
                    )
                )

                print(
                    f"\nИзменение признака {feature}: "
                    f"{delta * 100:+.1f}%"
                )
                print(
                    f"Новое значение: {float(new_project[feature]):.3f}"
                )
                show_prediction(pred_class, probabilities, expected_cost)

                if "fuzzy_risk" in feature_cols:
                    print(
                        "\nПримечание: fuzzy_risk не пересчитывается "
                        "полноценной нечёткой системой в what-if сценарии."
                    )

            except (ValueError, IndexError, KeyError) as error:
                print(f"Ошибка ввода или расчёта: {error}")

        elif choice == "2":
            print("\nВыполняем 500 симуляций с уровнем шума 5%...")

            mc_results = monte_carlo_simulation(
                model,
                scaler,
                base_project,
                feature_cols,
                cost_matrix,
                n_runs=500,
                noise_level=0.05,
            )

            counts = (
                mc_results["predicted_class"]
                .value_counts()
                .reindex([0, 1, 2], fill_value=0)
            )

            print("\nРаспределение предсказанных классов:")
            for class_id, class_name in CLASS_NAMES.items():
                count = int(counts[class_id])
                print(
                    f"{class_name}: {count} из {len(mc_results)} "
                    f"({count / len(mc_results):.1%})"
                )

            high_risk_probability = (
                mc_results["predicted_class"] == 2
            ).mean()

            print(
                "\nДоля симуляций с высоким риском: "
                f"{high_risk_probability:.1%}"
            )
            print(
                "Средние ожидаемые бизнес-потери: "
                f"{mc_results['expected_business_cost'].mean():.3f}"
            )

            os.makedirs("reports", exist_ok=True)
            output_path = "reports/monte_carlo_results.csv"
            mc_results.to_csv(output_path, index=False)

            print(f"Результаты симуляций сохранены: {output_path}")

            if "fuzzy_risk" in feature_cols:
                print(
                    "Примечание: fuzzy_risk в симуляциях остаётся "
                    "базовым значением и не пересчитывается."
                )

        elif choice == "3":
            print("Выход из анализа.")
            break

        else:
            print("Введите 1, 2 или 3.")


if __name__ == "__main__":
    main()
