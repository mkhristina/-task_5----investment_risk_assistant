
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix


MODEL_PATH = "models/best_model.joblib"
DATA_PATH = "data/data_with_fuzzy.csv"
OUTPUT_PATH = "reports/business_risk_analysis.csv"

CLASS_NAMES = {
    0: "Низкий риск",
    1: "Средний риск",
    2: "Высокий риск",
}

# Строка — фактический класс, столбец — предсказанный.
# Ошибка пропуска высокого риска существенно дороже
# ложного предупреждения о высоком риске.
DEFAULT_COST_MATRIX = np.array([
    [0, 1, 2],
    [5, 0, 1],
    [10, 5, 0],
])


def calculate_business_costs(y_true, y_pred, cost_matrix=None):
    """Рассчитывает матрицу ошибок и стоимость классификации."""

    if cost_matrix is None:
        cost_matrix = DEFAULT_COST_MATRIX

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    labels = [0, 1, 2]
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    total_cost = 0
    for actual_class in labels:
        for predicted_class in labels:
            total_cost += (
                matrix[actual_class, predicted_class]
                * cost_matrix[actual_class, predicted_class]
            )

    number_of_samples = len(y_true)
    mean_cost = (
        total_cost / number_of_samples
        if number_of_samples
        else 0.0
    )

    return {
        "confusion_matrix": matrix,
        "total_cost": float(total_cost),
        "mean_cost": float(mean_cost),
        "number_of_samples": number_of_samples,
    }


def predict_by_minimum_cost(probabilities, cost_matrix):
    """Выбирает класс с минимальной ожидаемой стоимостью ошибки."""

    probabilities = np.asarray(probabilities)
    expected_costs = probabilities @ cost_matrix
    predictions = np.argmin(expected_costs, axis=1)

    return predictions, expected_costs


def analyze_saved_model():
    """Проверяет модель на доступных размеченных данных."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Модель не найдена: {MODEL_PATH}. "
            "Сначала запусти ml_model.py."
        )

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Данные не найдены: {DATA_PATH}. "
            "Сначала запусти fuzzy_model.py."
        )

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]
    cost_matrix = np.asarray(
        bundle.get("cost_matrix", DEFAULT_COST_MATRIX)
    )

    df = pd.read_csv(DATA_PATH)

    if "risk_class" not in df.columns:
        raise ValueError("В данных отсутствует столбец risk_class.")

    missing_features = [
        col for col in feature_cols
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Не хватает признаков модели: {missing_features}"
        )

    X = df[feature_cols].copy()
    y_true = df["risk_class"].astype(int).to_numpy()

    # Используем сохранённый scaler и именно те признаки,
    # на которых обучалась итоговая модель.
    X_scaled = scaler.transform(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_scaled)
        predictions, expected_costs = predict_by_minimum_cost(
            probabilities,
            cost_matrix,
        )
    else:
        predictions = model.predict(X_scaled)
        expected_costs = None

    result = calculate_business_costs(
        y_true,
        predictions,
        cost_matrix,
    )

    print("\n=== Бизнес-анализ модели ===")
    print(f"Модель: {bundle.get('model_name', 'Не указана')}")
    print(f"Количество проектов: {result['number_of_samples']}")
    print(f"Общая стоимость ошибок: {result['total_cost']:.2f}")
    print(f"Средняя стоимость на проект: {result['mean_cost']:.4f}")

    print("\nМатрица ошибок:")
    print("Строки — фактический класс, столбцы — предсказанный.")
    print(result["confusion_matrix"])

    rows = []

    for actual_class in [0, 1, 2]:
        for predicted_class in [0, 1, 2]:
            count = int(np.sum(
                (y_true == actual_class)
                & (predictions == predicted_class)
            ))

            rows.append({
                "Фактический класс": CLASS_NAMES[actual_class],
                "Предсказанный класс": CLASS_NAMES[predicted_class],
                "Количество": count,
                "Стоимость одной ошибки": int(
                    cost_matrix[actual_class, predicted_class]
                ),
                "Общая стоимость": int(
                    count * cost_matrix[actual_class, predicted_class]
                ),
            })

    os.makedirs("reports", exist_ok=True)
    pd.DataFrame(rows).to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nПодробный отчёт сохранён: {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        analyze_saved_model()
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"Ошибка бизнес-анализа: {error}")
