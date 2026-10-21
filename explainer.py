
import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_PATH = "models/best_model.joblib"
DATA_PATH = "data/data_with_fuzzy.csv"
REPORTS_DIR = "reports"
MAX_EXPLAIN_ROWS = 100
BACKGROUND_ROWS = 50


def explain_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Не найдена модель: {MODEL_PATH}. Сначала запусти ml_model.py."
        )

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Не найдены данные: {DATA_PATH}. Сначала запусти fuzzy_model.py."
        )

    print("Загружаем модель и данные...")
    bundle = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    # Модель сохранена вместе со скалером и признаками.
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"В данных отсутствуют признаки: {missing}")

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()

    if X.empty:
        raise ValueError("Нет строк без пропущенных значений для SHAP.")

    # Для ускорения используем воспроизводимую выборку.
    sample_size = min(MAX_EXPLAIN_ROWS, len(X))
    X_sample = X.sample(
        n=sample_size,
        random_state=42
    )

    background_size = min(BACKGROUND_ROWS, len(X))
    X_background = X.sample(
        n=background_size,
        random_state=7
    )

    # В модель подаются именно масштабированные признаки.
    X_scaled = pd.DataFrame(
        scaler.transform(X_sample),
        columns=feature_cols,
        index=X_sample.index,
    )
    background_scaled = pd.DataFrame(
        scaler.transform(X_background),
        columns=feature_cols,
        index=X_background.index,
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f"Модель: {bundle.get('model_name', 'не указана')}")
    print(f"Набор признаков: {bundle.get('features', 'не указан')}")
    print(f"Признаков: {len(feature_cols)}")
    print(f"Объясняем проектов: {sample_size}")

    # Функция возвращает вероятности классов для SHAP.
    def predict_fn(data):
        if isinstance(data, pd.DataFrame):
            values = data[feature_cols]
        else:
            values = pd.DataFrame(data, columns=feature_cols)

        return model.predict_proba(values)

    print("Создаём SHAP explainer...")
    explainer = shap.PermutationExplainer(
        predict_fn,
        background_scaled,
        feature_names=feature_cols,
        seed=42,
    )

    print("Вычисляем SHAP значения...")
    shap_values = explainer(
        X_scaled,
        max_evals=2 * len(feature_cols) + 1,
    )

    # В актуальных версиях SHAP классификация обычно имеет
    # размерность (строки, признаки, классы).
    values = shap_values.values
    if values.ndim != 3:
        raise ValueError(
            f"Неожиданная размерность SHAP-значений: {values.shape}"
        )

    classes = list(model.classes_)
    if 2 not in classes:
        raise ValueError(
            "В модели нет класса 2 — невозможно построить "
            "графики для высокого риска."
        )

    high_class_index = classes.index(2)

    print("Строим summary plot для высокого риска...")
    plt.figure()
    shap.plots.beeswarm(
        shap_values[:, :, high_class_index],
        show=False,
        max_display=min(10, len(feature_cols)),
    )
    plt.title("SHAP: факторы высокого риска (класс 2)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(REPORTS_DIR, "shap_summary.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close("all")

    print("Строим waterfall plot...")
    plt.figure()
    shap.plots.waterfall(
        shap_values[0, :, high_class_index],
        show=False,
        max_display=min(10, len(feature_cols)),
    )
    plt.title("SHAP: объяснение одного проекта, класс 2")
    plt.tight_layout()
    plt.savefig(
        os.path.join(REPORTS_DIR, "shap_waterfall.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close("all")

    print("Строим графики важности для всех классов...")
    fig, axes = plt.subplots(
        nrows=len(classes),
        ncols=1,
        figsize=(10, 5 * len(classes)),
    )

    if len(classes) == 1:
        axes = [axes]

    class_names = {
        0: "Низкий риск",
        1: "Средний риск",
        2: "Высокий риск",
    }

    for axis, class_label, class_index in zip(
        axes, classes, range(len(classes))
    ):
        plt.sca(axis)
        shap.plots.bar(
            shap_values[:, :, class_index].abs.mean(0),
            show=False,
            max_display=min(10, len(feature_cols)),
        )
        axis.set_title(
            f"Важность признаков: {class_names.get(class_label, class_label)}"
        )

    fig.tight_layout()
    fig.savefig(
        os.path.join(REPORTS_DIR, "shap_bar_classes.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close("all")

    print("\nSHAP-анализ завершён!")
    print("Созданы файлы:")
    print(" - reports/shap_summary.png")
    print(" - reports/shap_waterfall.png")
    print(" - reports/shap_bar_classes.png")


if __name__ == "__main__":
    explain_model()
