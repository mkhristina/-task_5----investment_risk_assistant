
import os
import subprocess
import sys
import warnings

import joblib
import numpy as np
import pandas as pd


# ============================================================
# ПУТИ К ФАЙЛАМ
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================
# ПРЕДУПРЕЖДЕНИЯ
# ============================================================

warnings.filterwarnings(
    "ignore",
    message="X has feature names, but MLPClassifier was fitted without feature names",
    category=UserWarning,
)


# ============================================================
# РУССКИЕ НАЗВАНИЯ ПРИЗНАКОВ
# ============================================================

FEATURE_NAMES_RU = {
    "project_cost": "Стоимость проекта",
    "expected_profit": "Ожидаемая прибыль",
    "payback_period": "Срок окупаемости, лет",
    "team_experience": "Опыт команды, лет",
    "market_volatility": "Рыночная волатильность",
    "technology_readiness": "Технологическая готовность",
    "competition_level": "Уровень конкуренции",
    "strategic_fit": "Стратегическое соответствие",
    "regulatory_risk": "Регуляторный риск",
    "profitability": "Рентабельность",
    "roi_index": "Индекс ROI",
    "financial_index": "Финансовый индекс риска",
    "market_index": "Рыночный индекс риска",
    "organizational_index": "Организационный индекс риска",
    "fuzzy_risk": "Итоговый нечёткий риск",
}

CLASS_NAMES_RU = {
    0: "Низкий риск",
    1: "Средний риск",
    2: "Высокий риск",
}


# ============================================================
# ЗАПУСК СКРИПТОВ
# ============================================================

def run_script(script_name):
    script_path = os.path.join(BASE_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"\nФайл не найден: {script_name}")
        return

    print(f"\nЗапуск {script_name}...\n")

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR,
    )

    if result.returncode == 0:
        print(f"\nФайл {script_name} выполнен.")
    else:
        print(
            f"\nФайл {script_name} завершился "
            f"с кодом ошибки {result.returncode}."
        )


def run_dashboard():
    dashboard_path = os.path.join(BASE_DIR, "dashboard.py")

    if not os.path.exists(dashboard_path):
        print("Файл dashboard.py не найден.")
        return

    print("\nЗапускаю интерактивный дашборд...")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            dashboard_path,
        ],
        cwd=BASE_DIR,
    )


def run_pipeline():
    scripts = [
        "data_generator.py",
        "data_cleaner.py",
        "eda.py",
        "fuzzy_model.py",
        "ml_model.py",
        "explainer.py",
        "whatif_mc.py",
        "report_generator.py",
        "business_logic.py",
    ]

    for script in scripts:
        print("\n" + "=" * 60)
        print(f"ЭТАП: {script}")
        print("=" * 60)

        script_path = os.path.join(BASE_DIR, script)

        if not os.path.exists(script_path):
            print(f"Файл {script} не найден. Конвейер остановлен.")
            return

        result = subprocess.run(
            [sys.executable, script_path],
            cwd=BASE_DIR,
        )

        if result.returncode != 0:
            print(
                f"\nЭтап {script} завершился с ошибкой. "
                "Конвейер остановлен."
            )
            return

    print("\nВесь конвейер успешно завершён.")


# ============================================================
# ВВОД ЧИСЛОВЫХ ЗНАЧЕНИЙ
# ============================================================

def input_number(prompt, minimum=None, maximum=None):
    while True:
        raw_value = input(f"{prompt}: ").strip().replace(",", ".")

        try:
            value = float(raw_value)

            if not np.isfinite(value):
                print("Введи конечное числовое значение.")
                continue

            if minimum is not None and value < minimum:
                print(f"Значение должно быть не меньше {minimum}.")
                continue

            if maximum is not None and value > maximum:
                print(f"Значение должно быть не больше {maximum}.")
                continue

            return value

        except ValueError:
            print("Введи число. Например: 100000 или 12.5.")


# ============================================================
# СОЗДАНИЕ ЧЕТЫРЁХ НЕЧЁТКИХ СИСТЕМ
# ============================================================

def create_fuzzy_simulations():
    from skfuzzy import control as ctrl
    from fuzzy_model import create_fuzzy_system

    # Финансовая система
    financial_system = create_fuzzy_system(
        [
            "profitability_risk",
            "roi_risk",
            "payback_risk",
        ],
        "financial_risk",
    )

    # Рыночная система
    market_system = create_fuzzy_system(
        [
            "market_volatility",
            "regulatory_risk",
            "competition_risk",
        ],
        "market_risk",
    )

    # Организационная система
    organization_system = create_fuzzy_system(
        [
            "team_risk",
            "technology_risk",
            "strategy_risk",
        ],
        "organizational_risk",
    )

    # Итоговый агрегатор
    aggregator_system = create_fuzzy_system(
        [
            "financial_index",
            "market_index",
            "organizational_index",
        ],
        "fuzzy_risk",
    )

    return {
        "financial": ctrl.ControlSystemSimulation(financial_system),
        "market": ctrl.ControlSystemSimulation(market_system),
        "organization": ctrl.ControlSystemSimulation(
            organization_system
        ),
        "aggregator": ctrl.ControlSystemSimulation(aggregator_system),
    }


def calculate_fuzzy_risk(project, simulations=None):
    from fuzzy_model import calculate_indices

    if simulations is None:
        simulations = create_fuzzy_simulations()

    row = pd.Series(project, dtype=float)

    return calculate_indices(row, simulations)


# ============================================================
# ОЦЕНКА НОВОГО ИНВЕСТИЦИОННОГО ПРОЕКТА
# ============================================================

def predict_new_project():
    print("\n" + "=" * 60)
    print("   ОЦЕНКА НОВОГО ИНВЕСТИЦИОННОГО ПРОЕКТА")
    print("=" * 60)

    print(
        "Вводи числовые значения. "
        "Дробную часть можно отделять точкой или запятой.\n"
    )

    if not os.path.exists(MODEL_PATH):
        print("Обученная модель не найдена.")
        print("Сначала запусти пункт 5 — обучение ML-моделей.")
        return

    try:
        # Загружаем обученную модель
        bundle = joblib.load(MODEL_PATH)

        # Ввод параметров проекта
        project = {
            "project_cost": input_number(
                "Стоимость проекта",
                minimum=0,
            ),
            "expected_profit": input_number(
                "Ожидаемая прибыль",
            ),
            "payback_period": input_number(
                "Срок окупаемости, лет",
                minimum=0,
            ),
            "team_experience": input_number(
                "Опыт команды, лет",
                minimum=0,
            ),
            "market_volatility": input_number(
                "Рыночная волатильность (0–100)",
                minimum=0,
                maximum=100,
            ),
            "technology_readiness": input_number(
                "Технологическая готовность (1–9)",
                minimum=1,
                maximum=9,
            ),
            "competition_level": input_number(
                "Уровень конкуренции (1–10)",
                minimum=1,
                maximum=10,
            ),
            "strategic_fit": input_number(
                "Стратегическое соответствие (1–10)",
                minimum=1,
                maximum=10,
            ),
            "regulatory_risk": input_number(
                "Регуляторный риск (0–100)",
                minimum=0,
                maximum=100,
            ),
            "profitability": input_number(
                "Показатель profitability",
            ),
            "roi_index": input_number(
                "Показатель roi_index",
            ),
        }

        # Рассчитываем нечёткие индексы
        print("\nРассчитываю нечёткие показатели...")

        simulations = create_fuzzy_simulations()
        fuzzy_values = calculate_fuzzy_risk(
            project,
            simulations,
        )

        if not isinstance(fuzzy_values, dict):
            raise TypeError(
                "Нечёткая модель вернула неожиданный результат."
            )

        project.update(
            {
                key: float(value)
                for key, value in fuzzy_values.items()
            }
        )

        # Список признаков, использованных при обучении
        feature_cols = bundle.get("feature_cols")

        if not feature_cols:
            raise ValueError(
                "В файле модели нет списка feature_cols. "
                "Повтори обучение через пункт 5."
            )

        missing_features = [
            feature
            for feature in feature_cols
            if feature not in project
        ]

        if missing_features:
            readable_names = [
                FEATURE_NAMES_RU.get(name, name)
                for name in missing_features
            ]

            raise ValueError(
                "Не хватает значений для признаков: "
                + ", ".join(readable_names)
                + ". Возможно, сохранённая модель обучена "
                  "на другом наборе признаков."
            )

        # Строго соблюдаем порядок признаков при обучении
        X = pd.DataFrame(
            [[project[name] for name in feature_cols]],
            columns=feature_cols,
        )

        model = bundle.get("model")
        scaler = bundle.get("scaler")

        if model is None:
            raise ValueError(
                "В файле модели отсутствует обученная модель."
            )

        # Преобразуем признаки тем же scaler, что использовался
        # при обучении.
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.to_numpy()

        # Передаём массив в модель, чтобы избежать предупреждения
        # о названиях признаков у MLPClassifier.
        X_for_model = np.asarray(X_scaled)

        # Получаем вероятности классов
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_for_model)[0]
        else:
            probabilities = None

        # Выбираем класс с учётом матрицы стоимости ошибок
        cost_matrix = bundle.get("cost_matrix")
        expected_costs = None

        if probabilities is not None and cost_matrix is not None:
            cost_matrix = np.asarray(cost_matrix, dtype=float)

            if cost_matrix.shape == (
                len(probabilities),
                len(probabilities),
            ):
                expected_costs = probabilities @ cost_matrix
                prediction = int(np.argmin(expected_costs))
            else:
                prediction = int(model.predict(X_for_model)[0])
        else:
            prediction = int(model.predict(X_for_model)[0])

        # Вывод результата
        print("\n" + "=" * 60)
        print("   РЕЗУЛЬТАТ ОЦЕНКИ")
        print("=" * 60)

        print(
            f"\nМодель: {bundle.get('model_name', 'Не указана')}"
        )

        class_names = bundle.get("class_names")

        if class_names is not None and prediction < len(class_names):
            risk_label = str(class_names[prediction])
        else:
            risk_label = CLASS_NAMES_RU.get(
                prediction,
                f"Класс {prediction}",
            )

        print(f"Класс риска: {risk_label}")

        if probabilities is not None:
            print("\nВероятности классов:")

            for index, probability in enumerate(probabilities):
                label = CLASS_NAMES_RU.get(
                    index,
                    f"Класс {index}",
                )
                print(f"  {label}: {probability:.2%}")

        if expected_costs is not None:
            print("\nОжидаемая стоимость каждого решения:")

            for index, cost in enumerate(expected_costs):
                label = CLASS_NAMES_RU.get(
                    index,
                    f"Класс {index}",
                )
                print(f"  {label}: {cost:.4f}")

        print("\nПризнаки, используемые моделью:")

        for feature in feature_cols:
            russian_name = FEATURE_NAMES_RU.get(
                feature,
                feature,
            )
            print(f"  • {russian_name}: {project[feature]}")

        print("\nПоказатели нечёткой модели:")

        fuzzy_feature_names = [
            "financial_index",
            "market_index",
            "organizational_index",
            "fuzzy_risk",
        ]

        for feature in fuzzy_feature_names:
            if feature in project:
                russian_name = FEATURE_NAMES_RU.get(
                    feature,
                    feature,
                )
                print(
                    f"  • {russian_name}: "
                    f"{project[feature]:.2f}"
                )

        print("\nОценка проекта завершена.")

    except Exception as error:
        print("\nНе удалось оценить проект.")
        print(f"Причина: {error}")
        print(
            "\nПроверь сообщение об ошибке выше. "
            "Если проблема повторится, пришли её полный текст."
        )


# ============================================================
# МЕНЮ ПРОГРАММЫ
# ============================================================

def show_menu():
    print("\n" + "=" * 60)
    print("   СИСТЕМА ОЦЕНКИ ИНВЕСТИЦИОННЫХ РИСКОВ")
    print("=" * 60)
    print("1. Генерация синтетических данных")
    print("2. Очистка данных")
    print("3. Исследовательский анализ данных (EDA)")
    print("4. Построение нечёткой модели")
    print("5. Обучение и сравнение ML-моделей")
    print("6. SHAP-анализ")
    print("7. What-if и Monte Carlo")
    print("8. Формирование PDF- и Excel-отчётов")
    print("9. Запуск интерактивного дашборда")
    print("10. Выполнить весь конвейер")
    print("11. Бизнес-анализ стоимости ошибок")
    print("12. Оценить новый инвестиционный проект")
    print("0. Выход")
    print("=" * 60)


def main():
    actions = {
        "1": lambda: run_script("data_generator.py"),
        "2": lambda: run_script("data_cleaner.py"),
        "3": lambda: run_script("eda.py"),
        "4": lambda: run_script("fuzzy_model.py"),
        "5": lambda: run_script("ml_model.py"),
        "6": lambda: run_script("explainer.py"),
        "7": lambda: run_script("whatif_mc.py"),
        "8": lambda: run_script("report_generator.py"),
        "9": run_dashboard,
        "10": run_pipeline,
        "11": lambda: run_script("business_logic.py"),
        "12": predict_new_project,
    }

    while True:
        show_menu()
        choice = input("Выбери пункт меню: ").strip()

        if choice == "0":
            print("\nПрограмма завершена.")
            break

        action = actions.get(choice)

        if action is None:
            print("Неверный пункт меню. Попробуй ещё раз.")
            continue

        action()


if __name__ == "__main__":
    main()
