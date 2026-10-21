
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fuzzy_model import create_fuzzy_system, calculate_indices
import skfuzzy.control as ctrl


MODEL_PATH = "models/best_model.joblib"

COST_MATRIX = np.array([
    [0, 1, 2],
    [5, 0, 1],
    [10, 5, 0],
])


def run_script(script_name):
    """Запускает Python-скрипт из папки проекта."""
    if not os.path.exists(script_name):
        print(f"\nФайл {script_name} не найден.")
        return

    print(f"\nЗапуск: {script_name}")

    try:
        result = subprocess.run([sys.executable, script_name], check=False)

        if result.returncode == 0:
            print(f"\nСкрипт {script_name} завершился.")
        else:
            print(f"\nСкрипт завершился с кодом {result.returncode}.")

    except KeyboardInterrupt:
        print("\nЗапуск прерван пользователем.")


def run_dashboard():
    """Запускает интерактивный Streamlit-дашборд."""
    if not os.path.exists("dashboard.py"):
        print("\nФайл dashboard.py не найден.")
        return

    print("\nЗапускаю интерактивный дашборд.")
    print("Для остановки нажми Ctrl+C.")

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "dashboard.py"],
            check=False,
        )
    except KeyboardInterrupt:
        print("\nДашборд остановлен.")


def create_fuzzy_simulations():
    """Создаёт нечёткие системы, используемые в fuzzy_model.py."""
    financial = create_fuzzy_system(
        ["profitability_risk", "roi_risk", "payback_risk"],
        "financial_risk",
    )
    market = create_fuzzy_system(
        ["market_volatility", "regulatory_risk", "competition_risk"],
        "market_risk",
    )
    organization = create_fuzzy_system(
        ["team_risk", "technology_risk", "strategy_risk"],
        "organizational_risk",
    )
    aggregator = create_fuzzy_system(
        ["financial_index", "market_index", "organizational_index"],
        "fuzzy_risk",
    )

    return {
        "financial": ctrl.ControlSystemSimulation(financial),
        "market": ctrl.ControlSystemSimulation(market),
        "organization": ctrl.ControlSystemSimulation(organization),
        "aggregator": ctrl.ControlSystemSimulation(aggregator),
    }


def input_number(label, minimum=None):
    """Запрашивает число и при необходимости проверяет нижнюю границу."""
    while True:
        raw = input(f"{label}: ").strip().replace(",", ".")

        try:
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError

            if minimum is not None and value < minimum:
                print(f"Значение должно быть не меньше {minimum}.")
                continue

            return value

        except ValueError:
            print("Введи корректное число, например 12.5.")


def predict_new_project():
    """Запрашивает параметры проекта и рассчитывает прогноз риска."""
    if not os.path.exists(MODEL_PATH):
        print("\nСохранённая модель не найдена.")
        print("Сначала выполни пункты 1, 2, 4 и 5 по порядку.")
        return

    print("\n" + "=" * 52)
    print("   ОЦЕНКА НОВОГО ИНВЕСТИЦИОННОГО ПРОЕКТА")
    print("=" * 52)
    print("Вводи числовые значения. Дробную часть можно")
    print("отделять точкой или запятой.\n")

    try:
        project = {
            "project_cost": input_number("Стоимость проекта"),
            "expected_profit": input_number("Ожидаемая прибыль"),
            "payback_period": input_number("Срок окупаемости, лет", 0),
            "team_experience": input_number("Опыт команды, лет", 0),
            "market_volatility": input_number("Рыночная волатильность (0–100)"),
            "technology_readiness": input_number(
                "Технологическая готовность (1–9)"
            ),
            "competition_level": input_number("Уровень конкуренции (1–10)"),
            "strategic_fit": input_number(
                "Стратегическое соответствие (1–10)"
            ),
            "regulatory_risk": input_number("Регуляторный риск (0–100)"),
            "profitability": input_number("Показатель profitability"),
            "roi_index": input_number("Показатель roi_index"),
        }

        if project["project_cost"] <= 0:
            print("\nСтоимость проекта должна быть больше нуля.")
            return

        if project["payback_period"] <= 0:
            print("\nСрок окупаемости должен быть больше нуля.")
            return

        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        scaler = bundle["scaler"]
        feature_cols = bundle["feature_cols"]
        class_names = bundle.get(
            "class_names",
            {
                0: "Низкий риск",
                1: "Средний риск",
                2: "Высокий риск",
            },
        )
        cost_matrix = np.asarray(bundle.get("cost_matrix", COST_MATRIX))

        simulations = create_fuzzy_simulations()
        fuzzy_indices = calculate_indices(
            pd.Series(project),
            simulations,
        )

        model_input = {
            **project,
            **fuzzy_indices,
        }

        X_new = pd.DataFrame(
            [{column: model_input[column] for column in feature_cols}],
            columns=feature_cols,
        )

        X_scaled = scaler.transform(X_new)
        probabilities = model.predict_proba(X_scaled)
        classes = model.classes_.astype(int)

        ordered_costs = cost_matrix[np.ix_(classes, classes)]
        expected_costs = probabilities @ ordered_costs
        predicted_class = int(classes[np.argmin(expected_costs[0])])

        print("\n" + "=" * 52)
        print("   РЕЗУЛЬТАТ ОЦЕНКИ")
        print("=" * 52)
        print(f"Категория риска: {class_names[predicted_class]}")
        print(f"Нечёткий индекс риска: {fuzzy_indices['fuzzy_risk']:.2f}")
        print("\nНечёткие индексы:")
        print(f"  Финансовый:       {fuzzy_indices['financial_index']:.2f}")
        print(f"  Рыночный:         {fuzzy_indices['market_index']:.2f}")
        print(
            "  Организационный:  "
            f"{fuzzy_indices['organizational_index']:.2f}"
        )

        print("\nВероятности классов:")
        for class_id, probability in zip(classes, probabilities[0]):
            print(
                f"  {class_names[int(class_id)]}: "
                f"{probability * 100:.1f}%"
            )

        print(
            "\nОжидаемая стоимость ошибки для выбранного решения: "
            f"{expected_costs[0, np.argmin(expected_costs[0])]:.3f}"
        )
        print("\nЭто прогноз модели, а не гарантия фактического риска.")

    except KeyboardInterrupt:
        print("\nВвод проекта отменён.")
    except (KeyError, ValueError, OSError) as error:
        print(f"\nНе удалось оценить проект: {error}")


def show_menu():
    """Отображает главное меню."""
    print("\n" + "=" * 52)
    print("   СИСТЕМА ОЦЕНКИ ИНВЕСТИЦИОННЫХ РИСКОВ")
    print("=" * 52)
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
    print("=" * 52)


def run_pipeline():
    """Последовательно запускает основные этапы проекта."""
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

    required_files = {
        "data_generator.py": ["data/data_raw.csv"],
        "data_cleaner.py": ["data/data_clean.csv"],
        "fuzzy_model.py": ["data/data_with_fuzzy.csv"],
        "ml_model.py": [
            "data/model_comparison.csv",
            "models/best_model.joblib",
        ],
        "report_generator.py": [
            "reports/Investment_Report.pdf",
            "reports/Investment_Report.xlsx",
        ],
        "business_logic.py": [
            "reports/business_risk_analysis.csv",
        ],
    }

    for script in scripts:
        print("\n" + "-" * 52)
        run_script(script)

        for filename in required_files.get(script, []):
            if not os.path.exists(filename):
                print(f"\nНе найден ожидаемый файл: {filename}")
                print("Конвейер остановлен. Проверь вывод предыдущего скрипта.")
                return

    print("\nОсновной конвейер завершён.")
    print("Результаты находятся в папках data, models и reports.")


def main():
    """Главный цикл приложения."""
    script_actions = {
        "1": "data_generator.py",
        "2": "data_cleaner.py",
        "3": "eda.py",
        "4": "fuzzy_model.py",
        "5": "ml_model.py",
        "6": "explainer.py",
        "7": "whatif_mc.py",
        "8": "report_generator.py",
        "11": "business_logic.py",
    }

    while True:
        show_menu()

        try:
            choice = input("Выбери пункт меню: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nРабота программы завершена.")
            break

        if choice in script_actions:
            run_script(script_actions[choice])
        elif choice == "9":
            run_dashboard()
        elif choice == "10":
            confirm = input(
                "\nБудут повторно запущены генерация данных и все этапы.\n"
                "Продолжить? (да/нет): "
            ).strip().lower()

            if confirm in ("да", "д", "yes", "y"):
                run_pipeline()
            else:
                print("Конвейер не запускался.")
        elif choice == "12":
            predict_new_project()
        elif choice == "0":
            print("\nРабота программы завершена.")
            break
        else:
            print("\nНеизвестный пункт. Введи число от 0 до 12.")


if __name__ == "__main__":
    main()
