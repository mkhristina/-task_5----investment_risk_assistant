
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = "models/best_model.joblib"
DATA_PATH = "data/data_with_fuzzy.csv"

CLASS_NAMES = {
    0: "Низкий риск",
    1: "Средний риск",
    2: "Высокий риск",
}

CLASS_DESCRIPTIONS = {
    0: "Проект имеет относительно низкий риск. Проверьте исходные предположения.",
    1: "Проект требует дополнительного анализа и проверки ключевых предположений.",
    2: "Проект имеет высокий уровень риска. Рекомендуется углублённая проверка.",
}


@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Не найдена модель: {MODEL_PATH}. Сначала запустите ml_model.py."
        )

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Не найдены данные: {DATA_PATH}. Сначала запустите fuzzy_model.py."
        )

    bundle = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

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

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"В данных отсутствуют признаки: {missing}")

    return model, scaler, df, feature_cols, cost_matrix


def calculate_derivatives(row):
    """Пересчитывает производные показатели, если они есть в данных."""

    row = row.copy()

    cost = max(float(row["project_cost"]), 0.01)
    profit = float(row["expected_profit"])
    payback = max(float(row["payback_period"]), 0.01)

    if "profitability" in row.index:
        row["profitability"] = profit / cost * 100

    if "roi_index" in row.index:
        row["roi_index"] = profit / (cost / payback)

    if "risk_density" in row.index:
        row["risk_density"] = (
            float(row["market_volatility"])
            + float(row["regulatory_risk"])
            + float(row["competition_level"]) * 10
        ) / 3

    if "capability_score" in row.index:
        row["capability_score"] = (
            float(row["team_experience"])
            * float(row["technology_readiness"])
            * float(row["strategic_fit"])
        )

    return row


def predict_project(input_dict, model, scaler, df, feature_cols, cost_matrix):
    """Возвращает класс по бизнес-потерям и вероятности классов."""

    base_row = df.iloc[0].copy()

    for key, value in input_dict.items():
        if key in base_row.index:
            base_row[key] = value

    base_row = calculate_derivatives(base_row)

    # Для нового проекта fuzzy_risk пока берётся из базовой строки.
    # Полноценный пересчёт требует повторного запуска fuzzy-системы.
    input_frame = pd.DataFrame(
        [[float(base_row[col]) for col in feature_cols]],
        columns=feature_cols,
    )

    scaled = scaler.transform(input_frame)
    probabilities = model.predict_proba(scaled)[0]
    classes = np.asarray(model.classes_, dtype=int)

    # cost_matrix[истинный класс, предсказанный класс]
    expected_costs = []
    for predicted_class in classes:
        cost = sum(
            probabilities[i] * cost_matrix[true_class, predicted_class]
            for i, true_class in enumerate(classes)
        )
        expected_costs.append(cost)

    predicted_class = int(classes[np.argmin(expected_costs)])

    probability_map = {
        int(class_id): float(prob)
        for class_id, prob in zip(classes, probabilities)
    }

    return predicted_class, probability_map, base_row, min(expected_costs)


def main():
    st.set_page_config(
        page_title="Инвестиционный риск-ассистент",
        page_icon="📊",
        layout="wide",
    )

    st.title("Инвестиционный комитет: оценка рисков")
    st.caption(
        "Учебная система оценки риска инвестиционных проектов "
        "на синтетических данных."
    )

    try:
        model, scaler, df, feature_cols, cost_matrix = load_resources()
    except (FileNotFoundError, ValueError, KeyError) as error:
        st.error(str(error))
        st.info(
            "Проверьте наличие файлов модели и данных, "
            "а затем запустите соответствующие модули проекта."
        )
        st.stop()

    tab_data, tab_new, tab_whatif, tab_model = st.tabs(
        ["Данные", "Новый проект", "What-if анализ", "О модели"]
    )

    with tab_data:
        st.header("Пример очищенных данных")
        st.write(f"Количество записей: {len(df)}")
        st.dataframe(df.head(20), use_container_width=True)

        if "risk_class" in df.columns:
            st.subheader("Распределение исходных классов")
            counts = (
                df["risk_class"]
                .value_counts()
                .reindex([0, 1, 2], fill_value=0)
            )
            st.bar_chart(
                pd.DataFrame({
                    "Количество проектов": counts,
                }).rename(index=CLASS_NAMES)
            )

    with tab_new:
        st.header("Оценка нового проекта")
        st.write("Введите параметры проекта.")

        with st.form("new_project_form"):
            col1, col2 = st.columns(2)

            with col1:
                project_cost = st.number_input(
                    "Стоимость проекта",
                    min_value=1.0,
                    max_value=100.0,
                    value=50.0,
                    step=1.0,
                )
                expected_profit = st.number_input(
                    "Ожидаемая прибыль",
                    min_value=0.5,
                    max_value=150.0,
                    value=20.0,
                    step=0.5,
                )
                payback_period = st.number_input(
                    "Срок окупаемости, лет",
                    min_value=1.0,
                    max_value=10.0,
                    value=5.0,
                    step=0.1,
                )
                team_experience = st.number_input(
                    "Опыт команды, лет",
                    min_value=0.0,
                    max_value=20.0,
                    value=10.0,
                    step=0.1,
                )

            with col2:
                market_volatility = st.number_input(
                    "Волатильность рынка",
                    min_value=0.0,
                    max_value=100.0,
                    value=40.0,
                    step=1.0,
                )
                technology_readiness = st.number_input(
                    "Готовность технологии",
                    min_value=1,
                    max_value=9,
                    value=5,
                )
                competition_level = st.number_input(
                    "Уровень конкуренции",
                    min_value=1,
                    max_value=10,
                    value=5,
                )
                strategic_fit = st.number_input(
                    "Соответствие стратегии",
                    min_value=1,
                    max_value=10,
                    value=7,
                )
                regulatory_risk = st.number_input(
                    "Регуляторный риск",
                    min_value=0.0,
                    max_value=100.0,
                    value=20.0,
                    step=1.0,
                )

            submitted = st.form_submit_button("Оценить проект")

        if submitted:
            user_data = {
                "project_cost": project_cost,
                "expected_profit": expected_profit,
                "payback_period": payback_period,
                "team_experience": team_experience,
                "market_volatility": market_volatility,
                "technology_readiness": technology_readiness,
                "competition_level": competition_level,
                "strategic_fit": strategic_fit,
                "regulatory_risk": regulatory_risk,
            }

            pred_class, probs, row, expected_cost = predict_project(
                user_data, model, scaler, df, feature_cols, cost_matrix
            )

            st.subheader("Результат оценки")
            st.metric(
                "Класс риска",
                CLASS_NAMES.get(pred_class, str(pred_class)),
            )
            st.write(CLASS_DESCRIPTIONS.get(pred_class, ""))
            st.write(f"Ожидаемые бизнес-потери: {expected_cost:.3f}")

            st.write("**Вероятности классов модели**")
            probability_df = pd.DataFrame(
                {
                    "Класс": [
                        CLASS_NAMES.get(i, str(i))
                        for i in sorted(probs)
                    ],
                    "Вероятность": [
                        probs[i] for i in sorted(probs)
                    ],
                }
            ).set_index("Класс")

            st.bar_chart(probability_df)

            st.caption(
                "Важно: для нового проекта используется базовое значение "
                "fuzzy_risk из набора данных. Оно не пересчитывается "
                "автоматически по введённым параметрам."
            )

    with tab_whatif:
        st.header("Анализ чувствительности")
        st.write(
            "Измените один входной параметр и сравните предсказание "
            "с базовым проектом из набора данных."
        )

        editable_features = [
            col for col in feature_cols if col != "fuzzy_risk"
        ]

        param = st.selectbox(
            "Параметр для изменения",
            editable_features,
        )

        base_value = float(df.iloc[0][param])
        delta = st.slider(
            "Изменение параметра, %",
            min_value=-50.0,
            max_value=50.0,
            value=0.0,
            step=5.0,
        )

        if st.button("Рассчитать сценарий"):
            baseline_class, baseline_probs, _, baseline_cost = (
                predict_project(
                    {},
                    model,
                    scaler,
                    df,
                    feature_cols,
                    cost_matrix,
                )
            )

            changed_value = base_value * (1 + delta / 100)
            if param in ["project_cost", "payback_period"]:
                changed_value = max(changed_value, 0.01)

            scenario_class, scenario_probs, _, scenario_cost = (
                predict_project(
                    {param: changed_value},
                    model,
                    scaler,
                    df,
                    feature_cols,
                    cost_matrix,
                )
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Базовый проект")
                st.write(
                    f"Класс: **{CLASS_NAMES[baseline_class]}**"
                )
                st.write(
                    f"Вероятность высокого риска: "
                    f"{baseline_probs.get(2, 0.0):.1%}"
                )
                st.write(f"Ожидаемые потери: {baseline_cost:.3f}")

            with col2:
                st.subheader("Изменённый сценарий")
                st.write(f"{param}: {changed_value:.2f}")
                st.write(
                    f"Класс: **{CLASS_NAMES[scenario_class]}**"
                )
                st.write(
                    f"Вероятность высокого риска: "
                    f"{scenario_probs.get(2, 0.0):.1%}"
                )
                st.write(f"Ожидаемые потери: {scenario_cost:.3f}")

            st.caption(
                "Это сценарный анализ, а не прогноз реального дефолта. "
                "fuzzy_risk остаётся базовым и не пересчитывается "
                "нечёткой системой."
            )

    with tab_model:
        st.header("Информация о модели")
        st.write(f"**Алгоритм:** {model.__class__.__name__}")
        st.write(f"**Количество признаков:** {len(feature_cols)}")
        st.write(f"**Признаки:** {', '.join(feature_cols)}")
        st.write(f"**Количество записей:** {len(df)}")
        st.write(
            "**Правило выбора класса:** минимизация ожидаемых "
            "бизнес-потерь по матрице стоимости ошибок."
        )
        st.write(
            "**Ограничения:** данные синтетические; качество на них "
            "не гарантирует качество на реальных инвестиционных проектах."
        )


if __name__ == "__main__":
    main()
