
import os
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = "data/data_with_fuzzy.csv"
RESULTS_PATH = "data/model_comparison.csv"
MODEL_PATH = "models/best_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_TRIALS = 30

BASE_FEATURES = [
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

FUZZY_FEATURE = "fuzzy_risk"
TARGET = "risk_class"

# Строки — истинный класс, столбцы — предсказанный класс.
# Недооценка высокого риска обходится дороже ложной тревоги.
COST_MATRIX = np.array([
    [0, 1, 2],
    [5, 0, 1],
    [10, 5, 0],
])


def cost_sensitive_predict(model, X):
    """Выбирает класс с минимальными ожидаемыми бизнес-потерями."""
    probabilities = model.predict_proba(X)
    classes = model.classes_

    # Матрица потерь приводится к порядку классов модели.
    costs = COST_MATRIX[np.ix_(classes.astype(int), classes.astype(int))]
    expected_costs = probabilities @ costs
    best_positions = np.argmin(expected_costs, axis=1)

    return classes[best_positions]


def calculate_metrics(y_true, predictions, probabilities, classes):
    """Считает классификационные метрики и бизнес-потери."""
    roc_auc = np.nan
    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(
                y_true,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
    except ValueError:
        pass

    cm = confusion_matrix(y_true, predictions, labels=[0, 1, 2])
    actual_predicted_costs = COST_MATRIX[
        np.asarray(y_true, dtype=int),
        np.asarray(predictions, dtype=int),
    ]

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "roc_auc": roc_auc,
        "mcc": matthews_corrcoef(y_true, predictions),
        "business_cost_total": int(actual_predicted_costs.sum()),
        "business_cost_mean": float(actual_predicted_costs.mean()),
        "class_0_recall": recall_score(
            y_true, predictions, labels=[0], average=None, zero_division=0
        )[0],
        "class_1_recall": recall_score(
            y_true, predictions, labels=[1], average=None, zero_division=0
        )[0],
        "class_2_recall": recall_score(
            y_true, predictions, labels=[2], average=None, zero_division=0
        )[0],
        "confusion_matrix": cm.tolist(),
    }


def tune_mlp(X_train, y_train, cv):
    """Подбирает параметры MLP с масштабированием внутри CV."""
    print("\nПодбор гиперпараметров MLP через Optuna (30 попыток)...")

    def objective(trial):
        hidden_size = trial.suggest_int("hidden_layer_sizes", 50, 200)
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        learning_rate_init = trial.suggest_float(
            "learning_rate_init", 1e-4, 1e-2, log=True
        )

        pipeline = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(hidden_size,),
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=700,
                early_stopping=True,
                random_state=RANDOM_STATE,
            ),
        )

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
        )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print(f"Лучший CV F1 для MLP: {study.best_value:.4f}")
    print(f"Параметры MLP: {study.best_params}")

    return study.best_params


def train_models():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Не найден {DATA_PATH}. Сначала запусти fuzzy_model.py."
        )

    df = pd.read_csv(DATA_PATH)

    required = BASE_FEATURES + [FUZZY_FEATURE, TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"В данных отсутствуют столбцы: {missing}")

    df = df.dropna(subset=required).copy()
    df[TARGET] = df[TARGET].astype(int)

    if not set(df[TARGET].unique()).issubset({0, 1, 2}):
        raise ValueError("risk_class должен содержать классы 0, 1 и 2.")

    y = df[TARGET]
    X_all = df[BASE_FEATURES + [FUZZY_FEATURE]]

    X_train_all, X_test_all, y_train, y_test = train_test_split(
        X_all,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )

    best_mlp_params = tune_mlp(X_train_all, y_train, cv)

    model_factories = {
        "MLP": lambda: MLPClassifier(
            hidden_layer_sizes=(best_mlp_params["hidden_layer_sizes"],),
            alpha=best_mlp_params["alpha"],
            learning_rate_init=best_mlp_params["learning_rate_init"],
            max_iter=700,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": lambda: GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    feature_sets = {
        "without_fuzzy": BASE_FEATURES,
        "with_fuzzy": BASE_FEATURES + [FUZZY_FEATURE],
    }

    results = []
    trained_models = {}

    print("\nОбучение и сравнение 6 вариантов моделей...")

    for feature_set_name, feature_cols in feature_sets.items():
        X_train = X_train_all[feature_cols]
        X_test = X_test_all[feature_cols]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for model_name, factory in model_factories.items():
            print(f"\nОбучаем {model_name} ({feature_set_name})...")

            estimator = factory()
            estimator.fit(X_train_scaled, y_train)

            probabilities = estimator.predict_proba(X_test_scaled)

            # Сравнение проводится с учётом асимметричной стоимости ошибок.
            predictions = cost_sensitive_predict(estimator, X_test_scaled)

            metrics = calculate_metrics(
                y_test,
                predictions,
                probabilities,
                estimator.classes_,
            )

            row = {
                "model": model_name,
                "features": feature_set_name,
                "n_features": len(feature_cols),
                **metrics,
            }
            results.append(row)

            trained_models[(model_name, feature_set_name)] = {
                "model": estimator,
                "scaler": scaler,
                "feature_cols": feature_cols.copy(),
                "model_name": model_name,
                "features": feature_set_name,
            }

            print(
                f"F1={metrics['f1']:.4f} | "
                f"ROC-AUC={metrics['roc_auc']:.4f} | "
                f"Средняя бизнес-потеря={metrics['business_cost_mean']:.4f}"
            )

    results_df = pd.DataFrame(results)
    os.makedirs("data", exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    # Выбираем лучшую модель по F1, затем по средней бизнес-потере.
    ranking = results_df.sort_values(
        ["f1", "business_cost_mean"],
        ascending=[False, True],
    )
    best_row = ranking.iloc[0]
    best_key = (best_row["model"], best_row["features"])
    best_bundle = trained_models[best_key]

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)
    print(
        results_df[
            [
                "model",
                "features",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "mcc",
                "business_cost_mean",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nМатрица ошибок лучшей модели:")
    best_model = best_bundle["model"]
    best_scaler = best_bundle["scaler"]
    best_features = best_bundle["feature_cols"]
    best_test_scaled = best_scaler.transform(X_test_all[best_features])

    best_predictions = cost_sensitive_predict(best_model, best_test_scaled)
    print(
        confusion_matrix(y_test, best_predictions, labels=[0, 1, 2])
    )

    # Повторно обучаем выбранную модель на всех доступных данных.
    X_full = df[best_features]
    y_full = df[TARGET]

    final_scaler = StandardScaler()
    X_full_scaled = final_scaler.fit_transform(X_full)

    final_model = clone(best_model)
    final_model.fit(X_full_scaled, y_full)

    bundle = {
        "model": final_model,
        "scaler": final_scaler,
        "feature_cols": best_features,
        "model_name": best_row["model"],
        "features": best_row["features"],
        "cost_matrix": COST_MATRIX,
        "class_names": {
            0: "Низкий риск",
            1: "Средний риск",
            2: "Высокий риск",
        },
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    print("\n" + "=" * 60)
    print(f"Лучшая модель: {best_row['model']}")
    print(f"Набор признаков: {best_row['features']}")
    print(f"F1 macro: {best_row['f1']:.4f}")
    print(f"Средняя бизнес-потеря: {best_row['business_cost_mean']:.4f}")
    print(f"Сравнение сохранено: {RESULTS_PATH}")
    print(f"Лучшая модель сохранена: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train_models()
