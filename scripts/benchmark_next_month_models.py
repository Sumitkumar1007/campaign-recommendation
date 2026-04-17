from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import warnings

import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import ConvergenceWarning

from project_paths import (
    BENCHMARK_DIR,
    CATBOOST_INFO_DIR,
    FEATURE_DATA_DIR,
    SCHEDULE_DATA_DIR,
    ensure_parent_dir,
)

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


DAY_COLUMNS = ["D-5", "D-4", "D-3", "D-2", "D-1", "D+1", "D+2", "D+3", "D+4", "D+5"]
NON_FEATURE_COLUMNS = DAY_COLUMNS + ["TARGET_MONTH", "TARGET_MONTH_PERIOD", "TARGET_RISK"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark next-month schedule models on the same source-month split."
    )
    parser.add_argument(
        "--feature-file",
        default=str(FEATURE_DATA_DIR / "strategy_monthly_features.csv"),
        help="Monthly aggregated feature CSV.",
    )
    parser.add_argument(
        "--schedule-file",
        default=str(SCHEDULE_DATA_DIR / "strategy_schedule_dataset_all_months.csv"),
        help="Schedule-style strategy target CSV.",
    )
    parser.add_argument(
        "--results-json",
        default=str(BENCHMARK_DIR / "next_month_model_benchmark.json"),
        help="Detailed benchmark JSON output path.",
    )
    parser.add_argument(
        "--results-csv",
        default=str(BENCHMARK_DIR / "next_month_model_benchmark_summary.csv"),
        help="Benchmark summary CSV output path.",
    )
    parser.add_argument(
        "--target-offset-months",
        type=int,
        default=1,
        help="How many months ahead the target schedule should come from.",
    )
    parser.add_argument(
        "--train-source-months",
        nargs="*",
        default=["NOV-2025", "DEC-2025", "JAN-2026"],
        help="Source months used for training.",
    )
    parser.add_argument(
        "--validation-source-months",
        nargs="*",
        default=["FEB-2026"],
        help="Source months used for validation.",
    )
    parser.add_argument(
        "--test-source-months",
        nargs="*",
        default=[],
        help="Source months used for testing.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=16,
        help="Parallel workers for fitting the per-day models.",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=300,
        help="Number of trees for Random Forest.",
    )
    parser.add_argument(
        "--et-n-estimators",
        type=int,
        default=300,
        help="Number of trees for Extra Trees.",
    )
    parser.add_argument(
        "--xgb-n-estimators",
        type=int,
        default=300,
        help="Number of trees for XGBoost.",
    )
    parser.add_argument(
        "--lgbm-n-estimators",
        type=int,
        default=500,
        help="Number of trees for LightGBM.",
    )
    parser.add_argument(
        "--catboost-iterations",
        type=int,
        default=500,
        help="Number of boosting rounds for CatBoost.",
    )
    parser.add_argument(
        "--gb-n-estimators",
        type=int,
        default=200,
        help="Number of boosting stages for Gradient Boosting.",
    )
    parser.add_argument(
        "--knn-n-neighbors",
        type=int,
        default=15,
        help="Number of neighbors for KNN.",
    )
    parser.add_argument(
        "--label-powerset-n-estimators",
        type=int,
        default=300,
        help="Number of trees for the Label Powerset wrapper model.",
    )
    parser.add_argument(
        "--logistic-max-iter",
        type=int,
        default=300,
        help="Maximum iterations for logistic baseline.",
    )
    parser.add_argument(
        "--mlp-hidden-layers",
        default="256,128",
        help="Comma-separated hidden layer sizes for the MLP baseline.",
    )
    parser.add_argument(
        "--mlp-max-iter",
        type=int,
        default=100,
        help="Maximum iterations for the MLP baseline.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "catboost",
            "lightgbm",
            "xgboost",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "mlp",
            "gaussian_nb",
            "knn",
            "label_powerset",
        ],
        help="Subset of models to benchmark.",
    )
    return parser.parse_args()


def month_to_period(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%b-%Y", errors="coerce").dt.to_period("M")


def prepare_dataset(
    feature_file: Path,
    schedule_file: Path,
    target_offset_months: int,
) -> pd.DataFrame:
    features = pd.read_csv(feature_file).copy()
    schedule = pd.read_csv(schedule_file).copy()

    features["SOURCE_MONTH"] = features["MONTH"]
    features["SOURCE_MONTH_PERIOD"] = month_to_period(features["SOURCE_MONTH"])
    features["TARGET_MONTH_PERIOD"] = features["SOURCE_MONTH_PERIOD"] + target_offset_months
    features = features.drop(columns=["MONTH"])

    schedule["TARGET_MONTH"] = schedule["MONTH"]
    schedule["TARGET_MONTH_PERIOD"] = month_to_period(schedule["TARGET_MONTH"])
    schedule = schedule.rename(
        columns={
            "Loan_number": "APAC_CARD_NUMBER",
            "RISK": "TARGET_RISK",
        }
    )
    schedule = schedule.drop(columns=["MONTH"])

    return features.merge(
        schedule[
            ["APAC_CARD_NUMBER", "TARGET_MONTH", "TARGET_MONTH_PERIOD", "TARGET_RISK", *DAY_COLUMNS]
        ],
        on=["APAC_CARD_NUMBER", "TARGET_MONTH_PERIOD"],
        how="left",
    )


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.drop(columns=NON_FEATURE_COLUMNS, errors="ignore").copy()
    feature_df["RISK"] = feature_df["RISK"].fillna("UNKNOWN")
    feature_df["SOURCE_MONTH"] = feature_df["SOURCE_MONTH"].fillna("UNKNOWN")
    feature_df = pd.get_dummies(
        feature_df,
        columns=["SOURCE_MONTH", "RISK"],
        dummy_na=False,
    )
    feature_df = feature_df.drop(columns=["APAC_CARD_NUMBER", "SOURCE_MONTH_PERIOD"], errors="ignore")
    return feature_df.fillna(0)


def split_by_source_month(
    dataset: pd.DataFrame,
    months: list[str],
    require_target: bool,
) -> pd.DataFrame:
    selected = dataset[dataset["SOURCE_MONTH"].isin(months)].copy()
    if require_target:
        selected = selected.dropna(subset=["TARGET_MONTH"])
    return selected


def build_model_factory(model_name: str, args: argparse.Namespace):
    mlp_hidden_layers = tuple(
        int(part.strip()) for part in args.mlp_hidden_layers.split(",") if part.strip()
    )
    if model_name == "logistic":
        return lambda: LogisticRegression(
            max_iter=args.logistic_max_iter,
            solver="lbfgs",
            random_state=42,
        )
    if model_name == "random_forest":
        return lambda: RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            random_state=42,
            n_jobs=1,
            class_weight="balanced_subsample",
        )
    if model_name == "extra_trees":
        return lambda: ExtraTreesClassifier(
            n_estimators=args.et_n_estimators,
            random_state=42,
            n_jobs=1,
            class_weight="balanced_subsample",
        )
    if model_name == "xgboost":
        if XGBClassifier is None:
            return None
        return lambda: XGBClassifier(
            n_estimators=args.xgb_n_estimators,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2,
            objective="multi:softmax",
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=1,
            random_state=42,
        )
    if model_name == "lightgbm":
        if LGBMClassifier is None:
            return None
        return lambda: LGBMClassifier(
            n_estimators=args.lgbm_n_estimators,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multiclass",
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        )
    if model_name == "catboost":
        if CatBoostClassifier is None:
            return None
        return lambda: CatBoostClassifier(
            iterations=args.catboost_iterations,
            learning_rate=0.05,
            depth=8,
            loss_function="MultiClass",
            random_seed=42,
            verbose=False,
            thread_count=1,
            train_dir=str(CATBOOST_INFO_DIR / "benchmark"),
        )
    if model_name == "gradient_boosting":
        return lambda: GradientBoostingClassifier(
            n_estimators=args.gb_n_estimators,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
    if model_name == "mlp":
        return lambda: MLPClassifier(
            hidden_layer_sizes=mlp_hidden_layers,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=512,
            learning_rate_init=1e-3,
            max_iter=args.mlp_max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
    if model_name == "gaussian_nb":
        return lambda: GaussianNB()
    if model_name == "knn":
        return lambda: KNeighborsClassifier(
            n_neighbors=args.knn_n_neighbors,
            weights="distance",
            n_jobs=1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def fit_day_model(
    day: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_factory,
):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train[day].astype(str))
    model = model_factory()
    if model.__class__.__name__ == "XGBClassifier":
        model.set_params(num_class=len(encoder.classes_))
    model.fit(X_train, y_encoded)
    return day, model, encoder


def evaluate_models(models: dict, encoders: dict, X: pd.DataFrame, y: pd.DataFrame) -> dict:
    if X.empty:
        return {"rows": 0}

    pred_columns: dict[str, pd.Series] = {}
    per_day_accuracy: dict[str, float] = {}
    for day in DAY_COLUMNS:
        y_pred_encoded = models[day].predict(X)
        y_pred = encoders[day].inverse_transform(y_pred_encoded)
        pred_columns[day] = pd.Series(y_pred, index=y.index)
        per_day_accuracy[day] = float(accuracy_score(y[day], y_pred))

    pred_df = pd.DataFrame(pred_columns)
    return {
        "rows": int(len(X)),
        "exact_match_accuracy": float((pred_df == y).all(axis=1).mean()),
        "average_day_accuracy": float(pd.Series(per_day_accuracy).mean()),
        "per_day_accuracy": per_day_accuracy,
    }


def benchmark_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    model_factory = build_model_factory(model_name, args)
    if model_factory is None:
        return {
            "status": "skipped",
            "reason": f"{model_name} is not installed in the recommendation venv",
        }

    started = time.perf_counter()
    fitted_models = Parallel(n_jobs=args.n_jobs, verbose=10, prefer="threads")(
        delayed(fit_day_model)(day, X_train, y_train, model_factory)
        for day in DAY_COLUMNS
    )
    fit_seconds = time.perf_counter() - started

    models = {day: model for day, model, _ in fitted_models}
    encoders = {day: encoder for day, _, encoder in fitted_models}
    return {
        "status": "completed",
        "fit_seconds": round(fit_seconds, 3),
        "train_metrics": evaluate_models(models, encoders, X_train, y_train),
        "validation_metrics": evaluate_models(models, encoders, X_validation, y_validation),
        "test_metrics": evaluate_models(models, encoders, X_test, y_test),
    }


def benchmark_label_powerset(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    y_train_seq = y_train.astype(str).agg("|".join, axis=1)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_seq)

    model = RandomForestClassifier(
        n_estimators=args.label_powerset_n_estimators,
        random_state=42,
        n_jobs=args.n_jobs,
        class_weight="balanced_subsample",
    )
    started = time.perf_counter()
    model.fit(X_train, y_train_encoded)
    fit_seconds = time.perf_counter() - started

    def evaluate_split(X: pd.DataFrame, y: pd.DataFrame) -> dict:
        if X.empty:
            return {"rows": 0}
        pred_encoded = model.predict(X)
        pred_sequences = label_encoder.inverse_transform(pred_encoded)
        pred_df = pd.DataFrame(
            [seq.split("|") for seq in pred_sequences],
            columns=DAY_COLUMNS,
            index=y.index,
        )
        per_day_accuracy = {
            day: float(accuracy_score(y[day], pred_df[day])) for day in DAY_COLUMNS
        }
        return {
            "rows": int(len(X)),
            "exact_match_accuracy": float((pred_df == y).all(axis=1).mean()),
            "average_day_accuracy": float(pd.Series(per_day_accuracy).mean()),
            "per_day_accuracy": per_day_accuracy,
            "unique_train_sequences": int(len(label_encoder.classes_)),
        }

    return {
        "status": "completed",
        "fit_seconds": round(fit_seconds, 3),
        "train_metrics": evaluate_split(X_train, y_train),
        "validation_metrics": evaluate_split(X_validation, y_validation),
        "test_metrics": evaluate_split(X_test, y_test),
    }


def save_results(results: dict, results_json: Path, results_csv: Path) -> None:
    results_json.write_text(json.dumps(results, indent=2))

    summary_rows = []
    for model_name, model_result in results["models"].items():
        summary_rows.append(
            {
                "model": model_name,
                "status": model_result.get("status"),
                "fit_seconds": model_result.get("fit_seconds"),
                "validation_exact_match_accuracy": model_result.get("validation_metrics", {}).get(
                    "exact_match_accuracy"
                ),
                "validation_average_day_accuracy": model_result.get(
                    "validation_metrics", {}
                ).get("average_day_accuracy"),
                "test_exact_match_accuracy": model_result.get("test_metrics", {}).get(
                    "exact_match_accuracy"
                ),
                "test_average_day_accuracy": model_result.get("test_metrics", {}).get(
                    "average_day_accuracy"
                ),
                "reason": model_result.get("reason"),
            }
        )
    pd.DataFrame(summary_rows).to_csv(results_csv, index=False)


def load_existing_results(results_json: Path) -> dict | None:
    if not results_json.exists() or results_json.stat().st_size == 0:
        return None
    return json.loads(results_json.read_text())


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    results_json = ensure_parent_dir(args.results_json)
    results_csv = ensure_parent_dir(args.results_csv)
    dataset = prepare_dataset(
        Path(args.feature_file),
        Path(args.schedule_file),
        args.target_offset_months,
    )

    train_df = split_by_source_month(dataset, args.train_source_months, require_target=True)
    validation_df = split_by_source_month(
        dataset, args.validation_source_months, require_target=True
    )
    test_df = split_by_source_month(dataset, args.test_source_months, require_target=True)

    X_train = build_feature_matrix(train_df)
    X_validation = build_feature_matrix(validation_df).reindex(columns=X_train.columns, fill_value=0)
    X_test = build_feature_matrix(test_df).reindex(columns=X_train.columns, fill_value=0)

    y_train = train_df[DAY_COLUMNS].fillna("-")
    y_validation = validation_df[DAY_COLUMNS].fillna("-")
    y_test = test_df[DAY_COLUMNS].fillna("-")

    default_results = {
        "setup": {
            "target_offset_months": args.target_offset_months,
            "train_source_months": args.train_source_months,
            "validation_source_months": args.validation_source_months,
            "test_source_months": args.test_source_months,
            "n_jobs": args.n_jobs,
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "test_rows": int(len(test_df)),
        },
        "models": {},
    }
    results = load_existing_results(results_json) or default_results

    save_results(results, results_json, results_csv)

    for model_name in args.models:
        print(f"Benchmarking {model_name}...")
        try:
            if model_name == "label_powerset":
                model_result = benchmark_label_powerset(
                    X_train,
                    y_train,
                    X_validation,
                    y_validation,
                    X_test,
                    y_test,
                    args,
                )
            else:
                model_result = benchmark_model(
                    model_name,
                    X_train,
                    y_train,
                    X_validation,
                    y_validation,
                    X_test,
                    y_test,
                    args,
                )
        except Exception as exc:
            model_result = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        results["models"][model_name] = model_result
        save_results(results, results_json, results_csv)

    print(f"Saved detailed benchmark to {results_json}")
    print(f"Saved benchmark summary to {results_csv}")


if __name__ == "__main__":
    main()
