from __future__ import annotations

from typing import Any

import joblib
import mlflow.pyfunc
import pandas as pd


DAY_COLUMNS = ["D-5", "D-4", "D-3", "D-2", "D-1", "D+1", "D+2", "D+3", "D+4", "D+5"]
NON_FEATURE_COLUMNS = DAY_COLUMNS + ["TARGET_MONTH", "TARGET_MONTH_PERIOD", "TARGET_RISK"]
RISK_TOP_K = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}


class CatBoostStrategyPyfuncModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper for the saved CatBoost next-month strategy bundle.

    The model expects already prepared monthly/rolling feature rows, not raw
    communication rows. Use the project inference pipeline to fetch and prepare
    data, then pass the prepared rows to this model.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.bundle: dict[str, Any] = joblib.load(context.artifacts["model_bundle"])

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        input_df = pd.DataFrame(model_input).copy()
        X = self._build_feature_matrix(input_df).reindex(
            columns=self.bundle["feature_columns"],
            fill_value=0,
        )

        output = pd.DataFrame(index=input_df.index)
        if "APAC_CARD_NUMBER" in input_df.columns:
            output["Loan_number"] = input_df["APAC_CARD_NUMBER"].astype(str)
        if "SOURCE_MONTH" in input_df.columns:
            output["SOURCE_MONTH_USED"] = input_df["SOURCE_MONTH"].astype(str)
            source_period = pd.to_datetime(
                output["SOURCE_MONTH_USED"],
                format="%b-%Y",
                errors="coerce",
            ).dt.to_period("M")
            target_offset_months = int(self.bundle.get("target_offset_months", 1))
            output["MONTH"] = (
                source_period + target_offset_months
            ).dt.to_timestamp().dt.strftime("%b-%Y").str.upper()

        risks = input_df.get("RISK", pd.Series("LOW", index=input_df.index))
        output["SOURCE_RISK"] = risks.fillna("LOW").astype(str).values
        for day in DAY_COLUMNS:
            output[day] = self._predict_top_k_by_risk(
                self.bundle["models"][day],
                self.bundle["label_encoders"][day],
                X,
                risks,
            )
        output["D"] = "-"
        ordered_columns = [
            col
            for col in [
                "SOURCE_RISK",
                "Loan_number",
                "SOURCE_MONTH_USED",
                "MONTH",
                "D-5",
                "D-4",
                "D-3",
                "D-2",
                "D-1",
                "D",
                "D+1",
                "D+2",
                "D+3",
                "D+4",
                "D+5",
            ]
            if col in output.columns
        ]
        return output[ordered_columns]

    @staticmethod
    def _build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
        feature_df = df.drop(columns=NON_FEATURE_COLUMNS, errors="ignore").copy()
        feature_df["RISK"] = feature_df.get("RISK", pd.Series("UNKNOWN", index=feature_df.index)).fillna("UNKNOWN")
        feature_df["SOURCE_MONTH"] = feature_df.get(
            "SOURCE_MONTH",
            pd.Series("UNKNOWN", index=feature_df.index),
        ).fillna("UNKNOWN")
        feature_df = pd.get_dummies(
            feature_df,
            columns=["SOURCE_MONTH", "RISK"],
            dummy_na=False,
        )
        feature_df = feature_df.drop(
            columns=["APAC_CARD_NUMBER", "SOURCE_MONTH_PERIOD"],
            errors="ignore",
        )
        return feature_df.fillna(0)

    @staticmethod
    def _predict_top_k_by_risk(model: Any, encoder: Any, X: pd.DataFrame, risks: pd.Series) -> pd.Series:
        probabilities = model.predict_proba(X)
        output: list[str] = []
        for row_probs, risk in zip(probabilities, risks.fillna("LOW").astype(str), strict=False):
            k = RISK_TOP_K.get(risk.upper(), 1)
            top_indices = row_probs.argsort()[-k:][::-1]
            labels = [str(encoder.classes_[index]) for index in top_indices]
            output.append("|".join(labels))
        return pd.Series(output, index=X.index)
