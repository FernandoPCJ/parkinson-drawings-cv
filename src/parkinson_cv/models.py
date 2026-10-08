"""Model factories that plug into the cross-validation loop (`cv.oof_predictions`)."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .cv import labels


def logistic_fit_predict(features: list[str]):
    """Logistic regression on tabular features, as a `fit_predict(train_df, test_df)`.

    - StandardScaler puts every feature on the same scale (mean 0, std 1) using ONLY the
      training fold, so nothing about the test fold leaks into the model.
    - class_weight="balanced" makes both classes count equally, so the model cannot win
      by always answering the majority class (the trap of baseline 1).
    - The hard prediction is "probability >= 0.5"; the score fed to AUC is the probability.
    """

    def fit_predict(train_df, test_df):
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=1000, class_weight="balanced"))
        model.fit(train_df[features].to_numpy(float), labels(train_df))
        proba = model.predict_proba(test_df[features].to_numpy(float))[:, 1]
        return (proba >= 0.5).astype(int), proba

    return fit_predict
