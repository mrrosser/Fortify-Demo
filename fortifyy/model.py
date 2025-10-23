
from typing import Dict, List
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, mean_absolute_error, mean_squared_error, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
def _make_preprocessor(num_features: List[str], cat_features: List[str]) -> ColumnTransformer:
    num_t = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_t = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer(transformers=[("num", num_t, num_features), ("cat", cat_t, cat_features)])
def build_classification_pipeline(num_features: List[str], cat_features: List[str]) -> Pipeline:
    pre = _make_preprocessor(num_features, cat_features)
    clf = GradientBoostingClassifier(random_state=42)
    return Pipeline(steps=[("pre", pre), ("clf", clf)])
def build_calibrated_classifier(num_features: List[str], cat_features: List[str]) -> Pipeline:
    return CalibratedClassifierCV(build_classification_pipeline(num_features, cat_features), cv=3, method="isotonic")
def build_regression_pipeline(num_features: List[str], cat_features: List[str]) -> Pipeline:
    pre = _make_preprocessor(num_features, cat_features)
    reg = GradientBoostingRegressor(random_state=42)
    return Pipeline(steps=[("pre", pre), ("reg", reg)])
def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins+1); inds = np.digitize(y_prob, bins) - 1; ece = 0.0
    for b in range(n_bins):
        mask = inds == b
        if not np.any(mask): continue
        conf = y_prob[mask].mean(); acc = y_true[mask].mean()
        ece += (abs(acc - conf) * (mask.sum() / len(y_true)))
    return float(ece)
def evaluate_classifier(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    prob = model.predict_proba(X)[:,1]; pred = (prob >= 0.5).astype(int)
    return {"roc_auc": float(roc_auc_score(y, prob)), "avg_precision": float(average_precision_score(y, prob)), "accuracy": float(accuracy_score(y, pred)), "brier": float(brier_score_loss(y, prob)), "ece_10": expected_calibration_error(y.values, prob, 10)}
def evaluate_regressor(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    pred = model.predict(X); import numpy as np
    return {"mae": float(mean_absolute_error(y, pred)), "rmse": float(np.sqrt(mean_squared_error(y, pred)))}
def top_permutation_importances(model: Pipeline, X: pd.DataFrame, y: pd.Series, feature_names: List[str], n_top: int = 12):
    r = permutation_importance(model, X, y, n_repeats=5, random_state=42)
    import pandas as pd
    imp = pd.DataFrame({"feature": feature_names, "importance": r.importances_mean})
    return imp.sort_values("importance", ascending=False).head(n_top)
