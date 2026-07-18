"""
wdl_model.py: calibrated XGBoost win/draw/loss classifier.

Trains on historical match features (Elo, form, h2h) to predict W/D/L
probabilities. Wraps the raw XGBoost output with isotonic calibration fit on a
held-out validation slice, so the numbers are honest probabilities, not
overconfident softmax scores.

The calibrated result is used by the hybrid inversion (model.py) to set the
home-vs-away tilt of the Poisson scoreline grid. Total goals are anchored
independently from the ratings model so draw-probability error doesn't corrupt
prop markets.

Usage
-----
  train_model(train_df, val_df)  ->  (model, calibrator)
  predict_proba(model, calibrator, X)  ->  (p_home, p_draw, p_away)
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

FEATURES = [
    "neutral", "tournament_weight",
    "home_elo", "away_elo", "elo_diff",
    "home_win5", "away_win5",
    "home_gd5",  "away_gd5",
    "home_win10","away_win10",
    "home_rest_days", "away_rest_days",
    "h2h_n", "h2h_home_winrate", "h2h_home_gd",
]

TRAIN_START = "2006-01-01"
VAL_START   = "2023-01-01"


def split_by_date(dataset, train_start=TRAIN_START, val_start=VAL_START, cutoff=None):
    """Time-based split: train / val, optionally capped at `cutoff`."""
    import pandas as pd
    train = dataset[
        (dataset["date"] >= pd.Timestamp(train_start)) &
        (dataset["date"] <  pd.Timestamp(val_start))
    ].copy()
    val_end = pd.Timestamp(cutoff) if cutoff else dataset["date"].max()
    val = dataset[
        (dataset["date"] >= pd.Timestamp(val_start)) &
        (dataset["date"] <  val_end)
    ].copy()
    return train, val


def _fill_na(df):
    """Fill NaN feature values with sensible defaults."""
    defaults = {
        "home_win5": 0.5, "away_win5": 0.5,
        "home_gd5":  0.0, "away_gd5":  0.0,
        "home_win10":0.5, "away_win10":0.5,
        "home_rest_days": 30.0, "away_rest_days": 30.0,
        "h2h_n": 0.0, "h2h_home_winrate": 0.5, "h2h_home_gd": 0.0,
    }
    return df.fillna(defaults)


def train_model(train, val):
    """
    Fit an XGBoost multi-class classifier on train, using val for early stopping.
    Returns (xgb_model, calibrators) where calibrators is a list of three
    IsotonicRegression objects (one per class: home, draw, away).
    """
    train = _fill_na(train)
    val   = _fill_na(val)

    X_train = train[FEATURES].astype(float)
    y_train = train["label"].astype(int)
    X_val   = val[FEATURES].astype(float)
    y_val   = val["label"].astype(int)

    model = xgb.XGBClassifier(
        objective        = "multi:softprob",
        num_class        = 3,
        n_estimators     = 600,
        learning_rate    = 0.05,
        max_depth        = 5,
        subsample        = 0.85,
        colsample_bytree = 0.85,
        reg_lambda       = 1.0,
        eval_metric      = "mlogloss",
        early_stopping_rounds = 50,
        tree_method      = "hist",
        n_jobs           = -1,
        random_state     = 42,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    # Calibrate: fit one IsotonicRegression per class on the validation set.
    # This maps raw softmax scores to calibrated probabilities.
    raw = model.predict_proba(X_val)            # shape (n, 3)
    calibrators = []
    for cls in range(3):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw[:, cls], (y_val == cls).astype(float))
        calibrators.append(ir)

    return model, calibrators


def predict_proba(model, calibrators, X):
    """
    Return calibrated (p_home, p_draw, p_away) from a single-row feature DataFrame.
    Renormalizes after calibration so the three probs sum to 1.
    """
    raw = model.predict_proba(X)[0]             # (3,)
    cal = np.array([calibrators[c].predict([raw[c]])[0] for c in range(3)])
    cal = np.clip(cal, 1e-4, 1.0)
    cal /= cal.sum()
    return float(cal[0]), float(cal[1]), float(cal[2])


def evaluate(model, calibrators, val):
    """Print accuracy + log-loss (raw vs calibrated) on the val slice."""
    val = _fill_na(val)
    X_val = val[FEATURES].astype(float)
    y_val = val["label"].astype(int).values

    raw   = model.predict_proba(X_val)
    cal   = np.column_stack([
        calibrators[c].predict(raw[:, c]) for c in range(3)
    ])
    cal   = np.clip(cal, 1e-4, 1.0)
    cal   /= cal.sum(axis=1, keepdims=True)

    pred  = cal.argmax(axis=1)
    base  = np.tile(np.bincount(y_val, minlength=3) / len(y_val), (len(y_val), 1))

    print(f"  Validation accuracy  : {accuracy_score(y_val, pred):.3f}")
    print(f"  Raw log-loss         : {log_loss(y_val, raw):.3f}")
    print(f"  Calibrated log-loss  : {log_loss(y_val, cal):.3f}  "
          f"(baseline {log_loss(y_val, base, labels=[0,1,2]):.3f})")
