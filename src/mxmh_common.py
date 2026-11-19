import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

#---------------------------------------------------------------------------
# Experiment Settings
#---------------------------------------------------------------------------
RANDOM_STATE = 42   # seed for the train/test split and all CV shuffles
TEST_SIZE = 0.2     # 80/20 train/test split
N_SPLITS = 5        # folds in both the inner and outer CV
N_REPEATS = 10      # outer CV repeats -> 50 outer folds
SCORING = "r2"      # metric used to choose alpha in the inner CV

#---------------------------------------------------------------------------
# File Paths
#---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent  # absolute path for the folder with this file
DATA_DIR = HERE.parent / "data"
RAW_DATA_PATH = DATA_DIR / "mxmh_survey_results.csv"
CLEAN_DATA_PATH = DATA_DIR / "mxmh_survey_results_clean.csv"
RESULTS_DIR = HERE.parent / "results"

#---------------------------------------------------------------------------
# Variables in the Data Set
#---------------------------------------------------------------------------

# Mental health scores are the target variables
TARGETS = ["Anxiety", "Depression", "Insomnia", "OCD"]

# Columns not used in the analysis:
#   Timestamp, Permissions: administrative fields
#   BPM: about 15% missing and has implausible values
#   Music effects: an outcome of listening, not a predictor
UNUSED_COLUMNS = ["Timestamp", "Permissions", "BPM", "Music effects"]

NUMERIC_COLS = ["Age", "Hours per day"]
BINARY_COLS = ["While working", "Instrumentalist", "Composer", "Exploratory", "Foreign languages"]

GENRES = ["Classical", "Country", "EDM", "Folk", "Gospel", "Hip hop", "Jazz", "K pop",
          "Latin", "Lofi", "Metal", "Pop", "R&B", "Rap", "Rock", "Video game music"]
FREQ_COLS = [f"Frequency [{g}]" for g in GENRES]

FREQ_LEVELS = ["Never", "Rarely", "Sometimes", "Very frequently"]
STREAMING_LEVELS = ["Spotify", "Apple Music", "I do not use a streaming service.",
                    "Other streaming service", "Pandora", "YouTube Music"]
GENRE_LEVELS = GENRES

ALL_FEATURES = NUMERIC_COLS + BINARY_COLS + FREQ_COLS + ["Primary streaming service", "Fav genre"]

#---------------------------------------------------------------------------
# Data Set Access
#---------------------------------------------------------------------------

def load_raw():
    """Read the raw (original) data set file and return a Pandas frame."""
    return pd.read_csv(RAW_DATA_PATH)

def save_clean_dataset():
    """Drop unused columns and incomplete rows, then save the cleaned dataset."""
    df = load_raw()
    df = df.drop(columns=UNUSED_COLUMNS).dropna()
    df.to_csv(CLEAN_DATA_PATH, index=False)
    return df

def load_clean_dataset():
    """Read the cleaned dataset and return two data frames: X (all features) and Y (all targets)."""
    df = pd.read_csv(CLEAN_DATA_PATH)
    X = df[ALL_FEATURES].copy()
    Y = df[TARGETS].copy()
    return X, Y

def split_train_test(X, Y):
    """Split into training and test sets (80/20, fixed seed). Returns X_train, X_test, Y_train, Y_test."""
    return train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

#---------------------------------------------------------------------------
# Feature Encoding
#---------------------------------------------------------------------------

def make_feature_encoder(columns=ALL_FEATURES, freq_encoding="ordinal"):
    """Build the ColumnTransformer that encodes the raw survey columns.

    Only the groups present in `columns` are encoded; all other columns are dropped.
      numeric:   Age, Hours per day, passed through unchanged
      binary:    No = 0, Yes = 1
      frequency: "ordinal" -> Never = 0, Rarely = 1, Sometimes = 2, Very frequently = 3
                 "onehot"  -> 3 dummies per genre, Never is the reference
      streaming: one-hot, Spotify is the reference
      fav genre: one-hot, Classical is the reference
    """
    columns = list(columns)
    parts = []

    numeric_cols = [c for c in NUMERIC_COLS if c in columns]
    if numeric_cols:
        parts.append(("numeric", "passthrough", numeric_cols))

    binary_cols = [c for c in BINARY_COLS if c in columns]
    if binary_cols:
        parts.append(("binary", OrdinalEncoder(categories=[["No", "Yes"]] * len(binary_cols)), binary_cols))

    freq_cols = [c for c in FREQ_COLS if c in columns]
    if freq_cols:
        if freq_encoding == "ordinal":
            # Never = 0, Rarely = 1, Sometimes = 2, Very frequently = 3 (order of FREQ_LEVELS)
            enc = OrdinalEncoder(categories=[FREQ_LEVELS] * len(freq_cols))
        elif freq_encoding == "onehot":
            enc = OneHotEncoder(categories=[FREQ_LEVELS] * len(freq_cols), drop="first", sparse_output=False)
        else:
            raise ValueError(f"freq_encoding must be 'ordinal' or 'onehot', got {freq_encoding!r}")
        parts.append(("frequency", enc, freq_cols))

    if "Primary streaming service" in columns:
        # Spotify is the reference (dropped) level
        enc = OneHotEncoder(categories=[STREAMING_LEVELS], drop=["Spotify"], sparse_output=False)
        parts.append(("streaming", enc, ["Primary streaming service"]))

    if "Fav genre" in columns:
        # Classical is the reference (dropped) level
        enc = OneHotEncoder(categories=[GENRE_LEVELS], drop=["Classical"], sparse_output=False)
        parts.append(("fav_genre", enc, ["Fav genre"]))

    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)

#---------------------------------------------------------------------------
# Models
#---------------------------------------------------------------------------

# Alpha grids for Lasso and Ridge: log-spaced values. The ranges were chosen so that
# the alpha selected by the inner CV is always inside the grid, never at an edge.
LASSO_ALPHAS = np.logspace(start=-4, stop=1, num=26)   # 1e-4 to 10
RIDGE_ALPHAS = np.logspace(start=-2, stop=6, num=33)   # 1e-2 to 1e6

# Inner CV: used by GridSearchCV to tune alpha within each outer training fold
INNER_CV = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

def make_models(columns=ALL_FEATURES, freq_encoding="ordinal"):
    """Return the models to compare, as {name: unfitted estimator}.

    Each model is a Pipeline: encode -> (scale) -> model. Lasso and Ridge are wrapped in
    GridSearchCV, so alpha is tuned by INNER_CV every time the model is fitted.
    """
    # Dummy Regressor model as (Mean) baseline: Encode -> Model (no scaling needed)
    mean_baseline_estimator = Pipeline([
        ("encode", make_feature_encoder(columns, freq_encoding)),
        ("model", DummyRegressor(strategy="mean")),
    ])

    # Linear Regression: Encode -> Scale -> Model
    linear_estimator = Pipeline([
        ("encode", make_feature_encoder(columns, freq_encoding)),
        ("scale", StandardScaler()),
        ("model", LinearRegression()),
    ])

    # Lasso: Encode -> Scale -> Model, with alpha tuned by inner CV
    lasso_pipeline = Pipeline([
        ("encode", make_feature_encoder(columns, freq_encoding)),
        ("scale", StandardScaler()),
        ("model", Lasso(max_iter=50_000)),
    ])
    lasso_estimator = GridSearchCV(lasso_pipeline,
                                   {"model__alpha": LASSO_ALPHAS}, cv=INNER_CV,
                                   scoring=SCORING, refit=True)

    # Ridge: Encode -> Scale -> Model, with alpha tuned by inner CV
    ridge_pipeline = Pipeline([
        ("encode", make_feature_encoder(columns, freq_encoding)),
        ("scale", StandardScaler()),
        ("model", Ridge()),
    ])
    ridge_estimator = GridSearchCV(ridge_pipeline,
                                   {"model__alpha": RIDGE_ALPHAS}, cv=INNER_CV,
                                   scoring=SCORING, refit=True)

    models = {
        "Mean baseline": mean_baseline_estimator,
        "Linear": linear_estimator,
        "Lasso": lasso_estimator,
        "Ridge": ridge_estimator,
    }
    return models

#---------------------------------------------------------------------------
# Evaluation
#---------------------------------------------------------------------------

# Outer CV: measures performance. With a fixed seed it gives the same 50 folds on every use.
OUTER_CV = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

_SCORERS = {"R2": "r2", "RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error"}

def outer_fold_sizes(n_rows, n_splits=N_SPLITS):
    """Return (n_train, n_test): the training and held-out sizes of each outer fold."""
    n_test = n_rows / n_splits
    return n_rows - n_test, n_test

def outer_cv_scores(estimator, X, y):
    """Score one estimator on every outer fold.

    Returns one row per fold (fold, R2, RMSE, MAE).
    The outer folds and the tuning inside each fold both run serially (n_jobs=1).
    """
    res = cross_validate(estimator, X, y, cv=OUTER_CV, scoring=_SCORERS, n_jobs=1)
    return pd.DataFrame({
        "fold": np.arange(len(res["test_R2"])),
        "R2": res["test_R2"],
        "RMSE": -res["test_RMSE"],
        "MAE": -res["test_MAE"],
    })

def corrected_ci(values, n_train, n_test, alpha=0.05):
    """Mean and (1 - alpha) CI of per-fold values, with the Nadeau & Bengio (2003) variance
    correction.

    Folds in repeated cross-validation share training data, so a plain t-test overstates
    confidence. The corrected variance is (1/J + n_test/n_train) * s^2 for J folds
    (n_folds below).

    p_corrected and share_positive test against 0, so they are meaningful for R2 or for
    differences between models, not for RMSE or MAE.
    """
    values = np.asarray(values, dtype=float)
    n_folds = len(values)
    mean = values.mean()
    se = np.sqrt((1 / n_folds + n_test / n_train) * values.var(ddof=1))
    t_crit = stats.t.ppf(1 - alpha / 2, n_folds - 1)
    t_stat = mean / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t_stat), n_folds - 1) if se > 0 else np.nan
    return {"mean": mean, "ci_low": mean - t_crit * se, "ci_high": mean + t_crit * se,
            "p_corrected": p, "share_positive": float((values > 0).mean())}

#---------------------------------------------------------------------------
# Nested CV Analysis
#---------------------------------------------------------------------------

def run_nested_cv():
    """Run nested CV for every target and model on the training set.

    Saves the per-fold scores to results/nested_cv_folds.csv and returns them as a
    DataFrame with columns fold, R2, RMSE, MAE, target, model.
    """
    X, Y = load_clean_dataset()
    X_train, X_test, Y_train, Y_test = split_train_test(X, Y)
    del X_test, Y_test  # the test data is not used in nested cross-validation
    n_train, n_test = outer_fold_sizes(len(X_train))
    print(f"Outer folds: {OUTER_CV.get_n_splits()}  (train {n_train:.0f}, held out {n_test:.0f} per fold)")

    folds = []
    for target in TARGETS:
        for name, model in make_models().items():
            start = time.time()
            scores = outer_cv_scores(model, X_train, Y_train[target])
            scores["target"], scores["model"] = target, name
            folds.append(scores)
            print(f"{target:10s} {name:14s} mean R2 = {scores.R2.mean():.3f}   ({time.time() - start:.0f}s)")
    folds = pd.concat(folds, ignore_index=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    folds.to_csv(RESULTS_DIR / "nested_cv_folds.csv", index=False)
    return folds
