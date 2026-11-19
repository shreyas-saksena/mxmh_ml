import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    SplineTransformer,
    StandardScaler,
)

#---------------------------------------------------------------------------
# Experiment Settings
#---------------------------------------------------------------------------
RANDOM_STATE = 42   # seed for the train/test split and all CV shuffles
TEST_SIZE = 0.2     # 80/20 train/test split
N_SPLITS = 5        # folds in both the inner and outer CV
N_REPEATS = 10      # outer CV repeats -> 50 outer folds
SCORING = "r2"      # metric used to choose alpha in the inner CV
N_BOOTSTRAP = 2000  # bootstrap resamples of the test set for the final evaluation

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

# Feature sets for the age vs music comparison. "Music" is every feature except Age.
AGE_ONLY = ["Age"]
MUSIC_ONLY = [c for c in ALL_FEATURES if c != "Age"]
AGE_PLUS_MUSIC = ALL_FEATURES
FEATURE_SETS = {
    "Age only": AGE_ONLY,
    "Music only": MUSIC_ONLY,
    "Age + music": AGE_PLUS_MUSIC,
}

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

def make_feature_encoder(columns=ALL_FEATURES, freq_encoding="ordinal", age_spline=False):
    """Build the ColumnTransformer that encodes the raw survey columns.

    Only the groups present in `columns` are encoded; all other columns are dropped.
      numeric:   Age, Hours per day, passed through unchanged
                 (with age_spline=True, Age enters through a cubic spline with 5 knots,
                 so a linear model can fit a curved age effect)
      binary:    No = 0, Yes = 1
      frequency: "ordinal" -> Never = 0, Rarely = 1, Sometimes = 2, Very frequently = 3
                 "onehot"  -> 3 dummies per genre, Never is the reference
      streaming: one-hot, Spotify is the reference
      fav genre: one-hot, Classical is the reference
    """
    columns = list(columns)
    parts = []

    numeric_cols = [c for c in NUMERIC_COLS if c in columns]
    if age_spline and "Age" in numeric_cols:
        numeric_cols.remove("Age")
        parts.append(("age_spline", SplineTransformer(n_knots=5, degree=3), ["Age"]))
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

# Random forest grid for the tuned forest. The number of trees is fixed: more trees only
# stabilize a forest, they don't make it overfit. min_samples_leaf limits tree complexity
# (so max_depth is left unlimited). As with alpha, the ranges were widened until the
# selected values were inside the grid.
RF_N_ESTIMATORS = 300
RF_MAX_FEATURES = [0.05, 0.1, 0.2, 0.33, 0.6, 1.0]   # share of features tried at each split
RF_MIN_SAMPLES_LEAF = [1, 5, 10, 20, 40, 80]

# Inner CV: used by GridSearchCV to tune hyperparameters within each outer training fold
INNER_CV = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

def make_models(columns=ALL_FEATURES, freq_encoding="ordinal", age_spline=False):
    """Return the models to compare, as {name: unfitted estimator}.

    Each model is a Pipeline: encode -> (scale) -> model. Lasso, Ridge and the tuned random
    forest are wrapped in GridSearchCV, so their hyperparameters are tuned by INNER_CV every
    time the model is fitted. `columns`, `freq_encoding` and `age_spline` are passed to
    make_feature_encoder.
    """
    # Dummy Regressor model as (Mean) baseline: Encode -> Model (no scaling needed)
    mean_baseline_estimator = Pipeline([
        ("encode", make_feature_encoder(columns, freq_encoding, age_spline)),
        ("model", DummyRegressor(strategy="mean")),
    ])

    # Linear Regression: Encode -> Scale -> Model
    linear_estimator = Pipeline([
        ("encode", make_feature_encoder(columns, freq_encoding, age_spline)),
        ("scale", StandardScaler()),
        ("model", LinearRegression()),
    ])

    # Lasso: Encode -> Scale -> Model, with alpha tuned by inner CV
    lasso_pipeline = Pipeline([
        ("encode", make_feature_encoder(columns, freq_encoding, age_spline)),
        ("scale", StandardScaler()),
        ("model", Lasso(max_iter=50_000)),
    ])
    lasso_estimator = GridSearchCV(lasso_pipeline,
                                   {"model__alpha": LASSO_ALPHAS}, cv=INNER_CV,
                                   scoring=SCORING, refit=True)

    # Ridge: Encode -> Scale -> Model, with alpha tuned by inner CV
    ridge_pipeline = Pipeline([
        ("encode", make_feature_encoder(columns, freq_encoding, age_spline)),
        ("scale", StandardScaler()),
        ("model", Ridge()),
    ])
    ridge_estimator = GridSearchCV(ridge_pipeline,
                                   {"model__alpha": RIDGE_ALPHAS}, cv=INNER_CV,
                                   scoring=SCORING, refit=True)

    # Random forest with scikit-learn defaults (100 trees, leaf size 1, all features):
    # Encode -> Model. Trees only use the order of values, so no scaling.
    rf_default_estimator = Pipeline([
        ("encode", make_feature_encoder(columns, freq_encoding, age_spline)),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)),
    ])

    # Tuned random forest: Encode -> Model, with max_features and min_samples_leaf
    # tuned by inner CV
    rf_tuned_pipeline = Pipeline([
        ("encode", make_feature_encoder(columns, freq_encoding, age_spline)),
        ("model", RandomForestRegressor(n_estimators=RF_N_ESTIMATORS,
                                        random_state=RANDOM_STATE, n_jobs=1)),
    ])
    rf_tuned_estimator = GridSearchCV(rf_tuned_pipeline,
                                      {"model__max_features": RF_MAX_FEATURES,
                                       "model__min_samples_leaf": RF_MIN_SAMPLES_LEAF},
                                      cv=INNER_CV, scoring=SCORING, refit=True)

    models = {
        "Mean baseline": mean_baseline_estimator,
        "Linear": linear_estimator,
        "Lasso": lasso_estimator,
        "Ridge": ridge_estimator,
        "RF default": rf_default_estimator,
        "RF tuned": rf_tuned_estimator,
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
    The outer folds run in parallel on all CPU cores (n_jobs=-1); the tuning inside each
    fold runs serially. Results do not depend on n_jobs because every split and forest
    uses a fixed seed.
    """
    res = cross_validate(estimator, X, y, cv=OUTER_CV, scoring=_SCORERS, n_jobs=-1)
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

#---------------------------------------------------------------------------
# Age vs Music Analysis
#---------------------------------------------------------------------------

def run_age_vs_music():
    """Run nested CV for each feature set in FEATURE_SETS, on the same outer folds as
    run_nested_cv, to measure how much music habits add beyond age.

    Models: the mean baseline (once per target), Ridge with a spline for age, and the tuned
    random forest with raw age (trees need no spline).

    Saves the per-fold scores to results/age_vs_music_folds.csv and returns them as a
    DataFrame with columns fold, R2, RMSE, MAE, target, model, feature_set.
    """
    X, Y = load_clean_dataset()
    X_train, X_test, Y_train, Y_test = split_train_test(X, Y)
    del X_test, Y_test  # the test data is not used in nested cross-validation

    folds = []
    for target in TARGETS:
        scores = outer_cv_scores(make_models()["Mean baseline"], X_train, Y_train[target])
        scores["target"], scores["model"], scores["feature_set"] = target, "Mean baseline", "No features"
        folds.append(scores)

        for set_name, columns in FEATURE_SETS.items():
            models = {"Ridge": make_models(columns, age_spline=True)["Ridge"],
                      "RF tuned": make_models(columns)["RF tuned"]}
            for name, model in models.items():
                start = time.time()
                scores = outer_cv_scores(model, X_train[columns], Y_train[target])
                scores["target"], scores["model"], scores["feature_set"] = target, name, set_name
                folds.append(scores)
                print(f"{target:10s} {name:8s} {set_name:12s} mean R2 = {scores.R2.mean():.3f}"
                      f"   ({time.time() - start:.0f}s)")
    folds = pd.concat(folds, ignore_index=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    folds.to_csv(RESULTS_DIR / "age_vs_music_folds.csv", index=False)
    return folds

#---------------------------------------------------------------------------
# Permutation Importance
#---------------------------------------------------------------------------

# Held-out folds for permutation importance: 5-fold x 2 repeats = 10 folds on the training
# set. These are the first 10 of the 50 outer folds.
IMPORTANCE_CV = RepeatedKFold(n_splits=N_SPLITS, n_repeats=2, random_state=RANDOM_STATE)
N_PERMUTATIONS = 10  # times each variable is shuffled in each held-out fold

def run_permutation_importance():
    """Permutation importance of each survey variable for the tuned random forest.

    For each target, max_features and min_samples_leaf are chosen once by INNER_CV on the
    whole training set. Then, in each IMPORTANCE_CV fold, a forest with those settings is
    fitted on the training part, and each variable is shuffled N_PERMUTATIONS times in the
    held-out part. The importance is the mean drop in R2. A variable such as Fav genre is
    shuffled as one column, so its importance covers all of its encoded columns.

    Saves the per-fold importances to results/permutation_importance_folds.csv and returns
    them as a DataFrame with columns target, fold, variable, importance.
    """
    X, Y = load_clean_dataset()
    X_train, X_test, Y_train, Y_test = split_train_test(X, Y)
    del X_test, Y_test  # the test data is not used here

    rows = []
    for target in TARGETS:
        start = time.time()
        y = Y_train[target]
        search = make_models()["RF tuned"].set_params(refit=False, n_jobs=-1).fit(X_train, y)
        forest = clone(search.estimator).set_params(**search.best_params_)

        for fold, (train_idx, test_idx) in enumerate(IMPORTANCE_CV.split(X_train)):
            fitted = clone(forest).fit(X_train.iloc[train_idx], y.iloc[train_idx])
            result = permutation_importance(fitted, X_train.iloc[test_idx], y.iloc[test_idx],
                                            n_repeats=N_PERMUTATIONS, scoring=SCORING,
                                            random_state=RANDOM_STATE, n_jobs=-1)
            rows += [{"target": target, "fold": fold, "variable": variable, "importance": importance}
                     for variable, importance in zip(X_train.columns, result.importances_mean)]
        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        print(f"{target:10s} {params}   ({time.time() - start:.0f}s)")
    folds = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    folds.to_csv(RESULTS_DIR / "permutation_importance_folds.csv", index=False)
    return folds

#---------------------------------------------------------------------------
# Final Test Evaluation
#---------------------------------------------------------------------------

def run_test_evaluation():
    """Fit every model on the full training set and score it once on the held-out test set.

    Run this only after all model and hyperparameter decisions are final. Tuned models
    choose their hyperparameters with INNER_CV on the training set only.

    Saves the selected hyperparameters to results/final_selected_hyperparameters.csv and
    returns (Y_test, predictions, selected), where predictions[target][model] holds the
    test-set predictions and selected is a DataFrame of the chosen hyperparameters.
    """
    X, Y = load_clean_dataset()
    X_train, X_test, Y_train, Y_test = split_train_test(X, Y)

    predictions = {target: {} for target in TARGETS}
    selected = []
    for target in TARGETS:
        for name, model in make_models().items():
            model.fit(X_train, Y_train[target])
            predictions[target][name] = model.predict(X_test)
            if hasattr(model, "best_params_"):  # tuned models (GridSearchCV)
                params = {k.replace("model__", ""): v for k, v in model.best_params_.items()}
                selected.append({"Target": target, "Model": name, **params})
    selected = pd.DataFrame(selected)

    RESULTS_DIR.mkdir(exist_ok=True)
    selected.to_csv(RESULTS_DIR / "final_selected_hyperparameters.csv", index=False)
    return Y_test, predictions, selected

def _metrics(y, pred):
    """R2, RMSE and MAE along the last axis, so it works on one sample or many resamples."""
    resid = y - pred
    sse = (resid ** 2).sum(axis=-1)
    sst = ((y - y.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1)
    return {"R2": 1 - sse / sst,
            "RMSE": np.sqrt((resid ** 2).mean(axis=-1)),
            "MAE": np.abs(resid).mean(axis=-1)}

def bootstrap_test_metrics(y_true, predictions, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    """Test-set metrics with participant-level bootstrap 95% CIs.

    The test participants are resampled with replacement n_boot times; every model uses the
    same resamples.

    predictions: {model name: test-set predictions}. Returns one row per model and metric
    (R2, RMSE, MAE) with the estimate and 95% percentile CI.
    """
    y = np.asarray(y_true, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))  # one row of indices per resample
    point = {name: _metrics(y, np.asarray(pred)) for name, pred in predictions.items()}
    boot = {name: _metrics(y[idx], np.asarray(pred)[idx]) for name, pred in predictions.items()}

    rows = []
    for name in predictions:
        for metric in ["R2", "RMSE", "MAE"]:
            values = boot[name][metric]
            rows.append({"model": name, "metric": metric, "estimate": float(point[name][metric]),
                         "ci_low": np.percentile(values, 2.5), "ci_high": np.percentile(values, 97.5)})
    return pd.DataFrame(rows)
