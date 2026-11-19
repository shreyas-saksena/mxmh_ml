# mxmh_ml

Machine learning analysis of the Music & Mental Health (MxMH) survey data set. It asks how well
a person's age and music habits predict their self-reported Anxiety, Depression, Insomnia and
OCD scores (0–10).

## What the Repo Provides

The analysis compares models with repeated nested cross-validation on the training set. The
held-out test set is used once, at the end. Every estimate is reported with a 95% confidence
interval.

| Notebook (`src/`) | What it does | Results (`results/`) |
|---|---|---|
| `mxmh_data_cleaning.ipynb` | Drops unused columns and rows with missing values (736 → 725 rows) | `data/mxmh_survey_results_clean.csv` |
| `mxmh_nested_cv.ipynb` | Compares Mean baseline, Linear, Lasso, Ridge, RF default and RF tuned with nested CV (5-fold × 10 repeats outer, 5-fold inner) | `nested_cv_folds.csv`, `nested_cv_performance.csv` |
| `mxmh_age_vs_music.ipynb` | Measures how much music habits add to predictions beyond age (Ridge and RF tuned) | `age_vs_music_folds.csv`, `age_vs_music_by_set.csv`, `age_vs_music_increment.csv` |
| `mxmh_permutation_importance.ipynb` | Ranks the survey variables by their permutation importance in the tuned random forest | `permutation_importance_folds.csv`, `permutation_importance_summary.csv` |
| `mxmh_test_evaluation.ipynb` | Fits each model on the full training set and scores it once on the test set, with bootstrap CIs | `final_selected_hyperparameters.csv`, `test_set_results_table.csv` |

Most of the code is in `src/mxmh_common.py`: settings, feature encoding, models, cross-validation
and the corrected confidence intervals (Nadeau & Bengio, 2003). The notebooks are kept short.
They run an analysis and build its tables.

The results of every analysis are saved in `results/`, so you can read them without running
anything. All random steps use a fixed seed (42), so re-running gives identical results.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) to manage Python and its dependencies.

1. Install uv (macOS / Linux):

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repo and create the environment:

   ```sh
   git clone <repo-url> mxmh_ml
   cd mxmh_ml
   uv sync
   ```

   `uv sync` installs Python 3.14 if needed. It creates `.venv/` with the exact versions in
   `uv.lock`: pandas 3.1.1, NumPy 2.5.5, SciPy 1.18.1 and scikit-learn 1.9.1.

## Running the Notebooks

**VS Code:** open a notebook in `src/` and select the `.venv` Python kernel.

**Jupyter Lab:**

```sh
uv run --with jupyterlab jupyter lab src/
```

Run the notebooks in the order of the table above. You only need to re-run data cleaning if
`data/mxmh_survey_results.csv` changes.

Each analysis notebook starts with `RERUN = True`, which re-runs the analysis and overwrites its
saved results. Set `RERUN = False` to build the tables from the saved results in seconds.

## Data

`data/mxmh_survey_results.csv` is the MxMH survey: 736 responses about music listening habits
(favourite genre, listening frequency for 16 genres, hours per day, streaming service, and
more) along with self-reported mental health scores.
