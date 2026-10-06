# Student Academic Outcome Prediction

This project explores predicting a student's final academic outcome (`Dropout`, `Enrolled`, or `Graduate`) from information intended to be available at enrollment. It is prepared for the Gupio Campus Placement AI/ML Development Assignment, Option 1.

## Dataset

- The notebook uses the assignment-provided CSV at `data/data.csv`, with semicolon-separated fields.
- **Dataset title:** Predict Students' Dropout and Academic Success (UCI dataset ID 697).
- **Repository/source:** [UCI Machine Learning Repository — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).
- **Official UCI citation:** Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89.
- `data/data.csv` is the assignment-provided copy of the dataset; the project reads this local file and does not download or substitute the data.
- The dataset itself is excluded from Git by `.gitignore`; place the supplied file at `data/data.csv` before running the notebook.
- No other dataset is downloaded or used.

## Problem statement

Train a three-class classifier for the `Target` column, using only enrollment-time predictors. The supplied data contains 4,424 records and 37 columns, including the target.

## Objective

Provide an auditable multiclass classification workflow that avoids post-enrollment information, compares a linear baseline with a tree ensemble, and evaluates class-level performance rather than relying on accuracy alone.

## Project structure

```text
student-academic-outcome-prediction/
|-- data/
|   `-- data.csv
|-- models/
|   `-- student_academic_outcome_model.pkl
|-- notebooks/
|   `-- student_outcome_prediction.ipynb
|-- app.py
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Technologies

Python, Jupyter Notebook, Pandas, NumPy, Matplotlib, Scikit-learn, and Streamlit.

## Dataset inspection and data quality

Observed from the supplied file:

- Shape: 4,424 rows × 37 columns.
- Dtypes: 29 integer, 7 floating-point, and 1 string target column.
- Missing values: none in the inspected columns.
- Duplicate rows: 0.
- Target counts: Graduate 2,209 (49.93%), Dropout 1,421 (32.12%), Enrolled 794 (17.95%).

Many integer columns represent coded categories; their integer dtype does not imply a numeric distance or ranking.

## Leakage prevention

The primary model excludes exactly these 12 post-enrollment semester-performance variables:

- `Curricular units 1st sem (credited)`
- `Curricular units 1st sem (enrolled)`
- `Curricular units 1st sem (evaluations)`
- `Curricular units 1st sem (approved)`
- `Curricular units 1st sem (grade)`
- `Curricular units 1st sem (without evaluations)`
- `Curricular units 2nd sem (credited)`
- `Curricular units 2nd sem (enrolled)`
- `Curricular units 2nd sem (evaluations)`
- `Curricular units 2nd sem (approved)`
- `Curricular units 2nd sem (grade)`
- `Curricular units 2nd sem (without evaluations)`

These features are measured after enrollment and are unavailable at the intended prediction time. All other non-target columns are retained.

## EDA

The notebook visualizes target counts and compares outcome shares with age at enrollment and tuition-fee status. In this dataset, the tuition-current group has a larger observed Graduate share; the group with fees not up to date has a larger observed Dropout share. These are descriptive associations, not causal findings.

## Preprocessing and split strategy

- Stratified 80/20 train/test split with `random_state=42`.
- Model selection uses five-fold `StratifiedKFold` cross-validation on training data only, scored by macro F1.
- Numerical predictors are median-imputed and standardized.
- Encoded categorical predictors are most-frequent-imputed and one-hot encoded with unknown categories ignored.
- Preprocessing is inside each model's Scikit-learn `Pipeline`, so transformers are fitted only on the training partition/fold.
- Class weighting is enabled to account for the uneven target frequencies.

## Models and evaluation

Models compared: balanced Logistic Regression (linear baseline) and balanced Random Forest (tree ensemble; 300 trees). The final model is selected by training-only cross-validation macro F1; the held-out test set is reported once and does not drive selection.

### Training-only cross-validation

| Model | Mean macro F1 | Std. dev. |
|---|---:|---:|
| Random Forest | 0.5707 | 0.0138 |
| Logistic Regression | 0.5671 | 0.0202 |

The mean CV macro-F1 difference is small; Random Forest is selected because it has the higher observed mean.

### Held-out test metrics

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5684 | 0.5578 | 0.5569 | 0.5448 | 0.5851 |
| Random Forest | 0.6260 | 0.5586 | 0.5520 | 0.5538 | 0.6188 |

Per-class test precision / recall / F1:

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Dropout | 0.6805 | 0.5775 | 0.6248 | 284 |
| Logistic Regression | Enrolled | 0.2956 | 0.5094 | 0.3741 | 159 |
| Logistic Regression | Graduate | 0.6973 | 0.5837 | 0.6355 | 442 |
| Random Forest | Dropout | 0.6393 | 0.6303 | 0.6348 | 284 |
| Random Forest | Enrolled | 0.3411 | 0.2767 | 0.3056 | 159 |
| Random Forest | Graduate | 0.6954 | 0.7489 | 0.7211 | 442 |

Confusion matrices (true classes in rows; predictions in columns; order: Dropout, Enrolled, Graduate):

**Logistic Regression**

|  | Predicted Dropout | Predicted Enrolled | Predicted Graduate |
|---|---:|---:|---:|
| True Dropout | 164 | 65 | 55 |
| True Enrolled | 21 | 81 | 57 |
| True Graduate | 56 | 128 | 258 |

**Random Forest**

|  | Predicted Dropout | Predicted Enrolled | Predicted Graduate |
|---|---:|---:|---:|
| True Dropout | 179 | 36 | 69 |
| True Enrolled | 39 | 44 | 76 |
| True Graduate | 62 | 49 | 331 |

Random Forest has the higher held-out accuracy and macro F1, with better Dropout and Graduate recall. Logistic Regression has slightly higher macro recall and substantially better Enrolled recall. Both models struggle most with Enrolled; Random Forest's Enrolled recall is particularly low. This class-specific result is important despite its overall macro-F1 advantage.

The target distribution is uneven: Graduate is the largest class, and Enrolled is the smallest. Test supports are 442 Graduate, 284 Dropout, and 159 Enrolled. Class weighting is used during model fitting, but the observed Enrolled recall remains low for the selected Random Forest; this is a limitation despite its better overall macro F1.

## Final model

Random Forest is selected because it had the higher training-only cross-validation macro F1 (0.5707 vs. 0.5671). On the untouched test set, it also had higher macro F1 (0.5538 vs. 0.5448) and accuracy (0.6260 vs. 0.5684), and greater recall for Dropout and Graduate. It does not dominate every class: its Enrolled recall is 0.2767 versus 0.5094 for Logistic Regression. The selection therefore reflects the specified overall macro-F1 criterion, not accuracy alone.

The notebook saves the complete fitted preprocessing/model pipeline to `models/student_academic_outcome_model.pkl`, reloads it with joblib, and checks that the reloaded pipeline reproduces predictions on the same held-out sample records.

## Feature importance

The selected Random Forest's top encoded predictors in this run were Admission grade, Age at enrollment, Previous qualification (grade), Tuition fees up to date, GDP, Unemployment rate, and Inflation rate. The notebook prints and plots the complete top-15 list. Feature importance indicates predictive association and does not prove causation.

## Sample prediction

One actual held-out student record (row index 1853) had these selected enrollment-time inputs and prediction:

| Input feature | Value |
|---|---:|
| Age at enrollment | 20 |
| Previous qualification (grade) | 160.0 |
| Admission grade | 160.0 |
| Tuition fees up to date | 1 |
| Gender | 1 |
| Application mode | 44 |
| Course | 9003 |

**Predicted class:** Graduate. **Observed target for this held-out row:** Graduate.

The notebook generates five held-out examples through the same fitted pipeline. Actual target and model prediction from the saved notebook output:

| Test row index | Actual Target | Predicted Target |
|---:|---|---|
| 1853 | Graduate | Graduate |
| 2399 | Graduate | Graduate |
| 510 | Enrolled | Graduate |
| 242 | Graduate | Graduate |
| 3392 | Graduate | Graduate |

## Conclusion and limitations

The selected enrollment-time model provides a modest overall macro-F1 on this one held-out split and does not reliably identify the Enrolled class. Results are specific to this supplied dataset and split; external validation, probability calibration, decision-threshold review, and further training-only model development would be needed before any operational use. The observational data does not establish causation, and the underlying dataset provenance/license must be verified before redistribution.

## How to run

1. Use Python and install the versions verified in the current notebook environment: `python -m pip install -r requirements.txt`.
2. Place the assignment-provided, semicolon-delimited file at `data/data.csv`.
3. Open `notebooks/student_outcome_prediction.ipynb` in Jupyter or VS Code.
4. Run all cells from top to bottom. The notebook loads the supplied file, checks leakage, reproduces preprocessing/training/evaluation, generates predictions, then saves and verifies the fitted pipeline.
5. Start the interview-demo dashboard with `streamlit run app.py`. It loads the existing saved pipeline and executed notebook metrics; it does not retrain the model.
