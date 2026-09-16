# Leak Detection Classifier

A machine learning project for classifying water-pipeline leak types from sensor data using a hybrid Decision Tree + K-Nearest Neighbors (DTKNN) pipeline.

The project began as a three-person course project in Spring 2025. This repository contains a later refinement of the original implementation, with improved cross-validation methodology, leakage-safe preprocessing, reproducible dependencies, and clearer attribution of individual contributions.

## Problem

The task is to classify sensor observations into five categories:

- `in` — indoor leakage
- `out` — outdoor leakage
- `noise` — electrical or mechanical noise
- `other` — environmental noise
- `normal` — normal / no leak

The current pipeline uses `lrate` together with frequency-domain features sampled at 10 Hz intervals.

## Dataset

The project uses the AIHub **Leak Detection Dataset** (dataset ID 138):

https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=138

The original AIHub CSV files are not included in this repository because redistribution of the source data is restricted by AIHub's data-use policy.

After obtaining the dataset from AIHub, place the following files in the repository root:

```text
Project_2_Classification_training_data.csv
Project_2_Classification_testing_data.csv
```

The current model uses:

```text
lrate
0HZ, 10HZ, 20HZ, ..., 5110HZ
```

Additional metadata such as `site`, `sid`, and `ldate` is used only to construct cross-validation groups and is not used as a predictive feature.

## Approach

The final pipeline combines Decision Tree feature importance with KNN classification.

```text
Training data
    |
    |-- Group-aware cross-validation
    |
Decision Tree
    |
    |-- Rank features by importance
    |-- Select top-N features
    |
StandardScaler
    |
KNN
    |
Predicted leak type
```

`DecisionTreeClassifier` and `KNeighborsClassifier` are provided by scikit-learn. The hybrid architecture uses the Decision Tree as a feature selector and KNN as the final classifier.

### 1. Hyperparameter Selection

`src/model_selection.py` selects the standalone model hyperparameters using 5-fold `StratifiedGroupKFold`.

Rows sharing the same:

```text
(site, sid, ldate)
```

are assigned to the same fold so that samples from the same measurement group cannot appear in both training and validation portions of a fold.

For KNN, feature scaling is fitted only on each fold's training partition.

Current selected hyperparameters:

```text
KNN neighbors:        1
Decision Tree depth: 32
```

### 2. Feature Selection

`src/main.py` searches over different numbers of top-ranked Decision Tree features.

For every cross-validation fold:

1. Decision Tree feature importance is fitted only on the fold's training data.
2. The top-N features are selected.
3. `StandardScaler` is fitted only on the selected training features.
4. KNN is trained on the transformed training partition.
5. Performance is measured on the validation partition.

This keeps both feature selection and preprocessing inside the cross-validation boundary.

The best current configuration selects:

```text
83 features
```

with a mean group-aware validation accuracy of:

```text
0.9141
```

### 3. Final Evaluation

After model selection, feature selection is fitted on the full training set and the final KNN model is evaluated on the provided test split.

Current results:

| Metric | Score |
| --- | ---: |
| Accuracy | **0.9282** |
| Macro Precision | **0.9284** |
| Macro Recall | **0.9282** |

The confusion matrix is implemented manually using NumPy with the conventional orientation:

```text
rows    = actual class
columns = predicted class
```

Macro precision, macro recall, and accuracy are also calculated manually. The custom implementation was cross-checked against `sklearn.metrics`.

## Model Selection Results

The current group-aware cross-validation results are:

| Model | Selected Hyperparameter | Mean Validation Accuracy |
| --- | ---: | ---: |
| KNN | K = 1 | 0.8923 |
| Decision Tree | max_depth = 32 | 0.7195 |
| DTKNN | 83 selected features | **0.9141** |

In the current group-aware cross-validation setup, the hybrid pipeline achieves higher mean validation accuracy than either standalone model while using a reduced feature set for KNN classification.

## My Contribution

This was originally a team project.

My main responsibility was algorithm implementation and performance analysis. I:

- implemented the multiclass confusion matrix without using a built-in confusion-matrix function;
- implemented macro precision, macro recall, and accuracy calculations from the confusion matrix;
- evaluated multiple combinations of input features using KNN and Decision Tree classifiers;
- performed cross-validation-based hyperparameter tuning;
- contributed to identifying the initial feature set used by the classification pipeline;
- analyzed model performance and the effects of feature dimensionality.

### Team Contributions

**Yonghee Jang** proposed the hybrid DTKNN concept: using Decision Tree feature importance for feature selection followed by KNN classification. He also contributed to structuring the overall presentation and project narrative.

**Jinwon Park** analyzed the frequency-domain waveforms and proposed refining the frequency range used during feature exploration. He also contributed to interpreting and presenting the final DTKNN results.

## Repository Refinement

After the original course project, I independently revisited and refined this repository for reproducibility and methodological correctness.

The current version includes:

- removal of feature-selection leakage from cross-validation;
- fitting feature scaling only on each fold's training partition;
- replacement of row-level cross-validation with `StratifiedGroupKFold`;
- grouping by `site + sid + ldate` to prevent measurement-group overlap across CV folds;
- separation of model-selection logic from final test evaluation;
- standardized confusion-matrix orientation;
- verification of the manually implemented metrics against scikit-learn;
- portable repository-relative dataset paths;
- reproducible dependency management with Pipenv;
- removal of dataset files and local coursework artifacts from version control.

These refinements were made after the original team project and are separate from the original team contributions described above.

## Repository Structure

```text
.
├── README.md
├── Pipfile
├── Pipfile.lock
├── src/
│   ├── main.py
│   └── model_selection.py
├── tests/
│   └── test_metrics.py
└── .gitignore
```

The AIHub CSV files are intentionally excluded from version control.

## Setup

The repository is configured for Python 3.13.

Install the locked dependencies with:

```bash
pipenv install --deploy
```

## Tests

The manually implemented confusion matrix and evaluation metrics are
cross-checked against `sklearn.metrics` using automated tests.

Run:

```bash
pipenv run python -m unittest discover -s tests -v
```

The test suite verifies that:

- the custom multiclass confusion matrix matches scikit-learn;
- macro recall matches scikit-learn;
- macro precision matches scikit-learn;
- accuracy matches scikit-learn.

## Run

First, reproduce the hyperparameter-selection experiment:

```bash
pipenv run python src/model_selection.py
```

Expected selections:

```text
Best K: 1
Best depth: 32
```

Then run the final DTKNN pipeline:

```bash
pipenv run python src/main.py
```

The script performs group-aware feature-count selection and then evaluates the final classifier on the provided test dataset.

## Notes and Limitations

This project was developed as a machine-learning course project rather than as a production leak-monitoring system.

The hyperparameter and feature-count searches cover predefined candidate ranges, so the selected values should not be interpreted as globally optimal. Decision Tree feature importance can also vary with the training data, and performance on this dataset does not by itself establish performance under different sensors, sites, or operating conditions.
