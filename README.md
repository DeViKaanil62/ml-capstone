# 23CSE301 Machine Learning Capstone Project

This project compares supervised machine learning models for two prediction problems:

* **Regression:** Estimate a diamond's price from its physical measurements and quality characteristics.
* **Classification:** Classify telescope observations as gamma-ray signal (`g`) or hadron background (`h`).

The implemented work is contained in two Jupyter notebooks. Clustering and a standalone GUI are not currently implemented in this repository.

## Team Members

| Name              |   Roll Number   |
| :---------------- | :--------------: |
| Devika Anil Kumar | CB.SC.U4CSE24215 |
| H Dharshan        | CB.SC.U4CSE24223 |
| Naveen S S       | CB.SC.U4CSE24264 |

## Project Structure

```text
.
├── data/
│   ├── diamond.csv
│   └── telescope_data.csv
├── models/
│   ├── regression.ipynb
│   └── classification.ipynb
├── app/              # Reserved for future application code
├── requirements.txt
└── README.md
```

## Datasets

### Diamond Price Regression

The local dataset is `data/diamond.csv`. It contains 53,940 diamond records with the following columns:

* **Features:** carat, cut, color, clarity, depth, table, x, y, and z.
* **Target:** price in US dollars.
* **Cleaning:** rows with zero or negative `x`, `y`, or `z` measurements are removed.
* **Preprocessing:** median imputation and standardisation for numeric columns; most-frequent imputation and one-hot encoding for categorical columns.
* **Split:** 80% training and 20% test data, using `random_state=42`.

The notebook compares ten regressors: Multiple Linear Regression, Ridge, Lasso, ElasticNet, Polynomial Regression, Decision Tree, Random Forest, Gradient Boosting, Support Vector Regression, and K-Nearest Neighbours Regression.

Source: [Diamond Price Prediction Dataset](https://www.kaggle.com/datasets/ronil8/diamond-price-prediction-dataset).

### Telescope Classification

The local dataset is `data/telescope_data.csv`. It contains 18,905 observations and ten numeric telescope measurements: `fLength`, `fWidth`, `fSize`, `fConc`, `fConc1`, `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, and `fDist`.

* **Target:** `class`, where `g` is gamma signal and `h` is hadron background.
* **Feature engineering:** adds `length_width_ratio` from `fLength / fWidth`.
* **Preprocessing:** median imputation and standardisation.
* **Split:** stratified 80/20 train/test split, using `random_state=42`.
* **Models:** Logistic Regression, K-Nearest Neighbours, Gaussian Naive Bayes, Decision Tree, and Support Vector Machine with an RBF kernel.
* **Metrics:** accuracy, weighted precision, weighted recall, weighted F1 score, classification reports, and confusion matrices.

## Setup

Use Python 3. Install the dependencies from the project root:

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
python -m pip install -r requirements.txt
```

The required packages are NumPy, pandas, scikit-learn, Matplotlib, and Seaborn.

## Run The Notebooks

From the project root, start Jupyter or open the notebooks in VS Code:

```bash
python -m pip install jupyter
jupyter notebook
```

Run the notebooks top-to-bottom in this order:

1. `models/regression.ipynb`
2. `models/classification.ipynb`

Both notebooks use relative paths to load files from `data/`. Run all cells again when using a fresh kernel so that the displayed results match the current environment.

## Evaluation

### Regression

Regression results are compared using $R^2$, root mean squared error (RMSE), and mean absolute error (MAE). The final comparison is ranked by test-set $R^2$.

### Classification

Classification results are compared using accuracy, weighted precision, weighted recall, and weighted F1 score. Each model also produces a classification report and confusion matrix for the `g` and `h` classes.
