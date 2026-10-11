# 23CSE301 Machine Learning Capstone Project

This project implements end-to-end machine learning pipelines for three tracks:

* **Regression:** Estimate a diamond's price from its physical measurements and quality characteristics.
* **Classification:** Classify telescope observations as gamma-ray signal (`g`) or hadron background (`h`).
* **Clustering:** Unsupervised taxonomic clustering of confirmed exoplanets into physical classes (terrestrial / sub-Neptune / gas giant).

The implemented work is contained in three Jupyter notebooks (one per track). A standalone GUI is not currently implemented in this repository.

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
│   ├── telescope_data.csv
│   └── exoplanets.csv
├── models/
│   ├── regression.ipynb
│   ├── classification.ipynb
│   └── clustering.ipynb
├── app/              # Reserved for future application code (bonus GUI)
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

### Exoplanet Clustering

The local dataset is `data/exoplanets.csv`. It contains 6,445 confirmed exoplanets with 10 columns:

* **Identifier:** `pl_name` (used as index, not clustered).
* **Planet features:** `pl_rade` (radius), `pl_bmasse` (mass), `pl_orbper` (orbital period), `pl_orbsmax` (semi-major axis), `pl_orbeccen` (eccentricity), `pl_eqt` (equilibrium temperature).
* **Stellar features:** `st_teff`, `st_mass`, `st_rad`.
* **Missing values:** highest in `pl_orbeccen` (~16.5%), `pl_eqt` (~8.3%), `pl_orbsmax` (~7.6%); `pl_rade`/`pl_bmasse`/`st_mass` under 1%.
* **Preprocessing:** median imputation (in original units), `log10` for skewed positives (`pl_rade`, `pl_bmasse`, `pl_orbper`, `pl_orbsmax`, `pl_eqt`, `st_mass`, `st_rad`), linear for `pl_orbeccen` and `st_teff`, then `StandardScaler` on all 9 numeric features.
* **Split:** none — unsupervised clustering uses the full scaled matrix (`random_state=42` everywhere). Ground-truth labels do not exist and are never used during fitting.
* **Models:** K-Means (k = 2..10, Elbow + Silhouette; k=3 primary, k=4 alternative) and Agglomerative Hierarchical (Ward dendrogram truncated to top-30 merges; Ward / Complete / Average compared at k=3).

## Setup

Use Python 3. Install the dependencies from the project root:

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
python -m pip install -r requirements.txt
```

The required packages are NumPy, pandas, scikit-learn, Matplotlib, Seaborn, and SciPy (`jinja2` optional — enables styled DataFrame output, otherwise the notebook falls back to plain tables).

## Run The Notebooks

From the project root, start Jupyter or open the notebooks in VS Code:

```bash
python -m pip install jupyter
jupyter notebook
```

Run the notebooks top-to-bottom in this order:

1. `models/regression.ipynb`
2. `models/classification.ipynb`
3. `models/clustering.ipynb`

All notebooks use relative paths to load files from `data/`. Run all cells again when using a fresh kernel so that the displayed results match the current environment.

## Evaluation

### Regression

Regression results are compared using $R^2$, root mean squared error (RMSE), and mean absolute error (MAE). The final comparison is ranked by test-set $R^2$.

### Classification

Classification results are compared using accuracy, weighted precision, weighted recall, and weighted F1 score. Each model also produces a classification report and confusion matrix for the `g` and `h` classes.

### Clustering

Clustering is evaluated with Silhouette Score (higher is better), Davies-Bouldin Index (lower is better), and Calinski-Harabasz Index (higher is better), plus Elbow curve, Ward dendrogram, PCA 2D plots (all algorithms) and t-SNE 2D plots (K-Means k=3, Ward k=3).

Current results on the full 9-feature scaled matrix (`random_state=42`):

| Algorithm / Configuration | k | Silhouette (↑) | Davies-Bouldin (↓) | Calinski-Harabasz (↑) | Cluster sizes |
|---|---:|---:|---:|---:|---|
| K-Means (k=3) | 3 | 0.3324 | 1.3782 | 2042.51 | [4126, 1410, 909] |
| K-Means (k=4) | 4 | 0.3235 | 1.2587 | 1842.62 | [3502, 776, 892, 1275] |
| Agglomerative (Ward, k=3) | 3 | 0.3206 | 1.5286 | 1714.42 | [1279, 925, 4241] |
| Agglomerative (Complete, k=3) | 3 | 0.7715 | 0.4026 | 283.68 | [6430, 13, 2] |
| Agglomerative (Average, k=3) | 3 | 0.7715 | 0.4026 | 283.68 | [13, 2, 6430] |

Notes:

* **Selected model: K-Means (k=3)** — best balanced partition (≈64% terrestrials / sub-Neptunes, ≈22% hot Jupiters, ≈14% cold / wide-orbit giants). Complete/Average score higher on Silhouette/DB only because they collapse to one giant cluster + 2 outliers, so they are not selected.
* Visualisations in `models/clustering.ipynb`: distribution histograms (3×3) + Spearman heatmap, K-Means Elbow + Silhouette curves, Ward dendrogram with k=3 cut, PCA scatter per algorithm + scree bar, t-SNE scatter for K-Means-k3 and Ward-k3, and median-physical-property cluster profiles over all 9 features.
