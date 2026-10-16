# 23CSE301 Machine Learning Capstone Project

An end-to-end Machine Learning pipeline implementing regression, classification, and unsupervised clustering across real-world datasets, complete with exploratory data analysis, feature engineering, rigorous hyperparameter optimization, cross-validation, and an interactive multi-track Streamlit web application.

---

## Academic Information

* **Course:** 23CSE301 Machine Learning (Academic Year 2026–27)
* **Program:** B.Tech. Computer Science and Engineering (III Year)
* **Project Scope:** Review 1 (Regression + Classification Part A) & Review 2 (Classification Part B + Clustering + Bonus Web App)
* **Team Members:**

| Name | Roll Number | Primary Track Focus |
| :--- | :---: | :--- |
| **Devika Anil Kumar** | `CB.SC.U4CSE24215` | Classification Track (Cherenkov Telescope Signal vs. Hadron Background) |
| **H Dharshan** | `CB.SC.U4CSE24223` | Regression Track (Diamond Price Prediction & Valuation Modeling) |
| **Naveen S S** | `CB.SC.U4CSE24264` | Clustering Track (NASA Confirmed Exoplanet Taxonomic Clustering) |

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Track 1: Diamond Price Prediction (Regression)](#track-1-diamond-price-prediction-regression)
   * [Dataset & Problem Statement](#dataset--problem-statement-regression)
   * [Preprocessing & Feature Engineering](#preprocessing--feature-engineering-regression)
   * [Model Benchmarking & Comparison Table](#model-benchmarking--comparison-table-regression)
   * [5-Fold Cross-Validation](#5-fold-cross-validation-regression)
   * [Hyperparameter Tuning & Visualisations](#hyperparameter-tuning--visualisations-regression)
4. [Track 2: MAGIC Telescope Signal Classification (Classification)](#track-2-magic-telescope-signal-classification-classification)
   * [Dataset & Problem Statement](#dataset--problem-statement-classification)
   * [Preprocessing & Feature Engineering](#preprocessing--feature-engineering-classification)
   * [Consolidated 10-Algorithm Comparison Table](#consolidated-10-algorithm-comparison-table-classification)
   * [Final Model Selection & Optimization](#final-model-selection--optimization-classification)
5. [Track 3: Confirmed Exoplanet Taxonomy (Clustering)](#track-3-confirmed-exoplanet-taxonomy-clustering)
   * [Dataset & Problem Statement](#dataset--problem-statement-clustering)
   * [Preprocessing & Logarithmic Transformation](#preprocessing--logarithmic-transformation-clustering)
   * [Clustering Algorithms & Optimization](#clustering-algorithms--optimization-clustering)
   * [Validation Benchmark & Manifold Projections](#validation-benchmark--manifold-projections-clustering)
   * [Astrophysical Interpretation](#astrophysical-interpretation)
6. [Interactive Web Application (Bonus GUI)](#interactive-web-application-bonus-gui)
7. [Environment Setup & Installation](#environment-setup--installation)
8. [How to Run](#how-to-run)
9. [Methodology & Best Practices](#methodology--best-practices)

---

## Project Overview

This capstone project delivers three complete, self-contained machine learning workflows adhering to the 23CSE301 Capstone Guidelines:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               23CSE301 ML Capstone Hub                                 │
├────────────────────────────┬────────────────────────────┬──────────────────────────────┤
│ 1. Regression Track        │ 2. Classification Track    │ 3. Clustering Track          │
│ Diamond Price Estimation   │ Telescope Gamma Signal     │ Exoplanet Demographics       │
│ • 53,940 diamonds          │ • 18,905 observations      │ • 6,445 confirmed planets    │
│ • 10 Regressors evaluated  │ • 10 Classifiers evaluated │ • Unsupervised partitioning  │
│ • Top: SVR (R² = 0.979)    │ • Top: GradBoost (F1=0.88) │ • K-Means (k=3) & Ward Tree  │
│ • Deploy: RF-50 (joblib)   │ • Deploy: GradBoost-500    │ • Deploy: Scaled KMeans-3    │
└────────────────────────────┴────────────────────────────┴──────────────────────────────┘
                                          │
                                          ▼
                ┌──────────────────────────────────────────────────┐
                │          Streamlit Interactive GUI (app/)        │
                │   Tab 1: Regression | Tab 2: Classification      │
                │              Tab 3: Clustering                   │
                └──────────────────────────────────────────────────┘
```

---

## Repository Structure

```text
.
├── 23CSE301_ML_26_27_Capstone_Guidelines.pdf  # Course guidelines and evaluation rubrics
├── README.md                                  # Comprehensive project documentation
├── requirements.txt                           # Core dependencies for notebooks
├── app/                                       # Interactive Streamlit application
│   ├── app.py                                 # Web interface for all 3 tracks
│   └── requirements.txt                       # Application dependencies
├── data/                                      # Datasets for all tracks
│   ├── diamond.csv                            # Diamond price dataset (53,940 rows)
│   ├── telescope_data.csv                     # MAGIC telescope dataset (18,905 rows)
│   └── exoplanets.csv                         # NASA confirmed exoplanets (6,445 rows)
└── models/                                    # Jupyter Notebooks & trained artifacts
    ├── regression.ipynb                       # 10 regression models & validation
    ├── classification.ipynb                   # 10 classification models & tuning
    ├── clustering.ipynb                       # Unsupervised clustering & manifolds
    ├── diamond_rf_small.joblib                # Compact deployable Random Forest
    ├── telescope_gb.joblib                    # Champion Gradient Boosting pipeline
    └── exoplanet_kmeans.joblib                # K-Means clustering pipeline bundle
```

---

## Track 1: Diamond Price Prediction (Regression)

### Dataset & Problem Statement (Regression)

* **Dataset:** `data/diamond.csv` (Source: [Kaggle Diamond Price Prediction](https://www.kaggle.com/datasets/ronil8/diamond-price-prediction-dataset))
* **Size:** 53,940 records, 10 attributes (carat, cut, color, clarity, depth, table, price, x, y, z).
* **Goal:** Predict a diamond's continuous market price ($USD$) based on carat weight, cut quality, color grade, clarity rating, and physical 3D dimensions.

### Preprocessing & Feature Engineering (Regression)

1. **Cleaning:** Dropped duplicate entries and non-physical records with zero or negative dimensions ($x, y, z \le 0$).
2. **Feature Engineering:**
   * $\text{volume} = x \times y \times z$ (approximates the total diamond volume in $\text{mm}^3$, strongly correlating with mass and price).
   * $\text{length\_width\_ratio} = \frac{x}{y}$ (captures diamond cut symmetry and dimensional proportions).
3. **Pipeline (Train-Only Fit):** 
   * Numeric features: `SimpleImputer(strategy="median")` $\to$ `StandardScaler()`.
   * Categorical features (`cut`, `color`, `clarity`): `SimpleImputer(strategy="most_frequent")` $\to$ `OneHotEncoder(handle_unknown="ignore")`.
4. **Data Splitting:** 80% train ($N=43,136$), 20% test ($N=10,784$), `random_state=42`.

### Model Benchmarking & Comparison Table (Regression)

All 10 required regression algorithms were fitted on the identical preprocessed training matrix and evaluated against the held-out test split:

| Rank | Algorithm | Test $R^2$ Score | Test RMSE ($) | Test MAE ($) | Key Notes |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Support Vector Regressor (SVR)** | **0.9788** | **579.21** | 335.19 | Radial Basis Function kernel ($C=1000$, $\epsilon=0.1$); lowest overall RMSE. |
| **2** | **Random Forest Regression** | **0.9757** | 619.90 | **295.42** | 200 estimators; lowest absolute deviation (MAE) across all models. |
| **3** | **Gradient Boosting Regressor** | **0.9644** | 751.94 | 384.85 | Sequential ensemble with staged loss minimization. |
| **4** | **Decision Tree Regression** | **0.9630** | 766.69 | 361.61 | Single tree with cost-complexity pruning and depth constraints. |
| **5** | **K-Nearest Neighbors Regressor** | **0.9564** | 832.70 | 413.13 | Optimal $k=9$; sensitive to distance metrics and scaling. |
| **6** | **Multiple Linear Regression** | **0.9202** | 1125.16 | 735.22 | Baseline parametric model; interpretable weight coefficients. |
| **7** | **Polynomial Regression (Degree 2)** | **0.9202** | 1125.16 | 735.22 | Quadratic feature expansion; minimal gain over linear terms. |
| **8** | **Lasso Regression ($L_1$)** | **0.9184** | 1137.87 | 727.18 | Sparsity-inducing regularization; eliminates redundant features. |
| **9** | **Ridge Regression ($L_2$)** | **0.9165** | 1150.89 | 727.53 | Weight shrinkage regularizer addressing multicollinearity. |
| **10** | **ElasticNet Regression** | **0.9125** | 1178.09 | 735.10 | Convex combination of $L_1$ and $L_2$ penalties. |

### 5-Fold Cross-Validation (Regression)

As required by the capstone guidelines, the two leading models were validated using 5-fold cross-validation on the training set:

| Model | Mean CV $R^2$ | CV $R^2$ Std. Dev. | Fold $R^2$ Scores |
| :--- | :---: | :---: | :---: |
| **Support Vector Regressor (SVR)** | **0.9775** | $\pm 0.0012$ | `[0.9770, 0.9780, 0.9795, 0.9762, 0.9766]` |
| **Random Forest Regression** | **0.9720** | $\pm 0.0028$ | `[0.9701, 0.9718, 0.9771, 0.9690, 0.9719]` |

*Both models display high stability across folds ($\sigma \le 0.0028$), confirming that performance is not an artifact of train/test partition luck.*

### Hyperparameter Tuning & Visualisations (Regression)

* **Tuning:** Applied `GridSearchCV` on Decision Tree (`max_depth`, `min_samples_split`), Random Forest (`n_estimators`, `max_depth`), SVR ($C$, $\epsilon$), Gradient Boosting (`learning_rate`), and KNN ($k$).
* **Deployable Model Artifact:** While full SVR achieves the highest $R^2$, it has high inference latency on large sets. Random Forest (50 estimators, serialized to `models/diamond_rf_small.joblib`) retains $R^2 = 0.975$ with sub-millisecond inference and an MAE of $\approx \$305$, serving as the primary engine for the Streamlit app.
* **Visualizations in Notebook:** Residual vs. fitted plots, actual vs. predicted regression scatter, and tree-based Gini feature importance (identifying `carat` and `volume` as the dominant drivers of price).

---

## Track 2: MAGIC Telescope Signal Classification (Classification)

### Dataset & Problem Statement (Classification)

* **Dataset:** `data/telescope_data.csv` (Source: MAGIC Gamma Telescope Dataset, UCI ML Repository).
* **Size:** 18,905 observations, 10 continuous numeric features + binary target (`class`).
* **Goal:** Classify simulated Cherenkov shower images detected by ground-based atmospheric gamma telescopes into high-energy gamma-ray signals (`g`, primary signal) versus hadronic cosmic-ray background noise (`h`, hadron showers).

### Preprocessing & Feature Engineering (Classification)

1. **Feature Engineering:**
   * $\text{length\_width\_ratio} = \frac{\text{fLength}}{\text{fWidth}}$
   * *Astroparticle Justification:* Electromagnetic (gamma-ray) showers develop cleanly along the particle axis producing narrow, elongated elliptical shower images (high ratio), whereas hadronic showers experience larger transverse momentum kicks yielding wider, irregular patterns (low ratio).
2. **Preprocessing:** Train-only `SimpleImputer(strategy="median")` $\to$ `StandardScaler()`.
3. **Partition:** Stratified 80/20 train/test split (`random_state=42`) preserving the $\approx 65\% / 35\%$ class distribution.

### Consolidated 10-Algorithm Comparison Table (Classification)

Evaluating both Part A (Review 1) and Part B (Review 2) classifiers on the same stratified test split ($N=3,781$):

| Rank | Model | Part | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Gradient Boosting Classifier** | **Part B** | **0.8810** | **0.8815** | **0.8810** | **0.8784** | **0.9314** |
| **2** | **Random Forest Classifier** | **Part B** | 0.8770 | 0.8760 | 0.8770 | 0.8761 | **0.9327** |
| **3** | **Bagging Classifier (Decision Tree base)** | **Part B** | 0.8778 | 0.8786 | 0.8778 | 0.8749 | 0.9298 |
| **4** | **MLP Classifier (Neural Network)** | **Part B** | 0.8768 | 0.8776 | 0.8768 | 0.8738 | 0.9285 |
| **5** | **Support Vector Machine (SVC, RBF)** | **Part A** | 0.8717 | 0.8754 | 0.8717 | 0.8673 | 0.9209 |
| **6** | **Decision Tree Classifier** | **Part A** | 0.8474 | 0.8459 | 0.8474 | 0.8463 | 0.8895 |
| **7** | **AdaBoost Classifier** | **Part B** | 0.8426 | 0.8410 | 0.8426 | 0.8395 | 0.8966 |
| **8** | **K-Nearest Neighbours** | **Part A** | 0.8434 | 0.8495 | 0.8434 | 0.8357 | 0.8952 |
| **9** | **Logistic Regression** | **Part A** | 0.7932 | 0.7904 | 0.7932 | 0.7850 | 0.8392 |
| **10** | **Gaussian Naive Bayes** | **Part A** | 0.6683 | 0.7178 | 0.6683 | 0.6762 | 0.7534 |

*Key finding:* Part B ensemble and neural architectures raised average weighted F1 by $+0.067$ over Part A baselines.

### Final Model Selection & Optimization (Classification)

* **Selected Champion:** **Gradient Boosting Classifier**. Outperformed all other architectures across Accuracy and F1 while providing calibrated probability estimates.
* **Secondary Optimization Round (Review 2 Rubric A3):**
  * Tuned `learning_rate`, `max_depth`, and `n_estimators`.
  * Optimal hyperparameters: `{'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 500}`.
  * Performance gain:

| Metric | Before Extra Tuning | After Extra Tuning | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.8810 | **0.8828** | $+0.0019$ |
| **Weighted Precision** | 0.8815 | **0.8831** | $+0.0016$ |
| **Weighted Recall** | 0.8810 | **0.8828** | $+0.0019$ |
| **Weighted F1 Score** | 0.8784 | **0.8805** | $+0.0021$ |
| **ROC-AUC Score** | 0.9314 | **0.9333** | $+0.0019$ |

* **Class Breakdown on Final Tuned Model:**
  * Gamma signal (`g`): Precision $= 0.882$, Recall $= 0.947$, $F_1 = 0.913$ ($N=2,466$).
  * Hadron background (`h`): Precision $= 0.885$, Recall $= 0.762$, $F_1 = 0.819$ ($N=1,315$).
* **Serialized Model:** Exported to `models/telescope_gb.joblib` for live deployment.

---

## Track 3: Confirmed Exoplanet Taxonomy (Clustering)

### Dataset & Problem Statement (Clustering)

* **Dataset:** `data/exoplanets.csv` (Source: NASA Exoplanet Archive).
* **Size:** 6,445 confirmed exoplanets, 10 columns (`pl_name` as unique identifier + 9 physical/stellar attributes).
* **Features:** 
  * Planet: Radius (`pl_rade`), Mass (`pl_bmasse`), Orbital Period (`pl_orbper`), Semi-Major Axis (`pl_orbsmax`), Eccentricity (`pl_orbeccen`), Equilibrium Temp (`pl_eqt`).
  * Host Star: Effective Temp (`st_teff`), Mass (`st_mass`), Radius (`st_rad`).
* **Goal:** Perform purely unsupervised demographic clustering to discover physical planetary regimes without ground-truth labels.

### Preprocessing & Logarithmic Transformation (Clustering)

1. **Handling Skewness:** Highly skewed power-law distributions spanning multiple orders of magnitude were stabilized with $\log_{10}$ transformation:
   * $\log_{10}$ applied to: `pl_rade`, `pl_bmasse`, `pl_orbper`, `pl_orbsmax`, `pl_eqt`, `st_mass`, `st_rad`.
   * Linear pass retained for bounded features: `pl_orbeccen` ($[0, 1)$) and `st_teff` (normally distributed stellar temperatures).
2. **Standardization:** Median imputation followed by `StandardScaler` on all 9 transformed features (`random_state=42`).

### Clustering Algorithms & Optimization (Clustering)

1. **K-Means Optimization:**
   * Sweep from $k=2$ to $k=10$.
   * Analyzed both the **Elbow Method** (inertia curve inflection point) and **Silhouette Analysis**.
2. **Hierarchical Agglomerative Clustering:**
   * Constructed dendrograms comparing **Ward**, **Complete**, and **Average** linkage metrics.

### Validation Benchmark & Manifold Projections (Clustering)

Unsupervised validation metrics across evaluated configurations:

| Algorithm / Configuration | $k$ | Silhouette Score (↑) | Davies-Bouldin Index (↓) | Calinski-Harabasz Index (↑) | Cluster Membership Counts |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **K-Means (Selected)** | **3** | **0.3324** | **1.3782** | **2042.51** | **[4126, 1410, 909]** |
| K-Means | 4 | 0.3235 | 1.2587 | 1842.62 | [3502, 776, 892, 1275] |
| Agglomerative (Ward) | 3 | 0.3206 | 1.5286 | 1714.42 | [1279, 925, 4241] |
| Agglomerative (Complete) | 3 | 0.7715* | 0.4026* | 283.68 | [6430, 13, 2] *(Degenerate chaining)* |
| Agglomerative (Average) | 3 | 0.7715* | 0.4026* | 283.68 | [13, 2, 6430] *(Degenerate chaining)* |

*\*Note on Complete/Average linkage: While scoring deceptively high on Silhouette and Davies-Bouldin, they suffer from extreme outlier chaining ($99.8\%$ of data points collapsed into a single monolithic cluster with 2 outliers in separate clusters). Ward's linkage and K-Means yield physically meaningful, robust partitions.*

* **Manifold Dimensionality Reduction:**
  * **2D PCA:** Explains **61.73%** cumulative variance (PC1 = 39.59%, PC2 = 22.14%), cleanly separating planetary size and orbital distance.
  * **2D t-SNE:** Non-linear projection confirming dense cluster boundaries and neighborhood consistency.

### Astrophysical Interpretation

De-standardizing and inverse-transforming cluster centroids back to native astrophysical units reveals three distinct planetary archetypes:

```
                                  Exoplanet Demographics (K-Means k=3)
 ┌──────────────────────────────────────────────────┬───────────────────────┬────────────────────────┐
 │ Cluster 0: Terrestrial & Sub-Neptunes (64.0%)   │ Cluster 1: Hot        │ Cluster 2: Cold & Wide │
 │ 4,126 planets                                    │ Jupiters (21.9%)      │ Giants (14.1%)         │
 │ • Radius: 2.18 R⊕ (Earth radii)                  │ 1,410 planets         │ 909 planets            │
 │ • Mass: 5.60 M⊕ (Earth masses)                   │ • Radius: 12.67 R⊕    │ • Radius: 12.90 R⊕     │
 │ • Period: 9.80 days                              │ • Mass: 255.85 M⊕     │ • Mass: 881.81 M⊕      │
 │ • Equilibrium Temp: 749.5 K                      │ • Period: 4.29 days   │ • Period: 820.20 days  │
 │ • Low eccentricity (0.00)                        │ • Temp: 1264.0 K      │ • Eccentricity: 0.22   │
 └──────────────────────────────────────────────────┴───────────────────────┴────────────────────────┘
```

* **Serialized Bundle:** Saved to `models/exoplanet_kmeans.joblib` with preprocessing transform parameters for interactive inference.

---

## Interactive Web Application (Bonus GUI)

To satisfy the Review 2 Bonus requirements (+2 marks), the repository includes an interactive web application built with **Streamlit** (`app/app.py`):

```bash
streamlit run app/app.py
```

### Application Features

1. **Tab 1 — Diamond Price Predictor (Regression):**
   * Real-time sliders and selectors for carat, cut, color, clarity, depth, table, and 3D dimensions ($x, y, z$).
   * Automatic on-the-fly feature engineering ($\text{volume}$ and $\text{length\_width\_ratio}$).
   * Instant prediction powered by `diamond_rf_small.joblib`.
2. **Tab 2 — Cherenkov Telescope Signal Classifier (Classification):**
   * Numeric inputs for all 10 MAGIC Cherenkov shower parameters with preloaded realistic defaults.
   * Automatic calculation of $\text{length\_width\_ratio}$.
   * Predicts class (`gamma-ray signal (g)` vs `hadron background (h)`), shows calibrated probability meter $P(\text{gamma})$, and confidence indicators powered by `telescope_gb.joblib`.
3. **Tab 3 — Exoplanet Demographic Classifier (Clustering):**
   * Inputs for 9 planetary and stellar physical dimensions.
   * Replicates full notebook pipeline: median-fill $\to \log_{10}$ transformation $\to$ standardization $\to$ centroid distance computation.
   * Outputs assigned demographic cluster, physical archetype label, and Euclidean distances to all three cluster centroids.

---

## Environment Setup & Installation

### Prerequisites

* Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.14)
* `pip` package manager

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ml-capstone
```

### 2. Create and Activate a Virtual Environment

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install notebook and core training packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

To run the Streamlit GUI application, install the app dependencies:

```bash
pip install -r app/requirements.txt
```

---

## How to Run

### Running the Jupyter Notebooks

Launch Jupyter Notebook or JupyterLab from the project root:

```bash
jupyter notebook
```

Execute the notebooks in sequence from top to bottom:

1. **`models/regression.ipynb`:** Review 1 regression models, EDA, feature engineering, tuning, 5-fold CV, and residual diagnostics.
2. **`models/classification.ipynb`:** Review 1 Part A baselines + Review 2 Part B ensembles, 10-model benchmark table, secondary hyperparameter tuning, and ROC-AUC curves.
3. **`models/clustering.ipynb`:** Review 2 unsupervised exoplanet clustering, elbow/silhouette curves, Ward dendrogram, PCA/t-SNE projections, and astrophysical cluster profiling.

*All notebooks use relative file paths (`../data/` or `data/`) and will execute cleanly.*

### Launching the Streamlit Web Application

From the root directory:

```bash
streamlit run app/app.py
```

The browser will open automatically at `http://localhost:8501`.

---

## Methodology & Best Practices

* **Zero Data Leakage:** All feature encoders (`OneHotEncoder`) and scalers (`StandardScaler`) are strictly fitted on training splits only and subsequently applied (`transform`) to test splits. Unsupervised clustering operates without labels.
* **Deterministic Reproducibility:** Every stochastic process, cross-validation split, tree initialization, and t-SNE projection is pinned to `random_state=42`.
* **Metric Integrity:** Regressors are compared on the exact same held-out test split using $R^2$, RMSE, and MAE. Classifiers are compared on the exact same stratified test split using Accuracy, Weighted Precision, Weighted Recall, Weighted F1, and ROC-AUC.
* **Academic Integrity:** All data preprocessing strategies, feature engineering formulas, and astrophysical domain interpretations were independently developed according to the 23CSE301 Capstone specifications.

---
*Developed for 23CSE301 Machine Learning Capstone, Academic Year 2026–27.*
