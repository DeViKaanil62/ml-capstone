# 23CSE301 Machine Learning Capstone Project

## Project Description & Problem Statement
[Insert your specific problem statement here. Explain the goal of predicting, classifying, or clustering the assigned dataset.][cite: 1]

This repository contains an end-to-end Machine Learning pipeline encompassing three core tracks:
* **Regression:** Evaluates 10 algorithms to predict a diamond's price in US dollars from its physical measurements and quality characteristics.
* **Classification:** Evaluates 10 algorithms for categorical prediction.
* **Clustering:** Evaluates K-Means and Hierarchical clustering for unsupervised pattern discovery.

## Team Members

| 👤 Name | 🎓 Roll Number |
| :--- | :---: |
| DEVIKA ANIL KUMAR | CB.SC.U4CSE24215 |
| H DHARSHAN | CB.SC.U4CSE24223 |
| NAVEEN SS | CB.SC.U4CSE24264 |

## Dataset Description

### Regression Dataset: Diamond Price Prediction

The regression track uses the [Diamond Price Prediction Dataset](https://www.kaggle.com/datasets/ronil8/diamond-price-prediction-dataset) from Kaggle. The local dataset is stored at `data/diamond.csv` and contains 53,940 diamond records with no missing values.

* **Input features:**
  * **Carat:** Weight of the diamond.
  * **Cut:** Quality of the cut (`Fair`, `Good`, `Very Good`, `Premium`, or `Ideal`).
  * **Color:** Diamond color grade, ranging from `D` to `J`.
  * **Clarity:** Diamond clarity grade.
  * **Depth:** Total depth percentage.
  * **Table:** Width of the diamond's top facet relative to its widest point.
  * **X, Y, and Z:** Length, width, and depth dimensions, respectively.
* **Target variable:** **Price (in US dollars)**.
* **Problem statement:** Build and compare regression models that estimate a diamond's price using its carat, quality grades, proportions, and physical dimensions.


## Environment Setup & Requirements
To run this project, ensure you have Python 3 installed along with the following primary libraries:
* scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn

Install all dependencies using the provided requirements file:
`pip install -r requirements.txt`

## How to Run the Project
1. Clone this repository to your local machine.
2. Ensure the raw dataset files are placed in the `data/` directory (or run the provided download script).
3. Navigate to the `notebooks/` directory.
4. Run the Jupyter notebooks top-to-bottom: `regression.ipynb`, `classification.ipynb`, and `clustering.ipynb`.
5. Optional: Run the GUI application located in the `app/` directory.

## Results Summary

### Regression Track

The models are evaluated on their ability to predict `Price(in US dollars)`. Results will be compared using $R^2$, RMSE, and MAE, then ranked by test-set $R^2$.

| Model | $R^2$ Score | RMSE | MAE |
|---|---|---|---|
| [Best Model] | [Value] | [Value] | [Value] |

*(This table will summarize all 10 trained algorithms ranked by $R^2$ on the test split.)*

### Classification Track
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| [Best Model] | [Value] | [Value] | [Value] | [Value] | [Value] |
*(Table must consolidate all 10 algorithms from Part A and Part B)*[cite: 1]

### Clustering Track
| Model | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
|---|---|---|---|
| K-Means | [Value] | [Value] | [Value] |
| Hierarchical | [Value] | [Value] | [Value] |

