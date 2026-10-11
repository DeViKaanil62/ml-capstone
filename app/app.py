"""Diamond price predictor (Streamlit bonus app).

Loads a lightweight RandomForest pipeline (joblib) trained with the same
preprocessing as models/regression.ipynb and predicts price in USD.
Run: streamlit run app/app.py (from project root).
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

CUTS = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLORS = ["D", "E", "F", "G", "H", "I", "J"]
CLARITIES = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "diamond_rf_small.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Diamond Price Predictor")
st.title("Diamond Price Predictor")
st.caption(
    "Compact RandomForest (50 trees, joblib) deploy variant of the "
    "notebook's RF-200 (test R² 0.976 → 0.975). "
    "Enter 9 attributes; `volume` and `length_width_ratio` are engineered automatically."
)

with st.sidebar:
    st.header("About")
    st.write(
        "Model: `models/diamond_rf_small.joblib` — "
        "ColumnTransformer (median+scale / mode+one-hot) + RandomForest."
    )
    st.write("Dataset: `data/diamond.csv` (53,940 rows). Split 80/20, seed 42.")

col1, col2, col3 = st.columns(3)
with col1:
    carat = st.number_input("Carat", 0.2, 5.0, 0.7, 0.01)
    cut = st.selectbox("Cut", CUTS, index=4)
    color = st.selectbox("Color", COLORS, index=1)
with col2:
    clarity = st.selectbox("Clarity", CLARITIES, index=2)
    depth = st.number_input("Depth %", 43.0, 79.0, 61.8, 0.1)
    table = st.number_input("Table %", 43.0, 95.0, 57.0, 0.1)
with col3:
    x = st.number_input("Length (x, mm)", 0.1, 11.0, 5.7, 0.01)
    y = st.number_input("Width (y, mm)", 0.1, 59.0, 5.71, 0.01)
    z = st.number_input("Depth (z, mm)", 0.1, 32.0, 3.53, 0.01)

if st.button("Predict price", type="primary"):
    if min(x, y, z) <= 0:
        st.error("x, y, z must be positive (same cleaning as the notebook).")
    else:
        row = pd.DataFrame(
            [
                {
                    "carat": carat,
                    "cut": cut,
                    "color": color,
                    "clarity": clarity,
                    "depth": depth,
                    "table": table,
                    "x": x,
                    "y": y,
                    "z": z,
                    "volume": x * y * z,
                    "length_width_ratio": x / y,
                }
            ]
        )
        price = float(load_model().predict(row)[0])
        st.success(f"Estimated price: **${price:,.2f}**")
        st.caption("Engineered: volume = x·y·z, length_width_ratio = x/y.")
else:
    st.info("Adjust the inputs and click **Predict price**.")
