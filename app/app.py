"""ML Capstone bonus app: regression + classification.

Tab 1 loads a lightweight diamond-price RandomForest (joblib).
Tab 2 loads the selected telescope Gradient Boosting classifier (joblib).
Run: streamlit run app/app.py (from project root).
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

CUTS = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLORS = ["D", "E", "F", "G", "H", "I", "J"]
CLARITIES = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

TELESCOPE_FEATURES = [
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
]
TELESCOPE_DEFAULTS = {
    "fLength": 37.13,
    "fWidth": 17.14,
    "fSize": 2.74,
    "fConc": 0.354,
    "fConc1": 0.196,
    "fAsym": 4.06,
    "fM3Long": 15.34,
    "fM3Trans": 0.75,
    "fAlpha": 17.53,
    "fDist": 191.83,
}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REGRESSION_PATH = MODELS_DIR / "diamond_rf_small.joblib"
CLASSIFICATION_PATH = MODELS_DIR / "telescope_gb.joblib"


@st.cache_resource
def load_regression_model():
    return joblib.load(REGRESSION_PATH)


@st.cache_resource
def load_classification_model():
    return joblib.load(CLASSIFICATION_PATH)


st.set_page_config(page_title="ML Capstone App")
st.title("ML Capstone — Bonus App")

tab_reg, tab_clf = st.tabs(
    ["Diamond Price (Regression)", "Telescope Signal (Classification)"]
)

with tab_reg:
    st.caption(
        "Compact RandomForest (50 trees, joblib) deploy variant of the "
        "notebook's RF-200 (test R² 0.976 → 0.975). "
        "Enter 9 attributes; `volume` and `length_width_ratio` are engineered automatically."
    )
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
            price = float(load_regression_model().predict(row)[0])
            st.success(f"Estimated price: **${price:,.2f}**")
            st.caption("Engineered: volume = x·y·z, length_width_ratio = x/y.")
    else:
        st.info("Adjust the inputs and click **Predict price**.")

with tab_clf:
    st.caption(
        "Selected notebook model: Gradient Boosting (500 trees, depth 4, lr 0.1) — "
        "test accuracy ≈0.883, weighted F1 ≈0.881. "
        "`length_width_ratio = fLength / fWidth` is engineered automatically."
    )
    vals = {}
    cols = st.columns(3)
    for i, feat in enumerate(TELESCOPE_FEATURES):
        with cols[i % 3]:
            vals[feat] = st.number_input(
                feat, value=float(TELESCOPE_DEFAULTS[feat]), format="%.4f"
            )
    if vals["fWidth"] == 0:
        st.error("fWidth must be non-zero (used for length_width_ratio).")
    elif st.button("Classify observation", type="primary", key="clf_btn"):
        vals["length_width_ratio"] = vals["fLength"] / vals["fWidth"]
        row = pd.DataFrame([vals])
        model = load_classification_model()
        pred = str(model.predict(row)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(row)[0]
            labels = list(model.classes_)
            pg = float(proba[labels.index("g")]) if "g" in labels else None
        label = "gamma-ray signal (g)" if pred == "g" else "hadron background (h)"
        if pred == "g":
            st.success(f"Prediction: **{label}**")
        else:
            st.warning(f"Prediction: **{label}**")
        if proba is not None and pg is not None:
            st.progress(pg, text=f"P(gamma) = {pg:.3f}")
            st.caption(f"P(hadron) = {1 - pg:.3f}. Threshold 0.5; ROC-AUC ≈ 0.933.")
    else:
        st.info("Adjust the 10 measurements and click **Classify observation**.")

with st.sidebar:
    st.header("About")
    st.write(
        "Regression: `models/diamond_rf_small.joblib` — median+scale / mode+one-hot + RF-50. "
        "Data: `data/diamond.csv`, 80/20 split, seed 42."
    )
    st.write(
        "Classification: `models/telescope_gb.joblib` — median impute + Gradient Boosting "
        "(500, depth 4, lr 0.1). Data: `data/telescope_data.csv`, stratified 80/20, seed 42."
    )
