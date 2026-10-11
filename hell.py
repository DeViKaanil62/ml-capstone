import pandas as pd

# NASA Exoplanet Archive download URL
url = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    "query=select+*+from+pscomppars&format=csv"
)

# Download the complete dataset
print("Downloading NASA exoplanet dataset...")
df = pd.read_csv(url)

print("Original dataset shape:", df.shape)

# 10 selected columns
columns = [
    "pl_name",
    "pl_rade",
    "pl_bmasse",
    "pl_orbper",
    "pl_orbsmax",
    "pl_orbeccen",
    "pl_eqt",
    "st_teff",
    "st_mass",
    "st_rad",
]

# Check whether all requested columns exist
missing = [col for col in columns if col not in df.columns]

if missing:
    raise ValueError(f"Columns not found in dataset: {missing}")

# Filter columns only; retain every row
filtered_df = df[columns]

# Save filtered dataset
filtered_df.to_csv("exoplanets_10_columns.csv", index=False)

print("Filtered dataset shape:", filtered_df.shape)
print("Columns:", filtered_df.columns.tolist())
print("Saved as exoplanets_10_columns.csv")
