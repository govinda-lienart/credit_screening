import pandas as pd
import numpy as np
import random

# Recreate the synthetic dataset
np.random.seed(42)
n_samples = 1000
data = {
    "A1": [random.choice(["a", "b", "c", "d"]) for _ in range(n_samples)],
    "A2": np.random.randint(18, 70, n_samples),
    "A3": np.random.uniform(0, 10, n_samples),
    "A4": [random.choice(["u", "y"]) for _ in range(n_samples)],
    "A5": [random.choice(["g", "p"]) for _ in range(n_samples)],
    "A6": [random.choice(["cc", "no_cc"]) for _ in range(n_samples)],
    "A7": [random.choice(["v", "t"]) for _ in range(n_samples)],
    "A8": np.random.uniform(0, 5, n_samples),
    "A9": [random.choice(["t", "f"]) for _ in range(n_samples)],
    "A10": np.random.randint(300, 850, n_samples),
    "A11": [random.choice(["t", "f"]) for _ in range(n_samples)],
    "A12": [random.choice(["t", "f"]) for _ in range(n_samples)],
    "A14": np.random.randint(20000, 150000, n_samples),
}
data["Class"] = (
    (data["A10"] > 600).astype(int)
    + (data["A14"] > 80000).astype(int)
    - (data["A3"] > 7).astype(int)
    - (np.array(data["A9"]) == "t").astype(int)
) > 0
data["Class"] = data["Class"].astype(int)
df = pd.DataFrame(data)

# Introduce missing values
for _ in range(50):
    row_idx = np.random.randint(0, n_samples)
    col_idx = np.random.choice(df.columns)
    df.loc[row_idx, col_idx] = np.nan if random.random() > 0.5 else '?'

# Save locally
df.to_csv("synthetic_dataset_with_missing_values.csv", index=False)
print("File saved as 'synthetic_dataset_with_missing_values.csv'")