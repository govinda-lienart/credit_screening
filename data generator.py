# 1. Generate Synthetic Dataset
np.random.seed(42)
n_samples = 1000

data = {
    "A1": [random.choice(["a", "b", "c", "d"]) for _ in range(n_samples)],  # Categorical
    "A2": np.random.randint(18, 70, n_samples),  # Age (Numeric)
    "A3": np.random.uniform(0, 10, n_samples),  # Debt-to-income ratio (Numeric)
    "A4": [random.choice(["u", "y"]) for _ in range(n_samples)],  # Categorical
    "A5": [random.choice(["g", "p"]) for _ in range(n_samples)],  # Categorical
    "A6": [random.choice(["cc", "no_cc"]) for _ in range(n_samples)],  # Categorical
    "A7": [random.choice(["v", "t"]) for _ in range(n_samples)],  # Categorical
    "A8": np.random.uniform(0, 5, n_samples),  # Years of Employment (Numeric)
    "A9": [random.choice(["t", "f"]) for _ in range(n_samples)],  # Prior default (Categorical)
    "A10": np.random.randint(300, 850, n_samples),  # Credit Score (Numeric)
    "A11": [random.choice(["t", "f"]) for _ in range(n_samples)],  # Categorical
    "A12": [random.choice(["t", "f"]) for _ in range(n_samples)],  # Categorical
    "A14": np.random.randint(20000, 150000, n_samples),  # Income (Numeric)
}

# Strong relationship for synthetic binary target (Approval based on clear rules)
data["Class"] = (
    (data["A10"] > 600).astype(int)  # High credit score positively influences approval
    + (data["A14"] > 80000).astype(int)  # High income positively influences approval
    - (data["A3"] > 7).astype(int)  # High debt-to-income ratio negatively influences approval
    - (np.array(data["A9"]) == "t").astype(int)  # Prior default negatively influences approval
) > 0  # Decision threshold

data["Class"] = data["Class"].astype(int)  # Convert to binary