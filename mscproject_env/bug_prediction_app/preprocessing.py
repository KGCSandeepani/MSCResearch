import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.selected_features = [f"C{i}" for i in range(1, 22)] + [f"H{i}" for i in range(1, 20)]

    def preprocess(self, data: pd.DataFrame):
        # Check for missing columns
        missing = [col for col in self.selected_features if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {', '.join(missing)}")

        X = data[self.selected_features].copy()
        self.feature_names = X.columns
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, self.feature_names









