# Aerospace Predictive Maintenance - Week 2
# Author: Anit
# Description: Predict if an aircraft engine will fail within a particular cycle.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import json

# 1. Load dataset
df = pd.read_csv("aircraft_PM.csv")

# 2. Generate Remaining Useful Life (RUL)
df["max_cycle"] = df.groupby("engine_no")["cycle"].transform("max")
df["RUL"] = df["max_cycle"] - df["cycle"]
df.drop("max_cycle", axis=1, inplace=True)

# 3. Create binary target (1 = engine fails within 30 cycles)
df["label"] = (df["RUL"] <= 30).astype(int)

# 4. Feature selection
features = df.select_dtypes(include=[np.number]).drop(columns=["RUL", "label"]).columns
X = df[features]
y = df["label"]

# 5. Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# 9. Save report summary
summary = {
    "Model": "RandomForestClassifier",
    "Accuracy": acc,
    "Features": len(features),
    "Sample Size": len(df),
    "Comment": "Week 2 - Baseline model with sensor-based RUL feature and binary classification"
}

with open("report_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("✅ Model training complete. Accuracy:", round(acc * 100, 2), "%")
