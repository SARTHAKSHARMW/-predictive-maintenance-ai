import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# LOAD DATA
data = pd.read_csv("predictive_maintenance.csv")

# CLEAN DATA
data = data.drop(["UDI", "Product ID"], axis=1)
data = pd.get_dummies(data, columns=["Type"])
data = data.drop(["TWF", "HDF", "PWF", "OSF", "RNF"], axis=1)

# CREATE RUL
data["RUL"] = 250 - data["Tool wear [min]"]

# FEATURES
X = data.drop(["Machine failure", "RUL"], axis=1)
y_class = data["Machine failure"]
y_reg = data["RUL"]

# SCALE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MODELS
clf = RandomForestClassifier(n_estimators=100, random_state=42)
reg = RandomForestRegressor(n_estimators=100, random_state=42)

clf.fit(X_scaled, y_class)
reg.fit(X_scaled, y_reg)

# SAVE FILES
pickle.dump(clf, open("classifier.pkl", "wb"))
pickle.dump(reg, open("regressor.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Models saved!")