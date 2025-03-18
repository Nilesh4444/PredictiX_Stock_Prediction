import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Simulated historical stock prices (replace this with actual data)
prices = np.array([100, 102, 105, 107, 110, 115, 120, 125, 130]).reshape(-1, 1)

# Initialize and fit the scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(prices)

# Save the trained scaler
joblib.dump(scaler, 'scaler.pkl')

print("Scaler saved as 'scaler.pkl'")
