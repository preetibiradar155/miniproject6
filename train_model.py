import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Create sample house data
data = pd.DataFrame({
    "Area": [
        800, 1000, 1200, 1500, 1800,
        2000, 2200, 2500, 2800, 3000,
        1100, 1600, 2100, 2400, 2700,
        900, 1300, 1700, 2300, 2900
    ],
    "Bedrooms": [
        2, 2, 3, 3, 3, 4, 4, 4, 4, 5,
        2, 3, 4, 4, 5, 2, 3, 3, 4, 5
    ],
    "Age": [
        10, 8, 5, 7, 4, 3, 6, 2, 5, 1,
        12, 9, 4, 3, 2, 15, 6, 8, 5, 1
    ],
    "Price": [
        25, 30, 40, 50, 60,
        70, 72, 85, 90, 100,
        28, 48, 75, 82, 95,
        23, 42, 55, 78, 98
    ]
})

# Price is expressed in lakhs of rupees.

# 2. Select input and target
X = data[["Area", "Bedrooms", "Age"]].values
y = data["Price"].values

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Build the neural network
model = Sequential([
    Input(shape=(3,)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1, activation="linear")
])

# 6. Compile the model
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)

# 7. Train the model
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=25,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=4,
    callbacks=[early_stopping],
    verbose=1
)

# 8. Evaluate the model
predictions = model.predict(X_test, verbose=0).flatten()

print("\nModel training completed!")
print("Mean Absolute Error:",
      round(mean_absolute_error(y_test, predictions), 2), "lakhs")
print("R2 Score:", round(r2_score(y_test, predictions), 2))

# 9. Save model and scaler
model.save("house_price_model.keras")

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("\nSaved: house_price_model.keras")
print("Saved: scaler.pkl")

# 10. Test a sample house
sample = np.array([[1500, 3, 5]])
sample_scaled = scaler.transform(sample)
price = model.predict(sample_scaled, verbose=0)[0][0]

print("Sample predicted price:", round(float(price), 2), "lakhs")