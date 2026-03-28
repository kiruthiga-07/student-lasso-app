import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Title
# -------------------------------
st.title("📊 Student Performance Prediction using Lasso Regression")

st.write("This app predicts final exam scores and identifies important factors affecting performance.")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("student_data.xlsx")
    return data

data = load_data()

# -------------------------------
# Show Data
# -------------------------------
st.subheader("🔍 Dataset Preview")
st.dataframe(data.head())

# -------------------------------
# Feature Selection
# -------------------------------
features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Internet_Usage']
target = 'Final_Score'

X = data[features]
y = data[target]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Scaling
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Lasso Model
# -------------------------------
model = Lasso(alpha=0.5)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# -------------------------------
# Evaluation Metrics
# -------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Evaluation")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# -------------------------------
# Feature Importance
# -------------------------------
coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

st.subheader("📊 Feature Importance (Lasso Coefficients)")
st.dataframe(coefficients)

# -------------------------------
# Prediction Section
# -------------------------------
st.subheader("🎯 Predict Student Score")

hours = st.slider("Hours Studied", 0, 12, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
sleep = st.slider("Sleep Hours", 0, 12, 7)
previous = st.slider("Previous Scores", 0, 100, 60)
internet = st.slider("Internet Usage (hrs/day)", 0, 12, 3)

input_data = np.array([[hours, attendance, sleep, previous, internet]])
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)

st.success(f"📌 Predicted Final Score: {prediction[0]:.2f}")
