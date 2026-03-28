import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("📊 Student Performance Prediction using Lasso Regression")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("student_data.csv")
    except:
        data = pd.read_excel("student_data.xlsx")

    # Clean columns
    data.columns = (
        data.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.lower()
    )

    return data

data = load_data()

st.write("📌 Columns:", data.columns.tolist())
st.dataframe(data.head())

# -------------------------------
# FIX TARGET COLUMN (IMPORTANT)
# -------------------------------
# Try converting to numeric
data['final_result'] = pd.to_numeric(data['final_result'], errors='coerce')

# Drop rows where conversion failed
data = data.dropna(subset=['final_result'])

# -------------------------------
# Features
# -------------------------------
features = [
    'study_hours_per_day',
    'attendance_percentage',
    'previous_gpa',
    'midterm_marks',
    'assignment_score'
]

target = 'final_result'

# -------------------------------
# Split
# -------------------------------
X = data[features]
y = data[target]

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
# Model
# -------------------------------
model = Lasso(alpha=0.5)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# -------------------------------
# Evaluation
# -------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Evaluation")
st.write(f"MSE: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# -------------------------------
# Feature Importance
# -------------------------------
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

st.subheader("📊 Feature Importance")
st.dataframe(coef_df)

# -------------------------------
# Prediction
# -------------------------------
st.subheader("🎯 Predict Score")

study = st.slider("Study Hours", 0.0, 12.0, 5.0)
attendance = st.slider("Attendance %", 0.0, 100.0, 75.0)
gpa = st.slider("Previous GPA", 0.0, 10.0, 6.0)
midterm = st.slider("Midterm Marks", 0.0, 100.0, 50.0)
assignment = st.slider("Assignment Score", 0.0, 100.0, 60.0)

input_data = np.array([[study, attendance, gpa, midterm, assignment]])
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)

st.success(f"📌 Predicted Final Score: {prediction[0]:.2f}")
