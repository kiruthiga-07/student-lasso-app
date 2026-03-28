import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Student Lasso Predictor", layout="wide")
st.title("🎓 Student Performance Lasso Regression")

# 1. Load Dataset
@st.cache_data
def load_data():
    try:
        # Ensure 'student_data.csv' is in the same folder as this script
        df = pd.read_csv('student_data.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Error: 'student_data.csv' not found. Please upload it to your GitHub repo.")
        return None

df = load_data()

if df is not None:
    st.subheader("1. Data Preview (First 5 Rows)")
    st.write(df.head())

    # 2. Select Features and Target
    features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Internet_Usage']
    target = 'Final_Score'
    
    X = df[features]
    y = df[target]

    # 3. Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Build and Fit Lasso Model
    model = Lasso(alpha=0.5)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # 5. Evaluation Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    col1.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    col2.metric("R² Score", f"{r2:.2f}")

    # 6. Feature Importance (Coefficients)
    st.subheader("📊 Feature Importance")
    coeff_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
    # Sort by importance
    coeff_df = coeff_df.sort_values(by='Coefficient', ascending=False)
    
    st.bar_chart(coeff_df.set_index('Feature'))
    st.write("Variables with 0 coefficient were removed by the Lasso model.")
    st.table(coeff_df)

    # 7. Prediction Tool for Users
    st.subheader("🔮 Predict your Final Score")
    with st.form("prediction_form"):
        inputs = []
        for feat in features:
            val = st.number_input(f"Enter {feat}", value=float(df[feat].mean()))
            inputs.append(val)
        
        submit = st.form_submit_button("Predict")
        
        if submit:
            # Must scale the input just like the training data
            input_array = np.array(inputs).reshape(1, -1)
            scaled_input = scaler.transform(input_array)
            prediction = model.predict(scaled_input)
            st.success(f"Predicted Final Score: {prediction[0]:.2f}")
