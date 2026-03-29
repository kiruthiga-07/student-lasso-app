import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("🎓 Student Performance Lasso Regression")

# 1. Load Dataset
@st.cache_data
def load_data():
    try:
        # Load the new CSV file
        df = pd.read_csv('student_exam_scores.csv')
        # Clean column names (remove spaces and lowercase)
        df.columns = df.columns.str.strip().str.lower()
        return df
    except FileNotFoundError:
        st.error("❌ Error: 'student_exam_scores.csv' not found in your GitHub repo.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    st.subheader("1. Data Preview (First 5 Rows)")
    st.write(df.head())

    # 2. Select Features and Target based on YOUR CSV file
    # Note: 'internet_usage' is not in your current CSV, so we use the available columns
    features = ['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']
    target = 'exam_score'
    
    X = df[features]
    y = df[target]

    # 3. Split and Scale (80:20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Build and Fit Lasso Model (alpha = 0.5)
    model = Lasso(alpha=0.5)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # 5. Evaluate the Model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("2. Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    col2.metric("R² Score", f"{r2:.2f}")

    # 6. Feature Coefficients (Identifying Important Factors)
    st.subheader("3. Important Factors (Lasso Coefficients)")
    coeff_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
    coeff_df = coeff_df.sort_values(by='Coefficient', ascending=False)
    
    st.bar_chart(coeff_df.set_index('Feature'))
    st.write("Lasso automatically reduces irrelevant features to 0.00.")
    st.table(coeff_df)

    # 7. Prediction Tool
    st.divider()
    st.subheader("🔮 Predict Student Exam Score")
    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)
        user_inputs = []
        
        for i, feat in enumerate(features):
            with col_a if i % 2 == 0 else col_b:
                val = st.number_input(f"Enter {feat.replace('_', ' ').title()}", value=float(df[feat].mean()))
                user_inputs.append(val)
        
        submit = st.form_submit_button("Predict Score")
        
        if submit:
            # Prepare and scale the input
            input_array = np.array(user_inputs).reshape(1, -1)
            scaled_input = scaler.transform(input_array)
            prediction = model.predict(scaled_input)
            st.success(f"### Predicted Final Exam Score: {prediction[0]:.2f}")
