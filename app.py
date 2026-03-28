import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Student Lasso Predictor", layout="wide")
st.title("🎓 Student Performance Lasso Regression")

@st.cache_data
def load_data():
    try:
        df = pd.read_excel('student_data.xlsx', engine='openpyxl')
        df.columns = df.columns.str.strip().str.lower() # Clean and lowercase for safety
        return df
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

df = load_data()

if df is not None:
    st.subheader("1. Data Preview")
    st.write(df.head())

    # 2. MATCHED FEATURES (Updated to match your actual file)
    # Using the columns found in your file:
    features = ['study_hours_per_day', 'attendance_percentage', 'previous_gpa', 'assignment_score', 'midterm_marks']
    target = 'final_result'
    
    # Check if all these exist in the cleaned dataframe
    available_cols = list(df.columns)
    valid_features = [f for f in features if f in available_cols]
    
    if target not in available_cols:
        st.error(f"Target column '{target}' not found. Available: {available_cols}")
        st.stop()

    X = df[valid_features]
    y = df[target]

    # 3. Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Lasso Model (alpha=0.5)
    model = Lasso(alpha=0.5)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # 5. Metrics
    col1, col2 = st.columns(2)
    col1.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
    col2.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")

    # 6. Feature Importance
    st.subheader("📊 Feature Importance (Coefficients)")
    coeff_df = pd.DataFrame({'Feature': valid_features, 'Coefficient': model.coef_})
    st.bar_chart(coeff_df.set_index('Feature'))
    st.table(coeff_df)

    # 7. Prediction Form
    st.subheader("🔮 Predict Final Result")
    with st.form("predict_form"):
        user_inputs = []
        for col in valid_features:
            val = st.number_input(f"Enter {col}", value=float(df[col].mean()))
            user_inputs.append(val)
        
        if st.form_submit_button("Predict"):
            input_arr = np.array(user_inputs).reshape(1, -1)
            scaled_input = scaler.transform(input_arr)
            res = model.predict(scaled_input)
            st.success(f"Predicted Final Result: {res[0]:.2f}")
