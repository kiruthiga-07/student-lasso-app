import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Page Config
st.set_page_config(page_title="Student Lasso Predictor", layout="wide")
st.title("🎓 Student Performance Lasso Regression")

# 1. Load Dataset
@st.cache_data
def load_data():
    try:
        # Load the Excel file
        df = pd.read_excel('student_data.xlsx', engine='openpyxl')
        
        # CLEANING: Remove any leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()
        
        # CLEANING: Drop any completely empty rows
        df = df.dropna()
        
        return df
    except FileNotFoundError:
        st.error("❌ Error: 'student_data.xlsx' not found. Please ensure the file is in your GitHub repository.")
        return None
    except Exception as e:
        st.error(f"❌ An error occurred while loading the file: {e}")
        return None

df = load_data()

if df is not None:
    st.subheader("1. Data Preview (First 5 Rows)")
    st.write(df.head())

    # 2. Define Features and Target
    # Ensure these names match your Excel headers EXACTLY
    features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Internet_Usage']
    target = 'Final_Score'
    
    # Validation Check: Ensure columns exist after stripping whitespace
    actual_columns = list(df.columns)
    missing_cols = [col for col in features + [target] if col not in actual_columns]
    
    if missing_cols:
        st.error(f"❌ Column mismatch! Missing columns: {missing_cols}")
        st.info(f"Columns found in your file: {actual_columns}")
        st.stop() # Prevents the app from crashing further down

    # 3. Prepare Data
    X = df[features]
    y = df[target]

    # Split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling (Mandatory for Lasso)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Build and Fit Lasso Model
    # Alpha = 0.5 as per problem statement
    model = Lasso(alpha=0.5)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # 5. Evaluation Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("2. Model Evaluation")
    col1, col2 = st.columns(2)
    col1.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    col2.metric("R² Score", f"{r2:.2f}")

    # 6. Feature Importance (The "Lasso" Effect)
    st.subheader("3. Feature Importance (Coefficients)")
    coeff_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
    coeff_df = coeff_df.sort_values(by='Coefficient', ascending=False)
    
    st.bar_chart(coeff_df.set_index('Feature'))
    st.write("💡 Note: Features with a **0.00** coefficient were automatically removed by the Lasso model.")
    st.table(coeff_df)

    # 7. Prediction Tool
    st.divider()
    st.subheader("🔮 Predict Student Final Score")
    
    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)
        user_inputs = []
        
        for i, feat in enumerate(features):
            # Place inputs in two columns for better UI
            with col_a if i % 2 == 0 else col_b:
                val = st.number_input(f"Input {feat}", value=float(df[feat].mean()))
                user_inputs.append(val)
        
        submit = st.form_submit_button("Generate Prediction")
        
        if submit:
            # Scale the user input using the SAME scaler from training
            input_array = np.array(user_inputs).reshape(1, -1)
            scaled_input = scaler.transform(input_array)
            prediction = model.predict(scaled_input)
            st.success(f"### Predicted Final Exam Score: {prediction[0]:.2f}")
