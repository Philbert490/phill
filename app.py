# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------
# STREAMLIT APP CONFIGURATION
# -------------------------------------
st.set_page_config(page_title="Electricity Load Forecasting", layout="wide")

st.title("⚡ Electricity Load Forecasting")
st.markdown("""
This app predicts **electricity load** using Random Forest based on smart meter data.  
Upload your dataset (CSV) to explore and forecast electricity consumption across Rwandan cities.  
""")

# -------------------------------------
# FILE UPLOADER
# -------------------------------------
uploaded_file = st.file_uploader("📂 Upload your Smart Meter Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded successfully!")

    # Preview dataset
    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    # Basic info
    st.write("**Shape of dataset:**", data.shape)
    st.write("**Columns:**", list(data.columns))

    # -------------------------------------
    # EXPLORATORY DATA ANALYSIS
    # -------------------------------------
    st.subheader("🔎 Exploratory Data Analysis")
    st.write(data.describe())

    if "city" in data.columns:
        st.write("**Distribution of Data by City**")
        st.bar_chart(data["city"].value_counts())

    # -------------------------------------
    # FEATURE SELECTION
    # -------------------------------------
    st.subheader("⚙️ Feature Selection & Model Training")

    target = st.selectbox("Select Target Variable (e.g. load)", data.columns)
    features = st.multiselect("Select Feature Columns", [col for col in data.columns if col != target])

    if features and target:
        X = data[features]
        y = data[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Hyperparameters
        n_estimators = st.slider("Number of Trees (n_estimators)", 50, 500, 100, step=50)
        max_depth = st.slider("Maximum Depth", 2, 20, 10)

        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # -------------------------------------
        # MODEL EVALUATION
        # -------------------------------------
        st.subheader("📈 Model Evaluation")
        st.write("**Mean Absolute Error (MAE):**", mean_absolute_error(y_test, y_pred))
        st.write("**Mean Squared Error (MSE):**", mean_squared_error(y_test, y_pred))
        st.write("**R² Score:**", r2_score(y_test, y_pred))

        # Plot actual vs predicted
        st.write("**Prediction vs Actual (first 100 samples)**")
        fig, ax = plt.subplots()
        ax.plot(y_test.values[:100], label="Actual", marker="o")
        ax.plot(y_pred[:100], label="Predicted", marker="x")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel(target)
        ax.legend()
        st.pyplot(fig)

        # -------------------------------------
        # FEATURE IMPORTANCE
        # -------------------------------------
        st.subheader("🌟 Feature Importance")
        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance_df.set_index("Feature"))

else:
    st.info("👆 Please upload a CSV file to continue.")
