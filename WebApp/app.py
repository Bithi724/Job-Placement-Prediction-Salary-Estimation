import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r"../Data/ds_salaries.csv")
    return df

df = load_data()

st.set_page_config(page_title="Job Placement & Salary Estimation", page_icon=":bar_chart:", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Visualizations", "Model & Prediction", "About"])

# Home
if page == "Home":
    st.title("Job Placement Prediction & Salary Estimation")
    st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stApp {background-color: #f8f9fa;}
    </style>
    """, unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1519389950473-47ba0277781c", use_column_width=True)
    st.write("Welcome to the interactive dashboard for job placement and salary estimation analysis. Explore the data, visualize insights, and predict salaries based on your input!")

# Data Exploration
elif page == "Data Exploration":
    st.header("Dataset Overview")
    st.write("Shape:", df.shape)
    st.write("Sample Data:")
    st.dataframe(df.head(10))
    st.write("Summary Statistics:")
    st.dataframe(df.describe())
    st.write("Missing Values:")
    st.dataframe(df.isnull().sum())

# Visualizations
elif page == "Visualizations":
    st.header("Visualizations")
    st.subheader("Log-Transformed Salary Distribution")
    df['log_salary'] = np.log1p(df['salary_in_usd'])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['log_salary'], kde=True, color='skyblue', ax=ax)
    ax.set_title("Log-Transformed Salary Distribution")
    ax.set_xlabel("Log(Salary + 1)")
    st.pyplot(fig)

    st.subheader("Average Salary by Company Size & Experience Level")
    grouped = df.groupby(['company_size', 'experience_level'])['salary_in_usd'].mean().unstack()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    grouped.plot(kind='bar', colormap='viridis', ax=ax2)
    ax2.set_title("Average Salary by Company Size & Experience Level")
    ax2.set_ylabel("Average Salary (USD)")
    ax2.set_xlabel("Company Size")
    st.pyplot(fig2)

    st.subheader("Pairplot: Log Salary vs Remote Ratio")
    fig3 = sns.pairplot(df[['log_salary', 'remote_ratio']], diag_kind='kde')
    st.pyplot(fig3)

# Model & Prediction
elif page == "Model & Prediction":
    st.header("Salary Prediction")
    st.write("Train a model and predict salary based on your input.")
    # Data preprocessing
    df_cleaned = df.drop(columns=['salary', 'salary_currency'])
    df_cleaned['work_year'] = df_cleaned['work_year'].astype(str)
    categorical_cols = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size', 'work_year']
    df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)
    X = df_encoded.drop(columns=['salary_in_usd'])
    y = df_encoded['salary_in_usd']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # Model evaluation
    st.subheader("Model Performance")
    st.write("Linear Regression R²:", r2_score(y_test, y_pred_lr))
    st.write("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
    st.write("Random Forest R²:", r2_score(y_test, y_pred_rf))
    st.write("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

    st.subheader("Predict Salary")
    # User input for prediction
    input_dict = {}
    for col in categorical_cols:
        options = df[col].unique().tolist()
        input_dict[col] = st.selectbox(f"{col}", options)
    input_dict['remote_ratio'] = st.slider("Remote Ratio", int(df['remote_ratio'].min()), int(df['remote_ratio'].max()), int(df['remote_ratio'].mean()))
    # Prepare input for prediction
    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    # Align columns
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
    # Predict
    pred_lr = lr_model.predict(input_encoded)[0]
    pred_rf = rf_model.predict(input_encoded)[0]
    st.success(f"Predicted Salary (Linear Regression): ${pred_lr:,.2f}")
    st.success(f"Predicted Salary (Random Forest): ${pred_rf:,.2f}")

# About
elif page == "About":
    st.header("About This Project")
    st.write("""
    This web application was built using Streamlit and Python. It allows users to explore job placement and salary data, visualize insights, and predict salaries using machine learning models.\n\nAuthor: Bithi724\nGitHub: [Job-Placement-Prediction-Salary-Estimation](https://github.com/Bithi724/Job-Placement-Prediction-Salary-Estimation)
    """)
    st.markdown("---")
    st.write("For any queries, contact: your.email@example.com")
