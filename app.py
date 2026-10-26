import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📊",
    layout="wide"
)

# ---------- Custom styling ----------
st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    .result-card {
        padding: 1.5rem; border-radius: 14px; text-align: center;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white; margin-top: 0.5rem;
    }
    .result-card h1 {color: white; font-size: 3.2rem; margin: 0;}
    .result-card p {margin: 0; opacity: 0.9;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Student Performance Prediction")
st.write(
    "Predict a student's estimated final score using "
    "study hours, attendance, and previous exam score."
)

# 1. Create a sample dataset
@st.cache_data
def create_dataset():
    rng = np.random.default_rng(42)
    n = 200

    study_hours = rng.uniform(1, 8, n)
    attendance = rng.uniform(40, 100, n)
    previous_score = rng.uniform(30, 95, n)

    noise = rng.normal(0, 5, n)

    final_score = np.clip(
        5 * study_hours
        + 0.25 * attendance
        + 0.40 * previous_score
        + noise,
        0,
        100
    )

    return pd.DataFrame({
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Previous_Score": previous_score,
        "Final_Score": final_score
    })

df = create_dataset()

# 2. Select input features and target
X = df[["Study_Hours", "Attendance", "Previous_Score"]]
y = df["Final_Score"]

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the single ML algorithm
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# ---------- Sidebar: user input ----------
st.sidebar.header("🎓 Student Details")
study_hours = st.sidebar.slider("Study Hours (per day)", 1.0, 8.0, 4.0, 0.5)
attendance = st.sidebar.slider("Attendance (%)", 40, 100, 75)
previous_score = st.sidebar.slider("Previous Exam Score", 30, 95, 60)
predict_clicked = st.sidebar.button("🔮 Predict Score", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Algorithm: Linear Regression (only)")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Prediction", "📈 Model Performance", "🗂 Dataset", "ℹ️ About"]
)

# ----- Tab 1: Prediction -----
with tab1:
    st.subheader("Predicted Final Score")
    input_df = pd.DataFrame(
        [[study_hours, attendance, previous_score]], columns=X.columns
    )
    prediction = float(np.clip(model.predict(input_df)[0], 0, 100))

    if predict_clicked:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown(
                f"""<div class="result-card">
                    <p>Estimated Final Score</p>
                    <h1>{prediction:.1f}</h1>
                    <p>out of 100</p></div>""",
                unsafe_allow_html=True,
            )
        with col_b:
            if prediction >= 85:
                grade, msg = "A — Excellent", "Outstanding expected performance."
            elif prediction >= 70:
                grade, msg = "B — Good", "Good performance; a bit more effort can reach the top."
            elif prediction >= 55:
                grade, msg = "C — Average", "Improve study hours and attendance to do better."
            else:
                grade, msg = "D — Needs Improvement", "Needs more study time and regular attendance."
            st.metric("Estimated Grade", grade)
            st.progress(int(prediction))
            st.info(msg)

        st.markdown("#### Your Inputs")
        st.dataframe(input_df.round(2), use_container_width=True, hide_index=True)
    else:
        st.info("👈 Enter the student's details in the sidebar and click **Predict Score**.")

# ----- Tab 2: Model performance -----
with tab2:
    st.subheader("Model Evaluation (Test Data)")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{r2:.3f}")
    m2.metric("Mean Absolute Error", f"{mae:.2f}")
    m3.metric("RMSE", f"{rmse:.2f}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, y_pred, alpha=0.7, color="#4f46e5")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", label="Perfect prediction")
        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Predicted Score")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with c2:
        st.markdown("#### Feature Coefficients")
        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_
        })
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.barh(coef_df["Feature"], coef_df["Coefficient"], color="#7c3aed")
        ax2.set_xlabel("Coefficient value")
        ax2.grid(alpha=0.3, axis="x")
        st.pyplot(fig2)
        st.caption(f"Intercept: {model.intercept_:.2f}")

# ----- Tab 3: Dataset -----
with tab3:
    st.subheader("Sample Dataset (200 students)")
    st.dataframe(df.round(2), use_container_width=True, height=350)
    st.markdown("#### Summary Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)
    st.download_button(
        "⬇️ Download dataset (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        "student_data.csv",
        "text/csv",
    )

# ----- Tab 4: About -----
with tab4:
    st.subheader("About this Project")
    st.write(
        "This project predicts a student's final exam score using "
        "**Linear Regression**, the only Machine Learning algorithm used."
    )
    st.markdown("""
- **Inputs:** Study hours, attendance %, previous exam score
- **Output:** Estimated final score (0–100)
- **Data:** 200 synthetic students, 80/20 train-test split
- **Libraries:** Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib
""")
