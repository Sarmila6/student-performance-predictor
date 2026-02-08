import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("student_score_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict the Exam Score.")

# --- Input fields ---
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=50, value=10)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=15, value=7)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=70)
tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, max_value=10, value=1)
physical_activity = st.number_input("Physical Activity (hrs/week)", min_value=0, max_value=20, value=3)

gender = st.selectbox("Gender", ["Male", "Female"])
internet_access = st.selectbox("Internet Access", ["Yes", "No"])
school_type = st.selectbox("School Type", ["Public", "Private"])

motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
learning_disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])
parental_education_level = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])

# --- Create input row ---
input_dict = {
    "Hours_Studied": hours_studied,
    "Attendance": attendance,
    "Sleep_Hours": sleep_hours,
    "Previous_Scores": previous_scores,
    "Tutoring_Sessions": tutoring_sessions,
    "Physical_Activity": physical_activity,
    "Gender": gender,
    "Internet_Access": internet_access,
    "School_Type": school_type,
    "Motivation_Level": motivation_level,
    "Teacher_Quality": teacher_quality,
    "Family_Income": family_income,
    "Parental_Involvement": parental_involvement,
    "Access_to_Resources": access_to_resources,
    "Peer_Influence": peer_influence,
    "Learning_Disabilities": learning_disabilities,
    "Parental_Education_Level": parental_education_level,
    "Distance_from_Home": distance_from_home,
    "Extracurricular_Activities": extracurricular_activities,
}

input_df = pd.DataFrame([input_dict])

# Convert to dummy variables
input_df = pd.get_dummies(input_df, drop_first=True)

# Add missing columns
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Ensure same column order
input_df = input_df[model_columns]

# --- Predict ---
if st.button("Predict Exam Score"):
    prediction = model.predict(input_df)[0]
    st.success(f"📌 Predicted Exam Score: **{prediction:.2f}**")
