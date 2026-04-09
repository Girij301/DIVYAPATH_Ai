import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import streamlit as st
from app.module1.predictor import load_model, predict_grade
from app.module1.risk_engine import apply_college_rules, grade_to_probability, risk_level
from app.module2.resume_parser import extract_text
from app.module2.skill_matcher import analyze_resume
from app.module3.roadmap_generator import generate_roadmap
from app.module4.song_recommender import recommend_songs
from PIL import Image
import numpy as np

try:
    from app.module4.emotion_detector import detect_emotion
    emotion_detection_available = True
    emotion_error = None
except Exception as e:
    detect_emotion = None
    emotion_detection_available = False
    emotion_error = str(e)
   
    
st.set_page_config(
    page_title="DivyaPath AI",
    page_icon="🎓",
    layout="centered"
)

model, grade_enc, extra_enc, parent_enc = load_model()

# Initialize session state
if "resume_result" not in st.session_state:
    st.session_state["resume_result"] = None

st.markdown('<div style="text-align:center;font-size:2.6rem;font-weight:bold;">🎓 DivyaPath AI</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#666;">Your Academic & Career Mentor</div>', unsafe_allow_html=True)
st.markdown("---")

# Create tabs based on available features
if emotion_detection_available:
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎓 Student Performance", "📄 Resume Skill Analyzer", "🛣️ Career Roadmap", "🎧 Mood Based Song"]
    )
else:
    tab1, tab2, tab3 = st.tabs(
        ["🎓 Student Performance", "📄 Resume Skill Analyzer", "🛣️ Career Roadmap"]
    )
    tab4 = None
    st.info("💡 Mood detection feature is unavailable on this machine. You can use Student Performance, Resume Analyzer, and Roadmap features.")


# =========================
# Module 1 – Student Performance (UNCHANGED)
# =========================
with tab1:
    st.markdown("### Student Performance & Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider("Study Hours per Week", 0.0, 80.0, 20.0, 1.0)
        attendance = st.slider("Attendance Rate (%)", 0.0, 100.0, 80.0, 1.0)
        previous_grade = st.slider("Previous Grade (0–100)", 0.0, 100.0, 70.0, 1.0)

    with col2:
        extra = st.radio("Extracurricular Activities", ["Yes", "No"], index=1)
        parent_edu = st.selectbox(
            "Parent Education Level",
            ['Master', 'High School', 'Associate', 'Bachelor', 'Doctorate'],
            index=2
        )

    if st.button("🔮 Predict Performance", use_container_width=True):
        extra_val = 1 if extra == "Yes" else 0

        prob, grade, method = apply_college_rules(study_hours, attendance, previous_grade)

        if prob is None:
            grade = predict_grade(model, grade_enc, extra_enc, parent_enc,study_hours, attendance, previous_grade,extra, parent_edu)
            prob = grade_to_probability(grade)
            method = "ML Model"

        risk, icon, rtype = risk_level(prob)

        st.markdown("### 📋 Results")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Predicted Grade", grade)
        with m2:
            st.metric("Pass Probability", f"{round(prob*100,1)}%")
        with m3:
            st.metric("Risk Level", f"{icon} {risk}")

        st.write("Prediction Method:", method)

        st.markdown("### 🔍 Why this result?")
        shown = False

        if attendance < 55:
            st.write("- Your attendance is below the college minimum. This strongly increases risk.")
            shown = True
        if study_hours < 10:
            st.write("- Your study time is very low. This reduces your chance of success.")
            shown = True
        if previous_grade < 40:
            st.write("- Your previous performance was weak, so the system is cautious.")
            shown = True
        if attendance >= 75:
            st.write("- Good attendance is helping your chances.")
            shown = True
        if study_hours >= 20:
            st.write("- Strong study effort is improving your outcome.")
            shown = True

        if not shown:
            st.write("- Your current habits place you in a risky zone. Small improvements can change the outcome.")

        if 0.45 <= prob <= 0.60:
            st.info("⚠️ You are in the borderline zone. Improving attendance and study time can change your result.")

        new_study = study_hours + 10
        new_att = min(attendance + 10, 100)

        sim_prob, sim_grade, _ = apply_college_rules(new_study, new_att, previous_grade)
        if sim_prob is None:
            sim_grade = predict_grade(model, grade_enc, extra_enc, parent_enc,new_study, new_att, previous_grade, extra, parent_edu)
            sim_prob = grade_to_probability(sim_grade)

        st.markdown("### 🚀 What if you improve?")
        st.write(
            f"If you increase Study Hours to {new_study} and Attendance to {new_att}%, "
            f"your success chance can become about {round(sim_prob*100,1)}% "
            f"(Grade: {sim_grade})."
        )

# =========================
# Module 2 – Resume Skill Analyzer
# =========================
with tab2:
    st.markdown("### Resume Skill Analyzer")

    ROLE_OPTIONS = [
        "Data Scientist", "Data Analyst", "Machine Learning Engineer",
        "Web Developer", "Frontend Developer", "Backend Developer",
        "Software Engineer", "HR Manager", "Business Analyst"
    ]

    uploaded_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])
    target_role = st.selectbox("Target Job Role", ROLE_OPTIONS)

    if uploaded_file and st.button("🔍 Analyze Resume", use_container_width=True):
        with st.spinner("Analyzing your resume..."):
            resume_text = extract_text(uploaded_file)
            result = analyze_resume(resume_text, target_role)

        st.session_state["resume_result"] = result

        st.markdown("### 🧾 Resume Score Card")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Match %", f'{result["match_percent"]}%')
        with s2:
            st.metric("Skills Found", len(result["matched_skills"]))
        with s3:
            st.metric("Skills Missing", len(result["missing_skills"]))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### ✅ Matched Skills")
            for s in result["matched_skills"]:
                st.write(f"- {s}")

        with c2:
            st.markdown("#### ❌ Top Missing Skills")
            for s in result["missing_skills"][:8]:
                st.write(f"- {s}")

        st.markdown("### 🔍 Why this result?")
        match = result["match_percent"]
        missing = result["missing_skills"]
        matched = result["matched_skills"]

        shown = False
        if match < 40:
            st.write("- Your resume does not align well with this role yet.")
            shown = True
        if len(missing) > 10:
            st.write("- Many core skills for this role are missing.")
            shown = True
        if len(matched) >= 5:
            st.write("- You already have some relevant skills. This is a good base.")
            shown = True
        if match >= 70:
            st.write("- Your profile is already strong for this role.")
            shown = True

        if not shown:
            st.write("- Your profile is close. A few focused improvements can make a big difference.")

        if 45 <= match <= 60:
            st.info("⚠️ You are in the borderline zone. Adding 3–4 skills and 1–2 projects can move you ahead.")

        improved_match = min(match + 20, 95)
        st.markdown("### 🚀 What if you improve?")
        st.write(
            f"If you learn the top missing skills and add 1–2 projects, "
            f"your match score can grow from {match}% to around {improved_match}%."
        )

# =========================
# Module 3 – Career Roadmap
# =========================
with tab3:
    st.markdown("### Your Career Roadmap")

    if st.session_state["resume_result"] is None:
        st.info("First analyze your resume in the Resume tab.")
    else:
        result = st.session_state["resume_result"]

        student_profile = {
            "grade": "C",
            "risk": "Medium",
            "study_hours": 12,
            "attendance": 70
        }

        resume_profile = {
            "role": result["role"],
            "matched_skills": result["matched_skills"],
            "missing_skills": result["missing_skills"]
        }

        roadmap = generate_roadmap(student_profile, resume_profile)

        st.markdown("### 📅 3-Month Plan")
        for step in roadmap["3_month"]:
            st.write(f"- {step}")

        st.markdown("### 📅 6-Month Plan")
        for step in roadmap["6_month"]:
            st.write(f"- {step}")

        st.markdown("### 📅 12-Month Plan")
        for step in roadmap["12_month"]:
            st.write(f"- {step}")

# =========================
# Module 4 – Mood & Music

if tab4 is not None:
    with tab4:
        st.markdown("### 🎧 Mood Detection & Song Recommendation")

        uploaded_img = st.file_uploader(
            "Upload your face image",
            type=["jpg", "png", "jpeg"],
            key="mood_image"
        )

        if uploaded_img:
            st.image(uploaded_img, width=250)

            if st.button("🎯 Detect Emotion & Recommend Songs"):
                img_bytes = uploaded_img.read()
                try:
                    emotion, confidence = detect_emotion(img_bytes)
                    st.success(f"Detected Emotion: {emotion}")
                    st.write(f"Confidence: {round(confidence * 100, 2)}%")

                    songs = recommend_songs(emotion)

                    st.markdown("### 🎵 Songs for You")
                    if songs:
                        for s in songs:
                            st.write(f"🎶 {s['song']} – {s['artist']} ({s['language']})")
                    else:
                        st.info("No songs found for this mood.")
                except Exception as e:
                    st.error(f"Emotion detection failed: {str(e)}")
