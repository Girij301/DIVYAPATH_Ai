import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from flask import Flask, render_template, request, session, redirect, url_for, flash
from app.module1.predictor import load_model, predict_grade
from app.module1.risk_engine import apply_college_rules, grade_to_probability, risk_level
from app.module2.resume_parser import extract_text
from app.module2.skill_matcher import analyze_resume
from app.module3.roadmap_generator import generate_roadmap
from app.module4.song_recommender import recommend_songs
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from app.module4.emotion_detector import detect_emotion
    emotion_error = None
except Exception as e:
    detect_emotion = None
    emotion_error = str(e)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-key-change-in-production')

# Load model once
model, grade_enc, extra_enc, parent_enc = load_model()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/performance', methods=['GET', 'POST'])
def performance():
    if request.method == 'POST':
        try:
            study_hours = float(request.form['study_hours'])
            attendance = float(request.form['attendance'])
            previous_grade = float(request.form['previous_grade'])
            extra = request.form['extra']
            parent_edu = request.form['parent_edu']

            extra_val = 1 if extra == "Yes" else 0

            prob, grade, method = apply_college_rules(study_hours, attendance, previous_grade)

            if prob is None:
                grade = predict_grade(model, grade_enc, extra_enc, parent_enc, study_hours, attendance, previous_grade, extra_val, parent_edu)
                prob = grade_to_probability(grade)
                method = "ML Model"

            risk, icon, rtype = risk_level(prob)

            # Simulate improvement
            new_study = study_hours + 10
            new_att = min(attendance + 10, 100)
            sim_prob, sim_grade, _ = apply_college_rules(new_study, new_att, previous_grade)
            if sim_prob is None:
                sim_grade = predict_grade(model, grade_enc, extra_enc, parent_enc, new_study, new_att, previous_grade, extra_val, parent_edu)
                sim_prob = grade_to_probability(sim_grade)

            return render_template('performance.html', 
                                   grade=grade, prob=round(prob*100,1), risk=risk, icon=icon, method=method,
                                   study_hours=study_hours, attendance=attendance, previous_grade=previous_grade,
                                   sim_prob=round(sim_prob*100,1), sim_grade=sim_grade, new_study=new_study, new_att=new_att)
        except ValueError as e:
            flash(f'Error in prediction: {str(e)}. Please check your inputs.')
            return redirect(url_for('performance'))
        except Exception as e:
            flash(f'An unexpected error occurred: {str(e)}')
            return redirect(url_for('performance'))
    return render_template('performance.html')

@app.route('/resume', methods=['GET', 'POST'])
def resume():
    if request.method == 'POST':
        try:
            uploaded_file = request.files['resume']
            target_role = request.form['target_role']

            if uploaded_file:
                resume_text = extract_text(uploaded_file)
                result = analyze_resume(resume_text, target_role)
                session['resume_result'] = result
                return render_template('resume.html', result=result)
        except Exception as e:
            flash(f'Error analyzing resume: {str(e)}')
            return redirect(url_for('resume'))
    return render_template('resume.html')

@app.route('/roadmap')
def roadmap():
    try:
        if 'resume_result' not in session:
            flash('Please analyze your resume first.')
            return redirect(url_for('resume'))
        
        result = session['resume_result']
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
        roadmap_data = generate_roadmap(student_profile, resume_profile)
        return render_template('roadmap.html', roadmap=roadmap_data)
    except Exception as e:
        flash(f'Error generating roadmap: {str(e)}')
        return redirect(url_for('resume'))

@app.route('/mood', methods=['GET', 'POST'])
def mood():
    if request.method == 'POST':
        if detect_emotion is None:
            flash('Emotion detection is unavailable on this machine. Please use a compatible CPU or install a supported TensorFlow runtime.')
            return render_template('mood.html', error=emotion_error)

        try:
            uploaded_img = request.files['image']
            if uploaded_img:
                img_bytes = uploaded_img.read()
                emotion, confidence = detect_emotion(img_bytes)
                songs = recommend_songs(emotion)
                return render_template('mood.html', emotion=emotion, confidence=round(confidence*100,2), songs=songs)
        except Exception as e:
            flash(f'Error detecting emotion: {str(e)}')
            return redirect(url_for('mood'))
    return render_template('mood.html', error=emotion_error)

if __name__ == '__main__':
    app.run(debug=True)