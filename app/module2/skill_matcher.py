import pandas as pd
import ast
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "module2_resume"
MODEL_DIR = BASE_DIR / "models"

SKILL_PATH = DATA_DIR / "skills_master.csv"
JOB_PATH = DATA_DIR / "all_job_post.csv"
VECTORIZER_PATH = MODEL_DIR / "resume_vectorizer.pkl"

# Load data
skills_df = pd.read_csv(SKILL_PATH)
MASTER_SKILLS = [s.lower().strip() for s in skills_df["skill"].dropna().tolist()]

job_df = pd.read_csv(JOB_PATH)
job_df["job_skill_set"] = job_df["job_skill_set"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)

# Load vectorizer
vectorizer = joblib.load(VECTORIZER_PATH)

def extract_skills(resume_text):
    text = resume_text.lower()
    found = []
    for skill in MASTER_SKILLS:
        if skill in text:
            found.append(skill)
    return sorted(set(found))

def get_role_skills(target_role):
    role_jobs = job_df[job_df["job_title"].str.contains(target_role, case=False, na=False)]

    role_skills = []
    for skills in role_jobs["job_skill_set"]:
        role_skills.extend([s.lower().strip() for s in skills if isinstance(s, str)])

    return sorted(set(role_skills))

def semantic_match_score(resume_text, role_skills):
    if not role_skills:
        return 0

    role_text = " ".join(role_skills)

    vecs = vectorizer.transform([resume_text, role_text])
    sim = cosine_similarity(vecs[0], vecs[1])[0][0]

    return int(sim * 100)

def analyze_resume(resume_text, target_role):
    resume_skills = extract_skills(resume_text)
    role_skills = get_role_skills(target_role)

    if not role_skills:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "match_percent": 0,
            "role": target_role,
            "suggestions": ["Role not found in dataset. Try a different role name."]
        }

    matched = sorted(set(resume_skills) & set(role_skills))
    missing = sorted(set(role_skills) - set(resume_skills))

    rule_score = int((len(matched) / len(role_skills)) * 100)
    semantic_score = semantic_match_score(resume_text, role_skills)

    # Final blended score
    match_percent = int(0.6 * rule_score + 0.4 * semantic_score)

    suggestions = []
    if match_percent < 40:
        suggestions.append("Your resume is far from this role. Start with core fundamentals.")
    elif match_percent < 70:
        suggestions.append("You are on the right path. Focus on the missing skills.")
    else:
        suggestions.append("You are well aligned. Polish projects and experience.")

    if missing:
        suggestions.append(f"Learn top missing skills: {', '.join(missing[:5])}")

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "match_percent": match_percent,
        "role": target_role,
        "suggestions": suggestions
    }
