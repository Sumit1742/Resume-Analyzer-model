import streamlit as st
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Analyzer", layout="centered")

st.title("🔥 Smart Resume Analyzer + Job Matcher")

# -------- Load Model --------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
 # local model

model = load_model()

# -------- Extract Text from PDF --------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.lower()

# -------- Simple Skill Extraction --------
def extract_skills(text):
    skills_list = [
        "python", "machine learning", "deep learning",
        "nlp", "transformers", "bert",
        "pytorch", "tensorflow", "scikit-learn",
        "pandas", "numpy", "django",
        "mysql", "git", "streamlit",
        "data preprocessing", "model deployment"
    ]

    found_skills = []
    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)

    return found_skills

# -------- Inputs --------
uploaded_resume = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("📝 Paste Job Description")

# -------- Analyze Button --------
if st.button("🚀 Analyze Resume"):
    if uploaded_resume and job_description.strip():

        with st.spinner("Analyzing... Please wait"):
            resume_text = extract_text_from_pdf(uploaded_resume)
            jd_text = job_description.lower()

            # -------- Similarity --------
            resume_embedding = model.encode([resume_text])
            jd_embedding = model.encode([jd_text])
            score = cosine_similarity(resume_embedding, jd_embedding)[0][0]
            percentage = round(score * 100, 2)

            # -------- Skill Matching --------
            resume_skills = extract_skills(resume_text)
            jd_skills = extract_skills(jd_text)

            matched_skills = list(set(resume_skills) & set(jd_skills))
            missing_skills = list(set(jd_skills) - set(resume_skills))

        # -------- Display Score --------
        st.subheader("📊 Match Score")
        st.metric("Resume Match", f"{percentage}%")

        if percentage > 85:
            st.success("🔥 Excellent Match! Highly aligned.")
        if 85 > percentage > 65:
            st.success("🔥 Good Match! but still need improvement .")
        elif 65 > percentage > 45:
            st.warning("⚠ Moderate Match. Some improvements needed.")
        else:
            st.error("🚨 Low Match. Improve your resume alignment.")

        st.markdown("---")

        # -------- Show Skills --------
        st.subheader("✅ Matched Skills")
        if matched_skills:
            st.write(", ".join(matched_skills))
        else:
            st.write("No strong skill matches found.")

        st.subheader("❌ Missing Skills")
        if missing_skills:
            st.write(", ".join(missing_skills))
        else:
            st.write("No major missing skills.")

        st.markdown("---")

        # -------- Suggestions --------
        st.subheader("💡 Suggestions to Improve")
        if missing_skills:
            st.write("Consider adding these skills if relevant:")
            for skill in missing_skills:
                st.write(f"- {skill}")
        else:
            st.write("Your resume aligns well with this job description!")

    else:
        st.warning("Please upload resume and paste job description.")
