import streamlit as st
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import torch
import pdfplumber
import spacy

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load job family data
with open("final_with_software_dev_roles.json") as f:
    job_data = json.load(f)

# Course suggestions for missing skills
suggested_courses = {
    "tableau": "https://www.coursera.org/learn/data-visualization-tableau",
    "react": "https://www.udemy.com/course/react-the-complete-guide/",
    "pandas": "https://www.coursera.org/learn/data-analysis-with-python",
    "sql": "https://www.kaggle.com/learn/intro-to-sql",
    "excel": "https://www.udemy.com/course/microsoft-excel-all-in-one-package/",
    "machine learning": "https://www.coursera.org/learn/machine-learning",
    "python": "https://www.codecademy.com/learn/learn-python-3",
    "data visualization": "https://www.edx.org/learn/data-visualization",
    "power bi": "https://learn.microsoft.com/en-us/training/powerplatform/power-bi/",
}

# App config and tabs
st.set_page_config(page_title="Job Skill Matcher", layout="wide")
tabs = st.tabs([
    "🔍 Skills to Job Family",
    "🧭 Role-Based Recommendation",
    "📚 Explore",
    "📄 Resume → Career Match",
    "📊 Compare Roles"
])

# --- Utilities ---
def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def get_match_score(input_text, target_text, input_set, target_set):
    embedding_score = util.cos_sim(model.encode([input_text], convert_to_tensor=True)[0],
                                   model.encode([target_text], convert_to_tensor=True)[0]).item()
    jaccard_score = jaccard_similarity(input_set, target_set)
    return 0.7 * embedding_score + 0.3 * jaccard_score

def safe_split(skills):
    return (
        [s.strip() for s in skills.split(",") if s.strip()]
        if isinstance(skills, str)
        else [s.strip() for s in skills if isinstance(s, str) and s.strip()]
    )

# --- Tab 1: Skills to Job Family ---
with tabs[0]:
    st.header("🔍 Predict Job Family from Skills")
    skill_input = st.text_area("Enter your skills (comma-separated):", height=100)

    if skill_input:
        input_skills = [s.strip() for s in skill_input.split(",") if s.strip()]
        input_text = ", ".join(input_skills)
        input_skill_set = set(s.lower() for s in input_skills)

        ranked_families = []
        for family, content in job_data.items():
            base_list = safe_split(content["baseline_skills"])
            score = get_match_score(input_text, ", ".join(base_list), input_skill_set, set(s.lower() for s in base_list))
            ranked_families.append((family, round(score * 100, 2)))

        ranked_families.sort(key=lambda x: x[1], reverse=True)
        top_family, top_score = ranked_families[0]
        st.subheader(f"🔝 Best Matched Job Family: {top_family} ({top_score}%)")

        st.subheader("📊 Top 5 Matching Families + Role Analysis")
        for fam, score in ranked_families[:5]:
            with st.expander(f"{fam} - {score}% match"):
                roles = job_data[fam].get("roles", {})
                for role, role_skills in roles.items():
                    role_list = safe_split(role_skills)
                    matched = set(s.lower() for s in input_skills) & set(s.lower() for s in role_list)
                    missing = set(s.lower() for s in role_list) - set(s.lower() for s in input_skills)
                    percent = round(len(matched) / len(role_list) * 100, 2) if role_list else 0
                    st.markdown(f"**{role}** - Fit: {percent}%")
                    st.markdown(f"✅ Matched: {', '.join(f'**{s}**' for s in matched) if matched else 'None'}")
                    st.markdown(f"❌ Missing: {', '.join(missing) if missing else 'None'}")

# --- Tab 2: Role Recommendation ---
with tabs[1]:
    st.header("🧭 Find Related Roles from Your Current Role")
    current_role = st.text_input("Enter your current job role (e.g., Java Developer)")
    current_skills = st.text_area("Enter your current skills (comma-separated)", height=100)

    if current_role and current_skills:
        input_skills = [s.strip() for s in current_skills.split(",") if s.strip()]
        input_text = ", ".join(input_skills)
        input_skill_set = set(s.lower() for s in input_skills)

        matched_roles = []
        for family, content in job_data.items():
            for role, role_skills in content.get("roles", {}).items():
                role_list = safe_split(role_skills)
                role_set = set(s.lower() for s in role_list)
                score = get_match_score(input_text, ", ".join(role_list), input_skill_set, role_set)
                matched_roles.append((family, role, round(score * 100, 2), role_list))

        matched_roles.sort(key=lambda x: x[2], reverse=True)
        st.subheader("🔝 Top Roles You Can Transition Into")

        for fam, role, score, role_list in matched_roles[:10]:
            matched = set(s.lower() for s in input_skills) & set(s.lower() for s in role_list)
            missing = set(s.lower() for s in role_list) - set(s.lower() for s in input_skills)
            st.markdown(f"### **{role}** ({fam}) - Match Score: {score}%")
            st.markdown(f"✅ **Matched Skills:** {', '.join(f'**{s}**' for s in matched) if matched else 'None'}")
            st.markdown(f"❌ **Missing Skills:** {', '.join(missing) if missing else 'None'}")

            fig, ax = plt.subplots(figsize=(4, 1))
            ax.barh(["Skill Coverage"], [len(matched)], color="green", label="Matched")
            ax.barh(["Skill Coverage"], [len(missing)], left=[len(matched)], color="red", label="Missing")
            ax.set_xlim(0, len(matched) + len(missing) + 1)
            ax.set_xlabel("Number of Skills")
            ax.set_yticks([])
            ax.legend(loc="upper right")
            st.pyplot(fig)

# --- Tab 3: Explore ---
with tabs[2]:
    st.header("📚 Explore Job Families")
    job_family = st.selectbox("Select Job Family", sorted(job_data.keys()), key="explore_family")

    if job_family:
        st.subheader("Baseline Skills")
        st.write(job_data[job_family]["baseline_skills"])

        st.subheader("Baseline Skills Word Cloud")
        wc_text = ", ".join(safe_split(job_data[job_family]["baseline_skills"]))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wc_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        roles = job_data[job_family].get("roles", {})
        selected_role = st.selectbox("Select a Role", sorted(roles.keys()), key="explore_role")

        if selected_role:
            st.subheader("Skills for Selected Role")
            st.write(roles[selected_role])

# --- Tab 4: Resume Upload for Career Suggestions ---
with tabs[3]:
    st.header("📄 Upload Resume → Career Suggestions")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    if uploaded_file:
        with pdfplumber.open(uploaded_file) as pdf:
            resume_text = " ".join(page.extract_text() for page in pdf.pages if page.extract_text())

        doc = nlp(resume_text)
        extracted_skills = set()
        for chunk in doc.noun_chunks:
            cleaned = chunk.text.strip(" ,.;:()").lower()
            if 2 < len(cleaned) < 30:
                extracted_skills.add(cleaned)

        input_text = ", ".join(extracted_skills)
        input_skill_set = set(extracted_skills)

        ranked = []
        for family, content in job_data.items():
            base_list = safe_split(content["baseline_skills"])
            base_set = set(s.lower() for s in base_list)
            score = get_match_score(input_text, ", ".join(base_list), input_skill_set, base_set)
            ranked.append((family, round(score * 100, 2), base_list, base_set, content.get("roles", {})))

        ranked.sort(key=lambda x: x[1], reverse=True)

        st.subheader("📊 Top Matching Job Families from Resume")
        for fam, score, base_list, base_set, roles_dict in ranked[:5]:
            matched = input_skill_set & base_set
            missing = base_set - input_skill_set

            st.markdown(f"### **{fam}** - Match Score: {score}%")
            st.markdown(f"✅ Matched Skills: {', '.join(f'**{s}**' for s in matched) if matched else 'None'}")
            if missing:
                st.markdown("❌ Missing Skills:")
                for ms in missing:
                    course_link = suggested_courses.get(ms)
                    if course_link:
                        st.markdown(f"- {ms.title()} → [Suggested Course]({course_link})")
                    else:
                        st.markdown(f"- {ms.title()} → [Search on Google](https://www.google.com/search?q=learn+{ms.replace(' ', '+')})")

            st.markdown("#### 🔍 Best-Fit Roles in This Family")
            role_scores = []
            for role, role_skills in roles_dict.items():
                role_list = safe_split(role_skills)
                role_set = set(s.lower() for s in role_list)
                score = get_match_score(input_text, ", ".join(role_list), input_skill_set, role_set)
                role_scores.append((role, score, role_set))

            role_scores.sort(key=lambda x: x[1], reverse=True)
            for role, r_score, role_set in role_scores[:2]:
                r_matched = input_skill_set & role_set
                r_missing = role_set - input_skill_set
                st.markdown(f"**{role}** - Fit: {round(r_score * 100, 2)}%")
                st.markdown(f"✅ Matched: {', '.join(f'**{s}**' for s in r_matched) if r_matched else 'None'}")
                if r_missing:
                    st.markdown(f"❌ Missing: {', '.join(r_missing)}")

                fig, ax = plt.subplots(figsize=(4, 0.6))
                ax.barh(["Fit"], [len(r_matched)], color="green")
                ax.barh(["Fit"], [len(r_missing)], left=[len(r_matched)], color="red")
                ax.set_xlim(0, len(r_matched) + len(r_missing) + 1)
                ax.set_xticks([])
                ax.set_yticks([])
                st.pyplot(fig)

# --- Tab 5: Compare Roles ---
with tabs[4]:
    st.header("📊 Compare Two Job Roles Side-by-Side")
    fam = st.selectbox("Select Job Family", sorted(job_data.keys()), key="compare_family")
    roles = sorted(job_data[fam].get("roles", {}).keys())
    col1, col2 = st.columns(2)

    with col1:
        r1 = st.selectbox("Role 1", roles, key="compare_role1")
    with col2:
        r2 = st.selectbox("Role 2", roles, key="compare_role2")

    if r1 and r2 and r1 != r2:
        skills_r1 = set(s.lower() for s in safe_split(job_data[fam]["roles"][r1]))
        skills_r2 = set(s.lower() for s in safe_split(job_data[fam]["roles"][r2]))

        common = skills_r1 & skills_r2
        only_r1 = skills_r1 - skills_r2
        only_r2 = skills_r2 - skills_r1

        st.markdown(f"### 🔄 Comparing: **{r1}** vs **{r2}** in **{fam}**")
        st.markdown(f"**✅ Common Skills:** {', '.join(f'**{s}**' for s in common) if common else 'None'}")
        st.markdown(f"**🔵 {r1}-Only Skills:** {', '.join(only_r1) if only_r1 else 'None'}")
        st.markdown(f"**🟣 {r2}-Only Skills:** {', '.join(only_r2) if only_r2 else 'None'}")
