import asyncio
import os
import streamlit as st
from shared_data import JOB_DESCRIPTION
from pipeline import process_all_resumes

st.set_page_config(page_title="Resume Screener", layout="wide")
st.title("Intelligent Resume Screening Platform")

st.subheader("Job Description")
st.text(JOB_DESCRIPTION)

uploaded_files = st.file_uploader(
    "Upload candidate resumes (PDF) - You can select more than one",
    type="pdf",
    accept_multiple_files=True,
)

if uploaded_files and st.button("Screen Candidates"):
    os.makedirs("temp_uploads", exist_ok=True)
    files_to_process = []
    for file in uploaded_files:
        path = os.path.join("temp_uploads", file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        files_to_process.append((path, file.name))

    total = len(files_to_process)
    progress_bar = st.progress(0, text=f"Processed 0 / {total} resumes")
    completed = {"count": 0}

    def update_progress(filename):
        completed["count"] += 1
        progress_bar.progress(
            completed["count"] / total,
            text=f"Processed {completed['count']} / {total} resumes ({filename})",
        )

    with st.spinner("Parsing, scoring, and summarizing resumes..."):
        candidates = asyncio.run(
            process_all_resumes(files_to_process, on_done=update_progress)
        )

    progress_bar.empty()
    ranked = sorted(candidates, key=lambda c: c["match_score"], reverse=True)

    st.subheader("Ranked Candidates")
    st.dataframe(
        [
            {
                "name": c["name"],
                "match_score": c["match_score"],
                "semantic": c["semantic_score"],
                "skills_match": c["skill_score"],
                "experience": c["experience_score"],
                "skills": c["skills"],
                "experience_years": c["experience_years"],
                "email": c["email"],
            }
            for c in ranked
        ],
        use_container_width=True,
    )

    st.subheader("Top Match Detail")
    if ranked:
        top = ranked[0]
        st.success(f"🏆 {top['name']} --- {top['match_score']}% match")
        st.write(f"Matched skills: {top['skills']}")
        st.write(f"**AI Summary:** {top['summary']}")

    st.subheader("Resume Summaries")
    for c in ranked:
        with st.expander(f"{c['name']} --- {c['match_score']}%"):
            st.write(c["summary"])