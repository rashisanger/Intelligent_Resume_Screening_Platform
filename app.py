import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from extractor import extract_contact_info, extract_experience_years, extract_skills
from matcher import compute_match_score, rank_candidates, summarize_resume
from parser import extract_text_from_pdf
from shared_data import JOB_DESCRIPTION


st.set_page_config(page_title="Resume Screener", layout="wide")
st.title("Intelligent Resume Screening Platform")

st.subheader("Job Description")
jd_text_input = st.text_area(
    "Paste the Job Description",
    value=JOB_DESCRIPTION.strip(),
    height=200,
)
jd_text = jd_text_input.strip() or JOB_DESCRIPTION.strip()

uploaded_files = st.file_uploader(
    "Upload candidate resumes (PDF)",
    type="pdf",
    accept_multiple_files=True,
)

# One shared thread pool for the whole app session.
executor = ThreadPoolExecutor(max_workers=4)


def process_one(file_bytes, filename, jd_text):
    """Run the synchronous pipeline for one resume inside a worker thread."""
    os.makedirs("temp_uploads", exist_ok=True)
    path = os.path.join("temp_uploads", filename)
    with open(path, "wb") as f:
        f.write(file_bytes)

    text = extract_text_from_pdf(path)
    skills = extract_skills(text)
    experience = extract_experience_years(text)
    contact = extract_contact_info(text)
    score, breakdown = compute_match_score(text, skills, experience, jd_text)
    summary = summarize_resume(text)

    return {
        "name": filename,
        "match_score": score,
        "semantic_%": breakdown["semantic"],
        "skill_overlap_%": breakdown["skill_overlap"],
        "experience_%": breakdown["experience"],
        "skills": ", ".join(skills) if skills else "None found",
        "experience_years": experience,
        "email": contact["email"],
        "phone": contact["phone"],
        "summary": summary,
    }


async def process_all(files, jd_text):
    """Run process_one concurrently and update progress as each file finishes."""
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            executor,
            process_one,
            f.getbuffer().tobytes(),
            f.name,
            jd_text,
        )
        for f in files
    ]

    progress_bar = st.progress(0, text="Starting...")
    total = len(tasks)
    results = []

    for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
        result = await coro
        results.append(result)
        progress_bar.progress(i / total, text=f"Processed {i}/{total} resumes")

    return results


if uploaded_files and st.button("Screen Candidates"):
    with st.spinner("Loading models on first run..."):
        candidates = asyncio.run(process_all(uploaded_files, jd_text))

    ranked = rank_candidates(candidates)

    st.subheader("Ranked Candidates")
    st.dataframe(ranked, use_container_width=True)

    st.subheader("Top Match Detail")
    if ranked:
        top = ranked[0]
        st.success(f"Top match: {top['name']} - {top['match_score']}% match")
        st.write(f"**Matched skills:** {top['skills']}")
        st.write(
            f"**Score breakdown:** semantic {top['semantic_%']}% / "
            f"skill overlap {top['skill_overlap_%']}% / "
            f"experience {top['experience_%']}%"
        )
        st.write(f"**Contact:** {top['email']} / {top['phone']}")
        st.write(f"**Resume summary:** {top['summary']}")
