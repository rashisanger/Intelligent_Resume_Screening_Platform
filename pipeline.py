import asyncio
from parser import extract_text_from_pdf_async
from extractor import extract_skills, extract_experience_years, extract_contact_info
from matcher import compute_match_score_async
from summarizer import summarize_resume_async


async def process_one_resume(path, filename, on_done=None):
    """
    Runs the full pipeline for ONE resume: parse -> extract -> score -> summarize.
    """
    text = await extract_text_from_pdf_async(path)

    skills = extract_skills(text)
    years = extract_experience_years(text)
    contact = extract_contact_info(text)

    score_result, summary = await asyncio.gather(
        compute_match_score_async(text, skills, years),
        summarize_resume_async(text),
    )

    if on_done:
        on_done(filename)

    return {
        "name": filename,
        "match_score": score_result["match_score"],
        "semantic_score": score_result["semantic_score"],
        "skill_score": score_result["skill_score"],
        "experience_score": score_result["experience_score"],
        "skills": ", ".join(skills) if skills else "None found",
        "experience_years": years,
        "email": contact["email"],
        "summary": summary,
    }


async def process_all_resumes(files, on_done=None):
    tasks = [process_one_resume(path, name, on_done) for path, name in files]
    return await asyncio.gather(*tasks)