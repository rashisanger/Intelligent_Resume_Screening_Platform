import asyncio
from sentence_transformers import SentenceTransformer, util
from shared_data import JOB_DESCRIPTION, SKILL_KEYWORDS

# Loading takes ~1-2 minutes the first time (downloads ~80MB). If we called
# SentenceTransformer(...) inside a function, it would reload every time that
# function runs. Putting it here means it loads once when the file is
# imported, and every resume just reuses this same "model" object.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Same idea: turning the JD into an embedding is the slow part. We do it a
# single time here and reuse "jd_embedding" for every resume instead of
# re-encoding the JD again and again in a loop.
jd_embedding = model.encode(JOB_DESCRIPTION, convert_to_tensor=True)


def compute_match_score(resume_text, matched_skills, experience_years):
    """
    Combines three signals into one 0-100 score:
      - semantic similarity (70%) - does the resume "read" like a fit for the JD?
      - skill overlap        (20%) - how many required skills did we find?
      - experience           (10%) - how many years of experience, capped at 10?
    Returns a dict so the UI can show the breakdown, not just the total.
    """
    # Semantic similarity
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    similarity = util.cos_sim(jd_embedding, resume_embedding).item()  # -1 to 1
    semantic_score = max(0.0, similarity) * 100  # convert to 0-100

    # Skill overlap
    skill_score = (len(matched_skills) / len(SKILL_KEYWORDS)) * 100

    # Experience
    experience_score = min(experience_years / 10, 1.0) * 100

    total = (semantic_score * 0.70) + (skill_score * 0.20) + (experience_score * 0.10)

    return {
        "match_score": round(total, 1),
        "semantic_score": round(semantic_score, 1),
        "skill_score": round(skill_score, 1),
        "experience_score": round(experience_score, 1),
    }


async def compute_match_score_async(resume_text, matched_skills, experience_years):
    """
    Same as above, just run in the background so scoring many resumes
    doesn't block the app while each one is being encoded.
    """
    return await asyncio.to_thread(
        compute_match_score, resume_text, matched_skills, experience_years
    )


def rank_candidates(candidates):
    """
    candidates = list of dicts with at least a 'match_score' key.
    Returns the list sorted best-to-worst.
    """
    return sorted(candidates, key=lambda c: c['match_score'], reverse=True)

if __name__ == "__main__":
    from parser import extract_text_from_pdf
    from extractor import extract_skills, extract_experience_years

    text = extract_text_from_pdf("resumes/sample1.pdf")
    skills = extract_skills(text)
    years = extract_experience_years(text)
    print("Match result:", compute_match_score(text, skills, years))