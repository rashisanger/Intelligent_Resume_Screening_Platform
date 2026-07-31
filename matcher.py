from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import streamlit as st
except ImportError:
    class _StreamlitCacheShim:
        def cache_resource(self, func=None, **_kwargs):
            def decorator(inner_func):
                return inner_func

            return decorator(func) if func else decorator

    st = _StreamlitCacheShim()

from extractor import extract_skills
from shared_data import JOB_DESCRIPTION, REQUIRED_EXPERIENCE_YEARS


logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
SUMMARIZER_MODEL_NAME = "t5-small"
DEFAULT_SEMANTIC_WEIGHT = 0.50
DEFAULT_SKILL_WEIGHT = 0.35
DEFAULT_EXPERIENCE_WEIGHT = 0.15


class MatcherError(Exception):
    """Base exception for matcher-related failures."""


class InvalidInputError(MatcherError):
    """Raised when resume text or job description is missing/invalid."""


class ModelLoadError(MatcherError):
    """Raised when an optional ML model cannot be loaded."""


@dataclass
class MatchResult:
    match_score: float
    semantic_score: float
    skill_overlap_score: float
    experience_score: float = 0.0
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "match_score": self.match_score,
            "semantic_score": self.semantic_score,
            "skill_overlap_score": self.skill_overlap_score,
            "experience_score": self.experience_score,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
        }


@st.cache_resource
def load_model(model_name: str = DEFAULT_MODEL_NAME):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ModelLoadError(
            "sentence-transformers is not installed. "
            "Run `pip install sentence-transformers`."
        ) from exc

    try:
        logger.info("Loading sentence-transformer model '%s'...", model_name)
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load sentence-transformer model '{model_name}': {exc}"
        ) from exc


@st.cache_resource
def get_job_description_embedding(
    jd_text: str,
    model_name: str = DEFAULT_MODEL_NAME,
):
    """Encode one JD text and reuse it for every resume scored against it."""
    model = load_model(model_name)
    return model.encode(jd_text, convert_to_numpy=True)


@st.cache_resource
def get_summarizer():
    try:
        from transformers import pipeline
    except ImportError:
        return None

    try:
        return pipeline(
            "summarization",
            model=SUMMARIZER_MODEL_NAME,
            tokenizer=SUMMARIZER_MODEL_NAME,
        )
    except Exception:
        return None


def embed_texts(model_name: str, jd_text: str, resume_text: str):
    model = load_model(model_name)
    jd_embedding = get_job_description_embedding(jd_text, model_name)
    resume_embedding = model.encode(resume_text, convert_to_numpy=True)
    return jd_embedding, resume_embedding


def validate_text(value: Optional[str], field_name: str) -> str:
    if value is None or not isinstance(value, str) or not value.strip():
        raise InvalidInputError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _effective_job_description(jd_text: Optional[str] = None) -> str:
    cleaned = (jd_text or "").strip()
    return cleaned or JOB_DESCRIPTION.strip()


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-]*", (text or "").lower()))


def _to_float_list(vector: Sequence[float]) -> List[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    if vector and isinstance(vector[0], list):
        vector = vector[0]
    return [float(value) for value in vector]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_values = _to_float_list(left)
    right_values = _to_float_list(right)
    if not left_values or not right_values:
        return 0.0

    size = min(len(left_values), len(right_values))
    left_values = left_values[:size]
    right_values = right_values[:size]

    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(a * a for a in left_values))
    right_norm = math.sqrt(sum(b * b for b in right_values))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def clamp_percentage(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 2)


def _lexical_semantic_score(resume_text: str, job_description: str) -> float:
    jd_tokens = _tokens(job_description)
    resume_tokens = _tokens(resume_text)
    if not jd_tokens or not resume_tokens:
        return 0.0
    return clamp_percentage(100 * len(jd_tokens & resume_tokens) / len(jd_tokens))


def compute_semantic_score(
    resume_text: str,
    job_description: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> float:
    resume_text = validate_text(resume_text, "resume_text")
    job_description = validate_text(job_description, "job_description")

    try:
        jd_embedding, resume_embedding = embed_texts(
            model_name,
            job_description,
            resume_text,
        )
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            similarity = cosine_similarity([jd_embedding], [resume_embedding])[0][0]
        except ImportError:
            similarity = _cosine_similarity(jd_embedding, resume_embedding)
        return clamp_percentage(float(similarity) * 100)
    except Exception as exc:
        logger.info("Falling back to lexical semantic score: %s", exc)
        return _lexical_semantic_score(resume_text, job_description)


def compute_skill_overlap(resume_text: str, job_description: str) -> Dict[str, object]:
    resume_text = validate_text(resume_text, "resume_text")
    job_description = validate_text(job_description, "job_description")

    jd_skills = set(extract_skills(job_description))
    resume_skills = set(extract_skills(resume_text))

    if not jd_skills:
        return {
            "score": 0.0,
            "matched": [],
            "missing": [],
            "undefined": True,
        }

    matched = sorted(jd_skills & resume_skills)
    missing = sorted(jd_skills - resume_skills)
    overlap_score = clamp_percentage(100 * len(matched) / len(jd_skills))
    return {
        "score": overlap_score,
        "matched": matched,
        "missing": missing,
        "undefined": False,
    }


def _skill_overlap_score(
    resume_skills: Iterable[str],
    job_description: str,
) -> float:
    jd_skills = {skill.lower() for skill in extract_skills(job_description)}
    candidate_skills = {str(skill).lower() for skill in (resume_skills or [])}
    if not jd_skills:
        return 0.0
    return clamp_percentage(100 * len(jd_skills & candidate_skills) / len(jd_skills))


def _experience_score(experience_years: float) -> float:
    if REQUIRED_EXPERIENCE_YEARS <= 0:
        return 100.0
    try:
        years = float(experience_years)
    except (TypeError, ValueError):
        years = 0.0
    return clamp_percentage(100 * years / REQUIRED_EXPERIENCE_YEARS)


def compute_match_score(
    resume_text: str,
    resume_skills: Optional[Iterable[str]] = None,
    experience_years: float = 0,
    jd_text: Optional[str] = None,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Tuple[float, Dict[str, float]]:
    """Return final match score plus semantic/skills/experience breakdown."""
    if isinstance(resume_skills, str) and jd_text is None and experience_years == 0:
        jd_text = resume_skills
        resume_skills = None

    effective_jd = _effective_job_description(jd_text)
    resume_text = resume_text or ""
    candidate_skills = list(resume_skills or extract_skills(resume_text))

    try:
        semantic = compute_semantic_score(resume_text, effective_jd, model_name)
    except InvalidInputError:
        semantic = 0.0

    skill_overlap = _skill_overlap_score(candidate_skills, effective_jd)
    experience = _experience_score(experience_years)

    score = round(
        (semantic * DEFAULT_SEMANTIC_WEIGHT)
        + (skill_overlap * DEFAULT_SKILL_WEIGHT)
        + (experience * DEFAULT_EXPERIENCE_WEIGHT),
        2,
    )
    breakdown = {
        "semantic": semantic,
        "skill_overlap": skill_overlap,
        "experience": experience,
    }
    return score, breakdown


def summarize_resume(text: str) -> str:
    """Summarize a resume, using T5 when available and a simple fallback otherwise."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return "No resume text found."

    summarizer = get_summarizer()
    if summarizer is not None:
        try:
            summary = summarizer(
                "summarize: " + cleaned[:2000],
                max_length=80,
                min_length=25,
                do_sample=False,
            )
            if summary and summary[0].get("summary_text"):
                return summary[0]["summary_text"].strip()
        except Exception:
            pass

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    fallback = " ".join(sentence for sentence in sentences[:2] if sentence)
    if fallback:
        return fallback[:450]
    return " ".join(cleaned.split()[:70])


def rank_candidates(candidates: List[Dict], *, score_key: str = "match_score") -> List[Dict]:
    def _safe_score(candidate: Dict) -> float:
        value = candidate.get(score_key)
        if isinstance(value, (int, float)):
            return float(value)
        logger.warning(
            "Candidate missing/invalid '%s' (%r); treating as 0.",
            score_key,
            candidate.get("name", candidate),
        )
        return 0.0

    return sorted(candidates, key=_safe_score, reverse=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from extractor import extract_experience_years
    from parser import extract_text_from_pdf

    resume = extract_text_from_pdf("resumes/sample1.pdf")
    skills = extract_skills(resume)
    experience = extract_experience_years(resume)
    score, score_breakdown = compute_match_score(resume, skills, experience)

    print(f"JD Match Score: {score}%")
    print(score_breakdown)
