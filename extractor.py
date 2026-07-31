# extractor.py
"""
Resume data extraction module for the Intelligent Resume Screening Platform.

Uses **spaCy** (en_core_web_sm) for:
  - Named Entity Recognition (NER)  → candidate name, organisations
  - Tokenization                    → improved skill boundary detection

Also uses regex for:
  - Email & phone extraction
  - Years-of-experience patterns
  - Skill keyword matching (word-boundary safe)

Public API
----------
  extract_skills(text)             → list[str]
  extract_skills_by_category(text) → dict[str, list[str]]
  extract_name(text)               → str
  extract_email(text)              → str
  extract_phone(text)              → str
  extract_experience_years(text)   → float
  extract_education(text)          → list[dict]
  extract_full_profile(text)       → dict
"""

import re
from typing import Dict, List, Optional, Set

try:
    import spacy
except ImportError:
    spacy = None

from shared_data import (
    EDUCATION_KEYWORDS,
    EXPERIENCE_PATTERNS,
    SECTION_HEADERS,
    SKILL_KEYWORDS,
)

def _load_spacy_model():
    """Load spaCy once, falling back when the package or model is unavailable."""
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return spacy.blank("en")


# Load spaCy/tokenizer once at module level when available.
nlp = _load_spacy_model()


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase and collapse all whitespace into single spaces."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Contact information extraction
# ---------------------------------------------------------------------------

def extract_name(text: str) -> str:
    """Extract the candidate's name using spaCy NER.

    Heuristic: the first PERSON entity found in the top portion of the
    resume (first 500 characters) is very likely the candidate's name.
    Falls back to the first PERSON entity in the full text.
    """
    if not text:
        return "Unknown"

    if nlp is None:
        for line in text.splitlines():
            candidate = line.strip()
            if 2 <= len(candidate) <= 60 and not re.search(r"[@\d]", candidate):
                return candidate
        return "Unknown"

    # Check the top of the resume first (name is almost always at the top)
    top_section = text[:500]
    doc = nlp(top_section)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            # Filter out single-character or overly long entities
            if 2 <= len(name) <= 60:
                return name

    # Fallback: check full text
    doc_full = nlp(text[:2000])  # Limit to avoid slow processing
    for ent in doc_full.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            if 2 <= len(name) <= 60:
                return name

    return "Unknown"


def extract_email(text: str) -> str:
    """Extract the first email address found in the resume text."""
    if not text:
        return ""
    pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    match = re.search(pattern, text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    """Extract the first phone number found in the resume text.

    Handles common formats: +91-9876543210, (123) 456-7890, 123.456.7890, etc.
    """
    if not text:
        return ""
    patterns = [
        r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}",
        r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            phone = match.group(0).strip()
            # Must have at least 7 digits to be a real phone number
            digits = re.sub(r"\D", "", phone)
            if len(digits) >= 7:
                return phone
    return ""


def extract_contact_info(text: str) -> Dict[str, str]:
    """Return contact info in the shape app.py expects."""
    return {
        "email": extract_email(text),
        "phone": extract_phone(text),
    }


# ---------------------------------------------------------------------------
# Skill extraction
# ---------------------------------------------------------------------------

def extract_skills(text: str) -> List[str]:
    """Return a sorted list of display-name skills found in *text*.

    Uses spaCy for tokenization and word-boundary regex for matching
    against the curated SKILL_KEYWORDS dictionary.
    """
    if not text:
        return []

    cleaned = clean_text(text)

    # Use spaCy to tokenize — gives cleaner token boundaries than raw regex
    if nlp is None:
        tokenized_text = cleaned
    else:
        doc = nlp(cleaned)
        tokenized_text = " ".join(token.text for token in doc)

    detected_skills: Set[str] = set()

    for category_skills in SKILL_KEYWORDS.values():
        for match_key, display_name in category_skills.items():
            pattern = r"\b" + re.escape(match_key) + r"\b"
            if re.search(pattern, tokenized_text):
                detected_skills.add(display_name)

    return sorted(detected_skills)


def extract_skills_by_category(text: str) -> Dict[str, List[str]]:
    """Return detected skills grouped by their category.

    Categories with zero matches are omitted.
    """
    if not text:
        return {}

    cleaned = clean_text(text)
    if nlp is None:
        tokenized_text = cleaned
    else:
        doc = nlp(cleaned)
        tokenized_text = " ".join(token.text for token in doc)

    result: Dict[str, List[str]] = {}

    for category, category_skills in SKILL_KEYWORDS.items():
        matched: List[str] = []
        for match_key, display_name in category_skills.items():
            pattern = r"\b" + re.escape(match_key) + r"\b"
            if re.search(pattern, tokenized_text):
                matched.append(display_name)
        if matched:
            result[category] = sorted(matched)

    return result


# ---------------------------------------------------------------------------
# Experience extraction
# ---------------------------------------------------------------------------

def extract_experience_years(text: str) -> int:
    """Extract total years of experience from resume text.

    Scans for patterns like "5+ years of experience", "3 yrs exp", etc.
    Returns the maximum value found (candidates usually state their total).
    Returns 0 if no experience pattern is detected.
    """
    if not text:
        return 0

    cleaned = clean_text(text)
    years_found: List[float] = []

    for pattern in EXPERIENCE_PATTERNS:
        matches = re.findall(pattern, cleaned, re.IGNORECASE)
        for match in matches:
            try:
                years_found.append(float(match))
            except (ValueError, TypeError):
                continue

    return int(max(years_found)) if years_found else 0


def _find_section_text(text: str, section_key: str) -> str:
    """Extract the text belonging to a specific resume section.

    Looks for known section headers and captures text until the next
    section header or end of document.
    """
    cleaned = clean_text(text)
    headers = SECTION_HEADERS.get(section_key, [])
    if not headers:
        return ""

    # Build a pattern that matches any of this section's headers
    all_headers = []
    for header_list in SECTION_HEADERS.values():
        all_headers.extend(header_list)

    for header in headers:
        # Look for the header followed by content until the next header
        escaped_header = re.escape(header)
        # Build alternation of all OTHER headers for the stop boundary
        other_headers = [re.escape(h) for h in all_headers if h != header]
        stop_pattern = "|".join(other_headers) if other_headers else "$"

        pattern = (
            rf"(?:^|\n|\s)(?:{escaped_header})\s*[:\-]?\s*"
            rf"(.*?)"
            rf"(?=(?:{stop_pattern})|\Z)"
        )
        match = re.search(pattern, cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return ""


# ---------------------------------------------------------------------------
# Education extraction
# ---------------------------------------------------------------------------

def extract_education(text: str) -> List[Dict]:
    """Extract education qualifications from resume text.

    Returns a list of dicts, each with:
      - degree:  canonical display name (e.g. "B.Tech", "MBA")
      - level:   numeric score (1–5) for ranking
    Duplicates (same display name) are removed.
    """
    if not text:
        return []

    cleaned = clean_text(text)
    found: Dict[str, Dict] = {}  # keyed by display name to deduplicate

    for match_key, info in EDUCATION_KEYWORDS.items():
        pattern = r"\b" + re.escape(match_key) + r"\b"
        if re.search(pattern, cleaned):
            display = info["display"]
            # Keep the highest level if duplicate display names
            if display not in found or info["level"] > found[display]["level"]:
                found[display] = {
                    "degree": display,
                    "level": info["level"],
                }

    # Sort by level descending (most advanced first)
    return sorted(found.values(), key=lambda x: x["level"], reverse=True)


def get_max_education_level(text: str) -> int:
    """Return the highest education level score found in the text.

    Returns 0 if no education is detected.
    """
    education = extract_education(text)
    if not education:
        return 0
    return education[0]["level"]


# ---------------------------------------------------------------------------
# Organisation extraction (via spaCy NER)
# ---------------------------------------------------------------------------

def extract_organisations(text: str) -> List[str]:
    """Extract organisation names mentioned in the resume using spaCy NER."""
    if not text or nlp is None:
        return []

    doc = nlp(text[:3000])  # Limit to avoid slow processing
    orgs: Set[str] = set()
    for ent in doc.ents:
        if ent.label_ == "ORG":
            org_name = ent.text.strip()
            if len(org_name) >= 2:
                orgs.add(org_name)

    return sorted(orgs)


# ---------------------------------------------------------------------------
# Master profile extraction
# ---------------------------------------------------------------------------

def extract_full_profile(text: str) -> Dict:
    """Extract a complete candidate profile from resume text.

    Returns a dict containing all extracted fields — used by the
    matcher and ranker modules downstream.
    """
    if not text:
        return {
            "name": "Unknown",
            "email": "",
            "phone": "",
            "skills": [],
            "skills_by_category": {},
            "experience_years": 0,
            "education": [],
            "max_education_level": 0,
            "organisations": [],
            "raw_text": "",
        }

    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "skills_by_category": extract_skills_by_category(text),
        "experience_years": extract_experience_years(text),
        "education": extract_education(text),
        "max_education_level": get_max_education_level(text),
        "organisations": extract_organisations(text),
        "raw_text": text,
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from parser import extract_text_from_pdf

    sample_text = extract_text_from_pdf("resumes/sample1.pdf")
    profile = extract_full_profile(sample_text)

    print(f"Name:       {profile['name']}")
    print(f"Email:      {profile['email']}")
    print(f"Phone:      {profile['phone']}")
    print(f"Experience: {profile['experience_years']} years")
    print(f"Education:  {profile['education']}")
    print(f"Orgs:       {profile['organisations']}")
    print(f"\n=== Skills ({len(profile['skills'])}) ===")
    for skill in profile["skills"]:
        print(f"  - {skill}")
    print(f"\n=== Skills by Category ===")
    for cat, skills in profile["skills_by_category"].items():
        print(f"\n  [{cat}]")
        for s in skills:
            print(f"    - {s}")
