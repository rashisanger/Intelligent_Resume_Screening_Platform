import re
from rapidfuzz import fuzz
from shared_data import SKILL_KEYWORDS

FUZZY_MATCH_THRESHOLD = 82

def extract_skills(text):
    """
    Returns list of skills (from SKILL_KEYWORDS) found in the resume text.
    Uses RapidFuzz so small typos in the resume still count as a match.
    """
    text_lower = text.lower()
    words = [w.strip(".,()/:;") for w in text_lower.split()]

    found = []
    for skill in SKILL_KEYWORDS:
        if skill in text_lower:
            found.append(skill)
            continue
        for word in words:
            if fuzz.ratio(skill, word) >= FUZZY_MATCH_THRESHOLD:
                found.append(skill)
                break
    return found

def extract_experience_years(text):
    matches = re.findall(r'(\d+)\+?\s*years?', text.lower())
    years = [int(m) for m in matches]
    return max(years) if years else 0

def extract_contact_info(text):
    email = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    phone = re.findall(r'(\+?\d[\d \-]{8,}\d)', text)
    return {
        "email": email[0] if email else "Not found",
        "phone": phone[0] if phone else "Not found",
    }

if __name__ == "__main__":
    from parser import extract_text_from_pdf
    text = extract_text_from_pdf("resumes/sample1.pdf")
    print("Skills:", extract_skills(text))
    print("Experience:", extract_experience_years(text), "years")
    print("Contact:", extract_contact_info(text))