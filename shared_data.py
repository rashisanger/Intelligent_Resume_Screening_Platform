# shared_data.py
"""
Centralised data dictionaries for the Intelligent Resume Screening Platform.

Contains:
  - SKILL_KEYWORDS     — categorised skill matching patterns → display names
  - EXPERIENCE_PATTERNS — regex patterns for extracting years of experience
  - EDUCATION_KEYWORDS  — degree/qualification patterns and their level scores
  - SECTION_HEADERS     — common resume section headings for text segmentation
"""

# ---------------------------------------------------------------------------
# Job matching inputs
# ---------------------------------------------------------------------------

JOB_DESCRIPTION = """
We are looking for a Software Engineer with experience in Python,
Java, SQL, REST APIs, Git, Linux, Docker, AWS and problem solving.
Experience with Machine Learning is a plus.
"""

REQUIRED_EXPERIENCE_YEARS = 2


# ---------------------------------------------------------------------------
# Skill Keywords
# ---------------------------------------------------------------------------
# Maps each category to a dict of { "match_pattern": "Display Name" }.
# - match_pattern: the lowercased phrase to search for in resume text.
# - Display Name: the properly-cased name shown in results.
#
# Rules for adding new skills:
#   1. match_pattern must be lowercase.
#   2. Avoid single common English words (e.g. "go", "c", "r") — use
#      disambiguated forms like "golang", "c programming", "r language".
#   3. Keep Display Name in its canonical casing (e.g. "AWS", "Node.js").

SKILL_KEYWORDS = {
    "Programming Languages": {
        "python": "Python",
        "java": "Java",
        "c programming": "C",
        "c++": "C++",
        "c#": "C#",
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "golang": "Go",
        "rust": "Rust",
        "php": "PHP",
        "ruby": "Ruby",
        "swift": "Swift",
        "kotlin": "Kotlin",
        "scala": "Scala",
        "perl": "Perl",
        "r language": "R",
        "matlab": "MATLAB",
        "dart": "Dart",
    },

    "Web Development": {
        "html": "HTML",
        "css": "CSS",
        "react": "React",
        "angular": "Angular",
        "vue": "Vue",
        "next.js": "Next.js",
        "nuxt": "Nuxt",
        "svelte": "Svelte",
        "node.js": "Node.js",
        "express": "Express",
        "flask": "Flask",
        "django": "Django",
        "fastapi": "FastAPI",
        "spring boot": "Spring Boot",
        "ruby on rails": "Ruby on Rails",
        "graphql": "GraphQL",
        "rest api": "REST API",
        "bootstrap": "Bootstrap",
        "tailwind": "Tailwind CSS",
        "sass": "SASS",
        "webpack": "Webpack",
    },

    "Databases": {
        "sql": "SQL",
        "mysql": "MySQL",
        "postgresql": "PostgreSQL",
        "mongodb": "MongoDB",
        "sqlite": "SQLite",
        "oracle": "Oracle",
        "redis": "Redis",
        "cassandra": "Cassandra",
        "elasticsearch": "Elasticsearch",
        "dynamodb": "DynamoDB",
        "firebase": "Firebase",
    },

    "Cloud & DevOps": {
        "aws": "AWS",
        "azure": "Azure",
        "gcp": "GCP",
        "docker": "Docker",
        "kubernetes": "Kubernetes",
        "jenkins": "Jenkins",
        "terraform": "Terraform",
        "ansible": "Ansible",
        "linux": "Linux",
        "git": "Git",
        "github": "GitHub",
        "gitlab": "GitLab",
        "ci/cd": "CI/CD",
        "nginx": "Nginx",
        "heroku": "Heroku",
        "vercel": "Vercel",
    },

    "Data Science & AI": {
        "numpy": "NumPy",
        "pandas": "Pandas",
        "matplotlib": "Matplotlib",
        "seaborn": "Seaborn",
        "scikit-learn": "Scikit-learn",
        "tensorflow": "TensorFlow",
        "keras": "Keras",
        "pytorch": "PyTorch",
        "machine learning": "Machine Learning",
        "deep learning": "Deep Learning",
        "natural language processing": "NLP",
        "nlp": "NLP",
        "computer vision": "Computer Vision",
        "data analysis": "Data Analysis",
        "data visualization": "Data Visualization",
        "statistics": "Statistics",
        "opencv": "OpenCV",
        "hugging face": "Hugging Face",
        "langchain": "LangChain",
        "llm": "LLM",
    },

    "Analytics & Tools": {
        "power bi": "Power BI",
        "tableau": "Tableau",
        "excel": "Excel",
        "looker": "Looker",
        "google analytics": "Google Analytics",
        "jupyter": "Jupyter",
    },

    "Soft Skills & Methodologies": {
        "agile": "Agile",
        "scrum": "Scrum",
        "jira": "Jira",
        "kanban": "Kanban",
        "leadership": "Leadership",
        "communication": "Communication",
        "problem solving": "Problem Solving",
        "team management": "Team Management",
        "project management": "Project Management",
    },
}


# ---------------------------------------------------------------------------
# Experience Patterns
# ---------------------------------------------------------------------------
# Regex patterns (case-insensitive) for detecting years of experience.
# Each captures a numeric group for the number of years.

EXPERIENCE_PATTERNS = [
    r"(\d+)\+?\s*(?:years?|yrs?)[\s\w]*(?:of\s+)?(?:experience|exp)",
    r"(?:experience|exp)\s*(?:of\s+)?(\d+)\+?\s*(?:years?|yrs?)",
    r"(\d+)\+?\s*(?:years?|yrs?)\s+(?:in|of|working)",
    r"over\s+(\d+)\s+(?:years?|yrs?)",
    r"(\d+)\+?\s*(?:years?|yrs?)\s+(?:industry|professional|hands[- ]on)",
]


# ---------------------------------------------------------------------------
# Education Keywords
# ---------------------------------------------------------------------------
# Maps match patterns (lowercase) to a dict with:
#   - display: the canonical display name
#   - level:   numeric score used by the ranker (higher = more advanced)

EDUCATION_KEYWORDS = {
    "ph.d":           {"display": "Ph.D.",            "level": 5},
    "phd":            {"display": "Ph.D.",            "level": 5},
    "doctorate":      {"display": "Doctorate",        "level": 5},
    "master":         {"display": "Master's Degree",  "level": 4},
    "m.s.":           {"display": "M.S.",             "level": 4},
    "m.sc":           {"display": "M.Sc.",            "level": 4},
    "msc":            {"display": "M.Sc.",            "level": 4},
    "m.tech":         {"display": "M.Tech",           "level": 4},
    "mtech":          {"display": "M.Tech",           "level": 4},
    "mba":            {"display": "MBA",              "level": 4},
    "m.e.":           {"display": "M.E.",             "level": 4},
    "mca":            {"display": "MCA",              "level": 4},
    "bachelor":       {"display": "Bachelor's Degree","level": 3},
    "b.s.":           {"display": "B.S.",             "level": 3},
    "b.sc":           {"display": "B.Sc.",            "level": 3},
    "bsc":            {"display": "B.Sc.",            "level": 3},
    "b.tech":         {"display": "B.Tech",           "level": 3},
    "btech":          {"display": "B.Tech",           "level": 3},
    "b.e.":           {"display": "B.E.",             "level": 3},
    "bca":            {"display": "BCA",              "level": 3},
    "b.com":          {"display": "B.Com",            "level": 3},
    "diploma":        {"display": "Diploma",          "level": 2},
    "associate":      {"display": "Associate Degree", "level": 2},
    "certification":  {"display": "Certification",    "level": 1},
    "certificate":    {"display": "Certificate",      "level": 1},
}


# ---------------------------------------------------------------------------
# Section Headers
# ---------------------------------------------------------------------------
# Common resume section headings (lowercase).  Used by the extractor to
# locate and segment resume text into meaningful blocks.

SECTION_HEADERS = {
    "experience":       ["experience", "work experience", "professional experience",
                         "employment history", "work history", "career history"],
    "education":        ["education", "academic background", "qualifications",
                         "academic qualifications", "educational background"],
    "skills":           ["skills", "technical skills", "core competencies",
                         "key skills", "competencies", "technologies",
                         "tools & technologies", "tools and technologies"],
    "projects":         ["projects", "personal projects", "academic projects",
                         "key projects", "notable projects"],
    "certifications":   ["certifications", "certificates", "professional certifications",
                         "licenses & certifications"],
    "summary":          ["summary", "professional summary", "profile",
                         "about me", "objective", "career objective"],
}
