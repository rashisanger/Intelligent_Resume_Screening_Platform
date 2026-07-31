# parser.py
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

def extract_text_from_pdf(file_path):
    """Takes a path to a PDF resume, returns its full text as a string."""
    if fitz is None and PdfReader is None:
        raise ImportError(
            "PDF parsing requires PyMuPDF (`pip install pymupdf`) or pypdf."
        )

    if fitz is None:
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# --- Test it standalone ---
if __name__ == "__main__":
    sample_text = extract_text_from_pdf("resumes/sample1.pdf")
    print(sample_text)
