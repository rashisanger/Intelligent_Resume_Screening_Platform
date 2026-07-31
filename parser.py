import fitz
import asyncio

def extract_text_from_pdf(file_path):
    """
    Takes a path to a PDF resume, returns its full text as a string.
    """
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

async def extract_text_from_pdf_async(file_path):
    """
    Same as above, but runs in the background so the app doesn't freeze
    while a big PDF is being read. asyncio.to_thread just runs our normal
    function on a separate thread and lets us 'await' the result.
    """
    return await asyncio.to_thread(extract_text_from_pdf, file_path)

if __name__ == "__main__":
    sample_text = extract_text_from_pdf("resumes/sample1.pdf")
    print(sample_text[:500])