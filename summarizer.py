import os
import asyncio
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"  

def summarize_resume(resume_text):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize this resume in 2-3 short sentences for a recruiter: "
                    "mention the candidate's role, key skills, and years of experience. "
                    "No headers, no bullet points, just plain sentences."
                ),
            },
            {"role": "user", "content": resume_text[:6000]},
        ],
        temperature=0.3,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


async def summarize_resume_async(resume_text):
    try:
        return await asyncio.to_thread(summarize_resume, resume_text)
    except Exception as e:
        return f"(Could not generate summary: {e})"

if __name__ == "__main__":
    from parser import extract_text_from_pdf
    text = extract_text_from_pdf("resumes/sample1.pdf")
    print("Summary:", summarize_resume(text))