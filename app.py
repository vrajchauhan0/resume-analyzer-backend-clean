from flask import Flask, request, jsonify
import openai
import os
from dotenv import load_dotenv
import base64
import io
import pdfplumber


load_dotenv()


app = Flask(__name__)


# Max characters per input (≈3k tokens each) to avoid rate limit errors
MAX_CHAR_LIMIT = 12000

def extract_most_relevant_jd(jd_text):
    """
    Extract key sections (Responsibilities, Requirements, Summary) from JD
    If not found, fallback to first MAX_CHAR_LIMIT characters
    """
    sections = []
    lower = jd_text.lower()

    for keyword, chunk_size in [("responsibilities", 3000), ("requirements", 3000), ("summary", 2000)]:
        if keyword in lower:
            start = lower.find(keyword)
            sections.append(jd_text[start:start + chunk_size])

    extracted = "\n\n".join(sections)
    return extracted if extracted else jd_text[:MAX_CHAR_LIMIT]

@app.route("/analyze", methods=["POST"])
def analyze_resume():
    data = request.get_json()
    def extract_text_from_base64_pdf(base64_str):
        try:
            header, encoded = base64_str.split(",", 1)  # Remove `data:application/pdf;base64,`
            decoded = base64.b64decode(encoded)
            with pdfplumber.open(io.BytesIO(decoded)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            return text
        except Exception as e:
            return f"(Failed to extract text from resume: {str(e)})"

    resume_raw = extract_text_from_base64_pdf(data.get("resume", ""))
    jd_raw = data.get("job_description", "")

    if not resume_raw or not jd_raw:
        return jsonify({"error": "Missing resume or job_description"}), 400

    # Apply trimming
    resume = resume_raw[:MAX_CHAR_LIMIT]
    jd = extract_most_relevant_jd(jd_raw)

    # Construct the prompt
    prompt = f"""
You are a resume coach helping users improve their resume.
Compare the resume content below with the job description and provide the following:
1. A match score out of 10.
2. Keywords missing in the resume.
3. Suggestions to improve the resume's alignment with the JD.

Resume:
{resume}

Job Description:
{jd}
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a resume optimization assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        reply = response['choices'][0]['message']['content']
        return jsonify({"insights": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
