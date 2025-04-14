from flask import Flask, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
import io
import pdfplumber

# Load environment variables
load_dotenv()
client = OpenAI()  # picks up OPENAI_API_KEY from environment

app = Flask(__name__)

# Max characters per input (~3k tokens)
MAX_CHAR_LIMIT = 12000

def extract_most_relevant_jd(jd_text):
    sections = []
    lower = jd_text.lower()

    for keyword, chunk_size in [("responsibilities", 3000), ("requirements", 3000), ("summary", 2000)]:
        if keyword in lower:
            start = lower.find(keyword)
            sections.append(jd_text[start:start + chunk_size])

    extracted = "\n\n".join(sections)
    return extracted if extracted else jd_text[:MAX_CHAR_LIMIT]

def extract_text_from_base64_pdf(base64_str):
    try:
        header, encoded = base64_str.split(",", 1)
        decoded = base64.b64decode(encoded)
        with pdfplumber.open(io.BytesIO(decoded)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text
    except Exception as e:
        return f"(Failed to extract text from resume: {str(e)})"

@app.route("/analyze", methods=["POST"])
def analyze_resume():
    data = request.get_json()
    resume_raw = extract_text_from_base64_pdf(data.get("resume", ""))
    jd_raw = data.get("job_description", "")

    if not resume_raw or not jd_raw:
        return jsonify({"error": "Missing resume or job_description"}), 400

    resume = resume_raw[:MAX_CHAR_LIMIT]
    jd = extract_most_relevant_jd(jd_raw)

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
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a resume optimization assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        reply = response.choices[0].message.content
        return jsonify({"insights": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
