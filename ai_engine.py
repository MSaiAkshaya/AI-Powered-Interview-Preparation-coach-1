"""
ai_engine.py
------------
All AI/LLM logic for the Interview Prep Coach lives here, separate from the
Streamlit UI. This keeps prompts easy to find/tune and makes the app easier
to explain in a project report (clear separation of concerns).

IBM CLOUD EDITION
-----------------
This version is built for the AICTE IBM SkillsBuild Internship and uses:
  * IBM watsonx.ai              -> hosted foundation-model inference
  * IBM Granite Foundation Model -> the LLM that generates questions,
                                     analyzes answers, and builds roadmaps
  * IBM Cloud Project (project_id) -> the watsonx.ai project that scopes
                                       every inference call
  * IBM Cloud API Key           -> authenticates every request (IAM)

All of the above are configured through environment variables (see
.env.example) and are never hard-coded.
"""

import os
import json
import re

from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

load_dotenv()

# ---------------------------------------------------------------------------
# IBM Cloud / watsonx.ai configuration
# ---------------------------------------------------------------------------
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

# IBM Granite Foundation Model served through watsonx.ai.
# ibm/granite-3-8b-instruct is the current general-purpose instruct model;
# override via .env if your project has access to a different Granite variant
# (e.g. ibm/granite-3-2-8b-instruct, ibm/granite-13b-instruct-v2).
MODEL_ID = os.getenv(
    "MODEL_ID",
    "meta-llama/llama-3-3-70b-instruct")

_credentials = None
_model = None


def _get_credentials():
    """Builds (and caches) the IBM Cloud credentials object used for every
    watsonx.ai call. Raises a clear error if the API key is missing so the
    app can surface a helpful message instead of a raw SDK traceback."""
    global _credentials
    if _credentials is None:
        if not WATSONX_API_KEY:
            raise RuntimeError(
                "WATSONX_API_KEY is not set. Add your IBM Cloud API key to the "
                ".env file (see .env.example)."
            )
        _credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    return _credentials


def _get_model():
    """Builds (and caches) the ModelInference client bound to the IBM Granite
    Foundation Model and the IBM Cloud (watsonx.ai) project."""
    global _model
    if _model is None:
        if not WATSONX_PROJECT_ID:
            raise RuntimeError(
                "WATSONX_PROJECT_ID is not set. Add your watsonx.ai IBM Cloud "
                "project ID to the .env file (see .env.example)."
            )
        _model = ModelInference(
        model_id=MODEL_ID,
        credentials=_get_credentials(),
        project_id=WATSONX_PROJECT_ID,
        )
    return _model


# ---------------------------------------------------------------------------
# Helper: robust JSON extraction
# ---------------------------------------------------------------------------
def _extract_json(text: str):
    """
    LLMs sometimes wrap JSON in markdown fences or add stray text.
    This pulls out the first {...} or [...] block and parses it.
    """
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        text = match.group(1)

    return json.loads(text)


def _chat(messages, temperature=0.6, max_tokens=2048):
    """
    Sends a chat-style conversation to the IBM Granite model hosted on
    watsonx.ai and returns the generated text.

    `messages` follows the familiar [{"role": "system"/"user", "content": ...}]
    shape used throughout this file, and is translated internally into the
    parameters watsonx.ai's chat API expects.
    """
    model = _get_model()

    params = {
        GenParams.DECODING_METHOD: "greedy" if temperature == 0 else "sample",
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.TEMPERATURE: temperature,
    }

    response = model.chat(messages=messages, params=params)

    # ibm-watsonx-ai returns an OpenAI-style dict:
    # {"choices": [{"message": {"content": "..."}}], ...}
    return response["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# 1. Question Generation
# ---------------------------------------------------------------------------
def generate_questions(resume_text, jd_text, company_name, interview_type, difficulty, num_questions=5):
    """
    Generates tailored interview questions based on resume, job description,
    target company, interview type, and difficulty level.

    Returns: list of question strings.
    """
    company_clause = (
        f"The candidate is interviewing at '{company_name}'. Flavor the questions "
        f"to match the style, values, and typical interview approach you associate "
        f"with that company, based on your general knowledge. If you are not confident "
        f"about specifics of this company, fall back gracefully to strong general "
        f"questions for this role/industry without inventing false facts."
        if company_name and company_name.strip()
        else "No specific company was provided, so use strong general questions for this role/industry."
    )

    system_prompt = f"""You are an expert technical and behavioral interview coach.
Generate exactly {num_questions} interview questions based on the candidate's resume
and the job description below.

Interview type requested: {interview_type}
Difficulty level requested: {difficulty}

Rules:
- If interview type is "Technical": focus on role-relevant technical/problem-solving questions.
- If interview type is "Behavioral": focus on past experience, STAR-style situational questions.
- If interview type is "HR": focus on motivation, culture fit, career goals, salary/logistics-adjacent (non-numeric) questions.
- Adjust depth to the difficulty level (Beginner = conceptual/foundational, Intermediate = applied scenarios, Advanced = edge cases/system-level/leadership depth).
- {company_clause}
- Base questions on specific skills/projects mentioned in the resume where relevant, and on requirements in the job description.

Output STRICT JSON only, in this exact format, with no extra commentary:
{{"questions": ["question 1", "question 2", ...]}}
"""

    user_prompt = f"""RESUME:
{resume_text[:6000]}

JOB DESCRIPTION:
{jd_text[:4000]}
"""

    raw = _chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )

    try:
        data = _extract_json(raw)
        return data.get("questions", [])
    except (json.JSONDecodeError, AttributeError):
        # Fallback: split by lines if JSON parsing fails, so the app never hard-crashes
        lines = [l.strip("-• ").strip() for l in raw.split("\n") if l.strip()]
        return lines[:num_questions]


# ---------------------------------------------------------------------------
# 2. Answer Feedback
# ---------------------------------------------------------------------------
def analyze_answer(question, answer_text, jd_text, interview_type):
    """
    Analyzes a typed answer and returns structured feedback:
    - content feedback (relevance, structure/STAR if behavioral)
    - filler word / hesitation flag
    - pacing note (based on word count)
    - suggested improvement
    """
    word_count = len(answer_text.split())

    system_prompt = f"""You are an expert interview coach analyzing a candidate's written
answer to a mock interview question. The interview type is "{interview_type}".

Evaluate the answer below and output STRICT JSON only, in this exact format:
{{
  "relevance_score": <integer 1-10, how well the answer addresses the question and aligns with the job description>,
  "structure_feedback": "<1-2 sentences on structure; mention STAR method if interview_type is Behavioral and structure is missing/present>",
  "filler_word_flag": "<note if answer has noticeable hesitation markers like 'um', 'like', 'I think', 'maybe', 'basically' -- or say 'None noticeable' if clean>",
  "pacing_note": "<based on word count ({word_count} words), comment on whether the answer seems too short, too long, or well-scoped for this type of question. Typical good behavioral answers are 100-200 words.>",
  "strengths": "<1-2 sentences, what the candidate did well>",
  "improvement_suggestion": "<concrete, specific suggestion to improve this answer>"
}}

No commentary outside the JSON.
"""

    user_prompt = f"""JOB DESCRIPTION CONTEXT:
{jd_text[:2000]}

QUESTION:
{question}

CANDIDATE'S WRITTEN ANSWER:
{answer_text}
"""

    raw = _chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    try:
        return _extract_json(raw)
    except (json.JSONDecodeError, AttributeError):
        return {
            "relevance_score": None,
            "structure_feedback": "Could not parse structured feedback.",
            "filler_word_flag": "N/A",
            "pacing_note": f"Answer was {word_count} words.",
            "strengths": "N/A",
            "improvement_suggestion": raw[:500],
        }


# ---------------------------------------------------------------------------
# 3. Resume-to-Roadmap (Skill Gap Analysis)
# ---------------------------------------------------------------------------
def generate_roadmap(resume_text, jd_text):
    """
    Compares resume skills against JD requirements and produces a structured
    learning roadmap for the missing/weak skills.

    Returns dict: {
        "matched_skills": [...],
        "gap_skills": [
            {"skill": ..., "why_it_matters": ..., "topics_to_study": [...], "estimated_time": ...},
            ...
        ]
    }
    """
    system_prompt = """You are a career coach performing a skill-gap analysis between a
candidate's resume and a target job description.

Steps:
1. Identify key skills/technologies/competencies required by the job description.
2. Identify which of those are clearly present in the resume (matched_skills).
3. Identify which are missing or weakly represented (gap_skills).
4. For each gap skill, suggest what to study and a realistic estimated time to get
   interview-ready (not mastery) in it.

Output STRICT JSON only, in this exact format:
{
  "matched_skills": ["skill1", "skill2", ...],
  "gap_skills": [
    {
      "skill": "skill name",
      "why_it_matters": "1 sentence on why the JD needs this",
      "topics_to_study": ["topic1", "topic2", "topic3"],
      "estimated_time": "e.g. '3-5 days' or '1-2 weeks'"
    }
  ]
}

Limit gap_skills to the 6 most important gaps, ordered by priority (most important first).
No commentary outside the JSON.
"""

    user_prompt = f"""RESUME:
{resume_text[:6000]}

JOB DESCRIPTION:
{jd_text[:4000]}
"""

    raw = _chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=2048,
    )

    try:
        return _extract_json(raw)
    except (json.JSONDecodeError, AttributeError):
        return {"matched_skills": [], "gap_skills": [], "_raw_fallback": raw}
