"""
app.py
------
AI-Powered Interview Prep Coach
Main Streamlit application.

Run with: streamlit run app.py
"""

import os

import streamlit as st

from resume_parser import extract_text_from_pdf
import ai_engine

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Interview Prep Coach — IBM watsonx.ai",
    page_icon="🎤",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
defaults = {
    "resume_text": "",
    "jd_text": "",
    "company_name": "",
    "questions": [],
    "current_q_index": 0,
    "qa_log": [],  # list of dicts: {question, answer_text, feedback}
    "roadmap_data": None,
    "current_answer_text": None,   # typed answer for the CURRENT question, or None if not answered yet
    "current_feedback": None,      # feedback dict for the CURRENT question, or None if not answered yet
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_interview():
    st.session_state.questions = []
    st.session_state.current_q_index = 0
    st.session_state.qa_log = []
    st.session_state.current_answer_text = None
    st.session_state.current_feedback = None


def go_to_next_question():
    """Saves the current Q&A into the log, then advances the index and
    clears per-question state so the next question starts fresh."""
    if st.session_state.current_feedback is not None:
        idx = st.session_state.current_q_index
        st.session_state.qa_log.append(
            {
                "question": st.session_state.questions[idx],
                "answer_text": st.session_state.current_answer_text,
                "feedback": st.session_state.current_feedback,
            }
        )
    st.session_state.current_q_index += 1
    st.session_state.current_answer_text = None
    st.session_state.current_feedback = None


# ---------------------------------------------------------------------------
# Sidebar: Setup inputs (shared across both tabs)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📋 Setup")

    if not os.getenv("WATSONX_API_KEY") or not os.getenv("WATSONX_PROJECT_ID"):
        st.error(
            "IBM Cloud credentials not found. Add WATSONX_API_KEY and "
            "WATSONX_PROJECT_ID to a .env file in the project root (see "
            ".env.example)."
        )
    st.caption("Powered by IBM watsonx.ai · IBM Granite Foundation Model")

    resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    if resume_file is not None:
        with st.spinner("Reading resume..."):
            extracted = extract_text_from_pdf(resume_file)
        if extracted:
            st.session_state.resume_text = extracted
            st.success(f"Resume loaded ({len(extracted.split())} words)")
        else:
            st.warning("Couldn't extract text from this PDF. It may be a scanned image.")

    st.session_state.jd_text = st.text_area(
        "Paste the Job Description",
        value=st.session_state.jd_text,
        height=180,
        placeholder="Paste the full job description here...",
    )

    st.session_state.company_name = st.text_input(
        "Target company (optional)",
        value=st.session_state.company_name,
        placeholder="e.g. Google, Infosys, a startup name...",
    )

    st.caption(
        "Company-specific flavoring uses the AI's general knowledge. "
        "Results are stronger for well-known companies."
    )

    ready = bool(st.session_state.resume_text and st.session_state.jd_text.strip())
    if not ready:
        st.info("Upload a resume and paste a job description to get started.")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_interview, tab_roadmap = st.tabs(["🎤 Mock Interview", "🗺️ Skill Roadmap"])

# ===========================================================================
# TAB 1: MOCK INTERVIEW
# ===========================================================================
with tab_interview:
    st.title("🎤 AI Mock Interview")

    col1, col2 = st.columns(2)
    with col1:
        interview_type = st.selectbox(
            "Interview type", ["Behavioral", "Technical", "HR"]
        )
    with col2:
        difficulty = st.selectbox(
            "Difficulty level", ["Beginner", "Intermediate", "Advanced"]
        )

    num_questions = st.slider("Number of questions", min_value=3, max_value=8, value=5)

    if not ready:
        st.warning("Please complete setup in the sidebar first (resume + job description).")
    else:
        if st.button("🚀 Generate Interview Questions", type="primary"):
            with st.spinner("Generating tailored questions..."):
                questions = ai_engine.generate_questions(
                    resume_text=st.session_state.resume_text,
                    jd_text=st.session_state.jd_text,
                    company_name=st.session_state.company_name,
                    interview_type=interview_type,
                    difficulty=difficulty,
                    num_questions=num_questions,
                )
            if questions:
                reset_interview()
                st.session_state.questions = questions
                st.rerun()
            else:
                st.error("Couldn't generate questions. Please try again.")

    st.divider()

    # ------------------------------------------------------------------
    # Active interview flow
    # ------------------------------------------------------------------
    if st.session_state.questions:
        idx = st.session_state.current_q_index
        total = len(st.session_state.questions)

        if idx < total:
            st.subheader(f"Question {idx + 1} of {total}")
            current_question = st.session_state.questions[idx]
            st.markdown(f"### {current_question}")

            # If we haven't got feedback yet for this question, show the answer box
            if st.session_state.current_feedback is None:
                st.write("✍️ Type your answer:")
                typed_answer = st.text_area(
                    "Your answer",
                    key=f"answer_{idx}",
                    height=180,
                    label_visibility="collapsed",
                    placeholder="Type your answer to the question above, then click Submit Answer...",
                )

                if st.button("✅ Submit Answer", key=f"submit_{idx}"):
                    answer_text = (typed_answer or "").strip()

                    if answer_text:
                        with st.spinner("Analyzing your answer..."):
                            feedback = ai_engine.analyze_answer(
                                question=current_question,
                                answer_text=answer_text,
                                jd_text=st.session_state.jd_text,
                                interview_type=interview_type,
                            )
                        # Store in session state so it survives the rerun
                        # triggered by the Next Question button below.
                        st.session_state.current_answer_text = answer_text
                        st.session_state.current_feedback = feedback
                        st.rerun()
                    else:
                        st.warning("Please type an answer before submitting.")

            # If feedback exists for this question, display it + the Next button
            if st.session_state.current_feedback is not None:
                st.markdown("**Your answer:**")
                st.info(st.session_state.current_answer_text)

                feedback = st.session_state.current_feedback

                st.markdown("---")
                st.markdown("### 📝 Feedback")

                score = feedback.get("relevance_score")
                if score is not None:
                    st.metric("Relevance Score", f"{score}/10")

                fcol1, fcol2 = st.columns(2)
                with fcol1:
                    st.markdown(f"**Structure:** {feedback.get('structure_feedback', 'N/A')}")
                    st.markdown(f"**Strengths:** {feedback.get('strengths', 'N/A')}")
                with fcol2:
                    st.markdown(f"**Filler words/hesitation:** {feedback.get('filler_word_flag', 'N/A')}")
                    st.markdown(f"**Pacing:** {feedback.get('pacing_note', 'N/A')}")

                st.markdown(f"**💡 Suggestion:** {feedback.get('improvement_suggestion', 'N/A')}")

                st.button("➡️ Next Question", type="primary", on_click=go_to_next_question)
        else:
            st.success("🎉 Interview complete! Here's a summary of all your answers:")
            for i, entry in enumerate(st.session_state.qa_log, start=1):
                with st.expander(f"Q{i}: {entry['question']}"):
                    st.markdown(f"**Your answer:** {entry['answer_text']}")
                    fb = entry["feedback"]
                    st.markdown(f"**Score:** {fb.get('relevance_score', 'N/A')}/10")
                    st.markdown(f"**Suggestion:** {fb.get('improvement_suggestion', 'N/A')}")

            if st.button("🔄 Start a New Interview"):
                reset_interview()
                st.rerun()

# ===========================================================================
# TAB 2: SKILL ROADMAP
# ===========================================================================
with tab_roadmap:
    st.title("🗺️ Resume-to-Roadmap: Skill Gap Analysis")
    st.write(
        "Compares your resume against the job description to identify missing "
        "skills and suggests a focused study plan."
    )

    if not ready:
        st.warning("Please complete setup in the sidebar first (resume + job description).")
    else:
        if st.button("🔍 Analyze Skill Gaps", type="primary"):
            with st.spinner("Analyzing your resume against the job description..."):
                st.session_state.roadmap_data = ai_engine.generate_roadmap(
                    resume_text=st.session_state.resume_text,
                    jd_text=st.session_state.jd_text,
                )

        if st.session_state.roadmap_data:
            data = st.session_state.roadmap_data

            matched = data.get("matched_skills", [])
            gaps = data.get("gap_skills", [])

            if matched:
                st.markdown("### ✅ Skills You Already Have")
                st.write(", ".join(matched))

            if gaps:
                st.markdown("### 📚 Skill Gaps & Learning Roadmap")
                for i, gap in enumerate(gaps, start=1):
                    with st.expander(f"{i}. {gap.get('skill', 'Unknown skill')}  —  ⏱️ {gap.get('estimated_time', 'N/A')}"):
                        st.markdown(f"**Why it matters:** {gap.get('why_it_matters', 'N/A')}")
                        topics = gap.get("topics_to_study", [])
                        if topics:
                            st.markdown("**Topics to study:**")
                            for t in topics:
                                st.markdown(f"- {t}")
            elif not matched:
                st.info("No analysis yet, or the AI response couldn't be parsed. Try again.")
            else:
                st.success("No major skill gaps found — your resume covers the job description well!")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "🔷 Built on IBM Cloud — powered by IBM watsonx.ai and the IBM Granite "
    "Foundation Model. "
    "AICTE IBM SkillsBuild Internship project."
)