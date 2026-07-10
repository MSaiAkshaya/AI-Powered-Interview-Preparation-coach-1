"""
resume_parser.py
-----------------
Extracts plain text from an uploaded resume PDF using pdfplumber.
"""

import pdfplumber


def extract_text_from_pdf(uploaded_file):
    """
    uploaded_file: a Streamlit UploadedFile object (from st.file_uploader)
    Returns: extracted plain text (str)
    """
    text_parts = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    full_text = "\n".join(text_parts).strip()

    if not full_text:
        return ""

    return full_text
