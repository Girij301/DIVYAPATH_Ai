import re

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


def extract_text(uploaded_file):
    """
    Accepts a Flask uploaded file (PDF or TXT)
    Returns cleaned plain text of the resume
    """
    text = ""

    # Check file type
    filename = uploaded_file.filename.lower()
    if filename.endswith('.pdf'):
        # PDF handling
        if pdfplumber is None:
            raise ImportError("pdfplumber is not installed. Install it to read PDFs.")
        uploaded_file.seek(0)  # Reset file pointer
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    else:
        # TXT handling
        uploaded_file.seek(0)  # Reset file pointer
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    # Basic cleaning
    text = text.lower()
    text = re.sub(r"\s+", " ", text)          # remove extra spaces
    text = re.sub(r"[^a-z0-9+.# ]", " ", text)  # keep useful chars
    text = text.strip()

    return text
