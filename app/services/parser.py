import fitz  # PyMuPDF
import docx  # python-docx
import re
import os

def clean_text(text: str) -> str:
    """
    Cleans extracted text to prepare it for the AI model.
    Removes excessive whitespace and weird non-ASCII characters.
    """
    # Replace multiple spaces, tabs, and newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    # Keep only alphanumeric characters, basic punctuation, and common symbols used in tech
    text = re.sub(r'[^\w\s.,&+#@-]', '', text)
    return text.strip()

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts and cleans text from a PDF file."""
    text = ""
    try:
        # Opens the document once and concatenates text from each page
        with fitz.open(file_path) as doc:
            for page in doc:
                # Extracts plain text from the current page
                text += page.get_text("text") + " " 
        return clean_text(text)
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(file_path: str) -> str:
    """Extracts and cleans text from a DOCX file."""
    text = ""
    try:
        # Use the Document class to access the source DOCX file
        doc = docx.Document(file_path)
        # Iterate over the paragraphs in the document
        for para in doc.paragraphs:
            # Ignore empty lines
            if para.text.strip(): 
                text += para.text.strip() + " "
        return clean_text(text)
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"

# --- Local Testing Block ---
if __name__ == "__main__":
    # Define a path to a sample file (we will create this in the next step)
    sample_pdf_path = "data/test_resume.pdf" 
    
    if os.path.exists(sample_pdf_path):
        print(f"Extracting text from {sample_pdf_path}...\n")
        extracted = extract_text_from_pdf(sample_pdf_path)
        print("--- Extracted Text Preview ---")
        print(extracted[:500] + "...\n")
    else:
        print("Parser script is ready! Add a 'test_resume.pdf' to your data/ folder to test it.")