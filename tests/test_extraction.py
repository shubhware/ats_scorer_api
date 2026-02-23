import spacy
import fitz  # PyMuPDF

print("1. Loading Model 01 (Skill Extractor)...")
# Path points back to the app folder
nlp = spacy.load("../app/core/models/ner_model/model-best")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text() for page in doc])
    return text

print("2. Reading RESUME.pdf...")
resume_text = extract_text_from_pdf("../RESUME.pdf")

print("3. AI Extracting Skills...")
doc = nlp(resume_text)

# Get unique skills and sort them alphabetically
found_skills = sorted(list(set([ent.text for ent in doc.ents if ent.label_ == "SKILL"])))

print(f"\n--- 🎯 SKILLS FOUND ({len(found_skills)}) ---")
for skill in found_skills:
    print(f"- {skill}")