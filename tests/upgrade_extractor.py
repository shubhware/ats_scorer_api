import spacy
import fitz 
import os

print("1. Loading spaCy's foundational English AI...")
nlp = spacy.load("en_core_web_sm")

# --- THE GAZETTEER (Massive Tech Dictionary) ---
tech_skills = [
    # General & Web
    "python", "java", "c", "c++", "c#", "javascript", "html", "css", "sql", "git", "linux", 
    "react", "node.js", "oop", "dbms", "operating system", "networking", "web development",
    
    # Cyber Security & Tools from Sandeep's Resume
    "cyber security", "scapy", "npcap", "wireshark", "nmap", "metasploit", "penetration testing",
    "firewall", "intrusion detection", "malware analysis", "cryptography"
]

print("2. Hardwiring the Tech Vocabulary into the AI...")
# Create a Ruler that overrides the AI's guesses
ruler = nlp.add_pipe("entity_ruler", before="ner")

# Format the skills so spaCy understands multi-word terms (like "cyber security")
patterns = []
for skill in tech_skills:
    pattern = [{"LOWER": word} for word in skill.split()]
    patterns.append({"label": "SKILL", "pattern": pattern})

ruler.add_patterns(patterns)

# --- SAVE THE UPGRADED MODEL ---
save_path = "../app/core/models/ner_model/model-best"
os.makedirs(save_path, exist_ok=True)
nlp.to_disk(save_path)
print(f"3. Upgraded Model 01 saved to {save_path}!")

# --- TEST IT ON SANDEEP'S RESUME ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join([page.get_text() for page in doc])

print("\n4. Reading RESUME.pdf...")
resume_text = extract_text_from_pdf("../RESUME.pdf")

print("5. Extracting Skills...")
doc = nlp(resume_text)

# Set() removes duplicates, sorted() puts them in alphabetical order
found_skills = sorted(list(set([ent.text.title() for ent in doc.ents if ent.label_ == "SKILL"])))

print(f"\n--- 🎯 UPGRADED SKILLS FOUND ({len(found_skills)}) ---")
for skill in found_skills:
    print(f"- {skill}")