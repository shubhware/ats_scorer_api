from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import spacy
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF for resume parsing
import pandas as pd
import torch
import os
import math

# Initialize the Master API
app = FastAPI(title="Smart Campus Recruitment System", version="2.7")

# --- 1. BOOTING AI ENGINES ---
print("1. Loading Hybrid NER and Semantic Brain...")

# Load the base English model
nlp = spacy.load("en_core_web_sm")

# --- EXPANDED TECH DICTIONARY (GAZETTEER) ---
# Ensuring high-priority backend and system design terms are recognized
tech_skills = [
    # Programming & Web
    "python", "java", "c", "c++", "c#", "javascript", "html", "css", "sql", "git", "linux", 
    "react", "node.js", "oop", "dbms", "operating system", "networking", "web development",
    
    # Backend & System Design (Fix for Test #3)
    "golang", "go", "redis", "postgresql", "postgres", "mongodb", "system design", 
    "distributed systems", "data structures and algorithms", "dsa", "algorithms",
    
    # Cloud & DevOps
    "aws", "docker", "kubernetes", "cloud computing", "azure", "gcp", "ci/cd",
    
    # Cyber Security Stack
    "cyber security", "scapy", "npcap", "wireshark", "nmap", "metasploit", "penetration testing",
    "firewall", "intrusion detection", "vulnerability scanning", "sql injection", "xss", "owasp"
]

# Configure the Hybrid Entity Ruler
if "entity_ruler" not in nlp.pipe_names:
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    
    # Building explicit patterns for multi-word recognition
    patterns = []
    for skill in tech_skills:
        pattern = [{"LOWER": word} for word in skill.split()]
        patterns.append({"label": "SKILL", "pattern": pattern})
    
    ruler.add_patterns(patterns)

# Load the Transformer model for Semantic Similarity
master_brain = SentenceTransformer('all-MiniLM-L6-v2') 

# --- 2. LOADING DATABASES ---
print("2. Loading Market Trends & Course Catalog...")

# Model 03: Demand scores based on LinkedIn Market Data
trends_df = pd.read_csv("data/trends/top_market_skills.csv")

# Model 04: Coursera course embeddings for recommendation search
course_df = pd.read_pickle("app/core/models/course_vectors/course_catalog.pkl")
course_embeddings = torch.load("app/core/models/course_vectors/course_embeddings.pt")

# Mount the static directory for frontend access
app.mount("/static", StaticFiles(directory="static"), name="static")

def extract_text_from_pdf(pdf_path):
    """Utility to extract raw text from uploaded PDF resumes."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- 3. CORE API ROUTES ---

@app.get("/")
async def read_index():
    """Serves the main dashboard to the browser."""
    return FileResponse('static/index.html')

@app.get("/market-trends")
async def get_market_trends():
    """Returns the top 30 trending technical skills for the Market Trends Tab."""
    return {"trending_skills": trends_df.head(30).to_dict(orient="records")}

@app.post("/analyze-resume")
async def analyze_resume(
    job_description: str = Form(...), 
    file: UploadFile = File(...)
):
    # --- A. SETUP & EXTRACTION ---
    temp_pdf = f"temp_{file.filename}"
    with open(temp_pdf, "wb") as buffer:
        buffer.write(await file.read())
    
    resume_text = extract_text_from_pdf(temp_pdf)
    os.remove(temp_pdf)

    # --- B. SKILL GAP ANALYSIS (Hybrid NER) ---
    doc_resume = nlp(resume_text.lower())
    doc_jd = nlp(job_description.lower())
    
    # Extract unique skills from both sources
    resume_skills_lower = set([ent.text.lower().strip() for ent in doc_resume.ents if ent.label_ == "SKILL"])
    jd_skills_lower = set([ent.text.lower().strip() for ent in doc_jd.ents if ent.label_ == "SKILL"])
    
    # Mathematical Set Subtraction to find the gap
    missing_skills_list = list(jd_skills_lower - resume_skills_lower)

    # --- C. ATS SCORING WITH SIGMOID CALIBRATION ---
    # This addresses the semantic density issue for high-quality candidates
    res_emb = master_brain.encode(resume_text, normalize_embeddings=True)
    jd_emb = master_brain.encode(job_description, normalize_embeddings=True)
    raw_similarity = util.cos_sim(res_emb, jd_emb).item()

    # Apply Sigmoid Calibration (raw 0.52 similarity maps to ~85% score)
    calibrated_score = (1 / (1 + math.exp(-12 * (raw_similarity - 0.52)))) * 100
    final_score = round(min(99.5, max(5.0, calibrated_score)), 1)

    # --- D. OPTIMIZATION TIPS ---
    optimization_tips = []
    if not missing_skills_list and final_score < 90:
        optimization_tips.append("Your skills are a match! To boost your score above 90%, use specific technical verbs like 'Engineered' or 'Deployed'.")
        optimization_tips.append("Try listing your tech stack per project to increase semantic correlation with the Job Description.")

    # --- E. PRIORITIZED LEARNING ROADMAP (Model 03 & 04) ---
    missing_with_scores = []
    for skill in missing_skills_list:
        match = trends_df[trends_df['Technical Skill'] == skill]
        score = match['Market Demand Score'].values[0] if not match.empty else 0
        missing_with_scores.append((skill, score))
    
    # Sort: Highest market demand skills first
    missing_with_scores.sort(key=lambda x: x[1], reverse=True)

    recommendation_roadmap = []
    for skill, market_score in missing_with_scores:
        # Vector search for the best 2 courses for this specific skill
        query_vector = master_brain.encode(skill, convert_to_tensor=True)
        hits = util.semantic_search(query_vector, course_embeddings, top_k=2)[0]
        
        skill_courses = []
        for hit in hits:
            idx = hit['corpus_id']
            course = course_df.iloc[idx]
            skill_courses.append({
                "title": course['course_title'],
                "provider": course['course_organization'],
                "rating": course['course_rating'],
                "url": course['course_url']
            })
        
        recommendation_roadmap.append({
            "skill": skill.title(),
            "market_priority": "High" if market_score > 1000 else "Medium",
            "courses": skill_courses
        })

    # --- F. FINAL SYSTEM RESPONSE ---
    return {
        "candidate_file": file.filename,
        "ats_match_score": f"{final_score}%",
        "verdict": "Shortlisted" if final_score >= 75.0 else "Needs Improvement",
        "skills_found": sorted([s.title() for s in resume_skills_lower]),
        "skills_missing": sorted([s.title() for s in missing_skills_list]),
        "recommendation_roadmap": recommendation_roadmap,
        "optimization_tips": optimization_tips
    }