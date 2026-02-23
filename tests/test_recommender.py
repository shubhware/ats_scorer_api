import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

print("1. Loading the saved Course Database...")
# Load the text data and the mathematical vectors from your app folder
df = pd.read_pickle("app/core/models/course_vectors/course_catalog.pkl")
course_embeddings = torch.load("app/core/models/course_vectors/course_embeddings.pt")

print("2. Booting up the Master Brain...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_recommendations(missing_skill, top_k=3):
    print(f"\n--- 🎯 COURSE RECOMMENDATIONS FOR: '{missing_skill.upper()}' ---")
    
    # 1. Convert the missing skill into a math vector
    query_embedding = model.encode(missing_skill, convert_to_tensor=True)
    
    # 2. Instantly search all 994 courses using semantic similarity
    hits = util.semantic_search(query_embedding, course_embeddings, top_k=top_k)[0]
    
    # 3. Print the top results
    for i, hit in enumerate(hits):
        idx = hit['corpus_id']
        match_score = hit['score'] * 100
        course = df.iloc[idx]
        
        print(f"{i+1}. {course['course_title']} (Match: {match_score:.1f}%)")
        print(f"   Provider: {course['course_organization']}")
        print(f"   Rating  : {course['course_rating']} ⭐")
        print(f"   Link    : {course['course_url']}\n")

# --- THE REAL WORLD TESTS ---
# Let's test it on three completely different tech domains
get_recommendations("Penetration Testing and Cyber Security")
get_recommendations("React and Frontend Web Development")
get_recommendations("Data Structures and Algorithms in Java")