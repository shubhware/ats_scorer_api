from sentence_transformers import SentenceTransformer, util

# Loading the UNTOUCHED base model
print("Loading Master Brain...")
model = SentenceTransformer('all-MiniLM-L6-v2')

jd = "Looking for a B.Tech CSE student for a Cyber Security role. Must know Python, Linux, and network security."

good_resume = "Shubham Yadav. B.Tech Computer Science Engineering. Skills: Cyber Security, Python, Linux, Ethical Hacking."
bad_resume = "Experienced Chef. I make great Italian food and manage kitchen staff."

print("\n--- ATS MATCH SCORES ---")
# Score the Good Resume
good_emb = model.encode(good_resume, normalize_embeddings=True)
jd_emb = model.encode(jd, normalize_embeddings=True)
print(f"Your Resume Match: {util.cos_sim(good_emb, jd_emb).item() * 100:.2f}%")

# Score the Bad Resume
bad_emb = model.encode(bad_resume, normalize_embeddings=True)
print(f"Chef Resume Match: {util.cos_sim(bad_emb, jd_emb).item() * 100:.2f}%")