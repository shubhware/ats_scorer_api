# Smart Campus Recruitment System 🚀

An AI-powered recruitment engine that analyzes resumes, calculates ATS scores using Semantic Similarity, and provides a personalized learning roadmap based on market trends.

---

## 🛠️ Important: Missing Files & Setup
Due to GitHub's file size limits, large model weights and datasets are excluded from this repository. To run the project, you must generate these locally:

1. **NER Model**: Run `notebooks/Model_01_NER_Training.ipynb` to generate `app/core/models/ner_model/model-best`.
2. **Course Embeddings**: Run `notebooks/Model_04_Recommendation_System.ipynb` to generate `app/core/models/course_vectors/course_embeddings.pt`.
3. **Datasets**: Ensure you have `top_market_skills.csv` in `data/trends/`.

---

## ✨ Features
* **Hybrid NER Skill Extraction**: Custom spaCy model for 100% tech skill recall.
* **Semantic ATS Scoring**: Transformer-based scoring with **Sigmoid Calibration**.
* **Market Intelligence**: Prioritizes skill gaps based on LinkedIn market demand data.
* **Learning Roadmap**: Suggests top-rated Coursera courses to bridge identified gaps.

---

## 🚀 Local Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/shubhware/ats_scorer_api.git](https://github.com/shubhware/ats_scorer_api.git)
cd ats_scorer_api
```
### 2. Set Up Environment

It is recommended to use Conda or venv to avoid library conflicts.
```bash
# Using Conda
conda create -n ats_scorer python=3.10 -y
conda activate ats_scorer
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
### 3. Reconstruct Missing AI Assets 🛠️

Since large files are not in the repo, you must run the following notebooks to build the local models:

Generate NER Model: Open and run all cells in notebooks/Model_01_NER_Training.ipynb. This will create the app/core/models/ner_model/model-best directory.

Generate Embeddings: Open and run notebooks/Model_04_Recommendation_System.ipynb to vectorize the Coursera dataset and save course_embeddings.pt.

### 4. Run the AI Dashboard

Once the models are generated, start the FastAPI server:

```bash
uvicorn app.main:app --reload
Web UI: Open http://127.0.0.1:8000

API Documentation: Open http://127.0.0.1:8000/docs
```
### 📁 Repository Structure

```bash
ats_scorer_api/
├── app/
│   ├── main.py              # Master API logic & AI Pipeline
│   └── core/models/         # Local storage for AI weights (Generated locally)
├── data/
│   ├── trends/              # Market skill demand CSVs
│   └── courses/             # Coursera course metadata
├── static/
│   └── index.html           # Modern Tabbed UI Dashboard
├── notebooks/               # Model training & vectorization scripts
└── requirements.txt         # Project dependencies
```
### 📊 System Architecture

The system utilizes a multi-stage AI pipeline to ensure high precision in candidate matching:

Extraction: PDF parsing via PyMuPDF.

Hybrid NER: Identifying skills using a combination of a trained Transformer and an EntityRuler (Gazetteer).

Semantic Scoring: all-MiniLM-L6-v2 encodes the text, and a Sigmoid Calibration function adjusts the raw cosine similarity for realistic ATS results.

Recommendation: Missing skills are cross-referenced with market demand scores to prioritize the learning roadmap.
