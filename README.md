# resume-parser-demo
👔 TalentScout AI: Enterprise Resume Matcher

TalentScout AI is a local-first, privacy-focused resume screening prototype designed for enterprise use cases. Unlike standard keyword matchers, it uses a Hybrid Search Architecture that combines semantic understanding (Vector Embeddings) with precise keyword scoring (BM25) to rank candidates effectively.

🚀 Key Features

🧠 Hybrid Matching Engine: Combines BM25 (Keyword Precision) and SBERT (Semantic Understanding) for a balanced score.

⚡ Zero-Latency Search: Pre-computes embeddings during ingestion, enabling instant matching against Job Descriptions (JDs).

🛡️ Semantic Filtering: "Must-have" concept filters (e.g., "Leadership") that screen candidates based on meaning, not just keywords.

📂 Smart Parsing: Detects sections (Skills, Experience) and handles layout parsing better than standard text extractors.

🔒 Privacy First: Runs entirely locally on your CPU. No data is sent to OpenAI or cloud APIs.

🛠️ Installation

Clone the Repository

git clone [https://github.com/yourusername/talentscout-ai.git](https://github.com/yourusername/talentscout-ai.git)
cd talentscout-ai


Create a Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies

pip install -r requirements.txt


🏃‍♂️ Usage

Start the Application

streamlit run app.py


Workflow

Go to the "Indexing" tab.

Upload PDF resumes. The system will parse text, chunk sections, and generate vectors.

Switch to the "Matching" tab.

Paste a Job Description (JD).

(Optional) Open the sidebar to add Semantic Filters (e.g., "Financial Analysis").

Click "Analyze Candidates".

📊 How Scoring Works

The final match score is a weighted average designed to balance generic competence with specific requirements:

$$ FinalScore = (0.6 \times SemanticScore) + (0.4 \times KeywordScore) $$

Semantic Score: Derived from Cosine Similarity between the JD vector and the Candidate's Resume + Experience Section vectors.

Keyword Score: Derived from the BM25 algorithm, which rewards unique, specific keyword matches and penalizes common stopwords.

🏗️ Tech Stack

Frontend: Streamlit

Parsing: PyMuPDF (Fitz)

Vector DB: ChromaDB (Local Persistence)

Embedding Model: sentence-transformers/all-MiniLM-L6-v2

Keyword Search: Rank-BM25

🤝 Contributing

Contributions are welcome! Please read ARCHITECTURE.md to understand the core logic before submitting Pull Requests.

📄 License

MIT License
