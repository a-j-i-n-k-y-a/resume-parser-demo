 # 👔 TalentScout AI: Enterprise Resume Matcher
 
 TalentScout AI is a local-first, privacy-focused resume screening prototype built for rigorous enterprise hiring workflows. It pairs semantic intelligence with deterministic keyword guarantees so you can surface balanced, context-aware candidate rankings without sending sensitive data to third-party APIs.
 
 ## 🚀 Key Features
 - **Hybrid Matching Engine**: Marries BM25 keyword precision with SBERT semantic understanding for a nuanced, balanced score.
 - **Zero-Latency Search**: Pre-computes embeddings during ingestion so JD-to-resume matching runs instantly.
 - **Semantic Filtering**: Supports “must-have” concept filters (e.g., Leadership) that reason over meaning, not just literal words.
 - **Smart Parsing**: Detects resume sections (Skills, Experience, etc.) and handles complex layouts better than baseline extractors.
 - **Privacy First**: Runs entirely on your machine (CPU only). No calls to OpenAI or any cloud service.
 
 ## 🛠️ Installation
 ```bash
 git clone https://github.com/yourusername/talentscout-ai.git
 cd talentscout-ai
 
 python -m venv venv
 source venv/bin/activate  # Windows: venv\Scripts\activate
 
 pip install -r requirements.txt
 ```
 
 ## 🏃‍♂️ Usage
 ### Start the app
 ```bash
 streamlit run app.py
 ```
 
 ### Workflow
 1. Open the **Indexing** tab.
 2. Upload PDF resumes; the system parses, chunks sections, and generates embeddings.
 3. Switch to the **Matching** tab.
 4. Paste a Job Description (JD).
 5. (Optional) Use the sidebar to add semantic filters (e.g., “Financial Analysis”).
 6. Click **Analyze Candidates** to view ranked matches.
 
### Example Output

![Indexing view](./Screenshot%202025-11-20%20at%201.30.30%E2%80%AFPM.png)

![Matching results](./Screenshot%202025-11-20%20at%201.30.48%E2%80%AFPM.png)

 ## 📊 How Scoring Works
 The final score balances general semantic fit with specific keyword coverage:
 ```
 FinalScore = (0.6 × SemanticScore) + (0.4 × KeywordScore)
 ```
 - **SemanticScore**: Cosine similarity between the JD vector and the candidate’s resume + experience vectors.
 - **KeywordScore**: BM25 score rewarding distinctive keyword hits while down-weighting common stopwords.
 
 ## 🏗️ Tech Stack
 - **Frontend**: Streamlit
 - **Parsing**: PyMuPDF (fitz)
 - **Vector DB**: ChromaDB (local persistence)
 - **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
 - **Keyword Search**: Rank-BM25
 
 ## 🤝 Contributing
 Contributions welcome! Please review `ARCHITECTURE.md` before opening a PR to understand the hybrid search flow and data contracts.
 
 ## 📄 License
 MIT License
