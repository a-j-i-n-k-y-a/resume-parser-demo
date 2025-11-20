import json
from typing import Dict, List, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from parser import ParsedResume

class ResumeVectorStore:
    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path, settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(name="resumes_v2")
        # Load model once
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def add_resume(self, resume_id: str, parsed: ParsedResume) -> None:
        """
        Encodes the resume AND its specific sections during ingestion.
        Saves section embeddings in metadata to avoid re-computation at query time.
        """
        # 1. Encode Full Text
        full_text = parsed.raw_text
        full_embedding = self.model.encode(full_text).tolist()

        # 2. Encode Critical Sections (Skills, Experience)
        # We serialize these embeddings as strings to store in ChromaDB metadata
        # (Chroma metadata currently supports strings, ints, floats, bools - not lists)
        skills_text = parsed.sections.get("skills", "")
        exp_text = parsed.sections.get("experience", "")
        
        skills_emb = self.model.encode(skills_text).tolist() if skills_text else [0.0]*384
        exp_emb = self.model.encode(exp_text).tolist() if exp_text else [0.0]*384

        metadata = {
            "filename": parsed.metadata.get("filename", ""),
            "skills_text": skills_text[:1000], # Truncate for metadata storage limits if any
            "experience_text": exp_text[:1000],
            # Store raw text for BM25 ranking later
            "raw_text_chunk": full_text[:2000], 
            # Store vectors as JSON strings (Hack for metadata storage)
            "embedding_skills": json.dumps(skills_emb),
            "embedding_experience": json.dumps(exp_emb)
        }

        self.collection.add(
            ids=[resume_id],
            documents=[full_text],
            metadatas=[metadata],
            embeddings=[full_embedding],
        )

    def search(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode(query_text).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        
        hits = []
        if not results["ids"]:
            return hits
            
        for i in range(len(results["ids"][0])):
            hits.append({
                "resume_id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
                "document": results["documents"][0][i]
            })
        return hits