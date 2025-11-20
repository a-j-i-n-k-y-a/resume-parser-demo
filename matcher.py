import re
from typing import List, Dict, Any, Set, Tuple

from sentence_transformers import util as st_util

from vector_store import ResumeVectorStore


class ResumeMatcher:
    """
    Implements a simple hybrid scoring:
    FinalScore = 0.3 * KeywordMatch + 0.5 * SemanticMatch + 0.2 * EntityOverlap

    Enhancements over the basic prototype:
      - Section-aware semantic similarity: we up-weight the Skills / Experience sections.
      - spaCy-based NER for EntityOverlap (organizations / universities / locations).
    """

    def __init__(self, vector_store: ResumeVectorStore):
        self.vector_store = vector_store
        self._nlp = None  # lazy-loaded spaCy model

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return set(re.findall(r"\b[a-zA-Z0-9+\-#]+\b", text.lower()))

    def _keyword_overlap(self, jd: str, resume: str) -> Tuple[float, List[str], List[str]]:
        jd_tokens = self._tokenize(jd)
        resume_tokens = self._tokenize(resume)
        if not jd_tokens:
            return 0.0, [], []
        overlap = jd_tokens.intersection(resume_tokens)
        missing = jd_tokens - resume_tokens
        score = len(overlap) / len(jd_tokens)
        matched_sorted = sorted(overlap)
        missing_sorted = sorted(missing)
        return score, matched_sorted, missing_sorted

    def _get_nlp(self):
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy

            # Expecting that the user installs the small English model:
            # python -m spacy download en_core_web_sm
            self._nlp = spacy.load("en_core_web_sm")
        except Exception:
            self._nlp = None
        return self._nlp

    def _extract_entities(self, text: str) -> Set[str]:
        nlp = self._get_nlp()
        if nlp is None or not text.strip():
            return set()
        doc = nlp(text)
        entities: Set[str] = set()
        # Focus on ORG/universities/locations; can expand labels as needed
        target_labels = {"ORG", "GPE", "FAC"}
        for ent in doc.ents:
            if ent.label_ in target_labels:
                entities.add(ent.text.strip().lower())
        return entities

    def _entity_overlap_score(self, jd: str, resume: str) -> float:
        jd_entities = self._extract_entities(jd)
        resume_entities = self._extract_entities(resume)
        if not jd_entities:
            return 0.0
        overlap = jd_entities.intersection(resume_entities)
        return len(overlap) / len(jd_entities)

    def _section_text(self, hit: Dict[str, Any], name: str) -> str:
        md = hit.get("metadata") or {}
        key = f"section_{name}"
        val = md.get(key, "")
        return val or ""

    def _semantic_match_score(self, jd_text: str, hit: Dict[str, Any]) -> float:
        """
        Combine whole-resume semantic similarity with section-specific similarity.
        E.g. Skills and Experience get extra weight vs. everything else.
        """
        jd_emb = self.vector_store.model.encode(jd_text, convert_to_tensor=True)

        resume_text = hit["document"]
        resume_emb = self.vector_store.model.encode(resume_text, convert_to_tensor=True)
        base_sim = st_util.cos_sim(jd_emb, resume_emb).item()

        skills_text = self._section_text(hit, "skills")
        exp_text = self._section_text(hit, "experience")

        extras: List[float] = []
        if skills_text.strip():
            skills_emb = self.vector_store.model.encode(skills_text, convert_to_tensor=True)
            extras.append(st_util.cos_sim(jd_emb, skills_emb).item())
        if exp_text.strip():
            exp_emb = self.vector_store.model.encode(exp_text, convert_to_tensor=True)
            extras.append(st_util.cos_sim(jd_emb, exp_emb).item())

        if not extras:
            return float(base_sim)

        # Weight: 50% whole resume, 50% average of key sections
        section_mean = sum(extras) / len(extras)
        combined = 0.5 * base_sim + 0.5 * section_mean
        return float(combined)

    def match(self, jd_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # First-stage retrieval: semantic recall from ChromaDB
        hits = self.vector_store.search(jd_text, top_k=top_k * 3)
        if not hits:
            return []

        scored: List[Dict[str, Any]] = []
        for hit in hits:
            resume_text = hit["document"]

            semantic_sim = self._semantic_match_score(jd_text, hit)
            keyword_score, matched_keywords, missing_keywords = self._keyword_overlap(jd_text, resume_text)
            entity_overlap = self._entity_overlap_score(jd_text, resume_text)

            final_score = 0.3 * keyword_score + 0.5 * semantic_sim + 0.2 * entity_overlap

            scored.append(
                {
                    "resume_id": hit["resume_id"],
                    "semantic_score": float(semantic_sim),
                    "keyword_score": float(keyword_score),
                    "entity_overlap": float(entity_overlap),
                    "final_score": float(final_score),
                    "snippet": resume_text[:2000],
                    "match_details": {
                        "matched_keywords": matched_keywords,
                        "missing_keywords": missing_keywords,
                    },
                }
            )

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored[:top_k]


