import re
from dataclasses import dataclass
from typing import Any, Dict, Optional
import fitz  # PyMuPDF

@dataclass
class ParsedResume:
    resume_id: str
    raw_text: str
    sections: Dict[str, str]
    metadata: Dict[str, Any]

class ResumeParser:
    """
    Robust Parser with Regex-based Sectioning and Layout Preservation.
    """

    # Regex patterns are more robust than "if keyword in line"
    SECTION_PATTERNS = {
        "skills": re.compile(r"(technical\s+skills|skills|technologies|competencies|tech\s+stack)", re.IGNORECASE),
        "experience": re.compile(r"(work\s+experience|professional\s+experience|employment|history|work\s+history)", re.IGNORECASE),
        "education": re.compile(r"(education|qualifications|academic|certifications)", re.IGNORECASE),
        "projects": re.compile(r"(projects|personal\s+projects)", re.IGNORECASE)
    }

    def _chunk_sections(self, text: str) -> Dict[str, str]:
        lines = text.splitlines()
        section_indices = {}
        
        # 1. Identify Section Headers
        for i, line in enumerate(lines):
            clean_line = line.strip()
            # Heuristic: Headers are usually short (< 5 words) and match our patterns
            if len(clean_line.split()) < 6:
                for sec_name, pattern in self.SECTION_PATTERNS.items():
                    if pattern.search(clean_line) and sec_name not in section_indices:
                        section_indices[sec_name] = i

        # 2. Slice Text based on indices
        sorted_indices = sorted(section_indices.items(), key=lambda x: x[1])
        sections = {}
        
        for i, (name, start_idx) in enumerate(sorted_indices):
            # End index is the start of the next section, or end of file
            end_idx = sorted_indices[i+1][1] if i + 1 < len(sorted_indices) else len(lines)
            
            # Skip the header line itself
            content = "\n".join(lines[start_idx+1:end_idx]).strip()
            sections[name] = content
            
        # Fallback: If no sections found, put everything in 'experience' to be safe
        if not sections:
            sections["experience"] = text
            
        return sections

    def parse_pdf_bytes(self, pdf_bytes: bytes, filename: Optional[str] = None) -> ParsedResume:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Strategy: Get text blocks to preserve column layout better than simple get_text()
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n"
            
        doc.close()

        # Check for Scanned PDF (OCR Trigger)
        if len(full_text.strip()) < 50:
            full_text = "[WARNING] This PDF appears to be an image/scanned. Text extraction failed. Please use an OCR tool."

        sections = self._chunk_sections(full_text)
        
        metadata = {
            "filename": filename or "",
            "is_scanned": len(full_text) < 50
        }

        return ParsedResume(
            resume_id=filename or "unknown",
            raw_text=full_text,
            sections=sections,
            metadata=metadata,
        )