import streamlit as st
import pandas as pd
import altair as alt
from parser import ResumeParser
from vector_store import ResumeVectorStore
from matcher import ResumeMatcher

# --- Page Config ---
st.set_page_config(page_title="TalentScout - Enterprise Resume Matcher", layout="wide", page_icon="👔")

# --- Custom CSS for HR Style ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .keyword-badge {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-right: 5px;
        display: inline-block;
    }
    .missing-badge {
        background-color: #ffebee;
        color: #c62828;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-right: 5px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

def init_services():
    parser = ResumeParser()
    vector_store = ResumeVectorStore(db_path="./chroma_db")
    matcher = ResumeMatcher(vector_store=vector_store)
    return parser, vector_store, matcher

def make_grid(data):
    df = pd.DataFrame(data)
    if not df.empty:
        df['final_score'] = (df['final_score'] * 100).round(1)
        df = df[['resume_id', 'final_score', 'semantic_score', 'keyword_score']]
        df.columns = ['Candidate ID', 'Match %', 'Semantic Fit', 'Keyword Fit']
    return df

def main():
    parser, vector_store, matcher = init_services()

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        top_k = st.slider("Candidates to Display", 3, 20, 5)
        st.divider()
        st.info("Upload resumes to the Index before matching.")

    st.title("👔 TalentScout AI")
    st.caption("Enterprise-grade semantic resume parsing and matching system.")

    # --- Tab Layout ---
    tab1, tab2 = st.tabs(["📂 Resume Indexing", "🔍 Candidate Matching"])

    # --- Tab 1: Ingestion ---
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Upload New Resumes")
            uploaded_files = st.file_uploader("Drop PDF files here", type=["pdf"], accept_multiple_files=True)
            if uploaded_files and st.button("Process & Index"):
                with st.status("Processing resumes...", expanded=True) as status:
                    for uploaded in uploaded_files:
                        pdf_bytes = uploaded.read()
                        parsed = parser.parse_pdf_bytes(pdf_bytes, filename=uploaded.name)
                        vector_store.add_resume(uploaded.name, parsed)
                        st.write(f"✅ indexed {uploaded.name}")
                    status.update(label="Indexing Complete!", state="complete", expanded=False)
                    st.success(f"Successfully added {len(uploaded_files)} resumes to the knowledge base.")

    # --- Tab 2: Matching Dashboard ---
    with tab2:
        jd_col, results_col = st.columns([1, 2])

        with jd_col:
            st.subheader("Job Description")
            jd_text = st.text_area("Paste JD here...", height=300, placeholder="Paste the job description...")
            match_btn = st.button("Analyze Candidates", type="primary", use_container_width=True)

        if match_btn and jd_text:
            with st.spinner("AI is analyzing candidates..."):
                results = matcher.match(jd_text, top_k=top_k)
            
            with results_col:
                st.subheader("Match Analysis")
                
                if not results:
                    st.warning("No matches found.")
                else:
                    # 1. Top Candidates Table
                    st.dataframe(
                        make_grid(results), 
                        hide_index=True, 
                        use_container_width=True,
                        column_config={
                            "Match %": st.column_config.ProgressColumn(
                                "Match Score", min_value=0, max_value=100, format="%d%%"
                            )
                        }
                    )

                    # 2. Detailed Cards
                    st.markdown("### 📝 Detailed Candidate Reports")
                    for result in results:
                        score = int(result['final_score'] * 100)
                        
                        # Dynamic Color for Score
                        score_color = "green" if score > 75 else "orange" if score > 50 else "red"
                        
                        with st.expander(f"#{result['resume_id']} - {score}% Match"):
                            c1, c2, c3 = st.columns([1, 2, 1])
                            
                            with c1:
                                st.metric("Overall Score", f"{score}%")
                                st.metric("Context Match", f"{int(result['semantic_score']*100)}%")
                                st.metric("Keyword Match", f"{int(result['keyword_score']*100)}%")

                            with c2:
                                st.markdown("**✅ Matched Keywords (Top 10):**")
                                matched_badges = "".join([f"<span class='keyword-badge'>{k}</span>" for k in result['match_details']['matched_keywords'][:10]])
                                st.markdown(matched_badges, unsafe_allow_html=True)
                                
                                st.markdown("**❌ Missing Keywords (Top 10):**")
                                missing_badges = "".join([f"<span class='missing-badge'>{k}</span>" for k in result['match_details']['missing_keywords'][:10]])
                                st.markdown(missing_badges, unsafe_allow_html=True)
                                
                                st.markdown("**📄 Snippet:**")
                                st.info(result['snippet'])

                            with c3:
                                st.caption("Recommendation:")
                                if score > 80:
                                    st.success("Highly Recommended")
                                elif score > 60:
                                    st.warning("Potential Fit")
                                else:
                                    st.error("Low Match")

if __name__ == "__main__":
    main()