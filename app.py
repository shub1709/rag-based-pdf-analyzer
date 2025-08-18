import os
import streamlit as st
from src.utils import FileManager
from src.rag_pipeline import RAGPipeline

# Initialize pipeline
@st.cache_resource
def load_pipeline():
    return RAGPipeline(config_path="config.yaml")

pipeline = load_pipeline()

st.cache_resource.clear()

# Streamlit app layout
st.set_page_config(page_title="PDF RAG Summarizer", layout="wide")
st.title("📄 Ask Me Anything!!!")

# Sidebar - Upload PDF
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    # Save uploaded file
    save_path = os.path.join("data", "uploaded_pdfs", uploaded_file.name)
    FileManager.save_uploaded_file(uploaded_file, save_path)

    st.sidebar.success(f"Uploaded: {uploaded_file.name}")

    if st.sidebar.button("Process PDF"):
        with st.spinner("Processing PDF through RAG pipeline..."):
            result = pipeline.process_pdf(save_path)

        st.success("✅ Processing complete!")

        # # Display summary
        # st.subheader("📌 Overall Summary")
        # st.write(result.summary)

        # # Key insights
        # st.subheader("💡 Key Insights")
        # for insight in result.key_insights:
        #     st.markdown(f"- {insight}")

        # # Section summaries
        # if result.section_summaries:
        #     st.subheader("📚 Section-wise Summaries")
        #     for section, summary in result.section_summaries.items():
        #         with st.expander(section):
        #             st.write(summary)

        # # Metadata
        # st.subheader("📊 Document Metadata")
        # st.json(result.metadata)

# Q&A Section
st.sidebar.header("Ask Questions")
question = st.sidebar.text_input("Enter your question about the document")
if question:
    with st.spinner("Finding answer..."):
        answer = pipeline.answer_question(question)
    st.subheader("❓ Question")
    st.write(question)
    st.subheader("📝 Answer")
    st.write(answer)

    # --- Debug: show rewritten & expanded queries ---
    if st.sidebar.checkbox("Show Query Enhancements"):
        st.subheader("🔎 Query Enhancements")
        if hasattr(pipeline.llm_manager, "last_rewrite"):
            st.markdown(f"**Rewritten Query:** {pipeline.llm_manager.last_rewrite}")
        if hasattr(pipeline.llm_manager, "last_expansions"):
            st.markdown("**Expanded Queries:**")
            for q in pipeline.llm_manager.last_expansions:
                st.markdown(f"- {q}")



# Statistics
# if st.sidebar.checkbox("Show Pipeline Stats"):
#     st.subheader("⚙️ Pipeline Statistics")
#     stats = pipeline.get_statistics()
#     st.json(stats)

def show_retrieval_trace(pipeline, max_chars: int = 300):
    """
    Display retrieval trace: which queries pulled which chunks.
    Truncates chunk text for readability.
    """
    if not hasattr(pipeline, "last_retrieval_log"):
        st.info("No retrieval trace available.")
        return

    st.subheader("🔎 Retrieval Trace")
    for q, chunks in pipeline.last_retrieval_log.items():
        with st.expander(f"Query: {q}"):
            if not chunks:
                st.write("No chunks retrieved.")
                continue

            for idx, c in enumerate(chunks, start=1):
                page = c["metadata"].get("page", "?")
                text = c["content"][:max_chars] + ("..." if len(c["content"]) > max_chars else "")
                st.markdown(f"**Result {idx} (Page {page}):**")
                st.write(text)
                st.markdown("---")

if st.sidebar.checkbox("Show Retrieval Trace"):
    show_retrieval_trace(pipeline, max_chars=4000)