import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda
from operator import itemgetter
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile
import fitz # PyMuPDF
from PIL import Image
import io
import torch

@st.cache_resource
def get_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True}
    )


st.set_page_config(page_title="DocQuery | AI Document Assistant", page_icon="📄", layout="wide")


st.title("DocQuery")
st.subheader("Intelligent RAG-based Assistant powered by Mistral")

# Pydantic Model for Structured Output
class DocumentResponse(BaseModel):
    answer: str = Field(description="The main answer to the question based on the context")
    source_summary: str = Field(description="A brief summary of the sources used to answer the question")
    confidence_score: float = Field(description="Confidence score from 0 to 1 based on context relevance")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "pdf_registry" not in st.session_state:
    st.session_state.pdf_registry = {}

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Document Center")
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Documents"):
            all_splits = []
            with st.spinner("Analyzing and indexing documents..."):
                for uploaded_file in uploaded_files:
                    # Save uploaded file to registry
                    pdf_bytes = uploaded_file.getvalue()
                    file_name = uploaded_file.name
                    st.session_state.pdf_registry[file_name] = pdf_bytes
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(pdf_bytes)
                        tmp_path = tmp_file.name

                    try:
                        # PDF Processing
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()
                        
                        # Add filename to metadata
                        for doc in docs:
                            doc.metadata["source_name"] = file_name
                        
                        # Chunking (larger chunks = fewer embeddings = faster) but their is tradeoff
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
                        splits = text_splitter.split_documents(docs)
                        all_splits.extend(splits)
                    except Exception as e:
                        st.error(f"Error processing {file_name}: {e}")
                    finally:
                        os.remove(tmp_path)
                
                # Create/Add to Vector Store
                if all_splits:
                    embeddings = get_embedding_model()
                    
                    # Progress bar for embedding
                    progress_bar = st.progress(0, text="Generating embeddings...")
                    batch_size = 64
                    total_batches = (len(all_splits) + batch_size - 1) // batch_size
                    
                    # Embed in batches with progress
                    all_texts = [doc.page_content for doc in all_splits]
                    all_embeddings = []
                    for i in range(0, len(all_texts), batch_size):
                        batch = all_texts[i:i + batch_size]
                        batch_embeddings = embeddings.embed_documents(batch)
                        all_embeddings.extend(batch_embeddings)
                        progress = min((i + batch_size) / len(all_texts), 1.0)
                        progress_bar.progress(progress, text=f"Embedding chunks {min(i+batch_size, len(all_texts))}/{len(all_texts)}")
                    
                    progress_bar.progress(1.0, text="Building vector index...")
                    
                    
                    text_embedding_pairs = list(zip(all_texts, all_embeddings))
                    metadatas = [doc.metadata for doc in all_splits]
                    vector_store = FAISS.from_embeddings(text_embedding_pairs, embeddings, metadatas=metadatas)
                    st.session_state.vector_store = vector_store
                    
                    progress_bar.empty()
                    st.success(f"{len(uploaded_files)} documents indexed ({len(all_splits)} chunks) ✅")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # If assistant message has sources, render them
        if message["role"] == "assistant" and "sources" in message:
            st.markdown("---")
            st.caption("📄 Referenced Pages")
            source_cols = st.columns(len(message["sources"]))
            
            if "pdf_registry" in st.session_state:
                for i, src in enumerate(message["sources"]):
                    file_name = src.get("source_name")
                    if file_name and file_name in st.session_state.pdf_registry:
                        with source_cols[i]:
                            page_num = src["page"]
                            st.caption(f"From: {file_name}")
                            st.caption(f"Page {page_num + 1}")
                            
                            doc = fitz.open(stream=st.session_state.pdf_registry[file_name], filetype="pdf")
                            page = doc.load_page(page_num)
                            
                            # Precise Highlight Logic
                            # Break content into sentences and highlight unique phrases
                            content_text = src["content"]
                            sentences = [s.strip() for s in content_text.replace("\n", " ").split(".") if len(s.strip()) > 40]
                            
                            for sentence in sentences:
                                # Use a mid-section snippet (skip first few words which may be headers)
                                words = sentence.split()
                                if len(words) > 8:
                                    # Take a unique phrase from the middle of the sentence
                                    start_idx = min(3, len(words) // 4)
                                    snippet = " ".join(words[start_idx:start_idx + 12])
                                else:
                                    snippet = sentence
                                
                                search_results = page.search_for(snippet)
                                for rect in search_results:
                                    # Filter out matches in header/footer zones (top 8% and bottom 8% of page)
                                    page_height = page.rect.height
                                    if rect.y0 > page_height * 0.08 and rect.y1 < page_height * 0.92:
                                        highlight = page.add_highlight_annot(rect)
                                        highlight.set_colors(stroke=(1, 0.9, 0.2))  # Yellow
                                        highlight.update()
                            
                            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                            st.image(pix.tobytes("png"), use_container_width=True)
                            doc.close()
                            
                            with st.expander("Text"):
                                st.write(src["content"])
            
            if "metadata" in message:
                with st.expander("🔍 Analytics"):
                    st.write(f"**Sources:** {message['metadata']['summary']}")
                    st.write(f"**Confidence:** {message['metadata']['confidence']:.2f}")

if prompt := st.chat_input("Ask something about your document..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate Response
    if st.session_state.vector_store:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    
                    # 1. Get Context
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    source_docs = retriever.invoke(prompt)
                    context_text = "\n\n".join([d.page_content for d in source_docs])
                    
                    # 2. Setup Streaming
                    llm_streaming = ChatOllama(model="mistral", temperature=0.3, streaming=True)
                    
                    # We use a simpler prompt for streaming to avoid JSON parsing issues during stream
                    stream_template = """Answer the question based only on the context provided.
                    Be concise and accurate.
                    
                    Context: {context}
                    Question: {question}
                    
                    Answer:"""
                    
                    full_prompt = stream_template.format(context=context_text, question=prompt)
                    
                    # 3. Stream and display
                    with st.chat_message("assistant"):
                        # We use write_stream to handle the yield from llm
                        response_content = st.write_stream(llm_streaming.stream(full_prompt))
                        
                    # 4. Save to history with sources (including filename)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_content,
                        "sources": [
                            {
                                "page": d.metadata.get("page", 0), 
                                "content": d.page_content,
                                "source_name": d.metadata.get("source_name", "Unknown")
                            } 
                            for d in source_docs
                        ],
                        "metadata": {
                            "summary": "Generated from multiple document contexts.",
                            "confidence": 0.95 
                        }
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    else:
        with st.chat_message("assistant"):
            st.info("Please upload and process a PDF document first in the sidebar to start chatting.")
