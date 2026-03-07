import streamlit as st
import fitz

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA



# PDF LOADER
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents



# SPLIT TEXT

def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    return chunks



# EMBEDDINGS

def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings



# VECTOR DATABASE

def create_vector_store(chunks, embeddings):

    db = FAISS.from_documents(chunks, embeddings)

    return db



# RAG CHAIN

def build_rag_chain(vector_db):

    llm = Ollama(model="mistral")

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(),
        return_source_documents=True
    )

    return chain



# HIGHLIGHT TEXT IN PDF

def highlight_text(pdf_path, page_number, text):

    doc = fitz.open(pdf_path)

    page = doc[page_number]

    matches = page.search_for(text)

    for match in matches:
        highlight = page.add_highlight_annot(match)
        highlight.update()

    output = "highlighted_output.pdf"

    doc.save(output)

    return output



# QUERY ENGINE

def ask_question(chain, question, pdf_path):

    result = chain({"query": question})

    answer = result["result"]

    sources = result["source_documents"]

    source_doc = sources[0]

    page = source_doc.metadata["page"]

    snippet = source_doc.page_content[:200]

    highlighted_pdf = highlight_text(pdf_path, page, snippet)

    return answer, page, highlighted_pdf



# STREAMLIT PART
st.title("Offline Chat with PDF (RAG + Ollama)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF Uploaded Successfully")

    docs = load_pdf("temp.pdf")

    chunks = split_documents(docs)

    embeddings = create_embeddings()

    vector_db = create_vector_store(chunks, embeddings)

    chain = build_rag_chain(vector_db)

    question = st.text_input("Ask a question about the PDF")

    if question:

        answer, page, highlighted_pdf = ask_question(chain, question, "temp.pdf")

        st.subheader("Answer")

        st.write(answer)

        st.subheader("Source Page")

        st.write(page)

        with open(highlighted_pdf, "rb") as f:
            st.download_button(
                "Download Highlighted PDF",
                f,
                file_name="highlighted.pdf"
            )
