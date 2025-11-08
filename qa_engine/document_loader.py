import tempfile
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import shutil
import os

def load_documents(uploaded_files) -> List[Document]:
    """
    Accepts uploaded files, saves them properly, and loads them.
    Ensures files aren't deleted too early (important for PDFs).
    """
    documents = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        suffix = ".pdf" if filename.endswith(".pdf") else ".txt"

        # Save to a manually managed persistent temp directory
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, filename)

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the correct document loader
        try:
            if suffix == ".pdf":
                loader = PyMuPDFLoader(temp_file_path)
            else:
                loader = TextLoader(temp_file_path)

            docs = loader.load()
            documents.extend(docs)

        except Exception as e:
            print(f"⚠️ Failed to load {filename}: {e}")

        # Optional: Clean up manually later
        # shutil.rmtree(temp_dir)

    return documents

def chunk_documents(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits documents into chunks for better embedding and retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)