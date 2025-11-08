from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List
import os

# Use a simple small model for fast local embedding
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_local_embeddings(model_name: str = DEFAULT_MODEL_NAME):
    """
    Returns a local HuggingFace embedding model.
    """
    return HuggingFaceEmbeddings(model_name=model_name)

def create_vector_store(chunks: List[Document], persist: bool = False, persist_dir: str = "chroma_store"):
    """
    Creates a Chroma vector store from document chunks.
    Optionally persists it to disk.
    """
    embedding_model = get_local_embeddings()

    if persist:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
        vectordb.persist()
    else:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

    return vectordb