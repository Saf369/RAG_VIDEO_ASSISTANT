from openai.types import auto_file_chunking_strategy_param
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

CHROMA_DIR="vector_db"
COLLECTION_NAME="meeting_transcription"
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
def get_embed():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )
def build_vector_store(transcript:str,chunk_size:int=1000,chunk_overlap:int=200)->Chroma:
    embedding=get_embed()
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    docs=[Document(page_content=chunk,metadata={"chunk_index":i}) for i,chunk in enumerate(splitter.split_text(transcript))]
    if not os.path.exists(CHROMA_DIR):
        return Chroma.from_documents(
            documents=docs,
            collection_name=COLLECTION_NAME, 
            embedding_function=embedding, 
            persist_directory=CHROMA_DIR
        )
    vector_store=Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=CHROMA_DIR
    )
    vector_store.add_documents(docs)
    return vector_store
def load_vector_store()->Chroma:
    embedding=get_embed()
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError("Chroma directory not found")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=CHROMA_DIR
    )    
def get_retriever(vector_store:Chroma,k:int = 4):
    return vector_store.as_retriever(search_type="similarity",search_kwargs={"k":k})    