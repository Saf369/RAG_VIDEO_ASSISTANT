from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter  # fixed: langchain.text_splitter is deprecated
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcription"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embed() -> HuggingFaceEmbeddings:
    """Returns the HuggingFace embedding model (CPU)."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})


def build_vector_store(transcript: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Chroma:
    """Splits a transcript and stores/updates it in ChromaDB."""
    embedding = get_embed()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(splitter.split_text(transcript))
    ]

    if not os.path.exists(CHROMA_DIR):
        return Chroma.from_documents(
            documents=docs,
            collection_name=COLLECTION_NAME,
            embedding=embedding,          # fixed: was embedding_function (wrong kwarg)
            persist_directory=CHROMA_DIR
        )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=CHROMA_DIR
    )
    vector_store.add_documents(docs)
    return vector_store


def load_vector_store() -> Chroma:
    """Loads an existing ChromaDB vector store from disk."""
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(f"Chroma directory '{CHROMA_DIR}' not found. Run build_vector_store() first.")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embed(),
        persist_directory=CHROMA_DIR
    )


def get_retriever(vector_store: Chroma, k: int = 4):
    """Returns a similarity retriever for the given vector store."""
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})


def get_llm() -> ChatMistralAI:
    """Returns the Mistral LLM instance."""
    return ChatMistralAI(
        model="mistral-small-latest",
        api_key=os.environ["MISTRAL_API_KEY"],
        temperature=0.2
    )


def build_rag_chain(retriever) -> Runnable:
    """Builds a RAG chain: retriever → prompt → LLM → string output."""
    llm = get_llm()

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a helpful assistant that answers questions about video transcripts.

Use the following retrieved context to answer the user's question accurately.
If the answer is not in the context, say "I couldn't find that in the transcript."

<context>
{context}
</context>
        """.strip()),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )


def load_rag_chain() -> Runnable:
    """Loads the RAG chain from the persisted vector store."""
    return build_rag_chain(get_retriever(load_vector_store()))


def ask_question(query: str, chain: Runnable = None) -> str:
    """
    Asks a question using the RAG chain.
    Optionally pass a pre-built chain to avoid reloading the vector store on every call.
    """
    if chain is None:
        chain = load_rag_chain()
    return chain.invoke(query)