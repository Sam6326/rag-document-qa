"""
RAG Document Q&A App
====================

What this does (simple explanation):
  - You put text documents in the 'data/' folder
  - This app reads them, understands their content using AI
  - You type a question, it finds the most relevant parts of your documents
  - It sends those parts + your question to an AI and gives you an answer

How RAG works (step by step):
  1. Load documents from 'data/' folder
  2. Split them into small chunks (like cutting a book into paragraphs)
  3. Convert each chunk into numbers (vectors) that represent its meaning
  4. Store all those vectors in FAISS (a fast local database)
  5. When you ask a question:
     a. Convert your question into a vector too
     b. Find the document chunks most similar to your question
     c. Send those chunks + your question to the AI
     d. Get an answer that is grounded in your actual documents

Author: Sampath Kumar
Project: RAG Document Q&A
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (e.g. HUGGINGFACEHUB_API_TOKEN)
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# Change these if you want to use different folders or models
# ──────────────────────────────────────────────────────────────────────────────
DOCS_DIR = "data"                                          # Folder with your .txt files
FAISS_INDEX_PATH = "faiss_index"                           # Where FAISS saves its index
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # Free local embedding model
CHUNK_SIZE = 500                                           # Characters per chunk
CHUNK_OVERLAP = 50                                         # Overlap between chunks (for context)
TOP_K_RESULTS = 3                                          # How many chunks to retrieve per question


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD DOCUMENTS
# ──────────────────────────────────────────────────────────────────────────────
def load_documents(directory: str):
    """
    Load all .txt files from the given directory.

    Think of this like collecting all the pages of a book
    before you start reading them.
    """
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    if not Path(directory).exists():
        print(f"ERROR: Folder '{directory}' does not exist.")
        print(f"Create it and add some .txt files to it, then run again.")
        sys.exit(1)

    loader = DirectoryLoader(
        directory,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()

    if not docs:
        print(f"ERROR: No .txt files found in '{directory}/'")
        print("Add at least one .txt file and run again.")
        sys.exit(1)

    print(f"Loaded {len(docs)} document(s) from '{directory}/'")
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: SPLIT INTO CHUNKS
# ──────────────────────────────────────────────────────────────────────────────
def split_documents(docs):
    """
    Split documents into overlapping chunks.

    Why? Because LLMs have a limited context window (how much they can read
    at once). We split documents into small pieces so we can find EXACTLY
    which part is relevant to the question, instead of feeding the entire
    document every time.

    Overlap means consecutive chunks share some text, so a sentence that
    falls between two chunks is never cut off and lost.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: CREATE EMBEDDINGS + FAISS VECTOR STORE
# ──────────────────────────────────────────────────────────────────────────────
def get_embeddings():
    """
    Load the embedding model.

    An embedding model converts text into a list of numbers (a vector).
    Similar texts produce similar vectors. This is how the search works:
    your question becomes a vector, and we find document chunks whose
    vectors are closest to it.

    This model runs LOCALLY on your machine — no API key needed.
    First download is ~80MB. After that it is cached.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    print("(First time: downloads ~80MB. Takes 1-2 minutes. Then cached forever.)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def create_vector_store(chunks, embeddings, save_path: str):
    """
    Create a FAISS vector store from document chunks and save it.

    FAISS (Facebook AI Similarity Search) is a local database that stores
    vectors and finds the nearest ones to a query vector extremely fast.
    We save it to disk so we do not have to rebuild it every time.
    """
    from langchain_community.vectorstores import FAISS

    print("Creating vector store... (converting all chunks to vectors)")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(save_path)
    print(f"Vector store saved to '{save_path}/' — will be reused next time")
    return vector_store


def load_vector_store(embeddings, save_path: str):
    """Load an existing FAISS vector store from disk."""
    from langchain_community.vectorstores import FAISS

    vector_store = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True,  # Safe — we created this file ourselves
    )
    print(f"Loaded existing vector store from '{save_path}/'")
    return vector_store


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: SET UP THE LLM
# ──────────────────────────────────────────────────────────────────────────────
def get_llm():
    """
    Get the LLM (Large Language Model).

    Two options:
    A) If you have a HuggingFace token in your .env file:
       → Uses Mistral-7B via HuggingFace Inference API (better answers)
       → Get free token at: https://huggingface.co/settings/tokens

    B) No token:
       → Uses google/flan-t5-base running locally on your machine
       → Completely free, no internet needed after first download
       → Smaller model — answers are simpler but still RAG-grounded
    """
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if hf_token:
        from langchain_huggingface import HuggingFaceEndpoint

        print("HuggingFace token found — using Mistral-7B via Inference API")
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            task="text-generation",
            max_new_tokens=512,
            temperature=0.1,
            huggingfacehub_api_token=hf_token,
            do_sample=False,
        )
    else:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        from langchain_huggingface import HuggingFacePipeline

        model_name = "google/flan-t5-base"
        print(f"No HF token found — using local model: {model_name}")
        print("(First time: downloads ~250MB. Takes 2-3 minutes. Then cached.)")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        hf_pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
        )
        llm = HuggingFacePipeline(pipeline=hf_pipe)

    return llm


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: BUILD THE RAG CHAIN
# ──────────────────────────────────────────────────────────────────────────────
def build_rag_chain(vector_store, llm):
    """
    Build the RAG pipeline using LangChain LCEL (LangChain Expression Language).

    The chain works like this:
      question → retriever (finds relevant chunks) → prompt (formats everything)
      → LLM (generates answer) → output parser (cleans the text)

    This is the modern LangChain way of building chains. It is readable,
    composable, and easy to extend.
    """
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    # The retriever searches FAISS for top-k most similar chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )

    # Prompt template — tells the LLM exactly how to behave
    prompt_template = """You are a helpful assistant that answers questions based only on the provided context.

If the answer is clearly in the context, give a complete and accurate answer.
If the answer is NOT in the context, say exactly: "I don't have information about that in the provided documents."
Do not make up information. Do not use knowledge outside the context.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        """Join retrieved document chunks into a single context string."""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # LCEL chain: question → retrieval + passthrough → prompt → LLM → parse
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ──────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print()
    print("=" * 55)
    print("   RAG Document Q&A App")
    print("   Built with LangChain + FAISS + HuggingFace")
    print("=" * 55)
    print()

    # ── Load or create embeddings ──
    embeddings = get_embeddings()

    # ── Load or create vector store ──
    if Path(FAISS_INDEX_PATH).exists():
        print(f"Found existing vector store at '{FAISS_INDEX_PATH}/'")
        rebuild = input("Rebuild from documents? (y/n, default=n): ").strip().lower()
        if rebuild == "y":
            docs = load_documents(DOCS_DIR)
            chunks = split_documents(docs)
            vector_store = create_vector_store(chunks, embeddings, FAISS_INDEX_PATH)
        else:
            vector_store = load_vector_store(embeddings, FAISS_INDEX_PATH)
    else:
        print(f"No existing vector store found. Building from '{DOCS_DIR}/'...")
        docs = load_documents(DOCS_DIR)
        chunks = split_documents(docs)
        vector_store = create_vector_store(chunks, embeddings, FAISS_INDEX_PATH)

    # ── Load LLM ──
    print()
    llm = get_llm()

    # ── Build RAG chain ──
    print()
    print("Building RAG chain...")
    chain, retriever = build_rag_chain(vector_store, llm)

    # ── Ready ──
    print()
    print("=" * 55)
    print("  Ready! Ask questions about your documents.")
    print("  Commands: 'sources' = show last sources | 'exit' = quit")
    print("=" * 55)
    print()

    last_sources = []

    while True:
        try:
            question = input("Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if question.lower() == "sources":
            if last_sources:
                print("\nSources from last question:")
                for i, doc in enumerate(last_sources, 1):
                    src = doc.metadata.get("source", "Unknown")
                    print(f"  [{i}] {src}")
                    print(f"      \"{doc.page_content[:120]}...\"")
                print()
            else:
                print("No sources yet. Ask a question first.\n")
            continue

        # Run the RAG chain
        print("\nSearching documents and generating answer...")
        try:
            answer = chain.invoke(question)
            last_sources = retriever.invoke(question)

            print(f"\nAnswer:\n{answer}")

            print(f"\nSources ({len(last_sources)} chunks retrieved):")
            for i, doc in enumerate(last_sources, 1):
                src = doc.metadata.get("source", "Unknown").split("/")[-1]
                print(f"  [{i}] {src} — \"{doc.page_content[:100]}...\"")
            print()

        except Exception as e:
            print(f"\nError: {e}")
            print("Check your internet connection or HuggingFace token.\n")


if __name__ == "__main__":
    main()
