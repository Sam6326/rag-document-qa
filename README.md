# RAG Document Q&A App

**Ask questions about your own documents using AI — powered by LangChain, FAISS, and HuggingFace**

Built by: Sampath Kumar | Portfolio project demonstrating RAG architecture for AI Engineer roles

---

## What is this?

This is a **RAG (Retrieval-Augmented Generation)** application. You give it a folder of text documents, and you can ask it questions in plain English. It finds the relevant parts of your documents and uses an AI model to generate a grounded answer — it does not make things up.

---

## How RAG Works (Simple Explanation)

Imagine you want an AI to answer questions about a 500-page manual. The problem: AI models have a limited "reading window" and may hallucinate (make up) facts they do not know.

RAG solves this in 5 steps:

```
Your documents → Split into chunks → Convert to vectors (numbers) → Store in FAISS
                                                                            ↓
Your question → Convert to vector → Find closest chunks → Send to AI → Get answer
```

**Step by step:**
1. Load your `.txt` files from the `data/` folder
2. Split them into small overlapping chunks (500 characters each)
3. Convert each chunk to a vector using `sentence-transformers/all-MiniLM-L6-v2`
4. Store all vectors in **FAISS** — a fast local vector database by Meta/Facebook
5. When you ask a question:
   - Convert your question to a vector
   - Find the 3 most similar document chunks
   - Send those chunks + your question to the LLM
   - Get a factual answer grounded in your documents

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Framework | LangChain | Connects LLMs to external data cleanly |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Free, fast, runs locally |
| Vector Store | FAISS | Free, local, no server needed |
| LLM (free) | google/flan-t5-base | Runs locally, no API key needed |
| LLM (better) | Mistral-7B via HuggingFace API | Better answers, needs free HF token |
| Chain | LangChain LCEL | Modern composable pipeline |

---

## Requirements

- Python 3.9 or higher
- ~500MB disk space (for model downloads)
- Internet connection (first run only, to download models)
- A free HuggingFace account (optional but recommended)

---

## Setup — Step by Step

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-document-qa.git
cd rag-document-qa
```

### Step 2: Create a virtual environment

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

This installs: LangChain, FAISS, sentence-transformers, HuggingFace Transformers, and everything else needed.

### Step 4: Set up your environment file

```bash
cp .env.example .env
```

Open `.env` in any text editor. If you want better answers, add your free HuggingFace token:
- Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Sign up (free) → New token → Read permissions → Copy it
- Paste it into `.env` as: `HUGGINGFACEHUB_API_TOKEN=your_token_here`

**Without a token:** Uses `google/flan-t5-base` locally (free, simpler answers)  
**With a token:** Uses `Mistral-7B-Instruct` via HuggingFace API (better answers)

### Step 5: Add your documents

The `data/` folder already has two sample documents:
- `ai_overview.txt` — overview of AI, RAG, LLMs, agents
- `nlp_and_finance.txt` — sentiment analysis, VADER, FinBERT, NLP concepts

**To use your own documents:** Add any `.txt` file to the `data/` folder. Delete the `faiss_index/` folder if it exists (so it rebuilds from your new documents).

### Step 6: Run the app

```bash
python app.py
```

**First run:** Downloads models (~80MB embedding model + ~250MB LLM if no HF token). Takes 2-5 minutes. After that, cached permanently.

**Second run:** Loads from cache. Starts in seconds.

---

## Example Usage

```
====================================================
   RAG Document Q&A App
   Built with LangChain + FAISS + HuggingFace
====================================================

Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
Loaded existing vector store from 'faiss_index/'
No HF token found — using local model: google/flan-t5-base

Building RAG chain...

====================================================
  Ready! Ask questions about your documents.
  Commands: 'sources' = show last sources | 'exit' = quit
====================================================

Your question: What is RAG and how does it work?

Searching documents and generating answer...

Answer:
RAG is a technique that combines a vector database with a Large Language Model.
It works by converting documents into vectors, storing them in FAISS, and at
query time finding the most similar chunks to the question and sending them to
the LLM to generate a grounded answer.

Sources (3 chunks retrieved):
  [1] ai_overview.txt — "RAG is a technique that combines a vector database with a Large Language Model..."
  [2] ai_overview.txt — "Step 1 — Load your documents into the system. Step 2 — Break the documents..."
  [3] nlp_and_finance.txt — "REST APIs allow different software systems to communicate..."

Your question: What is FinBERT?

Answer:
FinBERT is a version of BERT that has been fine-tuned on financial text,
published by ProsusAI and available on HuggingFace as ProsusAI/FinBERT.
It classifies financial text as positive, negative, or neutral.

Your question: exit
Goodbye!
```

---

## Project Structure

```
rag-document-qa/
├── app.py              ← Main application (all RAG logic)
├── requirements.txt    ← Python dependencies
├── .env.example        ← Template for environment variables
├── .env                
├── .gitignore          
├── data/
│   ├── ai_overview.txt       ← Sample document 1
│   └── nlp_and_finance.txt   ← Sample document 2
└── faiss_index/        ← Auto-created on first run (NOT on GitHub)
    ├── index.faiss
    └── index.pkl
```

---

## Adding More Documents

Just add `.txt` files to the `data/` folder and delete the `faiss_index/` folder:

```bash
rm -rf faiss_index/    # Mac/Linux
# OR on Windows: delete the faiss_index folder manually
```

Then run the app again. It will rebuild the vector store from all documents.

**Supported formats:** Currently `.txt` files. PDF support is easy to add with `pypdf`.

---

## How to Add PDF Support (Optional)

Install: `pip install pypdf`

In `app.py`, change the loader in `load_documents()`:
```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    directory,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
)
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                 INDEXING PHASE (once)                │
│                                                     │
│  data/*.txt → TextLoader → RecursiveTextSplitter    │
│                              ↓                      │
│               HuggingFaceEmbeddings (MiniLM)        │
│                              ↓                      │
│               FAISS Vector Store (saved to disk)    │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                 QUERY PHASE (each question)          │
│                                                     │
│  User Question → HuggingFaceEmbeddings              │
│                      ↓                              │
│               FAISS similarity search               │
│                      ↓                              │
│          Top 3 most relevant chunks                 │
│                      ↓                              │
│  PromptTemplate (chunks + question)                 │
│                      ↓                              │
│  LLM (flan-t5-base or Mistral-7B)                  │
│                      ↓                              │
│          StrOutputParser → Answer                   │
└─────────────────────────────────────────────────────┘
```

## Troubleshooting

**"OSError: We couldn't connect to HuggingFace"**
You need internet to download models on first run. Connect and try again.

**"No .txt files found in data/"**
Make sure your documents are `.txt` format and are inside the `data/` folder.

**"ModuleNotFoundError"**
Run `pip install -r requirements.txt` again. Make sure your virtual environment is activated.

**Answers are short or vague**
Add a HuggingFace token to your `.env` file to use Mistral-7B instead of flan-t5-base.

---

## License

MIT License — free to use, modify, and share.
