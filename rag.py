import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import requests
from tqdm import tqdm
import time
import io
import contextlib



# --- DOWNLOAD MODEL ---
model_path = "granite-3.3-2b-instruct-BF16.gguf"

if not os.path.exists(model_path):
    print(f"Downloading {model_path}...")
    model_url = "https://huggingface.co/unsloth/granite-3.3-2b-instruct-GGUF/resolve/main/granite-3.3-2b-instruct-BF16.gguf"
    response = requests.get(model_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(model_path, 'wb') as f:
        for data in tqdm(response.iter_content(chunk_size=1024), total=total_size//1024):
            f.write(data)
    print("Download complete!")

# --- PREPARE DOCUMENTS ---
os.makedirs("docs", exist_ok=True)

documents = []
for file in os.listdir("docs"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join("docs", file))
        documents.extend(loader.load())
    elif file.endswith(".txt"):
        loader = TextLoader(os.path.join("docs", file))
        documents.extend(loader.load())

# --- TEXT SPLITTING ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = text_splitter.split_documents(documents)

# --- EMBEDDINGS & VECTORSTORE ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# --- LLAMA SETUP ---
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.7,
    max_tokens=4000,
    n_ctx=8196,
    n_gpu_layers=999,
    n_batch=256,
    verbose=False
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def simple_rag_chain(question):
    """Simple RAG implementation without complex chains"""
    # Retrieve documents
    docs = retriever.invoke(question)

    # Format context
    context = "\n\n".join([doc.page_content for doc in docs])

    full_prompt = f"""You are a technical expert. Fill in ONLY the blanks below. Do NOT add extra text.\n\n

    Template:\n
    **Definition:** [1 sentence]\n
    **Key Points:**\n
    • [point 1]\n
    • [point 2]\n
    • [point 3]\n\n

Context: {context}

Question: {question}

Answer: """

    answer = llm.invoke(full_prompt)

    return {
        "answer": answer,
        "context": docs,
        "question": question
    }

def ask_question_simple(question):
    """Using simple implementation - more readable"""
    start_time = time.time()

    result = simple_rag_chain(question)

    end_time = time.time()

    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("\nContext used:")
    for i, doc in enumerate(result['context'][:2]):
        print(f"\nDoc {i+1}: {doc.page_content[:200]}...")

    return result

def ask_question_simple_wrapper(q: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ask_question_simple(q)
    return buf.getvalue().strip()

# --- MAIN ---

if __name__ == "__main__":
    import sys

    # Pipes: echo "hi" | python rag.py)
    if not sys.stdin.isatty():
        question = sys.stdin.read().strip()
        print(ask_question(question))
        sys.exit(0)

    # Interactive
    print("Interactive mode – type '/bye' or Ctrl-D to quit.\n")
    try:
        while True:
            try:
                question = input("🤖 > ").strip()
            except EOFError:          # Ctrl-D
                break
            if question.lower() in {"/exit", "/quit", "/bye"}:
                break
            if not question:
                continue
            print(ask_question_simple_wrapper(question))
    except KeyboardInterrupt:         # Ctrl-C
        pass
    print("\nBye.")
