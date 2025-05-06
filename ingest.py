import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment
load_dotenv()

start_time = time.time()
# Load and read PDF documents
loader = DirectoryLoader(
    "docs",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader  # This loader can handle PDFs
)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks.")

if not docs:
    print("⚠️ No chunks created. Check your document content or adjust the splitter.")
    exit()

# Embeddings (local, no API key needed)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Store in Chroma
vectordb = Chroma.from_documents(docs, embedding, persist_directory="docs/chroma")
vectordb.persist()
end_time = time.time()
duration = end_time - start_time
print("***DURATION*****", duration)
print("✅ ChromaDB created and persisted successfully.")
