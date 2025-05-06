import os
import sys
import datetime
import json
from dotenv import load_dotenv, find_dotenv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append('../..')
_ = load_dotenv(find_dotenv())

# Load API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Missing GROQ_API_KEY in environment")

# Model selection
current_date = datetime.datetime.now().date()
llm_name = "mixtral-8x7b-32768" if current_date >= datetime.date(2023, 9, 2) else "llama2-70b-4096"
print(f"Using model: {llm_name}")

# === LangChain Vector DB ===
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

persist_directory = 'docs/chroma/'
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

print(f"Number of documents in vector DB: {vectordb._collection.count()}")

# === LLM Setup ===
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=groq_api_key)

# === QA Chain Setup ===
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum. Keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(similarity_top_k=4),  # Get top 4 documents
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# === Load questions from llamaparse_dataset.json ===
dataset_path = "llamaparse_dataset.json"
with open(dataset_path, "r") as f:
    dataset = json.load(f)

# === Process each question ===
for item in dataset:
    question = item.get("question")
    if not question:
        continue

    print(f"\n🔍 Asking: {question}")
    try:
        result = qa_chain({"query": question})

        # Get source documents text only
        source_texts = [doc.page_content for doc in result['source_documents']]

        # Store results in the dataset
        item["base rag answer"] = result["result"]
        item["base rag source docs"] = source_texts

        print("💬 Answer:", result["result"])
        print("📄 Source documents:", [text[:100] + "..." for text in source_texts])  # Print preview

    except Exception as e:
        print(f"⚠️ Failed to answer question: {e}")
        item["base rag answer"] = "Error"
        item["base rag source docs"] = []

# === Save updated dataset ===
with open(dataset_path, "w") as f:
    json.dump(dataset, f, indent=2)

print("\n✅ All questions answered and saved to llamaparse_dataset.json.")