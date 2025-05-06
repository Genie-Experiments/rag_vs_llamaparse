import os
import time
import json
import base64
from dotenv import load_dotenv
from llama_cloud_services.parse import ResultType
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.llama_parse import LlamaParse
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List, Dict, Any

# === Load API keys ===
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

# === Setup models ===
llm = Groq(
    model="llama3-8b-8192",
    api_key=groq_api_key,
    system_prompt="You are a helpful assistant. Only answer questions using the provided context. If the answer is not in the context, respond with 'I don't know'."
)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
parser = SentenceSplitter(chunk_size=300, chunk_overlap=50)

# === PDF files ===
files_to_load = [
    "add/path/to/files"
]

os.makedirs("extracted_images", exist_ok=True)


# === Load and parse each file with fallback ===
def load_and_parse_file(file_path: str, llama_cloud_api_key: str) -> List[Dict[str, Any]]:
    for mode in [ResultType.JSON, ResultType.MD]:
        try:
            reader = LlamaParse(api_key=llama_cloud_api_key, result_type=mode, verbose=True)
            docs = reader.load_data(file_path)
            if docs:
                print(f"✅ Parsed {len(docs)} docs from {file_path} using {mode.value}")
                return docs
        except Exception as e:
            print(f"⚠️ Error using {mode.value} for {file_path}: {e}")
    print(f"❌ Failed to parse any docs from {file_path}")
    return []


# === Parse documents and extract images ===
documents = []
start_time = time.time()

for file_path in files_to_load:
    docs = load_and_parse_file(file_path, llama_cloud_api_key)
    for doc_idx, doc in enumerate(docs):
        # Extract images from metadata if they exist
        elements = doc.metadata.get("elements", [])
        image_elements = [el for el in elements if el.get("type") == "image"]

        for idx, image in enumerate(image_elements):
            image_data = image.get("data")
            if image_data:
                try:
                    image_bytes = base64.b64decode(image_data)
                    filename = f"{os.path.basename(file_path).replace(' ', '_')}_img{doc_idx + 1}_{idx + 1}.png"
                    image_path = os.path.join("extracted_images", filename)
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                except Exception as e:
                    print(f"⚠️ Failed to extract image from {file_path}: {e}")

        # Check if document has text content
        if not doc.text.strip():
            print(f"⚠️ WARNING: No textual content extracted from {file_path} (document {doc_idx + 1})")
            continue

        documents.append(doc)

print(f"\n✅ Loaded {len(documents)} documents with textual content.")

# === Convert to nodes ===
nodes = parser.get_nodes_from_documents(documents)
print(f"✅ Generated {len(nodes)} nodes")

# === Build index and query engine ===
index = VectorStoreIndex(nodes, embed_model=embed_model)
query_engine = index.as_query_engine(llm=llm, similarity_top_k=4, verbose=True)

end_time = time.time()
print("\n⏱️ Parsing and Indexing Duration:", round(end_time - start_time, 2), "seconds")

# === Load questions ===
dataset_path = "llamaparse_dataset.json"
with open(dataset_path, "r") as f:
    dataset = json.load(f)

# === Run QA and append answers ===
for item in dataset:
    question = item.get("question")
    if not question:
        continue

    print(f"\n🔍 Asking: {question}")
    try:
        response = query_engine.query(question)

        # Get source documents text only
        source_texts = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes[:4]:  # Get top 4 sources
                source_texts.append(node.node.get_content())

        # Store results
        item["llamaparse answer"] = str(response)
        item["llamaparse source docs"] = source_texts  # Just store the text list
        print("💬 Answer:", response)
        print("📄 Source documents:", [text[:100] + "..." for text in source_texts])  # Print first 100 chars

    except Exception as e:
        print(f"⚠️ Failed to answer question: {e}")
        item["llamaparse answer"] = "Error"
        item["llamaparse source docs"] = []

# === Save updated dataset ===
with open(dataset_path, "w") as f:
    json.dump(dataset, f, indent=2)

print("\n✅ All questions answered and saved to llamaparse_dataset.json.")