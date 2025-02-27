import json
import faiss
import argparse
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
with open(r'config.json') as config_file:
    config_details = json.load(config_file)
openai_api_key = config_details['config_list'][0]['api_key']

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# todo: switch to open ai embedding
# embedding_model = OpenAIEmbeddings(api_key=openai_api_key)
# Load PDF and extract text
FAISS_VECTOR_STORE = "faiss_vector_store"
def extract_text_from_pdf(pdf_path):
    """Load and extract text from a PDF"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return " ".join([doc.page_content for doc in documents])


# Chunk the extracted text
def chunk_text(text, chunk_size=100, chunk_overlap=50):
    """Split text into smaller chunks for embedding"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   separators='.')
    return text_splitter.split_text(text)


def create_and_load_vector_store(pdf_path = "sample.pdf", vector_store_name = FAISS_VECTOR_STORE):
    # Example Usage
    pdf_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(pdf_text)

    print(f"Extracted {len(chunks)} text chunks from the PDF.")
    # print(f"\nRetrieved Policies:\n {chunks}")
    # Sample Guidelines
    # chunks = [
    #     "Terminated employees must have all access revoked within 24 hours.",
    #     "Contractors should only have limited access to project-related resources.",
    #     "Admins should not have access to HR data.",
    #     "Inactive accounts must be disabled after 90 days.",
    #     "All employees must enable multi-factor authentication (MFA)."
    # ]
    # Generate embeddings for chunks
    chunk_embeddings = embedding_model.embed_documents(chunks)
    chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)

    # Create FAISS index
    dimension = chunk_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(chunk_embeddings)

    # Create FAISS vector store with metadata
    vector_store = FAISS(
        index=faiss_index,
        embedding_function=embedding_model,
        docstore=InMemoryDocstore(
            {str(i): Document(page_content=chunks[i]) for i in range(len(chunks))}),
        index_to_docstore_id={i: str(i) for i in range(len(chunks))}
    )

    vector_store.save_local(vector_store_name)
    print("vector stored.")

    # Load the stored vector store
    vector_store = FAISS.load_local(vector_store_name,
                                    embedding_model,
                                    allow_dangerous_deserialization=True)
    # Search FAISS index
    # distances, indices = faiss_index.search(query_embedding, top_k)
    # guideline_texts = {i: text for i, text in enumerate(chunks)}
    # return [guideline_texts[i] for i in indices[0] if i in guideline_texts]

    # search FAISS vector store
    print('vector loaded')
    return vector_store


def retrieve_compliance_guidelines(vector_store, query, top_k=5):
    """Retrieve relevant access policies from FAISS"""

    results = vector_store.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]


def main(pdf_path, query):
    vector_store = create_and_load_vector_store(pdf_path)
    retrieved_policies = retrieve_compliance_guidelines(vector_store, query)
    return retrieved_policies


# Example Query

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run web automation task with a specified prompt.")
    parser.add_argument("--pdf_path", type=str, default="sample.pdf", help="path for pdf files.")
    parser.add_argument("--query", type=str,
                        default="retrieve guidelines for a terminated employee",
                        help="The prompt to run.")

    args = parser.parse_args()
    print(f"Received argument: {args.pdf_path}, {args.query}")
    policies = main(args.pdf_path, args.query)
    print("\nRetrieved Policies:\n", policies)
