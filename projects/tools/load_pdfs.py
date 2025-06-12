import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
import os

# === Config ===
PDF_DIR = "data/pdfs"  # Create this folder and put PDFs inside
COLLECTION_NAME = "second_brain"
CHUNK_SIZE = 500

# === Setup ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="chroma_storage")
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# === Load and chunk PDF text ===
def extract_chunks_from_pdf(file_path, chunk_size=500):
    doc = fitz.open(file_path)
    full_text = ""
    page_count = len(doc)

    print(f"🔍 Reading {page_count} pages from {file_path}")
    for i, page in enumerate(doc):
        page_text = page.get_text()
        print(f"Page {i+1}/{page_count}: {len(page_text)} characters")
        if len(page_text.strip()) == 0:
            print(f"⚠️ WARNING: No text found on page {i+1}")
        full_text += page_text + "\n"

    doc.close()

    if len(full_text.strip()) == 0:
        print("❌ ERROR: No text found in entire document.")
        return []

    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    print(f"✅ Total text length: {len(full_text)} characters, divided into {len(chunks)} chunks")
    return chunks

# === Embed and add to ChromaDB ===
def embed_and_store(chunks, source_tag):
    ids = [f"{source_tag}-chunk-{i}" for i in range(len(chunks))]
    embeddings = embedder.encode(chunks).tolist()
    metadatas = [{"source": source_tag} for _ in chunks]
    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
    print(f"✅ Stored {len(chunks)} chunks from {source_tag}")

# === Load all PDFs in folder ===
def load_all_pdfs(pdf_dir):
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, filename)
            print(f"📄 Processing: {filename}")
            chunks = extract_chunks_from_pdf(path)
            embed_and_store(chunks, source_tag=filename)

# === Run ===
if __name__ == "__main__":
    if not os.path.exists(PDF_DIR):
        print(f"Error: Folder '{PDF_DIR}' does not exist.")
    else:
        load_all_pdfs(PDF_DIR)
