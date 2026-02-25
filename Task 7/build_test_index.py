import os
import time
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Пути для тестового индекса
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base_test")
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "test_chroma_db")
COLLECTION_NAME = "starwars_modified_test"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

def load_documents(folder):
    docs = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            docs.append({"text": text, "metadata": {"source": filename}})
    print(f"Загружено {len(docs)} документов.")
    return docs

def split_documents(docs, chunk_size, overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunked_docs = []
    for doc in docs:
        chunks = text_splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                "text": chunk,
                "metadata": {"source": doc["metadata"]["source"], "chunk_id": i}
            })
    print(f"Создано {len(chunked_docs)} чанков.")
    return chunked_docs

def generate_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()
    return chunks

def create_chroma_index(chunks, persist_dir, collection_name):
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.create_collection(name=collection_name)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    embeddings = [chunk["embedding"] for chunk in chunks]
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    print(f"Индекс создан. Всего записей: {collection.count()}")
    return collection

if __name__ == "__main__":
    start = time.time()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    raw_docs = load_documents(KNOWLEDGE_BASE_DIR)
    chunks = split_documents(raw_docs, CHUNK_SIZE, CHUNK_OVERLAP)
    chunks_with_emb = generate_embeddings(chunks, embedding_model)
    collection = create_chroma_index(chunks_with_emb, CHROMA_PERSIST_DIR, COLLECTION_NAME)
    elapsed = time.time() - start
    print(f"Индексация завершена за {elapsed:.2f} сек.")