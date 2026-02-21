import os
import time
import json
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ------------------------------
# Конфигурация (пути для macOS)
# ------------------------------
# Определяем абсолютный путь к папке knowledge_base относительно текущего файла
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "..", "Task 2", "knowledge_base")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "starwars_modified"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# ------------------------------
# 1. Загрузка модели эмбеддингов
# ------------------------------
print(f"Загрузка модели эмбеддингов: {EMBEDDING_MODEL_NAME} ...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(f"Модель загружена. Размерность эмбеддингов: {embedding_model.get_sentence_embedding_dimension()}")


# ------------------------------
# 2. Чтение и чанкинг документов
# ------------------------------
def load_documents(folder: str) -> List[Dict]:
    docs = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            docs.append({
                "text": text,
                "metadata": {"source": filename}
            })
    print(f"Загружено {len(docs)} документов.")
    return docs


def split_documents(docs: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
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
                "metadata": {
                    "source": doc["metadata"]["source"],
                    "chunk_id": i
                }
            })
    print(f"Создано {len(chunked_docs)} чанков.")
    return chunked_docs


# ------------------------------
# 3. Генерация эмбеддингов
# ------------------------------
def generate_embeddings(chunks: List[Dict]) -> List[Dict]:
    texts = [chunk["text"] for chunk in chunks]
    print(f"Генерация эмбеддингов для {len(texts)} чанков...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()
    return chunks


# ------------------------------
# 4. Сохранение в ChromaDB
# ------------------------------
def create_chroma_index(chunks: List[Dict], persist_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))

    # Удаляем старую коллекцию, если есть
    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.create_collection(name=collection_name)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    embeddings = [chunk["embedding"] for chunk in chunks]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )
    print(f"Индекс создан. Всего записей: {collection.count()}")
    return collection


# ------------------------------
# 5. Тестовый поиск
# ------------------------------
def test_search(collection, query: str, top_k: int = 3):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    print(f"\nЗапрос: {query}")
    for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
    )):
        print(f"\n--- Результат {i + 1} (расстояние: {dist:.4f}) ---")
        print(f"Источник: {meta['source']} (чанк {meta['chunk_id']})")
        print(f"Текст: {doc[:200]}...")


# ------------------------------
# Основной процесс
# ------------------------------
if __name__ == "__main__":
    start_time = time.time()

    raw_docs = load_documents(KNOWLEDGE_BASE_DIR)
    if not raw_docs:
        print("Ошибка: не найдены .txt файлы в", KNOWLEDGE_BASE_DIR)
        exit(1)

    chunks = split_documents(raw_docs, CHUNK_SIZE, CHUNK_OVERLAP)
    chunks_with_emb = generate_embeddings(chunks)
    collection = create_chroma_index(chunks_with_emb, CHROMA_PERSIST_DIR, COLLECTION_NAME)

    elapsed = time.time() - start_time
    print(f"\nИндексация завершена за {elapsed:.2f} секунд.")

    # Метаинформация
    meta_info = {
        "model": EMBEDDING_MODEL_NAME,
        "embedding_dim": embedding_model.get_sentence_embedding_dimension(),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "num_documents": len(raw_docs),
        "num_chunks": len(chunks),
        "time_seconds": round(elapsed, 2)
    }
    with open(os.path.join(BASE_DIR, "index_meta.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    # Тестовые запросы
    print("\n=== Тестирование поиска ===")
    test_search(collection, "Кто такой Илья Звездин?")
    test_search(collection, "Что такое Поток?")
    test_search(collection, "Где находится Песчаная?")

    print("\n✅ Индекс успешно создан и протестирован.")