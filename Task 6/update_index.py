# update_index.py
import os
import time
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Конфигурация
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "Task 3", "chroma_db")
COLLECTION_NAME = "starwars_modified"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
NEW_DOCS_DIR = os.path.join(os.path.dirname(__file__), "new_docs")
LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "update.log")

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Настройка логирования
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def log(message, level="info"):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
    if level == "info":
        logging.info(message)
    elif level == "error":
        logging.error(message)

def get_existing_sources(collection):
    """Возвращает множество имён файлов, которые уже есть в индексе"""
    # Получаем все метаданные (может быть много, но у нас мало)
    results = collection.get(include=["metadatas"])
    sources = set()
    for meta in results['metadatas']:
        if meta and 'source' in meta:
            sources.add(meta['source'])
    return sources

def split_into_chunks(text, source_file):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    chunked = []
    for i, chunk in enumerate(chunks):
        chunked.append({
            "text": chunk,
            "metadata": {"source": source_file, "chunk_id": i}
        })
    return chunked

def main():
    log("Запуск обновления индекса...")

    # Загрузка модели эмбеддингов
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        log("Модель эмбеддингов загружена.")
    except Exception as e:
        log(f"Ошибка загрузки модели: {e}", "error")
        return

    # Подключение к ChromaDB
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
        collection = client.get_collection(COLLECTION_NAME)
        log(f"Подключение к коллекции {COLLECTION_NAME}, текущее количество записей: {collection.count()}")
    except Exception as e:
        log(f"Ошибка подключения к ChromaDB: {e}", "error")
        return

    # Получаем уже существующие источники
    existing_sources = get_existing_sources(collection)
    log(f"Найдено {len(existing_sources)} уникальных файлов в индексе.")

    # Сканируем папку new_docs
    if not os.path.exists(NEW_DOCS_DIR):
        log(f"Папка {NEW_DOCS_DIR} не существует. Создаю.", "warning")
        os.makedirs(NEW_DOCS_DIR, exist_ok=True)
        return

    new_files = []
    for filename in os.listdir(NEW_DOCS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(NEW_DOCS_DIR, filename)
            # Проверяем, есть ли уже этот файл в индексе
            if filename not in existing_sources:
                new_files.append(filepath)
            else:
                log(f"Файл {filename} уже есть в индексе, пропускаем.")

    if not new_files:
        log("Новых файлов не найдено. Завершение.")
        return

    log(f"Найдено {len(new_files)} новых файлов для обработки.")

    total_chunks = 0
    for filepath in new_files:
        filename = os.path.basename(filepath)
        log(f"Обработка файла: {filename}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()

            # Разбиваем на чанки
            chunks = split_into_chunks(text, filename)
            if not chunks:
                log(f"Файл {filename} не дал чанков, пропускаем.")
                continue

            # Генерируем эмбеддинги для всех чанков файла
            texts = [chunk["text"] for chunk in chunks]
            embeddings = embedding_model.encode(texts, show_progress_bar=False).tolist()

            # Подготовка данных для добавления
            ids = [f"{filename}_chunk_{chunk['metadata']['chunk_id']}_{int(time.time())}" for chunk in chunks]
            documents = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]

            # Добавляем в коллекцию
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )

            total_chunks += len(chunks)
            log(f"Добавлено {len(chunks)} чанков из файла {filename}.")

        except Exception as e:
            log(f"Ошибка при обработке файла {filename}: {e}", "error")

    log(f"Обновление завершено. Всего добавлено новых чанков: {total_chunks}. "
        f"Теперь в коллекции {collection.count()} записей.")

if __name__ == "__main__":
    main()