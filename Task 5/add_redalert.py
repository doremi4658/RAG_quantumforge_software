import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Пути
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "Task 3", "chroma_db")
COLLECTION_NAME = "starwars_modified"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Загружаем модель эмбеддингов
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Подключаемся к ChromaDB
client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
collection = client.get_collection(COLLECTION_NAME)

# Текст вредоносного файла (читаем его из файла, чтобы быть уверенным в содержимом)
malicious_path = os.path.join(os.path.dirname(__file__), "..", "Task 2", "knowledge_base", "вредоносный_файл.txt")
with open(malicious_path, "r", encoding="utf-8") as f:
    malicious_text = f.read().strip()

# Генерируем эмбеддинг
embedding = embedding_model.encode(malicious_text).tolist()

# Добавляем в коллекцию
collection.add(
    ids=["malicious_doc"],
    documents=[malicious_text],
    metadatas=[{"source": "вредоносный_файл.txt.txt", "chunk_id": 0}],
    embeddings=[embedding]
)

print("✅ Вредоносный документ успешно добавлен в базу.")
print(f"Теперь в коллекции {collection.count()} чанков.")