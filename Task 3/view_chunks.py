import chromadb
from chromadb.config import Settings

# Подключаемся к базе
client = chromadb.PersistentClient(path="./chroma_db", settings=Settings(anonymized_telemetry=False))
collection = client.get_collection("starwars_modified")

# Получаем все чанки
results = collection.get(include=["documents", "metadatas"])

# Выводим информацию
print(f"Всего чанков: {len(results['ids'])}")
print("-" * 50)

for i, (doc_id, doc, meta) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
    print(f"\n📄 Чанк #{i+1} (ID: {doc_id})")
    print(f"📁 Источник: {meta['source']} (чанк {meta['chunk_id']})")
    print(f"📝 Текст: {doc}")
    print("-" * 50)