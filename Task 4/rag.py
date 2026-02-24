import os
import requests
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Пути
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "Task 3", "chroma_db")
COLLECTION_NAME = "starwars_modified"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"  # или "llama3"

# Загружаем модель эмбеддингов
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Подключаемся к ChromaDB
client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
collection = client.get_collection(COLLECTION_NAME)

# Few-shot примеры
FEW_SHOT_EXAMPLES = """
Пример 1:
Вопрос: Кто такой Илья Звездин?
Контекст: Илья Звездин — человек, чувствительный к Потоку, сын Андрея Звездина. Обучался у Добрыни Светлова и Яромира Мудрого.
Ответ: Илья Звездин — сын Андрея Звездина, обучался у Добрыни Светлова и Яромира Мудрого, чувствителен к Потоку.

Пример 2:
Вопрос: Что такое Поток?
Контекст: Поток — энергетическое поле, дающее способности: телекинез, ускорение, предвидение.
Ответ: Поток — это энергетическое поле, которое даёт способности к телекинезу, ускорению и предвидению.
"""

# Chain-of-Thought инструкция
COT_INSTRUCTION = """
Ты — помощник, который всегда сначала размышляет, а потом отвечает.
Пожалуйста, следуй этим шагам:
1. Прочитай контекст из документов.
2. Найди информацию, релевантную вопросу.
3. Если информации нет, скажи "Я не знаю ответа на этот вопрос".
4. Если информация есть, сформулируй ответ на её основе.
5. В конце укажи источник (название файла).
"""

def search_chunks(query, top_k=5):
    """Поиск топ-k чанков по запросу"""
    query_emb = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return results

def build_prompt(query, retrieved_chunks):
    """Формирует промпт для LLM"""
    # Собираем контекст из найденных чанков
    context_parts = []
    for doc, meta in zip(retrieved_chunks['documents'][0], retrieved_chunks['metadatas'][0]):
        context_parts.append(f"[Из файла {meta['source']}]: {doc}")
    context = "\n\n".join(context_parts)

    prompt = f"""{COT_INSTRUCTION}

{FEW_SHOT_EXAMPLES}

Теперь ответь на следующий вопрос, используя только предоставленный контекст.

Контекст:
{context}

Вопрос: {query}

Твой ответ (сначала шаги размышления, потом ответ):"""
    return prompt

def ask_ollama(prompt):
    """Отправляет промпт в Ollama и возвращает ответ"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.3
    }
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Ошибка Ollama: {response.status_code}"

def ask_rag(question):
    """Основная функция: вопрос -> ответ"""
    # Поиск чанков
    chunks = search_chunks(question)

    # Если чанков нет или они слишком далеки (опционально проверяем расстояние)
    if not chunks['documents'][0]:
        return "Я не нашёл информации по вашему вопросу в базе знаний."

    # Формируем промпт
    prompt = build_prompt(question, chunks)

    # Отправляем в Ollama
    answer = ask_ollama(prompt)

    # Постобработка (можно добавить)
    return answer