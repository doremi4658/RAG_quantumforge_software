import os
import csv
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests

# ========== НАСТРОЙКИ ==========
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "test_chroma_db")
COLLECTION_NAME = "starwars_modified_test"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
GOLDEN_FILE = "golden_questions.txt"
LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "evaluation_log.csv")
TOP_K = 5
# ================================

# Загрузка модели эмбеддингов
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Подключение к ChromaDB
client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
collection = client.get_collection(COLLECTION_NAME)

def search_chunks(query, top_k=TOP_K):
    query_emb = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return results

def build_prompt(query, retrieved_chunks):
    context_parts = []
    for doc, meta in zip(retrieved_chunks['documents'][0], retrieved_chunks['metadatas'][0]):
        context_parts.append(f"[Из файла {meta['source']}]: {doc}")
    context = "\n\n".join(context_parts)

    system_instruction = """
    Ты — помощник, отвечающий строго на основе предоставленного КОНТЕКСТА (документов).
    КРИТИЧЕСКИ ВАЖНО:
    - Если в контексте нет информации, необходимой для ответа на вопрос, ты должен ответить ТОЛЬКО одной фразой: «Я не знаю ответа на этот вопрос».
    - НЕ используй свои общие знания, НЕ выдумывай, НЕ дополняй отсутствующую информацию.
    - Даже если вопрос кажется простым или относится к известным темам, НЕ ОТВЕЧАЙ, если этого нет в контексте.
    - Не объясняй, почему не знаешь, не перечисляй источники – только фраза отказа.
    """

    cot_instruction = """
    Ты — помощник, который сначала размышляет, а потом отвечает. Шаги:
    1. Прочитай контекст из документов.
    2. Найди информацию, релевантную вопросу.
    3. Если информации нет, скажи "Я не знаю ответа на этот вопрос".
    4. Если информация есть, сформулируй ответ на её основе.
    """

    prompt = f"{system_instruction}\n{cot_instruction}\nКонтекст:\n{context}\nВопрос: {query}\nТвой ответ:"
    return prompt

def ask_ollama(prompt):
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "temperature": 0.3}
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Ошибка {response.status_code}"
    except Exception as e:
        return f"Ошибка соединения: {e}"

def ask_rag(question):
    chunks = search_chunks(question)
    if not chunks['documents'][0]:
        return "Я не нашёл информации по вашему вопросу в базе знаний.", [], 0

    threshold = 0.3
    distances = chunks['distances'][0]
    if min(distances) > threshold:
        return "Я не знаю ответа на этот вопрос.", [], 0

    prompt = build_prompt(question, chunks)
    answer = ask_ollama(prompt)

    # Пост-фильтр (если ответ не содержит отказа, но расстояние большое)
    if "не знаю" not in answer.lower() and min(distances) > 0.3:
        return "Я не знаю ответа на этот вопрос.", [], 0

    return answer, chunks['metadatas'][0], len(answer)

def load_golden_questions(filepath):
    questions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(';')
            if len(parts) >= 2:
                q = parts[0].strip()
                expected = parts[1].strip()
                topic = parts[2].strip() if len(parts) > 2 else ''
                questions.append((q, expected, topic))
    return questions

def is_answer_correct(answer, expected, topic):
    answer_lower = answer.lower()
    # Если тема отсутствующая или удалённая – ожидаем отказ
    if topic in ["отсутствующая", "удалённая"]:
        return any(phrase in answer_lower for phrase in ["не знаю", "не нашёл", "нет информации"])
    else:
        # Существующая тема – ожидаем содержательный ответ
        if len(answer) < 20 or "не знаю" in answer_lower:
            return False
        # Проверяем наличие ключевых слов из expected (первые 3 слова)
        keywords = expected.split()[:3]
        return any(kw.lower() in answer_lower for kw in keywords)

def main():
    print("Запуск оценки RAG-системы...")
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    questions = load_golden_questions(GOLDEN_FILE)
    print(f"Загружено {len(questions)} вопросов.")

    with open(LOG_FILE, 'w', newline='', encoding='utf-8-sig') as csvfile:  # utf-8-sig для корректного отображения в Excel
        fieldnames = ['Время', 'Вопрос', 'Ожидание', 'Ответ', 'Источники', 'Длина_ответа', 'Чанков_найдено', 'Корректно']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for q, expected, topic in questions:
            print(f"\nВопрос: {q}")
            answer, sources, ans_len = ask_rag(q)
            chunks_found = len(sources)
            correct = is_answer_correct(answer, expected, topic)
            sources_str = ', '.join([s['source'] for s in sources]) if sources else ''

            writer.writerow({
                'Время': datetime.now().isoformat(),
                'Вопрос': q,
                'Ожидание': expected,
                'Ответ': answer,
                'Источники': sources_str,
                'Длина_ответа': ans_len,
                'Чанков_найдено': chunks_found,
                'Корректно': correct
            })
            print(f"Ответ: {answer[:100]}...")
            print(f"Корректно? {correct}")

    print(f"\nОценка завершена. Лог сохранён в {LOG_FILE}")

if __name__ == "__main__":
    main()