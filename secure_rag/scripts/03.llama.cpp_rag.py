#!/usr/bin/env python3
import requests
import json
import os
import sys

# --- Конфигурация ---
# URL твоего RAG API сервера (04.integration.py)
RAG_API_URL = "http://localhost:9000/search"
# URL твоего llama-server (обычно 8080)
LLAMA_SERVER_URL = "http://localhost:8080/completion"
# Количество релевантных чанков, которые нужно получить от RAG
K_RETRIEVED_CHUNKS = 3
# Модель LLM, которую ты используешь в llama-server
# Убедись, что это имя соответствует имени модели, загруженной в llama-server
LLM_MODEL_NAME = "saiga_yandexgpt_8b.Q4_K_M.gguf" # Пример: замени на твою модель

# --- Функции ---

def get_rag_context(query: str) -> list:
    """
    Отправляет запрос на RAG API сервер для получения релевантного контекста.
    """
    print(f"\n🔎 Отправляю запрос на RAG API: '{query}'...")
    try:
        response = requests.get(RAG_API_URL, params={"query": query, "k": K_RETRIEVED_CHUNKS})
        response.raise_for_status() # Вызовет исключение для ошибок HTTP (4xx или 5xx)
        data = response.json()
        
        if data and "results" in data:
            print(f"✅ Получено {len(data['results'])} релевантных документов.")
            return data["results"]
        else:
            print("⚠ RAG API вернул пустые или некорректные результаты.")
            return []
    except requests.exceptions.ConnectionError:
        print(f"❌ Ошибка подключения к RAG API серверу по адресу {RAG_API_URL}.")
        print("Убедитесь, что '04.integration.py' запущен и доступен.")
        return []
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка при запросе к RAG API: {e}")
        return []

def generate_llm_response(prompt: str) -> str:
    """
    Отправляет промпт на llama-server и возвращает ответ.
    """
    print(f"\n🧠 Отправляю промпт на llama-server...")
    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": prompt,
        "n_predict": 2048, # Максимальное количество токенов в ответе
        "temperature": 0.7,
        "stop": ["\nUser:", "\n###", "<|im_end|>", "<|endoftext|>"], # Остановки для чата
        "model": LLM_MODEL_NAME # Указываем модель, если llama-server поддерживает
    }
    
    try:
        response = requests.post(LLAMA_SERVER_URL, headers=headers, json=payload, stream=False)
        response.raise_for_status()
        
        # llama-server возвращает ответ в JSON, где "content" содержит текст
        result = response.json()
        if "content" in result:
            print("✅ Получен ответ от llama-server.")
            return result["content"].strip()
        else:
            print("⚠ llama-server вернул некорректный ответ (отсутствует 'content').")
            return "Не удалось получить ответ от LLM."
    except requests.exceptions.ConnectionError:
        print(f"❌ Ошибка подключения к llama-server по адресу {LLAMA_SERVER_URL}.")
        print("Убедитесь, что llama-server запущен и доступен.")
        return "Не удалось подключиться к LLM серверу."
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка при запросе к llama-server: {e}")
        return f"Ошибка при генерации ответа LLM: {e}"

def main():
    print("=== Запуск RAG-системы с LLAMA.cpp ===")
    print("Для выхода введите 'exit'.")

    while True:
        user_query = input("\nТвой вопрос (или 'exit'): ").strip()
        if user_query.lower() == 'exit':
            print("Завершение работы RAG-системы. До свидания!")
            break

        if not user_query:
            print("Пожалуйста, введите вопрос.")
            continue

        # 1. Получаем контекст из RAG API
        retrieved_docs = get_rag_context(user_query)

        context_text = ""
        if retrieved_docs:
            context_text = "\n\n### Контекст из документов:\n"
            for i, doc in enumerate(retrieved_docs):
                context_text += f"Документ {i+1} (Источник: {doc.get('source', 'Неизвестно')}):\n"
                context_text += doc.get("content", "") + "\n---\n"
        else:
            print("⚠ Контекст не найден. Ответ LLM может быть менее точным.")

        # 2. Формируем промпт для LLM
        # Используем шаблон, который хорошо работает с большинством LLM
        prompt_template = f"""
Ты — полезный ассистент, который отвечает на вопросы, используя предоставленный контекст.
Если контекст не содержит достаточной информации, отвечай, что не можешь найти ответ в предоставленных данных.
Не выдумывай информацию.

{context_text}

### Вопрос:
{user_query}

### Ответ:
"""
        # print("\n--- Сформированный промпт для LLM (для отладки) ---")
        # print(prompt_template)
        # print("---------------------------------------------------\n")

        # 3. Отправляем промпт на LLM и получаем ответ
        llm_response = generate_llm_response(prompt_template)
        
        print("\n--- Ответ LLM ---")
        print(llm_response)
        print("-------------------\n")

if __name__ == "__main__":
    main()

