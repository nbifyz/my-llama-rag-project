#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
import uvicorn # Добавлен явный импорт uvicorn

app = FastAPI()

# Конфигурация
DB_PATH = os.path.expanduser("~/secure_rag/vector_db")
# Локальный путь к модели эмбеддингов
EMBEDDING_MODEL_PATH = os.path.expanduser("~/models/embeding/BAAI-bge-m3")

# Инициализация базы при старте сервера
db = None # Инициализируем db как None
try:
    print(f"🔄 Инициализация модели эмбеддингов (локально): {EMBEDDING_MODEL_PATH}...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_PATH)
    
    print(f"🔄 Загрузка векторной базы из: {DB_PATH}...")
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    print(f"✅ Векторная база готова. Векторов: {db.index.ntotal}")
except Exception as e:
    print(f"❌ Ошибка загрузки векторной базы: {str(e)}")
    print("Убедитесь, что база создана с помощью '02.create_vector_db.py' и локальная модель эмбеддингов доступна.")
    # db останется None, что вызовет HTTPException при попытке поиска

@app.get("/search")
async def search(query: str, k: int = 3):
    """
    Эндпоинт для поиска релевантных документов в векторной базе.
    Принимает поисковый запрос и возвращает k наиболее релевантных чанков.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Векторная база не загружена. Проверьте логи сервера.")
    
    try:
        print(f"🔎 Получен запрос на поиск: '{query}' (k={k})")
        results = db.similarity_search(query, k=k)
        
        formatted_results = []
        for doc in results:
            source_info = doc.metadata.get("source", "unknown")
            source_info = source_info.replace(os.path.expanduser("~/secure_rag/md/"), "") # Обновлен путь для очистки
            formatted_results.append({
                "content": doc.page_content,
                "source": source_info
            })
        
        print(f"✅ Найдено {len(formatted_results)} релевантных документов.")
        return {
            "query": query,
            "results": formatted_results
        }
    except Exception as e:
        print(f"❌ Ошибка при выполнении поиска: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при выполнении поиска: {str(e)}")

if __name__ == "__main__":
    print("🚀 Запуск RAG API сервера...")
    uvicorn.run(app, host="0.0.0.0", port=9000)

