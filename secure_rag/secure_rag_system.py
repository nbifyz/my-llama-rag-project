#!/usr/bin/env python3
# secure_rag_system.py - Полностью защищенная RAG-система (без шифрования)
import os
import sys
import yaml
import pickle
import numpy as np
import uvicorn
import time
import hashlib
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional, Tuple
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from cryptography.fernet import Fernet

# Конфигурация (замените `your_user` на ваше имя пользователя в Linux!)
CONFIG = {
    "documents_path": "/home/user/secure_rag/documents",
    "vector_db_path": "/home/user/secure_rag/vector_db",
    "embedding_model": "bge-m3:567m",  # Модель для эмбеддингов
    "chunk_size": 512,
    "chunk_overlap": 128,
    "host": "127.0.0.1",
    "port": 9000,
    "encrypt_content": False,  # Шифрование отключено!
    "log_file": "/home/user/secure_rag/logs/rag_system.log"
}

# API Keys (замените на свои!)
API_KEYS = {
    "client-app": "SECURE_RAG_ACCESS_KEY_123!",
    "admin": "MASTER_KEY_ADMIN_!#456"
}

# Модели данных
class SearchResult(BaseModel):
    content: str
    source: str
    score: float
    is_encrypted: bool = False

class SearchRequest(BaseModel):
    query: str
    k: int = 3
    source_filter: Optional[str] = None

# Инициализация приложения
app = FastAPI(
    title="Secure RAG API",
    description="RAG-система для интеграции с OpenWebUI",
    version="3.0"
)

# Аутентификация
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key not in API_KEYS.values():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

# Загрузка и индексация документов
def load_and_index_documents(reindex: bool = False):
    """Загрузка документов и создание индексов (FAISS + BM25)"""
    if not reindex and os.path.exists(CONFIG['vector_db_path']):
        return
    
    print("⚙️ Начало индексации документов...")
    
    # Загрузка документов
    loader = DirectoryLoader(
        CONFIG['documents_path'],
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True
    )
    documents = loader.load()
    
    if not documents:
        raise RuntimeError("Документы не найдены!")
    
    # Добавляем метаданные
    for doc in documents:
        doc.metadata = {
            "source": doc.metadata['source'],
            "hash": hashlib.sha256(doc.page_content.encode()).hexdigest(),
            "timestamp": time.time()
        }
    
    # Разделение на чанки
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG['chunk_size'],
        chunk_overlap=CONFIG['chunk_overlap'],
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    
    # Создание векторной базы
    embeddings = OllamaEmbeddings(model=CONFIG['embedding_model'])
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # Сохранение
    os.makedirs(CONFIG['vector_db_path'], exist_ok=True)
    vector_db.save_local(CONFIG['vector_db_path'])
    
    # Индекс BM25
    bm25_index = BM25Okapi([c.page_content for c in chunks])
    with open(os.path.join(CONFIG['vector_db_path'], "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25_index, f)
    
    print(f"✅ База данных сохранена в {CONFIG['vector_db_path']}")

# Инициализация при запуске
@app.on_event("startup")
def startup_event():
    print("🔍 Принудительная переиндексация...")
    load_and_index_documents(reindex=True)  # Всегда пересоздаем индекс

# API Endpoints
@app.post("/search", response_model=List[SearchResult])
async def secure_search(
    request: SearchRequest,
    api_key: str = Security(get_api_key)
):
    """Гибридный поиск (семантический + BM25)"""
    try:
        # Загрузка FAISS
        embeddings = OllamaEmbeddings(model=CONFIG['embedding_model'])
        vector_db = FAISS.load_local(
            CONFIG['vector_db_path'],
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Семантический поиск
        semantic_results = vector_db.similarity_search_with_score(
            request.query, 
            k=request.k * 2,
            filter={"source": request.source_filter} if request.source_filter else None
        )
        
        # BM25
        with open(os.path.join(CONFIG['vector_db_path'], "bm25_index.pkl"), "rb") as f:
            bm25_index = pickle.load(f)
        
        # Комбинирование результатов
        combined_results = []
        for doc, score in semantic_results:
            bm25_score = bm25_index.get_scores(request.query)[0]
            combined_score = (1 - score) * 0.7 + bm25_score * 0.3
            
            combined_results.append({
                "doc": doc,
                "score": combined_score,
                "content": doc.page_content  # Контент уже не зашифрован
            })
        
        # Топ результатов
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        return [SearchResult(
            content=r['content'][:1000],
            source=r['doc'].metadata['source'],
            score=r['score'],
            is_encrypted=False
        ) for r in combined_results[:request.k]]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка поиска: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "active", "model": CONFIG['embedding_model']}

# Запуск
if __name__ == "__main__":
    # Проверка ОС (только Linux)
    if not sys.platform.startswith('linux'):
        print("⛔ ОШИБКА: Система предназначена только для Linux!")
        sys.exit(1)
    
    print(f"🚀 Запуск RAG-сервера на {CONFIG['host']}:{CONFIG['port']}")
    print(f"🔑 Доступные API-ключи: {list(API_KEYS.keys())}")
    
    uvicorn.run(
        app,
        host=CONFIG['host'],
        port=CONFIG['port'],
        log_config=None
    )
