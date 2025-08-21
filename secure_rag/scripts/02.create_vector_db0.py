#!/usr/bin/env python3
import os
import sys
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
import logging

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_documents(doc_path: str):
    """Загрузка документов с обработкой ошибок"""
    try:
        logger.info(f"Загрузка документов из: {doc_path}")
        loader = DirectoryLoader(doc_path, glob="**/*.md")
        documents = loader.load()
        logger.info(f"Успешно загружено документов: {len(documents)}")
        return documents
    except Exception as e:
        logger.error(f"Ошибка загрузки: {str(e)}", exc_info=True)
        raise

def create_vector_db(documents, db_path: str):
    """Создание и сохранение векторной базы"""
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        logger.info("Инициализация модели эмбеддингов BAAI/bge-m3 (локально)...")
        # Изменено: Указываем точный локальный путь к модели BAAI-bge-m3
        embeddings = SentenceTransformerEmbeddings(model_name="/home/user/models/embeding/BAAI-bge-m3")
        
        logger.info("Разбиение документов на чанки...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " "]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Документы разбиты на {len(chunks)} чанков.")
        
        logger.info("Создание векторного хранилища FAISS...")
        vector_db = FAISS.from_documents(chunks, embeddings)
        vector_db.save_local(db_path)
        
        logger.info(f"Векторная база успешно сохранена в: {db_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка создания базы: {str(e)}", exc_info=True)
        return False

def main():
    DOC_PATH = os.path.expanduser("~/secure_rag/md")
    DB_PATH = os.path.expanduser("~/secure_rag/vector_db")
    
    print("=== Начало создания векторной базы ===")
    try:
        if not os.path.exists(DOC_PATH):
            logger.error(f"Директория с документами не найдена: {DOC_PATH}")
            print(f"Пожалуйста, убедитесь, что документы находятся в '{DOC_PATH}'")
            return 1
        
        docs = load_documents(DOC_PATH)
        if not docs:
            logger.error("Нет документов для обработки в указанной директории.")
            print("Пожалуйста, убедитесь, что в директории есть .md файлы.")
            return 1
            
        if create_vector_db(docs, DB_PATH):
            print("\n=== Результат ===")
            print(f"✅ Векторная база успешно создана с моделью: BAAI/bge-m3 (локально)")
            print(f"• Документов обработано: {len(docs)}")
            print(f"• Векторная база сохранена по пути: {DB_PATH}")
            return 0
        else:
            print("\n❌ Ошибка при создании векторной базы.")
            return 1
            
    except Exception as e:
        logger.critical(f"Критическая ошибка в основной функции: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())

