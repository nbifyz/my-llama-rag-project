#!/usr/bin/env python3
import os
import sys
import logging
import json
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Конфигурация ---
# Базовая директория для хранения векторных баз
BASE_DB_DIR = os.path.expanduser("~/secure_rag/vector_dbs")
# Директория, где ожидается один Markdown-файл для добавления
LORE_BOOKS_DIR = os.path.expanduser("~/secure_rag/lore_books")
# Локальный путь к модели эмбеддингов
EMBEDDING_MODEL_PATH = os.path.expanduser("~/models/embeding/BAAI-bge-m3")

# --- Вспомогательные функции ---

def load_single_document(file_path: str) -> list:
    """
    Загружает один Markdown-файл как документ.
    """
    logger.info(f"Загрузка документа из: {file_path}")
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        if documents:
            # Добавляем метаданные для отслеживания источника
            for doc in documents:
                doc.metadata["source"] = os.path.basename(file_path)
            logger.info(f"Успешно загружен документ: {file_path}")
            return documents
        else:
            logger.warning(f"Документ пуст или не удалось загрузить: {file_path}")
            return []
    except Exception as e:
        logger.error(f"Ошибка загрузки документа {file_path}: {e}")
        return []

def get_added_lorebooks_path(db_path: str) -> str:
    """Возвращает путь к файлу журнала добавленных книг для данной базы."""
    return os.path.join(db_path, "added_lorebooks.json")

def load_added_lorebooks(json_path: str) -> list:
    """Загружает список имен добавленных книг из JSON-файла."""
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка чтения JSON-файла '{json_path}': {e}. Возвращаю пустой список.")
        return []
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при загрузке '{json_path}': {e}. Возвращаю пустой список.")
        return []

def save_added_lorebooks(json_path: str, added_files: list):
    """Сохраняет список имен добавленных книг в JSON-файл."""
    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(added_files, f, ensure_ascii=False, indent=4)
        logger.info(f"Список добавленных книг сохранен в: {json_path}")
    except Exception as e:
        logger.error(f"Ошибка сохранения списка добавленных книг в '{json_path}': {e}")

def list_existing_dbs() -> list:
    """Возвращает список имен существующих векторных баз данных."""
    if not os.path.exists(BASE_DB_DIR):
        return []
    
    dbs = [d for d in os.listdir(BASE_DB_DIR) if os.path.isdir(os.path.join(BASE_DB_DIR, d))]
    return dbs

def create_new_vector_db_from_documents(db_name: str, documents: list) -> bool:
    """
    Создает новую векторную базу данных FAISS из списка документов.
    Эта логика должна быть идентична 02.create_vector_db.py.
    """
    db_path = os.path.join(BASE_DB_DIR, db_name)
    if os.path.exists(db_path):
        logger.error(f"Директория для базы данных '{db_name}' уже существует: {db_path}")
        logger.error("Пожалуйста, выберите другое имя или используйте опцию добавления в существующую базу.")
        return False

    logger.info(f"Создание новой векторной базы данных '{db_name}' в: {db_path}")
    os.makedirs(db_path, exist_ok=True)

    logger.info("Инициализация модели эмбеддингов BAAI/bge-m3 (локально)...")
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_PATH)
        logger.info("Модель эмбеддингов успешно инициализирована.")
    except Exception as e:
        logger.error(f"Ошибка инициализации модели эмбеддингов: {e}")
        logger.error(f"Убедитесь, что модель эмбеддингов доступна по пути: {EMBEDDING_MODEL_PATH}")
        return False

    logger.info(f"Создание FAISS векторной базы данных из {len(documents)} документа(ов)...")
    try:
        vector_db = FAISS.from_documents(documents, embeddings)
        vector_db.save_local(db_path)
        logger.info(f"Векторная база данных '{db_name}' успешно создана и сохранена.")
        return True
    except Exception as e:
        logger.critical(f"Критическая ошибка при создании векторной базы данных: {e}", exc_info=True)
        logger.error("Убедитесь, что `faiss-gpu` или `faiss-cpu` установлен и совместим.")
        return False

def main():
    logger.info("=== Начало процесса добавления книги в векторную базу ===")
    
    # --- Шаг 1: Проверка директории lore_books на наличие одного файла ---
    if not os.path.exists(LORE_BOOKS_DIR):
        logger.error(f"Директория '{LORE_BOOKS_DIR}' не найдена.")
        logger.error("Пожалуйста, создайте эту директорию и поместите в нее Markdown-файл.")
        sys.exit(1)

    markdown_files = [f for f in os.listdir(LORE_BOOKS_DIR) if f.endswith(".md") and os.path.isfile(os.path.join(LORE_BOOKS_DIR, f))]
    
    if not markdown_files:
        logger.error(f"В директории '{LORE_BOOKS_DIR}' не найдено Markdown-файлов (.md).")
        logger.error("Пожалуйста, поместите ровно один Markdown-файл, который вы хотите добавить.")
        sys.exit(1)
    elif len(markdown_files) > 1:
        logger.error(f"В директории '{LORE_BOOKS_DIR}' найдено более одного Markdown-файла:")
        for f in markdown_files:
            logger.error(f"- {f}")
        logger.error("Пожалуйста, оставьте ровно один Markdown-файл, который вы хотите добавить, и уберите остальные.")
        sys.exit(1)
    
    file_to_add_name = markdown_files[0]
    file_to_add_path = os.path.join(LORE_BOOKS_DIR, file_to_add_name)
    logger.info(f"Обнаружен файл для добавления: '{file_to_add_name}'")

    # --- Шаг 2: Показать список существующих баз и запросить имя базы ---
    existing_dbs = list_existing_dbs()
    if existing_dbs:
        logger.info("\nДоступные векторные базы данных:")
        for db in existing_dbs:
            logger.info(f"- {db}")
    else:
        logger.info("\nВекторные базы данных пока не найдены. Вы можете создать новую.")

    db_name = input("\nВведите имя векторной базы данных (существующей или новой), в которую вы хотите добавить книгу: ").strip()
    if not db_name:
        logger.error("Имя базы данных не может быть пустым. Завершение.")
        sys.exit(1)

    # --- Шаг 3: Загрузка документа для добавления/создания ---
    document_to_add = load_single_document(file_to_add_path)
    if not document_to_add:
        logger.error("Не удалось загрузить документ для добавления. Завершение.")
        sys.exit(1)

    db_path = os.path.join(BASE_DB_DIR, db_name)
    lorebooks_json_path = get_added_lorebooks_path(db_path)
    
    success = False
    if db_name in existing_dbs:
        # База существует, добавляем в нее
        logger.info(f"База данных '{db_name}' найдена. Попытка добавить документ.")
        
        # Проверка на дубликат в журнале
        added_files = load_added_lorebooks(lorebooks_json_path)
        if file_to_add_name in added_files:
            logger.warning(f"Файл '{file_to_add_name}' уже присутствует в журнале для базы '{db_name}'. Пропускаю добавление.")
            logger.warning("Если вы хотите обновить файл, удалите его из директории lore_books, затем добавьте измененную версию. Это добавит новую версию.")
            sys.exit(0) # Успешное завершение, так как файл уже "добавлен"

        logger.info(f"Загрузка существующей векторной базы данных '{db_name}' из: {db_path}")
        try:
            embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_PATH)
            vector_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Векторная база '{db_name}' успешно загружена.")
            
            logger.info(f"Добавление документа '{file_to_add_name}' в базу '{db_name}'...")
            vector_db.add_documents(document_to_add)
            vector_db.save_local(db_path)
            logger.info(f"Документ '{file_to_add_name}' успешно добавлен в базу '{db_name}'.")
            
            added_files.append(file_to_add_name)
            save_added_lorebooks(lorebooks_json_path, added_files)
            success = True
        except Exception as e:
            logger.critical(f"Критическая ошибка при добавлении документа в существующую базу '{db_name}': {e}", exc_info=True)
            logger.error("Убедитесь, что `faiss-gpu` или `faiss-cpu` установлен и совместим.")
            success = False
    else:
        # База не существует, создаем новую
        logger.info(f"База данных '{db_name}' не найдена. Создаю новую базу данных.")
        if create_new_vector_db_from_documents(db_name, document_to_add):
            # Если новая база успешно создана, добавляем запись в ее журнал
            added_files = [file_to_add_name]
            save_added_lorebooks(lorebooks_json_path, added_files)
            success = True
        else:
            logger.error(f"Не удалось создать новую базу данных '{db_name}'.")
            success = False

    if success:
        logger.info("\n=== Результат ===")
        logger.info(f"✅ Процесс добавления/создания книги в базу '{db_name}' завершен успешно.")
    else:
        logger.error("\n❌ Процесс добавления/создания книги в векторную базу завершен с ошибкой.")
        sys.exit(1)

if __name__ == "__main__":
    main()
