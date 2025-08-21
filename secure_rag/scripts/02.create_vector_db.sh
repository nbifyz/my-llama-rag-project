#!/usr/bin/env python3
import os
import sys
import logging
import json
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Конфигурация ---
# Базовая директория для хранения векторных баз
BASE_DB_DIR = os.path.expanduser("~/secure_rag/vector_dbs")
# Локальный путь к модели эмбеддингов
EMBEDDING_MODEL_PATH = os.path.expanduser("~/models/embeding/BAAI-bge-m3")
# Директория по умолчанию для исходных документов
DEFAULT_SOURCE_DIR = os.path.expanduser("~/secure_rag/current")

# --- Вспомогательные функции ---

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

def main():
    logger.info("=== Начало процесса создания векторной базы ===")
    
    # --- Шаг 1: Запрос имени базы данных ---
    existing_dbs = list_existing_dbs()
    if existing_dbs:
        logger.info("\nДоступные векторные базы данных:")
        for db in existing_dbs:
            logger.info(f"- {db}")
    else:
        logger.info("\nВекторные базы данных пока не найдены. Вы можете создать новую.")

    db_name = input("\nВведите имя для новой векторной базы данных: ").strip()
    if not db_name:
        logger.error("Имя базы данных не может быть пустым. Завершение.")
        sys.exit(1)

    db_path = os.path.join(BASE_DB_DIR, db_name)

    # --- Шаг 2: Защита от перезаписи ---
    if os.path.exists(db_path):
        logger.warning(f"Внимание: Директория для базы данных '{db_name}' уже существует: {db_path}")
        overwrite_choice = input("Вы хотите перезаписать ее? Все существующие данные будут потеряны! (да/нет): ").strip().lower()
        if overwrite_choice != 'да':
            logger.info("Операция отменена пользователем. Завершение.")
            sys.exit(0)
        else:
            logger.info(f"Перезапись базы данных '{db_name}' подтверждена. Удаление старых данных...")
            # Удаляем старую директорию, чтобы создать новую чистую
            try:
                import shutil
                shutil.rmtree(db_path)
                logger.info("Старые данные успешно удалены.")
            except Exception as e:
                logger.error(f"Ошибка при удалении старых данных базы '{db_name}': {e}. Завершение.")
                sys.exit(1)

    os.makedirs(db_path, exist_ok=True) # Создаем директорию для новой/перезаписываемой базы

    # --- Шаг 3: Запрос директории с исходными .md файлами ---
    source_dir = input(f"Введите путь к директории с Markdown-файлами (по умолчанию: {DEFAULT_SOURCE_DIR}): ").strip()
    if not source_dir:
        source_dir = DEFAULT_SOURCE_DIR
        logger.info(f"Используется директория по умолчанию: {source_dir}")

    if not os.path.exists(source_dir):
        logger.error(f"Указанная директория '{source_dir}' не найдена. Завершение.")
        sys.exit(1)
    
    # --- Шаг 4: Запрос размеров чанков и оверлапа в ТОКЕНАХ ---
    try:
        chunk_size = int(input("Введите желаемый размер чанка в токенах (например, 500): ").strip())
        chunk_overlap = int(input("Введите желаемый размер перекрытия чанков в токенах (например, 50): ").strip())
        if chunk_size <= 0 or chunk_overlap < 0:
            raise ValueError("Размер чанка должен быть положительным, перекрытие - неотрицательным.")
        if chunk_overlap >= chunk_size:
            raise ValueError("Перекрытие чанков должно быть меньше размера чанка.")
    except ValueError as e:
        logger.error(f"Некорректный ввод для размера чанка или перекрытия: {e}. Завершение.")
        sys.exit(1)

    # --- Шаг 5: Загрузка документов ---
    logger.info(f"Загрузка Markdown-файлов из директории: {source_dir}")
    try:
        loader = DirectoryLoader(
            source_dir,
            glob="**/*.md", # Ищем все .md файлы, включая поддиректории
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            recursive=True # Рекурсивно ищем в поддиректориях
        )
        documents = loader.load()
        if not documents:
            logger.warning(f"В директории '{source_dir}' не найдено Markdown-файлов. База данных будет пустой.")
            # sys.exit(0) # Можно выйти, если пустая база не нужна
        else:
            logger.info(f"Найдено {len(documents)} документов для обработки.")
            # Добавляем метаданные для отслеживания источника
            for doc in documents:
                doc.metadata["source"] = os.path.basename(doc.metadata.get("source", "unknown_file"))

    except Exception as e:
        logger.critical(f"Критическая ошибка при загрузке документов из '{source_dir}': {e}", exc_info=True)
        sys.exit(1)

    # --- Шаг 6: Разделение документов на чанки ---
    logger.info(f"Разделение документов на чанки (размер: {chunk_size} токенов, перекрытие: {chunk_overlap} токенов)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, # Простая функция длины, но для токенов лучше использовать токенизатор
        # Можно добавить `tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode` для более точного подсчета токенов
    )
    texts = text_splitter.split_documents(documents)
    logger.info(f"Создано {len(texts)} чанков.")

    # --- Шаг 7: Инициализация модели эмбеддингов ---
    logger.info("Инициализация модели эмбеддингов BAAI/bge-m3 (локально)...")
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_PATH)
        logger.info("Модель эмбеддингов успешно инициализирована.")
    except Exception as e:
        logger.critical(f"Критическая ошибка инициализации модели эмбеддингов: {e}", exc_info=True)
        logger.error(f"Убедитесь, что модель эмбеддингов доступна по пути: {EMBEDDING_MODEL_PATH}")
        sys.exit(1)

    # --- Шаг 8: Создание и сохранение векторной базы данных ---
    logger.info(f"Создание FAISS векторной базы данных из {len(texts)} чанков...")
    try:
        # allow_dangerous_deserialization=True необходимо для загрузки FAISS баз, созданных LangChain
        vector_db = FAISS.from_documents(texts, embeddings)
        vector_db.save_local(db_path)
        logger.info(f"Векторная база данных '{db_name}' успешно создана и сохранена в: {db_path}")
    except Exception as e:
        logger.critical(f"Критическая ошибка при создании векторной базы данных: {e}", exc_info=True)
        logger.error("Убедитесь, что `faiss-gpu` или `faiss-cpu` установлен и совместим с вашей версией `numpy`.")
        sys.exit(1)

    # --- Шаг 9: Создание/обновление журнала added_lorebooks.json ---
    processed_file_names = [os.path.basename(doc.metadata.get("source", "unknown_file")) for doc in documents]
    save_added_lorebooks(get_added_lorebooks_path(db_path), processed_file_names)
    logger.info(f"Журнал добавленных книг для базы '{db_name}' обновлен.")

    logger.info("\n=== Результат ===")
    logger.info(f"✅ Процесс создания векторной базы '{db_name}' завершен успешно.")

if __name__ == "__main__":
    main()
