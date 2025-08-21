#!/usr/bin/env python3
import os
import sys
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

def main():
    # Конфигурация
    DB_PATH = os.path.expanduser("~/secure_rag/vector_db")
    
    print("\n" + "="*50)
    print("🔍 Тестирование векторной базы данных")
    print("="*50)

    # 1. Проверка существования файлов базы FAISS
    if not os.path.exists(f"{DB_PATH}/index.faiss"):
        print("\n❌ Ошибка: файлы базы не найдены!")
        print(f"Убедитесь, что путь корректен: {DB_PATH}")
        print("Сначала создайте базу командой:")
        print("python3 ~/secure_rag/scripts/02.create_vector_db.py")
        sys.exit(1)

    # 2. Загрузка базы с явным разрешением
    try:
        print("\n🔄 Загрузка векторной базы...")
        # Изменено: Указываем точный локальный путь к модели BAAI-bge-m3
        embeddings = SentenceTransformerEmbeddings(model_name="/home/user/models/embeding/BAAI-bge-m3")
        db = FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True  # Ваше решение
        )
        print(f"✅ Успешно загружено векторов: {db.index.ntotal}")
    except Exception as e:
        print(f"\n❌ Ошибка загрузки: {str(e)}")
        print("\nВозможные решения:")
        print(f"1. Удалите и пересоздайте базу: rm -rf {DB_PATH}")
        print(f"2. Проверьте путь к модели эмбеддингов: /home/user/models/embeding/BAAI-bge-m3")
        sys.exit(1)

    # 3. Тестовые запросы
    test_queries = ["оранжевый конь", "селедочные головы", "малиновые копыта"]
    print("\n" + "="*50)
    print("🔎 Тестовые запросы к базе:")
    
    for query in test_queries:
        try:
            results = db.similarity_search(query, k=1)
            print(f"\nЗапрос: '{query}'")
            if results:
                doc = results[0]
                source_info = doc.metadata.get('source', 'unknown').replace(os.path.expanduser("~/secure_rag/md/"), "")
                print(f"📄 Источник: {source_info}")
                print(f"📝 Содержание: {doc.page_content[:200]}...")
            else:
                print("⚠ Ничего не найдено")
        except Exception as e:
            print(f"⚠ Ошибка при запросе '{query}': {str(e)}")

    print("\n" + "="*50)
    print("✅ Тестирование завершено успешно!")
    print("="*50)
    sys.exit(0)

if __name__ == "__main__":
    main()

