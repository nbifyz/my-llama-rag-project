#!/usr/bin/env python3
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import requests
import json
import os
import subprocess
import sys
import asyncio
import logging
import socket
import threading
import webbrowser
import atexit # Для регистрации функции завершения

# --- 1. Конфигурация приложения ---
# Пути к скриптам и файлам
RAG_API_SERVER_SCRIPT = os.path.expanduser("~/secure_rag/scripts/04.integration.py")
LLAMA_SERVER_RUN_SCRIPT = os.path.expanduser("~/secure_rag/scripts/05.run_server_api.sh")
LOG_FILE = "07_rag_web_app.log"
TEMPLATES_DIR = "." # Директория для index.html

# Сетевые настройки
RAG_API_URL = "http://localhost:9000/search"
LLAMA_SERVER_URL = "http://localhost:8080/completion"
WEB_APP_URL = "http://localhost:8000"
WEB_APP_HOST = "0.0.0.0"
RAG_API_PORT = 9000
LLAMA_SERVER_PORT = 8080
WEB_APP_PORT = 8000

# Параметры RAG и LLM
K_RETRIEVED_CHUNKS = 3
LLM_MODEL_NAME = "mistral-7b-grok-Q4_K_M.gguf"

# --- 2. Настройка логирования ---
# Создаем логгер
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Форматтер для логов
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')

# Обработчик для записи в файл
file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Обработчик для вывода в консоль
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# Функция для записи финального сообщения в лог
def log_shutdown():
    logger.info("**** КОНЕЦ ЗАПИСИ ЛОГА ****")

# Регистрируем функцию, которая будет вызвана при выходе
atexit.register(log_shutdown)

# --- 3. Вспомогательные асинхронные функции ---

async def check_port_is_free(port: int, host: str = '127.0.0.1'):
    """Проверяет, свободен ли указанный порт."""
    logger.info(f"Проверка доступности порта {port}...")
    try:
        # Пытаемся создать серверный сокет на порту
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
        logger.info(f"Порт {port} свободен.")
        return True
    except OSError:
        logger.error(f"❌ ПОРТ {port} УЖЕ ЗАНЯТ. Освободите порт и перезапустите приложение.")
        return False

async def read_stream(stream, ready_signal: str, log_prefix: str):
    """Асинхронно читает поток, логирует и ищет сигнал готовности."""
    # Используем to_thread для блокирующего вызова readline
    while line := await asyncio.to_thread(stream.readline):
        stripped_line = line.strip()
        logger.info(f"[{log_prefix}] {stripped_line}")
        if ready_signal in stripped_line:
            return True
    return False

async def wait_for_server_ready(process, ready_signal: str, log_prefix: str, timeout: int):
    """
    Конкурентно отслеживает stdout и stderr процесса в ожидании сигнала готовности.
    """
    logger.info(f"Ожидание сигнала готовности ('{ready_signal}') от процесса {log_prefix} (PID: {process.pid}). Таймаут: {timeout}с.")
    
    # Создаем две задачи для одновременного чтения stdout и stderr
    stdout_task = asyncio.create_task(read_stream(process.stdout, ready_signal, f"{log_prefix}-stdout"))
    stderr_task = asyncio.create_task(read_stream(process.stderr, ready_signal, f"{log_prefix}-stderr"))

    try:
        # Ждем завершения первой из задач с сигналом готовности
        done, pending = await asyncio.wait(
            [stdout_task, stderr_task],
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Отменяем оставшуюся задачу, она нам больше не нужна
        for task in pending:
            task.cancel()
            
        if not done:
            raise asyncio.TimeoutError
            
        # Проверяем, был ли результат успешным
        task_result = done.pop().result()
        if not task_result:
            # Задача завершилась (поток закрылся), но сигнал не был найден
             raise RuntimeError(f"Поток вывода процесса {log_prefix} закрылся без сигнала готовности.")

        logger.info(f"✅ Сигнал готовности от {log_prefix} получен.")
        return True

    except asyncio.TimeoutError:
        logger.error(f"❌ Таймаут ожидания запуска процесса {log_prefix}.")
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка при ожидании запуска {log_prefix}: {e}")
        return False


# --- 4. Lifespan Manager (Управление жизненным циклом) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Запускает и останавливает фоновые процессы при старте и завершении приложения.
    """
    global rag_api_process, llama_server_process
    rag_api_process = None
    llama_server_process = None
    
    logger.info("--- Начало этапа запуска фоновых процессов ---")

    try:
        # Шаг 1: Проверка и запуск RAG API сервера
        logger.info("ШАГ 1: Запуск RAG API сервера")
        if not await check_port_is_free(RAG_API_PORT):
            raise RuntimeError("Не удалось запустить RAG API сервер: порт занят.")

        rag_api_process = subprocess.Popen(
            [sys.executable, RAG_API_SERVER_SCRIPT],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8'
        )

        if not await wait_for_server_ready(rag_api_process, "Uvicorn running on", "RAG_API", 30):
            raise RuntimeError("RAG API сервер не смог запуститься в установленное время.")
        
        logger.info("Результат: RAG API сервер успешно запущен.")
        await asyncio.sleep(3) # Даем время на окончательную инициализацию

        # Шаг 2: Проверка и запуск Llama.cpp сервера
        logger.info("ШАГ 2: Запуск Llama.cpp сервера")
        if not await check_port_is_free(LLAMA_SERVER_PORT):
            raise RuntimeError("Не удалось запустить Llama.cpp сервер: порт занят.")
            
        # Убедимся, что скрипт исполняемый
        if not os.access(LLAMA_SERVER_RUN_SCRIPT, os.X_OK):
             logger.warning(f"Скрипт {LLAMA_SERVER_RUN_SCRIPT} не является исполняемым. Попытка добавить права (chmod +x)...")
             os.chmod(LLAMA_SERVER_RUN_SCRIPT, 0o755)

        llama_server_process = subprocess.Popen(
            [LLAMA_SERVER_RUN_SCRIPT],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8'
        )
        
        if not await wait_for_server_ready(llama_server_process, "server is listening on", "Llama.cpp", 60):
            raise RuntimeError("Llama.cpp сервер не смог запуститься в установленное время.")
            
        logger.info("Результат: Llama.cpp сервер успешно запущен.")
        await asyncio.sleep(3)

        logger.info("--- Все фоновые процессы успешно запущены. Приложение готово. ---")

        # Шаг 3: Открытие браузера
        def open_browser():
            logger.info(f"Новый шаг: Открытие браузера по адресу {WEB_APP_URL}")
            try:
                webbrowser.open(WEB_APP_URL)
                logger.info("Результат: Команда на открытие браузера отправлена.")
            except Exception as e:
                logger.error(f"Не исполнено: Не удалось открыть браузер. Причина: {e}")
        
        threading.Timer(2, open_browser).start()
        
        yield # Основная работа приложения

    finally:
        # --- Остановка фоновых процессов при завершении ---
        logger.info("--- Начало этапа остановки фоновых процессов ---")
        if llama_server_process and llama_server_process.poll() is None:
            logger.info(f"Новый шаг: Остановка Llama.cpp сервера (PID: {llama_server_process.pid})")
            llama_server_process.terminate()
            try:
                llama_server_process.wait(timeout=5)
                logger.info("Результат: Llama.cpp сервер штатно остановлен.")
            except subprocess.TimeoutExpired:
                logger.warning("Llama.cpp сервер не ответил на terminate, принудительное завершение (kill).")
                llama_server_process.kill()
                logger.info("Результат: Llama.cpp сервер принудительно остановлен.")

        if rag_api_process and rag_api_process.poll() is None:
            logger.info(f"Новый шаг: Остановка RAG API сервера (PID: {rag_api_process.pid})")
            rag_api_process.terminate()
            try:
                rag_api_process.wait(timeout=5)
                logger.info("Результат: RAG API сервер штатно остановлен.")
            except subprocess.TimeoutExpired:
                logger.warning("RAG API сервер не ответил на terminate, принудительное завершение (kill).")
                rag_api_process.kill()
                logger.info("Результат: RAG API сервер принудительно остановлен.")
        
        logger.info("--- Все фоновые процессы остановлены. ---")

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- 5. Функции для взаимодействия с API ---

async def get_rag_context_async(query: str) -> list:
    """Асинхронно получает контекст от RAG API."""
    logger.info(f"Новый шаг: Получение контекста для запроса '{query}'")
    try:
        # Использование `asyncio.to_thread` для запуска синхронного кода в отдельном потоке
        response = await asyncio.to_thread(requests.get, RAG_API_URL, params={"query": query, "k": K_RETRIEVED_CHUNKS})
        response.raise_for_status()
        data = response.json()
        
        if data and "results" in data:
            logger.info(f"Исполнено: Получено {len(data['results'])} релевантных документов.")
            return data["results"]
        else:
            logger.warning("Не исполнено: RAG API вернул пустые или некорректные результаты.")
            return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Не исполнено: Ошибка при запросе к RAG API. Причина: {e}")
        return []

async def generate_llm_response_async(prompt: str) -> str:
    """Асинхронно генерирует ответ с помощью LLM."""
    logger.info("Новый шаг: Отправка промпта на Llama-сервер для генерации ответа.")
    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": prompt, "n_predict": 2048, "temperature": 0.7,
        "stop": ["\nUser:", "\n###", "<|im_end|>", "<|endoftext|>"],
        "model": LLM_MODEL_NAME
    }
    
    try:
        response = await asyncio.to_thread(requests.post, LLAMA_SERVER_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        if "content" in result:
            logger.info("Исполнено: Получен ответ от Llama-сервера.")
            return result["content"].strip()
        else:
            logger.warning("Не исполнено: Llama-сервер вернул ответ без поля 'content'.")
            return "Не удалось получить корректный ответ от LLM."
    except requests.exceptions.RequestException as e:
        logger.error(f"Не исполнено: Ошибка при запросе к Llama-серверу. Причина: {e}")
        return f"Ошибка при генерации ответа LLM: {e}"

# --- 6. Веб-эндпоинты FastAPI ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Отображает главную страницу."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "response_text": "Введите ваш вопрос и нажмите 'Спросить'."}
    )

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, user_query: str = Form(...)):
    """Обрабатывает запрос пользователя: RAG -> LLM -> Ответ."""
    logger.info(f"==== НАЧАЛО ОБРАБОТКИ ЗАПРОСА ПОЛЬЗОВАТЕЛЯ: '{user_query}' ====")
    
    # 1. Получаем контекст
    retrieved_docs = await get_rag_context_async(user_query)
    context_text = ""
    if retrieved_docs:
        context_text = "\n\n### Контекст из документов:\n"
        for i, doc in enumerate(retrieved_docs):
            source = doc.get('source', 'Неизвестно').replace(os.path.expanduser("~/secure_rag/md/"), "")
            context_text += f"Документ {i+1} (Источник: {source}):\n{doc.get('content', '')}\n---\n"
    else:
        logger.warning("Контекст для запроса не найден. Ответ будет сгенерирован без него.")

    # 2. Формируем промпт
    prompt = f"""Ты — полезный ассистент. Ответь на вопрос, используя предоставленный контекст. Если ответ в контексте отсутствует, сообщи об этом.
{context_text}
### Вопрос:
{user_query}
### Ответ:"""
    
    # 3. Генерируем ответ
    llm_response = await generate_llm_response_async(prompt)
    logger.info(f"Результат: Финальный ответ LLM для пользователя сгенерирован.")
    logger.info(f"==== КОНЕЦ ОБРАБОТКИ ЗАПРОСА '{user_query}' ====")
    
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "user_query": user_query, "response_text": llm_response}
    )

# --- 7. Запуск приложения ---
if __name__ == "__main__":
    logger.info("**** Старт начала записи ****")
    logger.info(f"Запуск веб-сервера Uvicorn на {WEB_APP_HOST}:{WEB_APP_PORT}")
    
    # Убедимся, что порт для веб-приложения свободен
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((WEB_APP_HOST, WEB_APP_PORT))
    except OSError:
         logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Порт {WEB_APP_PORT} для самого веб-приложения занят!")
         logger.error("Освободите порт и повторите попытку.")
         sys.exit(1) # Завершаем выполнение

    uvicorn.run(app, host=WEB_APP_HOST, port=WEB_APP_PORT)
