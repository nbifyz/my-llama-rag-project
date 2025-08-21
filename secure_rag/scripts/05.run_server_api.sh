#!/bin/bash

# Запуск llama.cpp сервера с HTTP API.
# Убедитесь, что модель указана правильно.
# --host 0.0.0.0 позволяет принимать подключения со всех IP-адресов.
# --port 8080 - порт, на котором будет работать API.
# --n-gpu-layers 12 - загружает 12 слоев модели на GPU.
# Если у вас недостаточно VRAM, уменьшите это число или установите -1 для всех слоев (если GPU позволяет).
# Если вы хотите загрузить все слои на GPU, используйте --n-gpu-layers -1 (для llama.cpp версии 1.1.0 и выше).
# В вашем случае, n-gpu-layers 12 - это хороший старт.

/opt/llama.cpp/build/bin/llama-server \
    --model "/home/user/models/GGUF/mistral-7b-grok-Q4_K_M.gguf" \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 8192 \
    --threads 16 \
    --n-gpu-layers 12
