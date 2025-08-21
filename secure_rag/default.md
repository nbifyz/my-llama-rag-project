
Теперь, если тебе когда-либо понадобится настроить это окружение на другой машине или пересоздать его, ты сможешь сделать это одной командой:
Bash

conda create -n new_rag_env python=3.10 -y
conda activate new_rag_env
pip install -r requirements.txt

(И, возможно, conda install pytorch cudatoolkit=11.8 -c pytorch -c nvidia -y перед pip install -r для PyTorch, так как conda лучше управляет CUDA-зависимостями).

