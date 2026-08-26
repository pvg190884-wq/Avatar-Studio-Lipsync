FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
WORKDIR /app

# Системные зависимости: ffmpeg для работы с видео/аудио, git для клонирования
RUN apt-get update && apt-get install -y ffmpeg git libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

# Клонируем официальный репозиторий MuseTalk
RUN git clone https://github.com/TMElyralab/MuseTalk.git
WORKDIR /app/MuseTalk

# Зависимости MuseTalk. requirements.txt в самом репозитории обычно
# покрывает основное, но mmcv/mmpose/mmdet ставятся отдельно через
# openmim, т.к. требуют точного соответствия версии torch/CUDA.
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir openmim && \
    mim install mmengine && \
    mim install "mmcv>=2.0.1" && \
    mim install "mmdet>=3.1.0" && \
    mim install "mmpose>=1.1.0"

# runpod SDK для serverless-хендлера
RUN pip install --no-cache-dir runpod

# Загрузка весов моделей (musetalk, sd-vae-ft-mse, whisper, dwpose,
# face-parse-bisent) — MuseTalk поставляет скрипт для этого в репозитории.
# ВНИМАНИЕ: точный путь/имя скрипта может отличаться в зависимости от
# версии репозитория на момент сборки — если этот шаг упадёт, нужно
# свериться с актуальным README в самом репозитории и поправить путь.
RUN bash download_weights.sh || \
    (echo "download_weights.sh не найден или упал — проверить актуальный способ загрузки весов в README репозитория" && exit 1)

# Фикс конфликта версий: openmim/mmcv/mmdet/mmpose (или runpod) подтягивают
# более новый huggingface_hub, чем допускает transformers (WhisperModel из
# MuseTalk/scripts/inference.py требует huggingface_hub<1.0,>=0.19.3).
# Ставим ПОСЛЕДНИМ шагом, чтобы никто из предыдущих pip install не
# перезаписал версию снова.
RUN pip install --no-cache-dir "huggingface_hub>=0.19.3,<1.0"

COPY handler.py /app/MuseTalk/handler.py
CMD ["python", "-u", "/app/MuseTalk/handler.py"]
