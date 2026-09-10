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

# runpod SDK для serverless-хендлера + pyyaml для генерации
# inference_config.yaml на лету (MuseTalk принимает пути к видео/аудио
# только через YAML-конфиг, не через прямые CLI-аргументы)
RUN pip install --no-cache-dir runpod pyyaml

# Загрузка весов моделей (musetalk, sd-vae-ft-mse, whisper, dwpose,
# face-parse-bisent) — MuseTalk поставляет скрипт для этого в репозитории.
# ВНИМАНИЕ: точный путь/имя скрипта может отличаться в зависимости от
# версии репозитория на момент сборки — если этот шаг упадёт, нужно
# свериться с актуальным README в самом репозитории и поправить путь.
RUN bash download_weights.sh || \
    (echo "download_weights.sh не найден или упал — проверить актуальный способ загрузки весов в README репозитория" && exit 1)

# download_weights.sh тихо не докачивает вес DWPose (внутри он использует
# устаревший аргумент `gdown --id`, который упал с ошибкой, но не остановил
# сборку) — качаем его вручную напрямую с HuggingFace.
RUN mkdir -p models/dwpose && \
    curl -L -o models/dwpose/dw-ll_ucoco_384.pth \
    https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth

# Тот же паттерн — download_weights.sh не докачивает и sd-vae-ft-mse
# (нужны config.json + diffusion_pytorch_model.bin). Качаем вручную.
RUN mkdir -p models/sd-vae && \
    curl -L -o models/sd-vae/config.json \
    https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json && \
    curl -L -o models/sd-vae/diffusion_pytorch_model.bin \
    https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin

# MuseTalk 1.5 (рекомендованная разработчиками версия — заметно чище
# по качеству, особенно в области рта, чем версия 1.0). Весит больше
# (~3.4 ГБ unet.pth), но именно это и нужно для нормального результата.
RUN mkdir -p models/musetalkV15 && \
    curl -L -o models/musetalkV15/musetalk.json \
    https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json && \
    curl -L -o models/musetalkV15/unet.pth \
    https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth

# И whisper — эта версия MuseTalk использует transformers.AutoFeatureExtractor,
# который ожидает полный HuggingFace-формат (config.json + pytorch_model.bin +
# preprocessor_config.json), а не старый одиночный файл tiny.pt.
RUN mkdir -p models/whisper && \
    curl -L -o models/whisper/config.json \
    https://huggingface.co/openai/whisper-tiny/resolve/main/config.json && \
    curl -L -o models/whisper/pytorch_model.bin \
    https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin && \
    curl -L -o models/whisper/preprocessor_config.json \
    https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json

# И последняя по списку — face-parse-bisent (79999_iter.pth + resnet18).
RUN mkdir -p models/face-parse-bisent && \
    curl -L -o models/face-parse-bisent/79999_iter.pth \
    https://huggingface.co/camenduru/MuseTalk/resolve/main/face-parse-bisent/79999_iter.pth && \
    curl -L -o models/face-parse-bisent/resnet18-5c106cde.pth \
    https://download.pytorch.org/models/resnet18-5c106cde.pth

# Фикс конфликта версий: openmim/mmcv/mmdet/mmpose (или runpod) подтягивают
# более новый huggingface_hub, чем допускает transformers (WhisperModel из
# MuseTalk/scripts/inference.py требует huggingface_hub<1.0,>=0.19.3).
# Ставим ПОСЛЕДНИМ шагом, чтобы никто из предыдущих pip install не
# перезаписал версию снова.
RUN pip install --no-cache-dir "huggingface_hub>=0.19.3,<1.0"

# ---------------------------------------------------------------------
# GFPGAN — финальный шаг постобработки для устранения размытия рта.
# MuseTalk генерирует область рта во внутреннем разрешении 256x256 и
# вклеивает обратно в кадр — при более высоком разрешении исходного
# видео это выглядит как размытие именно в области рта (задокументи-
# ровано в самом MuseTalk README как известное ограничение модели).
# GFPGAN восстанавливает резкость лица кадр за кадром поверх готового
# результата MuseTalk.
#
# ВНИМАНИЕ: basicsr (зависимость GFPGAN) исторически конфликтует с
# новыми версиями torchvision (импортирует убранный оттуда модуль
# torchvision.transforms.functional_tensor). На связке PyTorch 2.1.0 /
# torchvision ~0.16.x (эта версия базового образа) этот модуль ещё
# существует, так что конфликта быть не должно — но если сборка
# упадёт именно на этом шаге с ошибкой импорта, ищи
# "functional_tensor" в трейсбеке.
RUN pip install --no-cache-dir gfpgan facexlib basicsr opencv-python-headless
RUN mkdir -p /app/MuseTalk/gfpgan_weights && \
    curl -L -o /app/MuseTalk/gfpgan_weights/GFPGANv1.4.pth \
    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

COPY handler.py /app/MuseTalk/handler.py
CMD ["python", "-u", "/app/MuseTalk/handler.py"]
