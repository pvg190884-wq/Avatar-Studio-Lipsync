import runpod
import base64
import subprocess
import os
import uuid
import shutil

# Аналогично SadTalker-воркеру: НЕ загружаем тяжёлые модели при старте
# контейнера, чтобы холодный старт воркера успевал пройти проверку
# готовности со стороны RunPod (Testing-этап). MuseTalk грузит свои
# модели внутри собственного inference-скрипта по требованию, так что
# здесь просто нет глобальной загрузки на этапе импорта — уже "лениво"
# по устройству самого MuseTalk CLI.


def run_musetalk_inference(video_path, audio_path, result_dir):
    """Запускает inference MuseTalk через его CLI-скрипт.

    ВАЖНО: точное имя скрипта, названия аргументов и формат конфигурации
    могут отличаться от версии к версии репозитория MuseTalk — это
    первая черновая версия, скорее всего потребует правки после первого
    реального запуска (по аналогии с тем, как дорабатывался SadTalker).
    Нужно свериться с README/inference-примерами в самом репозитории и
    поправить cmd ниже под то, что там реально ожидается.
    """
    os.makedirs(result_dir, exist_ok=True)

    cmd = [
        "python", "-m", "scripts.inference",
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--result_dir", result_dir,
    ]

    proc = subprocess.run(
        cmd, cwd="/app/MuseTalk",
        capture_output=True, text=True, timeout=1800
    )
    if proc.returncode != 0:
        raise RuntimeError(f"MuseTalk упал: {proc.stderr[-3000:]}")

    for root, _, files in os.walk(result_dir):
        for f in files:
            if f.endswith(".mp4"):
                return os.path.join(root, f)
    raise RuntimeError("MuseTalk не создал видео")


def handler(event):
    input_data = event.get("input", {}) or {}

    # Тот же мягкий healthcheck, что и в SadTalker-воркере — чтобы
    # Testing-этап RunPod не считал пустой/тестовый запрос ошибкой.
    if input_data.get("healthcheck") is True or not input_data:
        return {"ok": True}

    job_id = str(uuid.uuid4())
    work_dir = f"/tmp/{job_id}"
    os.makedirs(work_dir, exist_ok=True)

    try:
        if "video_base64" not in input_data:
            return {"error": "нужен video_base64 (исходное видео с лицом)"}
        if "audio_base64" not in input_data:
            return {"error": "нужен audio_base64 (аудио-драйвер для липсинка)"}

        video_path = f"{work_dir}/source_video.mp4"
        with open(video_path, "wb") as f:
            f.write(base64.b64decode(input_data["video_base64"]))

        audio_path = f"{work_dir}/driven_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(base64.b64decode(input_data["audio_base64"]))

        result_dir = f"{work_dir}/results"
        output_video_path = run_musetalk_inference(video_path, audio_path, result_dir)

        with open(output_video_path, "rb") as vf:
            video_base64 = base64.b64encode(vf.read()).decode("utf-8")

        return {"video_base64": video_base64}

    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()[-2000:]}

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
  
