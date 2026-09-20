import runpod
import base64
import subprocess
import os
import uuid
import shutil
import yaml

# Аналогично SadTalker-воркеру: НЕ загружаем тяжёлые модели при старте
# контейнера, чтобы холодный старт воркера успевал пройти проверку
# готовности со стороны RunPod (Testing-этап). MuseTalk грузит свои
# модели внутри собственного inference-скрипта по требованию, так что
# здесь просто нет глобальной загрузки на этапе импорта — уже "лениво"
# по устройству самого MuseTalk CLI.


def get_media_duration_seconds(path):
    """Читает длительность файла через метаданные контейнера (ffprobe
    format=duration). Годится для аудио (WAV с простым заголовком), но
    НЕ для видео — см. get_video_frame_duration_seconds ниже."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Не удалось определить длительность файла {path}: {result.stderr}")


def get_video_frame_duration_seconds(video_path):
    """Возвращает длительность видео, посчитанную из РЕАЛЬНОГО числа
    декодированных кадров и частоты кадров — то же самое, что видит
    MuseTalk при покадровом чтении файла. Метаданные длительности
    контейнера (format=duration) у некоторых mp4 (особенно с телефона
    или веб-камеры) оказались занижены относительно реального числа
    кадров на десятые доли секунды — именно из-за этого предыдущая
    версия фикса (запас в 1с, посчитанный от format=duration) почти
    полностью съедалась этой погрешностью измерения и не оставляла
    MuseTalk реального запаса (см. диагностику в чате: расхождение
    ~0.84с на реальном видео пользователя)."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,avg_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True, text=True
    )
    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Не удалось посчитать кадры видео через ffprobe: {result.stderr}")

    nb_frames_str, frame_rate_str = lines[0], lines[1]
    try:
        nb_frames = int(nb_frames_str)
        num, den = frame_rate_str.split("/")
        fps = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        raise RuntimeError(f"Не удалось разобрать вывод ffprobe: {lines}")

    return nb_frames / fps


def pad_audio_to_video_length(audio_path, video_path, work_dir, margin_seconds=2.5):
    """Причина 'MuseTalk не создал видео' (найдено по логам stdout):
    внутри MuseTalk каждому кадру видео сопоставляется окно
    whisper-аудио-признаков с отступом вперёд для временного контекста.
    У последних кадров видео это окно выходит за пределы массива
    признаков, если у аудио нет достаточного запаса длительности сверх
    длительности видео.

    Длительность видео считается через get_video_frame_duration_seconds
    (по реальному числу кадров), а не через метаданные контейнера —
    на них нельзя полагаться (см. её docstring). Запас увеличен до
    2.5с — с большим избытком относительно наблюдавшихся расхождений
    (~0.84с погрешности измерения + ~0.2с самого требуемого MuseTalk
    контекста).

    Решение: дополняем аудио-дорожку тишиной в конце. Финальное видео
    при этом НЕ укорачивается: длительность результата определяется
    видео, а лишняя тишина в конце аудио отбрасывается собственной
    логикой сборки MuseTalk."""
    video_duration = get_video_frame_duration_seconds(video_path)
    audio_duration = get_media_duration_seconds(audio_path)
    target_duration = video_duration + margin_seconds

    if audio_duration >= target_duration:
        return audio_path

    padded_path = os.path.join(work_dir, "driven_audio_padded.wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", "apad",
                "-t", f"{target_duration:.3f}",
                padded_path,
            ],
            capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Не удалось дополнить аудио тишиной: {e.stderr[-1500:]}")

    return padded_path


def run_musetalk_inference(video_path, audio_path, work_dir, result_dir):
    """Запускает inference MuseTalk через его CLI-скрипт.
    ВАЖНО: у scripts.inference MuseTalk нет прямых аргументов
    --video_path/--audio_path — вместо этого он принимает YAML-файл
    через --inference_config, где перечисляются задачи (video_path +
    audio_path на каждую). Формируем такой конфиг на лету под один
    запрос."""
    os.makedirs(result_dir, exist_ok=True)

    inference_config = {
        "task_0": {
            "video_path": video_path,
            "audio_path": audio_path,
        }
    }
    config_path = f"{work_dir}/inference_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(inference_config, f)

    cmd = [
        "python", "-m", "scripts.inference",
        "--inference_config", config_path,
        "--result_dir", result_dir,
        # MuseTalk 1.5 (рекомендованная версия, заметно чище по качеству,
        # особенно в области рта, чем версия 1.0, которую использовали
        # раньше). Обязательно указывать --version v15 явно — иначе
        # inference.py по умолчанию попробует собрать пайплайн под v1.
        "--unet_config", "./models/musetalkV15/musetalk.json",
        "--unet_model_path", "./models/musetalkV15/unet.pth",
        "--version", "v15",
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

    # "Тихий" сбой: код завершения 0, но выходного файла нет. Оставлено
    # на случай, если pad_audio_to_video_length не покрыла все причины —
    # stdout/stderr дадут диагностику для следующего разбора.
    raise RuntimeError(
        "MuseTalk не создал видео.\n"
        f"--- stdout (конец) ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr (конец) ---\n{proc.stderr[-2000:]}"
    )


# ---------------------------------------------------------------------------
# GFPGAN — постобработка для устранения размытия в области рта.
#
# MuseTalk генерирует область рта во внутреннем разрешении 256x256 и
# вклеивает её обратно в исходный кадр — при более высоком разрешении
# видео это выглядит как заметное размытие именно там, где двигаются
# губы. Это задокументированное ограничение самой модели (не баг в
# нашем коде), и официально рекомендованное решение — прогнать готовое
# видео через модель восстановления лица (face restoration) как
# финальный шаг.
#
# Модель грузится один раз лениво (тот же паттерн, что XTTS/SadTalker
# в другом воркере) и переиспользуется между запросами на одном
# воркере.
# ---------------------------------------------------------------------------

_gfpgan_restorer = None


def get_gfpgan_restorer():
    global _gfpgan_restorer
    if _gfpgan_restorer is None:
        print("Загружаю GFPGAN...")
        from gfpgan import GFPGANer
        _gfpgan_restorer = GFPGANer(
            model_path="/app/MuseTalk/gfpgan_weights/GFPGANv1.4.pth",
            upscale=1,  # не увеличиваем разрешение кадра, только резкость лица
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,  # фон не трогаем — не нужен, экономит время
        )
        print("GFPGAN готова.")
    return _gfpgan_restorer


def enhance_video_with_gfpgan(input_video_path, work_dir):
    """Разбирает готовое видео на кадры, прогоняет каждый через GFPGAN
    для повышения резкости лица (в первую очередь — области рта),
    затем собирает обратно в видео с исходной аудиодорожкой и fps.
    Возвращает путь к улучшенному файлу."""
    import cv2

    restorer = get_gfpgan_restorer()

    frames_dir = os.path.join(work_dir, "gfpgan_frames_in")
    enhanced_dir = os.path.join(work_dir, "gfpgan_frames_out")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)

    # Узнаём реальный fps исходного видео, чтобы не потерять синхронизацию
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of",
         "default=noprint_wrappers=1:nokey=1", input_video_path],
        capture_output=True, text=True
    )
    fps = probe.stdout.strip() or "25/1"

    subprocess.run(
        ["ffmpeg", "-y", "-i", input_video_path, f"{frames_dir}/frame_%06d.png"],
        capture_output=True, text=True, check=True
    )

    frame_files = sorted(os.listdir(frames_dir))
    if not frame_files:
        raise RuntimeError("ffmpeg не извлёк ни одного кадра для GFPGAN")

    for fname in frame_files:
        img = cv2.imread(os.path.join(frames_dir, fname))
        _, _, restored_img = restorer.enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True
        )
        cv2.imwrite(os.path.join(enhanced_dir, fname), restored_img)

    enhanced_video_path = os.path.join(work_dir, "enhanced.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-r", fps, "-i", f"{enhanced_dir}/frame_%06d.png",
            "-i", input_video_path,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            enhanced_video_path,
        ],
        capture_output=True, text=True, check=True
    )

    return enhanced_video_path


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

        audio_path = pad_audio_to_video_length(audio_path, video_path, work_dir)

        result_dir = f"{work_dir}/results"
        output_video_path = run_musetalk_inference(video_path, audio_path, work_dir, result_dir)

        # Постобработка GFPGAN — не должна ронять всю задачу, если вдруг
        # упадёт по какой-то причине: в этом случае просто отдаём видео
        # без улучшения резкости, а не оставляем клиента совсем без
        # результата.
        try:
            output_video_path = enhance_video_with_gfpgan(output_video_path, work_dir)
        except Exception as e:
            print(f"GFPGAN-постобработка не удалась, отдаём видео без неё: {e}")

        with open(output_video_path, "rb") as vf:
            video_base64 = base64.b64encode(vf.read()).decode("utf-8")

        return {"video_base64": video_base64}

    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()[-2000:]}

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
