"""Патч для встроенного edge-case в musetalk/utils/audio_processor.py
(оригинальный репозиторий TMElyralab/MuseTalk).

Проблема: при вычислении аудио-признака для последнего кадра (или
нескольких последних кадров) видео округление во внутренней формуле
(math.floor) иногда даёт audio_index, при котором срез
whisper_feature[:, audio_index:audio_index+N] короче ожидаемого N.
Оригинальный код в этом случае просто вызывает exit() без аргументов —
это завершает процесс с кодом 0 (то есть "успех" для оболочки!), но
без записи итогового mp4. Именно поэтому генерация "тихо" падала:
subprocess.run() в handler.py видел returncode == 0 и никакой ошибки
не поднимал.

Важно: это НЕ связано с соотношением длин видео и аудио — num_frames
здесь считается из длительности самого АУДИО (см. код выше в файле),
так что ни подрезка видео, ни удлинение аудио тишиной (обе пробовали
раньше) эту проблему не решают в принципе — она воспроизводилась
стабильно на некоторых длительностях независимо от запаса.

Фикс: вместо падения — подрезаем audio_index так, чтобы срез
гарантированно помещался в границы массива (переиспользуем последнее
доступное окно признаков для самых крайних кадров — на слух
незаметно, это буквально доли секунды в самом конце ролика)."""

import sys

TARGET_PATH = "musetalk/utils/audio_processor.py"

OLD_BLOCK = '''                audio_index = math.floor(frame_index * whisper_idx_multiplier)
                audio_clip = whisper_feature[:, audio_index: audio_index + audio_feature_length_per_frame]
                assert audio_clip.shape[1] == audio_feature_length_per_frame
                audio_prompts.append(audio_clip)
            except Exception as e:
                print(f"Error occurred: {e}")
                print(f"whisper_feature.shape: {whisper_feature.shape}")
                print(f"audio_clip.shape: {audio_clip.shape}")
                print(f"num frames: {num_frames}, fps: {fps}, whisper_idx_multiplier: {whisper_idx_multiplier}")
                print(f"frame_index: {frame_index}, audio_index: {audio_index}-{audio_index + audio_feature_length_per_frame}")
                exit()'''

NEW_BLOCK = '''                audio_index = math.floor(frame_index * whisper_idx_multiplier)
                max_start_index = whisper_feature.shape[1] - audio_feature_length_per_frame
                if audio_index > max_start_index:
                    # Округление ушло за пределы массива признаков — не
                    # падаем, переиспользуем последнее доступное окно.
                    audio_index = max(0, max_start_index)
                audio_clip = whisper_feature[:, audio_index: audio_index + audio_feature_length_per_frame]
                audio_prompts.append(audio_clip)
            except Exception as e:
                print(f"Error occurred: {e}")
                print(f"whisper_feature.shape: {whisper_feature.shape}")
                print(f"num frames: {num_frames}, fps: {fps}, whisper_idx_multiplier: {whisper_idx_multiplier}")
                print(f"frame_index: {frame_index}, audio_index: {audio_index}")
                raise'''

with open(TARGET_PATH, "r", encoding="utf-8") as f:
    source = f.read()

if OLD_BLOCK not in source:
    print(
        f"ПАТЧ НЕ ПРИМЕНЁН: ожидаемый блок кода не найден в {TARGET_PATH}. "
        "Судя по всему, апстрим-репозиторий MuseTalk изменился — нужно "
        "вручную свериться с актуальным содержимым файла и обновить "
        "этот скрипт.",
        file=sys.stderr,
    )
    sys.exit(1)

source = source.replace(OLD_BLOCK, NEW_BLOCK)

with open(TARGET_PATH, "w", encoding="utf-8") as f:
    f.write(source)

print(f"Патч успешно применён к {TARGET_PATH}")
