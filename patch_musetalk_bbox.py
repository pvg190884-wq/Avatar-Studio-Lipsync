"""Патч для встроенного edge-case в musetalk/utils/preprocessing.py
(оригинальный репозиторий TMElyralab/MuseTalk), функция
get_landmark_and_bbox.

Проблема: в конце покадрового извлечения landmarks функция печатает
диагностическую сводку с рекомендуемым диапазоном bbox_shift:

    print(f"...{int(sum(average_range_minus) / len(average_range_minus))}"
          f"~{int(sum(average_range_plus) / len(average_range_plus))}...")

average_range_minus/average_range_plus заполняются только для кадров,
где ДЕТЕКТОР ЛИЦ НАШЁЛ лицо — если детектор не нашёл лицо ни на одном
кадре всего видео, оба списка остаются пустыми, и деление на
len(...) == 0 роняет весь процесс с ZeroDivisionError ПОСЛЕ того, как
вся дорогая покадровая обработка уже выполнена (см. историю чата:
ошибка "division by zero" ровно после 100% прогресса по кадрам).

Фикс: делаем эту диагностическую печать безопасной при пустых списках
и поднимаем понятную ошибку "лицо не найдено", а не криптичный
ZeroDivisionError — если лица действительно нет ни на одном кадре,
дальше по цепочке пайплайн всё равно не сможет сгенерировать
осмысленный результат, так что лучше сообщить об этом сразу и явно."""

import re
import sys

TARGET_PATH = "musetalk/utils/preprocessing.py"

with open(TARGET_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

anchor_substring = "average_range_minus) / len(average_range_minus)"
target_index = None
for i, line in enumerate(lines):
    if anchor_substring in line:
        target_index = i
        break

if target_index is None:
    print(
        f"ПАТЧ НЕ ПРИМЕНЁН: строка с делением на len(average_range_minus) "
        f"не найдена в {TARGET_PATH}. Судя по всему, апстрим-репозиторий "
        "MuseTalk изменился — нужно вручную свериться с актуальным "
        "содержимым файла и обновить этот скрипт.",
        file=sys.stderr,
    )
    sys.exit(1)

original_line = lines[target_index]
indent_match = re.match(r"^(\s*)", original_line)
indent = indent_match.group(1) if indent_match else ""

replacement = (
    f'{indent}if average_range_minus and average_range_plus:\n'
    f'{indent}    print(f"Total frame:\\u300c{{len(frames)}}\\u300d Manually adjust range : '
    f'[ -{{int(sum(average_range_minus) / len(average_range_minus))}}'
    f'~{{int(sum(average_range_plus) / len(average_range_plus))}} ] , '
    f'the current value: {{upperbondrange}}")\n'
    f'{indent}else:\n'
    f'{indent}    raise RuntimeError(\n'
    f'{indent}        f"Лицо не обнаружено ни на одном из {{len(frames)}} кадров видео. "\n'
    f'{indent}        "Проверьте, что лицо хорошо видно на протяжении всей записи."\n'
    f'{indent}    )\n'
)

lines[target_index] = replacement

with open(TARGET_PATH, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"Патч успешно применён к {TARGET_PATH} (строка {target_index + 1})")
