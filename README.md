# Best Photo AI

Инструмент для анализа личных медиабиблиотек и отбора лучших фото и видео из серий и похожих сцен.

Цель проекта:
- убрать мусорные и дублирующие медиафайлы;
- выбрать лучшие кадры и ролики;
- сохранить только ценные воспоминания;
- подготовить curated-библиотеку с возможностью отката.

Полная целевая архитектура описана в [docs/pipeline.md](D:/GitHub/Best_photo_ai/docs/pipeline.md).

## Установка окружения
Базовые зависимости:
```powershell
pip install -r D:\GitHub\Best_photo_ai\requirements.txt
```

CPU-профиль:
```powershell
pip install -r D:\GitHub\Best_photo_ai\requirements-cpu.txt
```

CUDA-профиль:
```powershell
pip install -r D:\GitHub\Best_photo_ai\requirements-cuda.txt
```

Проверка runtime:
```powershell
python D:\GitHub\Best_photo_ai\Scripts\check_runtime.py
```

Для быстрого `05_group_similar_images.py` нужен:
- CUDA-enabled `torch`
- `cuda_available = True`
- видимая NVIDIA GPU в `check_runtime.py`

Для ускорения photo-pipeline уже реализовано:
- batched CUDA semantic grouping в [05_group_similar_images.py](D:/GitHub/Best_photo_ai/Scripts/05_group_similar_images.py);
- disk-cache для CLIP и face embeddings в [05_group_similar_images.py](D:/GitHub/Best_photo_ai/Scripts/05_group_similar_images.py);
- parallel CPU execution в [06_compute_sharpness.py](D:/GitHub/Best_photo_ai/Scripts/06_compute_sharpness.py), [07_compute_composition.py](D:/GitHub/Best_photo_ai/Scripts/07_compute_composition.py), [08_compute_subject.py](D:/GitHub/Best_photo_ai/Scripts/08_compute_subject.py);
- batched aesthetic scoring с GPU support и score-cache в [09_compute_aesthetic.py](D:/GitHub/Best_photo_ai/Scripts/09_compute_aesthetic.py).

## Ключевые принципы
- Единица обработки: не отдельный файл, а `asset`.
- `asset` включает основной медиафайл и связанные service/sidecar-файлы.
- Viewer не должен содержать бизнес-логику классификации; он только показывает и фильтрует уже подготовленные данные.
- Перенос лучших файлов должен быть transactional: `move + manifest + rollback`.

## Текущий пайплайн
- [01_preflight_archives.py](D:/GitHub/Best_photo_ai/Scripts/01_preflight_archives.py): ранний поиск архивов, список `txt`, stop/continue и controlled unpack в staging перед построением asset-базы.
- [02_scan_takeout.py](D:/GitHub/Best_photo_ai/Scripts/02_scan_takeout.py): первичный индекс файлов и сборка asset.
- [03_find_exact_duplicates.py](D:/GitHub/Best_photo_ai/Scripts/03_find_exact_duplicates.py): каноникализация и exact dedupe.
- [04_prepare_photo_index.py](D:/GitHub/Best_photo_ai/Scripts/04_prepare_photo_index.py): подготовка `photo_index.csv` для photo-ветки.
- [05_group_similar_images.py](D:/GitHub/Best_photo_ai/Scripts/05_group_similar_images.py): группировка похожих фото и сцен.
- [06_compute_sharpness.py](D:/GitHub/Best_photo_ai/Scripts/06_compute_sharpness.py): резкость.
- [07_compute_composition.py](D:/GitHub/Best_photo_ai/Scripts/07_compute_composition.py): композиционные признаки.
- [08_compute_subject.py](D:/GitHub/Best_photo_ai/Scripts/08_compute_subject.py): качество и выраженность субъекта.
- [09_compute_aesthetic.py](D:/GitHub/Best_photo_ai/Scripts/09_compute_aesthetic.py): эстетическая оценка.
- [10_build_best.py](D:/GitHub/Best_photo_ai/Scripts/10_build_best.py): финальный score и `review_groups.csv`.
- [13_prepare_video_index.py](D:/GitHub/Best_photo_ai/Scripts/13_prepare_video_index.py): нормализованный вход video-ветки (`video_index.csv`).
- [review_app.py](D:/GitHub/Best_photo_ai/Scripts/review_app.py): review UI.

## Текущий статус
- photo pipeline уже работает на asset-centric архитектуре;
- review layer для фото стабилен и пригоден для настройки алгоритмов;
- базовый action/export слой уже есть, но ещё не доведён до полного workflow;
- archive preflight уже умеет stop/continue/save/unpack, но ещё не встроен в финальный пользовательский UX на всём пути;
- video branch пока находится на уровне нормализованного входа через [13_prepare_video_index.py](D:/GitHub/Best_photo_ai/Scripts/13_prepare_video_index.py).

## Ближайшие цели
- довести archive preflight до завершённого пользовательского сценария во всём pipeline;
- завершить transactional action/export workflow с rollback;
- развить video analysis branch поверх `video_index.csv`;
- затем продолжить refinement фото- и видео-алгоритмов.
