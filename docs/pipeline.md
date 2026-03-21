# Pipeline Specification

## Цель
Проект предназначен для анализа библиотек воспоминаний:
- локальных папок с фото и видео;
- экспортов Google Photos / iCloud / Яндекс Фото;
- других источников с медиафайлами и sidecar-метаданными.

Результат работы:
- отфильтрованные лучшие фото и видео;
- grouped scenes / series;
- curated-библиотека без лишних дублей;
- возможность отката перемещений.

## Текущий статус реализации
На март 2026 проект находится в фазе миграции от file-centric модели к asset-centric модели.

Уже реализовано:
- ранний шаг [01_preflight_archives.py](D:/GitHub/Best_photo_ai/Scripts/01_preflight_archives.py) для поиска архивов;
- asset-aware индекс и сборка sidecar-связей в [02_scan_takeout.py](D:/GitHub/Best_photo_ai/Scripts/02_scan_takeout.py);
- canonicalization/exact dedupe в [03_find_exact_duplicates.py](D:/GitHub/Best_photo_ai/Scripts/03_find_exact_duplicates.py);
- dual-key слой `asset_id + file_path` в photo-ветке;
- asset-aware ручное удаление во вьюере;
- GPU-aware и batched semantic grouping в [05_group_similar_images.py](D:/GitHub/Best_photo_ai/Scripts/05_group_similar_images.py);
- CPU/CUDA installation profiles и runtime check.

Пока ещё не завершено:
- интерактивный пользовательский диалог по найденным архивам;
- полноценный video pipeline;
- transactional export `move + manifest + rollback`;
- полный отказ downstream-логики от центральности `file_path`.

Текущий приоритет после завершения photo-ветки:
- перейти к action/export layer;
- формализовать `curation_plan.csv`;
- формализовать `move_manifest.csv`;
- перевести [11_build_photo_library.py](D:/GitHub/Best_photo_ai/Scripts/11_build_photo_library.py) и [12_cleanup_lower_rated_duplicates.py](D:/GitHub/Best_photo_ai/Scripts/12_cleanup_lower_rated_duplicates.py) на asset-level действия.

## Текущая нумерация шагов
1. [01_preflight_archives.py](D:/GitHub/Best_photo_ai/Scripts/01_preflight_archives.py)
2. [02_scan_takeout.py](D:/GitHub/Best_photo_ai/Scripts/02_scan_takeout.py)
3. [03_find_exact_duplicates.py](D:/GitHub/Best_photo_ai/Scripts/03_find_exact_duplicates.py)
4. [04_prepare_analysis_images.py](D:/GitHub/Best_photo_ai/Scripts/04_prepare_analysis_images.py)
5. [05_group_similar_images.py](D:/GitHub/Best_photo_ai/Scripts/05_group_similar_images.py)
6. [06_compute_sharpness.py](D:/GitHub/Best_photo_ai/Scripts/06_compute_sharpness.py)
7. [07_compute_composition.py](D:/GitHub/Best_photo_ai/Scripts/07_compute_composition.py)
8. [08_compute_subject.py](D:/GitHub/Best_photo_ai/Scripts/08_compute_subject.py)
9. [09_compute_aesthetic.py](D:/GitHub/Best_photo_ai/Scripts/09_compute_aesthetic.py)
10. [10_build_best.py](D:/GitHub/Best_photo_ai/Scripts/10_build_best.py)
11. [11_build_photo_library.py](D:/GitHub/Best_photo_ai/Scripts/11_build_photo_library.py)
12. [12_cleanup_lower_rated_duplicates.py](D:/GitHub/Best_photo_ai/Scripts/12_cleanup_lower_rated_duplicates.py)

## Базовые сущности

### Raw File
Отдельный файл, найденный в источнике.

Поля:
- абсолютный путь;
- имя;
- расширение;
- размер;
- тип: `media`, `sidecar`, `archive`, `unsupported`, `other`.

### Asset
Основная единица обработки.

`asset` включает:
- один primary media file;
- связанные service/sidecar-файлы;
- единый `asset_id`;
- общую судьбу при анализе, переносе, удалении и rollback.

Система должна работать на уровне `asset`, а не одиночных файлов.

### Dual-Key Transition
На переходном этапе часть артефактов хранит одновременно:
- `asset_id` как целевой логический ключ;
- `file_path` как operational key для существующих вычислений и файловой работы.

Переходная стратегия:
- новые шаги обязаны протаскивать `asset_id`;
- merge результатов должен предпочитать `asset_id`;
- `file_path` используется как fallback, пока вся ветка не станет полностью asset-native.

## Sidecar / service files
- Sidecar-файлы неотделимы от primary media.
- Перемещение, удаление и rollback выполняются только на уровне всего `asset`.
- Связь sidecar с primary определяется по локальному контексту пути и набору правил, а не только по имени файла.
- Если sidecar остался без основного файла, он считается `orphan_sidecar`, не идёт в основной анализ и попадает в audit-report.

## Итоговый идеальный пайплайн

### 1. Source Ingest
Вход:
- локальная папка;
- облачный коннектор;
- фильтры по датам, альбомам, типам медиа.

Выход:
- поток сырых файлов/объектов источника.

### 2. Raw Audit
Задачи:
- найти архивы;
- найти orphan sidecars;
- найти unsupported files;
- найти потенциально нерелевантные файлы;
- подготовить отчёт по проблемам входа.

Выход:
- `raw_files.csv`
- `audit_report.csv`
- `orphan_sidecars.csv`
- `archives_found.csv`

#### Архивы
Если найдены архивы, система должна показать пользователю развилку:
1. остановиться;
2. игнорировать и продолжить;
3. сохранить список архивов в `txt` с абсолютными путями;
4. автоматически распаковать архивы в staging-папку и продолжить анализ.

Текущее состояние:
- ранний поиск архивов уже вынесен в отдельный шаг;
- пока реализован отчёт в `archives_found.csv` и печать путей в stdout;
- интерактивная развилка для пользователя ещё не реализована.

### 3. Asset Assembly
Задачи:
- собрать `asset` из primary media и sidecars;
- присвоить стабильный `asset_id`;
- связать asset с источником, альбомом и метаданными.

Выход:
- `media_assets.csv`

### 4. Canonicalization / Exact Dedup
Это не просто удаление дублей.

Задачи:
- находить одинаковые primary media;
- выбирать canonical asset;
- переносить к нему максимально полный набор sidecar/metadata;
- помечать остальные как absorbed / redundant.

Особый случай:
- если у одного дубля есть `json`, а у другого нет, canonical asset должен сохранить более полный комплект метаданных.

Выход:
- `duplicate_assets.csv`
- `unique_assets.csv`

### 5. Branching
После canonicalization:
- фото идут в photo pipeline;
- видео идут в video pipeline;
- unsupported / other остаются только в audit.

## Photo Pipeline

### 6. Photo Technical Index
Задачи:
- размеры;
- размер файла;
- `pHash`;
- EXIF / json datetime;
- базовый `content_type_file`.

`content_type_file`:
- `people`
- `landscape`
- `document`

Эта классификация должна считаться вне viewer.

Выход:
- `photo_index.csv`

Текущее состояние:
- эта логика сейчас частично включена в [02_scan_takeout.py](D:/GitHub/Best_photo_ai/Scripts/02_scan_takeout.py) и [04_prepare_analysis_images.py](D:/GitHub/Best_photo_ai/Scripts/04_prepare_analysis_images.py);
- отдельная финальная таблица `photo_index.csv` ещё не выделена как самостоятельный артефакт.

### 7. Photo Measurable Features
Задачи:
- sharpness;
- resolution / pixels;
- face coverage;
- subject placement;
- tilt / edge penalties;
- composition features.

Текущее состояние:
- [06_compute_sharpness.py](D:/GitHub/Best_photo_ai/Scripts/06_compute_sharpness.py), [07_compute_composition.py](D:/GitHub/Best_photo_ai/Scripts/07_compute_composition.py), [08_compute_subject.py](D:/GitHub/Best_photo_ai/Scripts/08_compute_subject.py) уже переведены на parallel CPU execution;
- результаты этих шагов уже умеют протаскивать `asset_id`.

### 8. Photo Semantic Quality
Задачи:
- aesthetic score;
- subject quality;
- дополнительные semantic metrics по мере развития проекта.

Текущее состояние:
- [09_compute_aesthetic.py](D:/GitHub/Best_photo_ai/Scripts/09_compute_aesthetic.py) переведён на batched inference;
- при доступной CUDA semantic scoring идёт на GPU;
- batch size зависит от устройства.

### 9. Photo Grouping
Уровни группировки:
- exact / near-duplicate;
- strict group;
- merged scene group.

Используемые сигналы:
- `pHash`;
- semantic similarity;
- background similarity;
- face-identity veto для multi-person сцен;
- локальные мостики внутри серии.

Выход:
- `similar_pairs.csv`
- `similar_groups.csv`

Текущее состояние:
- [05_group_similar_images.py](D:/GitHub/Best_photo_ai/Scripts/05_group_similar_images.py) уже использует:
  - staged grouping;
  - CLIP semantic matching;
  - background similarity;
  - local pHash bridges;
  - face-identity veto;
  - disk-cache embeddings;
  - batched GPU precompute при доступной CUDA;
- `similar_pairs.csv` и `similar_groups.csv` уже содержат asset-aware поля.

### 10. Scene Content Aggregation
Нужно агрегировать теги на разных уровнях:
- `content_type_file`
- `content_type_group`
- `content_type_scene`

Это нужно для:
- фильтров viewer;
- контроля качества;
- будущих правил ranking/export.

Текущее состояние:
- `content_type_file`, `content_type_group`, `content_type_scene` уже рассчитываются и используются в review layer.

### 11. Photo Ranking
Задача:
- внутри строгой группы выбрать лучший кадр.

Score учитывает:
- measurable features;
- semantic quality;
- нормализацию внутри группы;
- тип сцены при необходимости.

Выход:
- `best_combined.csv`
- `review_groups.csv`

Текущее состояние:
- [10_build_best.py](D:/GitHub/Best_photo_ai/Scripts/10_build_best.py) уже умеет мерджить score-артефакты по `asset_id` с fallback на `file_path`;
- `review_groups.csv` уже содержит asset-aware поля для viewer.

## Video Pipeline

### 12. Video Technical Index
Минимальный набор:
- duration;
- fps;
- resolution;
- bitrate;
- audio presence.

### 13. Video Semantic Analysis
Минимальный набор:
- keyframes;
- visual quality;
- similarity between videos;
- связность с photo scenes.

### 14. Video Ranking
Задача:
- выбрать лучший ролик внутри серии или сцены.

Выход:
- `video_index.csv`
- `video_groups.csv`
- `video_best.csv`

## Review Layer

### 15. Unified Review UI
Viewer нужен для:
- контроля качества группировки;
- контроля качества тегов;
- ручного удаления и проверки;
- обучения правил.

Viewer не должен содержать business-логику классификации.

Viewer должен отображать:
- file/group/scene tags;
- строгие и объединённые сцены;
- фильтры по категориям контента.

Текущее состояние:
- viewer уже отображает file/group/scene tags;
- viewer уже умеет удалять asset вместе с sidecar-файлами;
- viewer пока остаётся review-инструментом и не является transactional action layer.

## Export / Cleanup

### 16. Export Plan
Сначала строится план, а не выполняется перенос сразу.

Выход:
- `curation_plan.csv`
- `move_manifest.csv` или `move_manifest.jsonl`

### 17. Transactional Move To Curated
Основной сценарий:
- лучшие asset’ы перемещаются, а не копируются;
- вместе с sidecar-файлами;
- с сохранением структуры папок/альбомов;
- с записью manifest для rollback.

Default mode:
- preserve album structure.

Текущее состояние:
- этот слой ещё не реализован полностью;
- текущие `11/12` шаги следует рассматривать как предварительные/временные по отношению к целевому export design.

### 18. Rollback
Rollback должен:
- читать manifest;
- возвращать весь `asset` целиком;
- восстанавливать primary и sidecars;
- логировать конфликты.

### 19. Cleanup / Archive Losers
Отдельный, не автоматический шаг:
- удалить неотобранное;
- архивировать неотобранное;
- оставить неотобранное на месте.

## Итоговые таблицы
- `raw_files.csv`
- `audit_report.csv`
- `orphan_sidecars.csv`
- `archives_found.csv`
- `media_assets.csv`
- `duplicate_assets.csv`
- `unique_assets.csv`
- `photo_index.csv`
- `photo_features.csv`
- `photo_semantic_scores.csv`
- `similar_pairs.csv`
- `similar_groups.csv`
- `best_combined.csv`
- `review_groups.csv`
- `video_index.csv`
- `video_groups.csv`
- `video_best.csv`
- `curation_plan.csv`
- `move_manifest.csv`

## Runtime Profiles
Проект должен поддерживать два профиля среды:

- CPU profile:
  - [requirements-cpu.txt](D:/GitHub/Best_photo_ai/requirements-cpu.txt)
- CUDA profile:
  - [requirements-cuda.txt](D:/GitHub/Best_photo_ai/requirements-cuda.txt)

Проверка среды:
- [check_runtime.py](D:/GitHub/Best_photo_ai/Scripts/check_runtime.py)

Критично для производительности:
- [05_group_similar_images.py](D:/GitHub/Best_photo_ai/Scripts/05_group_similar_images.py) и [09_compute_aesthetic.py](D:/GitHub/Best_photo_ai/Scripts/09_compute_aesthetic.py) должны использовать CUDA-enabled PyTorch на машинах с NVIDIA GPU.


