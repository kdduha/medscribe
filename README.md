# Medscribe — полный цикл данных → SFT → RAG

Репозиторий охватывает все этапы: анонимизация, сбор и подготовка датасета, supervised fine-tuning (локально и в Together.ai), построение RAG-индекса, запуск инференса и метрик. Ниже — практический гид по каждому шагу.

## 0. Пайплайн

```text
            СЫРЫЕ ДАННЫЕ (CSV/Excel)
                       │
                       ▼
          Анонимизация + подготовка к SFT/RAG
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
  ПОДГОТОВКА К SFT              ПОДГОТОВКА RAG
        │                             │
        ▼                             ▼
  ОБУЧЕНИЕ (SFT)                 FAISS-ИНДЕКС
  ├─ локально                         │
  └─ Together.ai                      │
        │                             │
        └──────────────┬──────────────┘
                       ▼
                  vLLM server
                       │
                       ▼
          RAG ИНФЕРЕНС / ВАЛИДАЦИЯ
          ├─ онлайн/батч
          └─ оффлайн eval
                       │
                       ▼
            CSV + JSON-логи + метрики
```

## 1. Установка окружения
- Python 3.10+
- `pip install -r requirements.txt`
- Переменные окружения для LLM:
  - `OPENAI_API_KEY` и (опционально) `OPENAI_BASE_URL` — для OpenAI-совместимых API или локального vLLM.
  - `TOGETHER_API_KEY` — для Together.ai fine-tuning.

## 2. Подготовка данных
Высокоуровневая схема:
1. (Опционально) Скачиваем `processed_result(s).csv` с Google Drive.
2. Объединяем найденные CSV в один `datasets/raw_compiled.csv`.
3. Генерируем промпты и таргеты для SFT/RAG.
4. Преобразуем JSONL в чат-формат и делим на train/eval/test.

Все шаги управляются Hydra-конфигом `configs/prepare_data.yaml`.

### 2.1 Запуск конвейера
```bash
python -m scripts.prepare_data \
  'steps=[download,compile,prompts]' \
  data.drive_folder='<DRIVE_FOLDER_URL_OR_ID>'
```

Другие опции:
- Только собрать CSV, если данные уже локально:
  ```bash
  python -m scripts.prepare_data 'steps=[compile]' \
    data.output_root='datasets/raw_data'
  ```
- Только построить промпты из готового CSV:
  ```bash
  python -m scripts.prepare_data \
    'steps=[prompts]' \
    prompts.input_file='datasets/raw_compiled.csv' \
    prompts.output_jsonl='datasets/train_prompts.jsonl'
  ```

### 2.2 Форматы и артефакты
- `datasets/raw_data/**/processed_result(s).csv` — исходники из Google Drive.
- `datasets/raw_compiled.csv` — объединённый CSV (+ колонки `origin_folder`, `modality`).
- `datasets/train_prompts.jsonl` — JSONL с полями `prompt_text`, `result_text`, `organ`, `finding_text`, `type`.
- `datasets/sft_train.jsonl`, `datasets/sft_eval.jsonl`, `datasets/sft_test_rag.jsonl` — чат-формат для обучения и проверки.
- `datasets/sft_dataset/` — HuggingFace DatasetDict (train/eval) с `messages`.
- `datasets/rag_train.jsonl` — пары `finding_text/result_text` для индекса.

#### Пользовательские колонки
Через `prompts.source_columns` можно сопоставить нестандартные имена (`organ`, `finding`, `result`). `organ_mapper` позволяет привести аббревиатуры к читаемым органам.

## 3. Подготовка к SFT
Флаг `sft_prep.enabled=true` автоматически:
- конвертирует `prompt_text/result_text` в `{"messages": [{"role": "user"...}, ...]}`;
- делает train/eval split;
- выгружает HF dataset (`datasets/sft_dataset`), совместимый с `trl.SFTTrainer`.

Пример ручного запуска:
```bash
python -m src.data_preparation.prepare_for_sft \
  source_jsonl='datasets/train_prompts.jsonl' \
  outputs.train_jsonl='datasets/sft_train.jsonl' \
  outputs.eval_jsonl='datasets/sft_eval.jsonl'
```

## 4. Локальный SFT (GPU)
Скрипт: `sft/train_sft.py`, конфиг по умолчанию `sft/configs/sft.yaml`.

```bash
python -m sft.train_sft \
  model.name_or_path='Qwen/Qwen2.5-7B-Instruct' \
  data.hf_dir='datasets/sft_dataset' \
  training.output_dir='outputs/sft_qwen2p5' \
  lora.enabled=true
```

Особенности:
- Используется `trl.SFTTrainer` с LoRA (`peft`)
- Если нет HF-dataset, укажите `data.train_path`/`data.eval_path` и поля (`input_field`, `target_field`) + шаблон `{prompt}\n\n{response}`
- Для inference модели удобно выгружать в GGUF или использовать vLLM (см. раздел 6.2)

## 5. SFT в Together.ai
Скрипт: `sft/together_finetune.py`, конфиги в `sft/configs/together*.yaml`

```bash
export TOGETHER_API_KEY=...
python -m sft.together_finetune \
  +model.name_or_path='Qwen/Qwen3-14B' \
  data.train_path='datasets/sft_train.jsonl' \
  data.eval_path='datasets/sft_eval.jsonl' \
  training.batch_size=8 \
  training.epochs=3 \
  job.wait=true
```

Требования к данным:
- JSONL из `prepare_for_sft` (каждый объект: `{"messages": [{"role": "user",...}, {"role": "assistant",...}]}`)
- Скрипт сам загрузит файлы в Together (или примет `file-***` id)

Опции:
- `lora.*` — параметры LoRA на стороне Together (`target_modules: all-linear` для Qwen/LLaMA)
- `training.*` — базовые гиперпараметры API
- `job.save_job_json` — куда сохранить идентификатор запущенной джобы

## 6. Подготовка и запуск RAG

### 6.1 FAISS-индекс
Используется `src/rag/retriever.FaissRetriever` с `sentence-transformers` (по умолчанию `intfloat/e5-base-v2`)

```bash
python -m scripts.build_index \
  jsonl='datasets/rag_train.jsonl' \
  out_dir='artifacts/rag_index' \
  model='intfloat/e5-base-v2' \
  device='cuda'
```

Формат входа: JSONL, поля `finding_text` и `result_text`. Скрипт пропускает пустые строки и сохраняет `index.faiss` + `meta.json` (со списком примеров и гиперпараметрами). Конфиг — `configs/rag_build_index.yaml`

### 6.2 LLM-сервер (OpenAI style API)
`scripts.rag_infer` и `scripts.rag_eval` обращаются к `OpenAICompatClient`. Можно:
- Использовать облачный OpenAI / Together / DeepInfra
- Запустить локально vLLM c вашим SFT-чекпоинтом:
  ```bash
  vllm serve Qwen/Qwen2.5-7B-Instruct \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --api-key dummy \
    --model-format hf \
    --served-model-name qwen-med \
    --response-format json_schema \
    --trust-remote-code
  ```
  Тогда в конфиге: `llm.base_url='http://localhost:8000/v1'`, `llm.api_key='dummy'`, `llm.model='qwen-med'`.

### 6.3 Инференс (одиночный и batch)
Скрипт: `scripts/rag_infer.py`, конфиг `configs/rag_infer.yaml`

**Одиночный запрос:**
```bash
python -m scripts.rag_infer \
  organ='головной мозг' \
  finding='МР-картина арахноидальной кисты правой височной доли' \
  index_dir='artifacts/rag_index' \
  llm.base_url='http://localhost:8000/v1' \
  llm.model='qwen-med' \
  json_output=true cot=true
```

**Batch (CSV или JSONL):**
```bash
python -m scripts.rag_infer \
  input.file='datasets/sft_test_rag.jsonl' \
  input.format='jsonl' \
  input.columns.id='id' \
  input.columns.organ='organ' \
  input.columns.finding='finding_text' \
  input.columns.reference='result_text' \
  top_k=5 json_output=true cot=false
```

Флаги:
- `top_k` — сколько примеров достать из индекса
- `cot=true` — добавить chain-of-thought инструкции (+ поле `reasoning` в ответе)
- `json_output=true` — включить строгий JSON-ответ (`result/has_finding/organ/modality`). Парсер (`src/rag/postprocess.py`) fallback-ит в сырой текст, если JSON невалидный

Артефакты:
- `outputs/team_name_results.csv` — акумулирует строки (может дозаписываться)
- `outputs/logs/log_<timestamp>_<id>.json` — полный ввод/вывод и примеры
- `outputs/metrics.json` — агрегированные метрики для текущего запуска

### 6.4 Опциональный reasoning parser
Если `json_output=true`, `parse_llm_response` использует `langchain JsonOutputParser` и Pydantic-схему (`result`, `reasoning`, `has_finding`, `organ`, `modality`). Это позволяет:
- сверять классификационные метрики (`compute_classification_accuracy`);
- пост-обрабатывать модальность (например, для UI).

## 7. Оценка датасетов (`scripts/rag_eval.py`)
Используется тот же RAG-пайплайн, но поверх тестового JSONL и с расчётом средних метрик.

```bash
python -m scripts.rag_eval \
  input_jsonl='datasets/sft_test_rag.jsonl' \
  index_dir='artifacts/rag_index' \
  llm.base_url='http://localhost:8000/v1' \
  llm.model='qwen-med' \
  json_output=true cot=true \
  outputs.csv='outputs/metrics_eval.csv' \
  outputs.metrics='outputs/metrics_eval.json'
```

Метрики (`src/validation/metrics.py`):
- Exact Match, BLEU, ROUGE-L, METEOR (нужен `nltk`), Levenshtein;
- Accuracy по наличию заключения, органу (exact + cosine через `sentence-transformers`), модальности.

## 8. Структура репозитория (ключевое)
- `scripts/anonymize.py` — анонимизация исходных CSV.
- `scripts/prepare_data.py` — единая точка входа для download/compile/prompts/SFT prep.
- `src/data_preparation/*` — модули пайплайна.
- `sft/train_sft.py`, `sft/together_finetune.py` — локальный и Together SFT.
- `scripts/build_index.py`, `scripts/rag_infer.py`, `scripts/rag_eval.py` — RAG.
- `src/rag/*` — ретривер, промпты, LLM-клиент, парсинг.
- `src/validation/metrics.py` — текстовые и классификационные метрики.
- `configs/*.yaml`, `sft/configs/*.yaml` — все Hydra-конфиги, переопределяются через CLI.

## 9. Частые сценарии
| Задача | Команда | Комментарий |
| --- | --- | --- |
| Сбор данных из Google Drive | `python -m scripts.prepare_data 'steps=[download,compile]' data.drive_folder='...'` | Загружает CSV и объединяет. |
| Генерация промптов + подготовка SFT | `python -m scripts.prepare_data 'steps=[prompts]'` | Заполняет `datasets/train_prompts.jsonl`, `datasets/sft_*`. |
| Локальный SFT | `python -m sft.train_sft data.hf_dir='datasets/sft_dataset'` | Работает из коробки с LoRA. |
| Together SFT | `python -m sft.together_finetune data.train_path='datasets/sft_train.jsonl'` | Авто-аплоад датасета. |
| Построение FAISS | `python -m scripts.build_index jsonl='datasets/rag_train.jsonl'` | Требует `faiss-cpu` или GPU. |
| Инференс RAG (batch) | `python -m scripts.rag_infer input.file='datasets/sft_test_rag.jsonl'` | Пишет CSV + JSON-логи. |
| Eval на датасете | `python -m scripts.rag_eval input_jsonl='datasets/sft_test_rag.jsonl'` | Возвращает агрегированные метрики. |
