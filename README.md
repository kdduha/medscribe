## запуск анонимайзера 
```shell
python -m scripts.anonymize datasets/raw-data datasets/anon-data
```

## сбор данных из заключений
`TODO`

## подготовка данных для sft
конфиг по умолчанию: `configs/pipeline.yaml`.

запуск всего пайплайна (download → compile → prompts):
```shell
python -m scripts.prepare_data 'steps=[download,compile,prompts]'
```

только генерация промптов (предполагается, что `datasets/compiled.csv` уже существует):
```shell
python -m scripts.prepare_data 'steps=[prompts]'
```

скачать только `processed_result(s).csv` и собрать единый CSV:
```shell
python -m scripts.prepare_data 'steps=[download,compile]' \
  data.drive_folder='<FOLDER_ID_OR_URL>'
```

переопределения конфига через cli:
```shell
python -m scripts.prepare_data 'steps=[compile,prompts]' \
  compile.output_csv='datasets/compiled.csv' \
  prompts.input_file='datasets/compiled.csv' \
  prompts.output_jsonl='datasets/train_prompts.jsonl'
```

## построение индекса и инференс

### FAISS-индекс
```bash
python -m scripts.build_index jsonl=datasets/train_prompts.jsonl out_dir=artifacts/rag_index
```
параметры также настраиваются в `configs/rag_build_index.yaml`.

### инференс (один запрос)
```bash
python -m scripts.rag_infer \
  organ='головной мозг' \
  finding='МР-картина арахноидальной кисты правой височной доли' \
  json_output=true cot=true
```
Выходы:
- CSV: `outputs/team_name_results.csv`
- логи промптов/ответов: `outputs/logs/*.json`
- метрики: `outputs/metrics.json`

### инференс из файла (batch)
поддерживается `CSV` и `JSONL`.

Ожидаемые поля (можно переопределить):
- `id` — идентификатор строки
- `organ` — орган (или `organ_abbr`)
- `finding` — текст находки (или `finding_text`)
- `result_text` — референс (опционально)

Пример запуска:
```bash
python -m scripts.rag_infer \
  input.file=datasets/train_prompts.jsonl \
  input.format=jsonl \
  input.columns.id=id \
  input.columns.organ=organ \
  input.columns.finding=finding_text \
  input.columns.reference=result_text \
  json_output=true cot=true
```

### оценка датасета (eval)
```bash
python -m scripts.rag_eval \
  input_jsonl=datasets/train_prompts.jsonl \
  json_output=true cot=true
```
Метрики: Accuracy (Exact Match), BLEU, ROUGE-L, METEOR (если доступен nltk), Levenshtein.
