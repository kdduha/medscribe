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
