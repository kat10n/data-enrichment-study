# data-enrichment-study

## Сравнение разметок A/B

Из корня репозитория, с Python 3.9+:

```bash
python3 scripts/compare_annotations.py
```

Дополнительные библиотеки не нужны. Отчёт и список расхождений появятся в
`artifacts/annotation_comparison/`. Скрипт не изменяет исходные ответы и не
принимает итоговые решения за аннотаторов.

[Инструкция, методика и проверка скрипта](docs/annotation_comparison.md).
