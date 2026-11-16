# Annotated data

Independent annotation exports for the new data collection.

| File | Dataset | Annotator | Tasks | Status |
| --- | --- | --- | ---: | --- |
| `telegram/text_A.json` | Text messages | A | 200 | First independent annotation; not adjudicated |

The file is an unchanged full JSON export from Label Studio. Each task contains one saved, non-cancelled annotation. The original texts and `data.sample_id` values match the 200-task annotation input.

Compare annotators by `data.sample_id`, not Label Studio's internal task IDs. Preserve the original A/B exports before resolving disagreements; store adjudicated labels separately. Annotator B's export and image annotations have not been added to this folder yet.

`annotations[].lead_time` contains the duration recorded by Label Studio in seconds. It is not claimed to be manually measured active time with every pause removed.

Text collection sources and preparation details: [../real/telegram/](../real/telegram/).

Export date: 2026-11-16.

SHA-256 of `telegram/text_A.json`: `874be3e871ceb8e1729650e0a2c697e21007bb52db4d9b099ea65d4ebf4c4bce`.
