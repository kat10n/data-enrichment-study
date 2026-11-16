# Annotated data

Annotator exports for the new data collection. Preserve the original exports before resolving disagreements; these files are not adjudicated ground truth.

| File | Dataset | Annotator | Tasks | Status |
| --- | --- | --- | ---: | --- |
| `telegram/text_A.json` | Text messages | A | 200 | First independent annotation; not adjudicated |
| `caltech/images_A.json` | Camera-trap images | A | 140 | Export with one explicit A-requested status correction; not adjudicated |

Both files use the full JSON export format from Label Studio. The text export is unchanged. The image export has one correction explicitly requested by annotator A on 2026-11-17: `IMG021` changed from `animal` to `empty`. It had no bounding box. No other task or annotation values were changed, including all timing values. The original image export is retained in Git history and as a separate local backup. The local Label Studio project was not modified by this file correction.

Each task contains one saved, non-cancelled annotation. Task data and `data.sample_id` values match the corresponding 200-text and 140-image annotation inputs.

Compare annotators by `data.sample_id`, not Label Studio's internal task IDs. Store adjudicated labels separately. Annotator B's exports have not been added yet; B should work from the neutral task inputs without consulting A's answers.

Some image interpretation and annotation questions were discussed with an AI assistant during annotation. The image export is not claimed to be entirely unaided annotation.

## Recorded time

`annotations[].lead_time` contains the duration recorded by Label Studio in seconds. It is not claimed to be manually measured active time with every pause removed.

| Export | Positive `lead_time` values | Total seconds | Total, rounded to seconds |
| --- | ---: | ---: | --- |
| Text A | 200 / 200 | 2123.301 | 00:35:23 |
| Images A | 140 / 140 | 7871.323 | 02:11:11 |

The optional custom `active_seconds` field is separate. Text A contains no entries in that field; Images A contains one zero entry for `IMG035`. That zero is not used as the task duration: its automatic `lead_time` is 163.544 seconds. All automatic durations are present and positive.

## Image export checks

After the user-requested correction, the export contains 73 frames marked `animal`, 67 marked `empty`, and 113 `animal` bounding boxes across 73 frames. All exported boxes have positive dimensions, valid original image dimensions, zero rotation, and coordinates within the image. Every frame marked `animal` has at least one box; every frame marked `empty` has none.

Review is still needed:

- Fourteen low-confidence tasks have no saved explanatory comment, although the codebook requests one: `IMG119`, `IMG108`, `IMG115`, `IMG117`, `IMG114`, `IMG030`, `IMG022`, `IMG127`, `IMG065`, `IMG121`, `IMG041`, `IMG031`, `IMG131`, `IMG099`.

Missing comments have not been invented or filled in. Details, the explicit correction record, file hashes, and per-task image durations are in [caltech/images_A_check.json](caltech/images_A_check.json). The checks assess export structure and timing availability, not the visual correctness of all labels.

## Sources and file integrity

Text collection sources and preparation details: [../real/telegram/](../real/telegram/).
Image collection: [../real/caltech/images/](../real/caltech/images/).

Export dates in Europe/Moscow: Text A — 2026-11-16; Images A — 2026-11-17. The original image download filename uses the UTC date 2026-11-16.

SHA-256 of `telegram/text_A.json`: `874be3e871ceb8e1729650e0a2c697e21007bb52db4d9b099ea65d4ebf4c4bce`.

SHA-256 of the original image export before the correction: `f340ac68aa253505e8d3610d682e6f4ef736dfa0c3d2abd2448778d5b2ef7af1`. The hash of the corrected `caltech/images_A.json` is recorded in `caltech/images_A_check.json`.
