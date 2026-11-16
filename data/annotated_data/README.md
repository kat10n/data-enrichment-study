# Annotated data

Original annotator exports for the new data collection. Preserve these files before resolving disagreements; they are not adjudicated ground truth.

| File | Dataset | Annotator | Tasks | Status |
| --- | --- | --- | ---: | --- |
| `telegram/text_A.json` | Text messages | A | 200 | First independent annotation; not adjudicated |
| `caltech/images_A.json` | Camera-trap images | A | 140 | First annotation export; issues pending review; not adjudicated |

Both files are unchanged full JSON exports from Label Studio. Each task contains one saved, non-cancelled annotation. Task data and `data.sample_id` values match the corresponding 200-text and 140-image annotation inputs.

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

The export contains 74 frames marked `animal`, 66 marked `empty`, and 113 `animal` bounding boxes across 73 frames. All exported boxes have positive dimensions, valid original image dimensions, zero rotation, and coordinates within the image.

Review is still needed:

- `IMG021` is marked `animal` but has no bounding box.
- Fourteen low-confidence tasks have no saved explanatory comment, although the codebook requests one: `IMG119`, `IMG108`, `IMG115`, `IMG117`, `IMG114`, `IMG030`, `IMG022`, `IMG127`, `IMG065`, `IMG121`, `IMG041`, `IMG031`, `IMG131`, `IMG099`.

These findings have not been corrected automatically. The original answers and timing values remain unchanged. Details and per-task image durations are in [caltech/images_A_check.json](caltech/images_A_check.json). The checks assess export structure and timing availability, not the visual correctness of all labels.

## Sources and file integrity

Text collection sources and preparation details: [../real/telegram/](../real/telegram/).
Image collection: [../real/caltech/images/](../real/caltech/images/).

Export dates in Europe/Moscow: Text A — 2026-11-16; Images A — 2026-11-17. The original image download filename uses the UTC date 2026-11-16.

SHA-256 of `telegram/text_A.json`: `874be3e871ceb8e1729650e0a2c697e21007bb52db4d9b099ea65d4ebf4c4bce`.

SHA-256 of `caltech/images_A.json`: `f340ac68aa253505e8d3610d682e6f4ef736dfa0c3d2abd2448778d5b2ef7af1`.
