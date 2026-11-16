#!/usr/bin/env python3
"""Compare Label Studio A/B exports; Python standard library only.

Run from any directory: python3 path/to/scripts/compare_annotations.py
"""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
import sys

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'data/annotated_data'
DEFAULT_INPUTS = {
    'text_A': BASE / 'telegram/text_A.json',
    'text_B': BASE / 'telegram/text_B.json',
    'images_A_original': BASE / 'photo/original_exports/images_A_2026-11-17.json',
    'images_A_current': BASE / 'photo/images_A.json',
    'images_B': BASE / 'photo/animals_B.json',
}


def require(condition, message):
    if not condition:
        raise ValueError(str(message))


def load(path):
    tasks = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(tasks, list), f'{path}: expected a full JSON task list, not JSON_MIN')
    result = {}
    for task in tasks:
        sid = task['data']['sample_id']
        require(sid not in result, ('duplicate_id', sid))
        require(len(task.get('annotations', [])) == 1, ('expected_one_annotation', sid))
        require(not task['annotations'][0].get('was_cancelled'), ('cancelled', sid))
        result[sid] = task
    return result


def unpack(task, kind):
    annotation = task['annotations'][0]
    results = annotation['result']
    field = 'label' if kind == 'text' else 'frame_status'

    def choice(name):
        entries = [r for r in results if r.get('from_name') == name]
        if not entries:
            return None
        require(len(entries) == 1, ('multiple_choice_results', task['data']['sample_id'], name))
        values = entries[0]['value'].get('choices', [])
        require(len(values) == 1, (task['data']['sample_id'], name, values))
        return values[0]

    flags = {r['id']: sorted(r['value'].get('choices', []))
             for r in results if r.get('from_name') == 'box_flags'}
    boxes = []
    for result in results:
        if result.get('type') != 'rectanglelabels':
            continue
        value = result['value']
        box = {key: float(value[key]) for key in ('x', 'y', 'width', 'height')}
        require(all(math.isfinite(v) for v in box.values()), ('nonfinite_box', task['data']['sample_id']))
        require(value.get('rotation', 0) == 0, ('rotated_box', task['data']['sample_id']))
        require(value.get('rectanglelabels') == ['animal'], ('unexpected_box_label', task['data']['sample_id']))
        require(min(box['x'], box['y']) >= -0.001 and box['width'] > 0 and box['height'] > 0,
                ('invalid_box', task['data']['sample_id']))
        require(box['x'] + box['width'] <= 100.001 and box['y'] + box['height'] <= 100.001,
                ('box_outside_image', task['data']['sample_id']))
        require(result.get('original_width', 0) > 0 and result.get('original_height', 0) > 0,
                ('missing_image_dimensions', task['data']['sample_id']))
        box.update({'id': result['id'], 'flags': flags.get(result['id'], []),
                    'original_width': result['original_width'],
                    'original_height': result['original_height']})
        boxes.append(box)
    record = {
        'status': choice(field),
        'confidence': choice('confidence'),
        'comment': '\n'.join(s for r in results if r.get('from_name') == 'comment'
                             for s in r['value'].get('text', [])),
        'view_adjusted': choice('view_adjusted'),
        'lead_time_seconds': annotation.get('lead_time'),
        'boxes': boxes,
    }
    valid = ('not_spam', 'spam', 'needs_review', 'excluded') if kind == 'text' else ('animal', 'empty', 'needs_review', 'excluded')
    require(record['status'] in valid, ('invalid_status', task['data']['sample_id'], record['status']))
    return record


def binary_metrics(rows, labels):
    eligible = [(a, b) for a, b in rows if a in labels and b in labels]
    n = len(eligible)
    matrix = [[sum(a == left and b == right for a, b in eligible) for right in labels] for left in labels]
    agreement = sum(a == b for a, b in eligible) / n if n else None
    expected = sum(sum(a == label for a, _ in eligible) * sum(b == label for _, b in eligible)
                   for label in labels) / n**2 if n else None
    kappa = (agreement - expected) / (1 - expected) if n and not math.isclose(expected, 1) else None
    return {'total_pairs': len(rows), 'comparable_pairs': n,
            'coverage': n / len(rows) if rows else None,
            'matching_pairs': sum(a == b for a, b in eligible),
            'disagreeing_pairs': sum(a != b for a, b in eligible),
            'nonbinary_pairs': len(rows) - n, 'agreement': agreement, 'cohen_kappa': kappa,
            'matrix_labels': labels, 'matrix_rows_A_columns_B': matrix,
            'status_counts_A': dict(Counter(a for a, _ in rows)),
            'status_counts_B': dict(Counter(b for _, b in rows))}


def iou(a, b):
    left = max(a['x'], b['x'])
    top = max(a['y'], b['y'])
    right = min(a['x'] + a['width'], b['x'] + b['width'])
    bottom = min(a['y'] + a['height'], b['y'] + b['height'])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = a['width'] * a['height'] + b['width'] * b['height'] - intersection
    return intersection / union


def optimal_match(matrix, b_count, threshold=0.5):
    """Min-cost maximum flow: maximize pair count first, then summed IoU."""
    a_count = len(matrix)
    source, sink = a_count + b_count, a_count + b_count + 1
    graph = [[] for _ in range(sink + 1)]

    def edge(u, v, cost):
        forward = [v, len(graph[v]), 1, cost]
        graph[u].append(forward)
        graph[v].append([u, len(graph[u]) - 1, 0, -cost])
        return forward

    links = []
    for i in range(a_count):
        require(len(matrix[i]) == b_count, 'Invalid IoU matrix dimensions')
        edge(source, i, 0)
        for j in range(b_count):
            if matrix[i][j] >= threshold:
                links.append((i, j, edge(i, a_count + j, -matrix[i][j])))
    for j in range(b_count):
        edge(a_count + j, sink, 0)
    while True:
        distance, parent = [math.inf] * len(graph), [None] * len(graph)
        distance[source] = 0
        for _ in range(len(graph) - 1):
            changed = False
            for u, edges in enumerate(graph):
                if not math.isfinite(distance[u]):
                    continue
                for k, (v, _, capacity, cost) in enumerate(edges):
                    if capacity and distance[u] + cost < distance[v] - 1e-12:
                        distance[v], parent[v] = distance[u] + cost, (u, k)
                        changed = True
            if not changed:
                break
        if parent[sink] is None:
            break
        node = sink
        while node != source:
            u, k = parent[node]
            selected = graph[u][k]
            selected[2] -= 1
            graph[node][selected[1]][2] += 1
            node = u
    pairs = tuple((i, j) for i, j, selected in links if selected[2] == 0)
    return len(pairs), sum(matrix[i][j] for i, j in pairs), pairs


def compare_boxes(a, b):
    matrix = [[iou(left, right) for right in b] for left in a]
    count, total, pairs = optimal_match(matrix, len(b))
    matched = [{'A_index': i, 'B_index': j, 'iou': matrix[i][j],
                'flags_A': a[i]['flags'], 'flags_B': b[j]['flags']} for i, j in pairs]
    return {'boxes_A': len(a), 'boxes_B': len(b), 'matched_count': count,
            'matched': matched,
            'unmatched_A': [i for i in range(len(a)) if i not in {i for i, _ in pairs}],
            'unmatched_B': [j for j in range(len(b)) if j not in {j for _, j in pairs}],
            'matching_f1': 2 * count / (len(a) + len(b)) if a or b else None,
            'mean_matched_iou': total / count if count else None,
            'median_matched_iou': median(item['iou'] for item in matched) if matched else None}


def box_summary(rows):
    comparable = [row for row in rows if row['A']['status'] in ('animal', 'empty') and row['B']['status'] in ('animal', 'empty')]
    a_count = sum(row['box_comparison']['boxes_A'] for row in comparable)
    b_count = sum(row['box_comparison']['boxes_B'] for row in comparable)
    matched = [pair for row in comparable for pair in row['box_comparison']['matched']]
    return {'scope': 'Frames with a binary status from both annotators; needs_review/excluded frames excluded and reported separately.',
            'iou_threshold': 0.5, 'matching': 'Maximum cardinality, then maximum summed IoU; one-to-one.',
            'comparable_frames': len(comparable),
            'frames_with_two_empty_box_sets': sum(not row['A']['boxes'] and not row['B']['boxes'] for row in comparable),
            'frames_with_any_boxes': sum(bool(row['A']['boxes'] or row['B']['boxes']) for row in comparable),
            'boxes_A': a_count, 'boxes_B': b_count, 'matched_pairs': len(matched),
            'unmatched_A': a_count - len(matched), 'unmatched_B': b_count - len(matched),
            'matching_f1': 2 * len(matched) / (a_count + b_count) if a_count + b_count else None,
            'mean_matched_iou': mean(pair['iou'] for pair in matched) if matched else None,
            'median_matched_iou': median(pair['iou'] for pair in matched) if matched else None,
            'pairs_below_review_iou_075': sum(pair['iou'] < 0.75 for pair in matched),
            'matched_pairs_with_different_flags': sum(pair['flags_A'] != pair['flags_B'] for pair in matched)}


def reasons(a, b, kind, comparison=None):
    result = []
    binary = ('spam', 'not_spam') if kind == 'text' else ('animal', 'empty')
    if a['status'] not in binary or b['status'] not in binary:
        result.append('unresolved_status')
    elif a['status'] != b['status']:
        result.append('label_disagreement')
    for name, annotation in [('A', a), ('B', b)]:
        if annotation['confidence'] == 'low':
            result.append('low_confidence_' + name)
        if (annotation['confidence'] == 'low' or annotation['status'] not in binary) and not annotation['comment'].strip():
            result.append('missing_comment_' + name)
        if kind == 'images' and annotation['status'] == 'animal' and not annotation['boxes']:
            result.append('animal_without_bbox_' + name)
        if kind == 'images' and annotation['status'] == 'empty' and annotation['boxes']:
            result.append('empty_with_bbox_' + name)
    if comparison:
        if comparison['unmatched_A'] or comparison['unmatched_B']:
            result.append('unmatched_boxes')
        if any(pair['iou'] < 0.75 for pair in comparison['matched']):
            result.append('matched_iou_below_075')
        if any(pair['flags_A'] != pair['flags_B'] for pair in comparison['matched']):
            result.append('different_box_flags')
    return result


def build(inputs, output):
    data = {name: load(path) for name, path in inputs.items()}
    require(data['text_A'].keys() == data['text_B'].keys(), 'Text A/B sample_id sets differ; align the input sets before comparison.')
    require(data['images_A_original'].keys() == data['images_A_current'].keys() == data['images_B'].keys(),
            'Image original A/current A/B sample_id sets differ; align the input sets before comparison.')
    text_rows, image_rows, original_image_rows = [], [], []
    for sid in sorted(data['text_A']):
        ta, tb = data['text_A'][sid], data['text_B'][sid]
        require(ta['data'] == tb['data'], ('different_text_input', sid))
        a, b = unpack(ta, 'text'), unpack(tb, 'text')
        text_rows.append({'sample_id': sid, 'dataset': 'text', 'text': ta['data']['text'],
                          'A': a, 'B': b, 'review_reasons': reasons(a, b, 'text')})
    for sid in sorted(data['images_A_current']):
        ta, tb, original = data['images_A_current'][sid], data['images_B'][sid], data['images_A_original'][sid]
        require(ta['data'] == tb['data'] == original['data'], ('different_image_input', sid))
        a, b, original_a = unpack(ta, 'images'), unpack(tb, 'images'), unpack(original, 'images')
        dimensions = {(box['original_width'], box['original_height']) for box in a['boxes'] + b['boxes'] + original_a['boxes']}
        require(len(dimensions) <= 1, ('different_image_dimensions_in_exports', sid))
        width, height = next(iter(dimensions)) if dimensions else (None, None)
        comparison = compare_boxes(a['boxes'], b['boxes'])
        row = {'sample_id': sid, 'dataset': 'images', 'image_url': ta['data']['image'],
               'width': width, 'height': height,
               'A': a, 'B': b, 'original_A_status': original_a['status'],
               'box_comparison': comparison, 'review_reasons': reasons(a, b, 'images', comparison)}
        image_rows.append(row)
        original_image_rows.append({**row, 'A': original_a})
    changed_ids = [row['sample_id'] for row in image_rows if row['A']['status'] != row['original_A_status']]
    stats = {
        'text': binary_metrics([(r['A']['status'], r['B']['status']) for r in text_rows], ['not_spam', 'spam']),
        'image_presence_original_A': binary_metrics([(r['A']['status'], r['B']['status']) for r in original_image_rows], ['empty', 'animal']),
        'image_presence_after_user_correction': binary_metrics([(r['A']['status'], r['B']['status']) for r in image_rows], ['empty', 'animal']),
        'bounding_boxes': box_summary(image_rows),
    }
    stats['bounding_boxes_original_A'] = box_summary(original_image_rows)
    review = [row for row in text_rows + image_rows if row['review_reasons']]
    stats['review'] = {'text': sum(row['dataset'] == 'text' for row in review),
                       'images': sum(row['dataset'] == 'images' for row in review),
                       'total': len(review),
                       'reason_counts': dict(Counter(reason for row in review for reason in row['review_reasons']))}
    first_reasons = {'label_disagreement', 'unresolved_status', 'unmatched_boxes', 'animal_without_bbox_A',
                     'animal_without_bbox_B', 'empty_with_bbox_A', 'empty_with_bbox_B'}
    for row in text_rows + image_rows:
        row['review_priority'] = 'first' if first_reasons.intersection(row['review_reasons']) else ('followup' if row['review_reasons'] else None)
    stats['review']['first_priority_text'] = sum(r['dataset'] == 'text' and r['review_priority'] == 'first' for r in review)
    stats['review']['first_priority_images'] = sum(r['dataset'] == 'images' and r['review_priority'] == 'first' for r in review)
    stats['time'] = {}
    for key, value in data.items():
        if key == 'images_A_original':
            continue
        durations = [t['annotations'][0].get('lead_time') for t in value.values()]
        numeric = [v for v in durations if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0]
        stats['time'][key] = {'tasks': len(value), 'valid_durations': len(numeric),
                              'missing_or_invalid': len(durations) - len(numeric), 'zero_durations': numeric.count(0),
                              'seconds': round(sum(numeric), 3)}
    report = {
        'generated_at': datetime.now(timezone(timedelta(hours=3), 'Europe/Moscow')).isoformat(timespec='seconds'),
        'source_hashes': {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in inputs.items()},
        'method': {
            'join_key': 'data.sample_id', 'verified_identical_task_data': True,
            'codebook': 'Working annotation codebook v0.1, sections 9–10, distributed in the annotation kit.',
            'baseline': 'First preserved A/B exports; corrected IMG021 shown as a separate secondary result.',
            'bbox_coordinates': 'Percent coordinates on the same image; equivalent IoU to original pixel coordinates. Dimensions are checked between exports; image files are not downloaded or decoded.',
            'bbox_matching': 'Exact maximum cardinality matching at IoU >= 0.50; ties maximize summed IoU.',
            'review_iou': 'IoU < 0.75 is an additional inspection heuristic, not a requirement that every pair meet 0.75. The working codebook orientation concerns aggregate median IoU. Match eligibility remains IoU >= 0.50.',
        },
        'limitations': [
            'Some A image judgments were discussed with an AI assistant. Results measure agreement of the received annotations; they are not claimed to be entirely unaided independent annotation.',
            'No separate completed pilot or frozen version 1.0 was verified. Do not describe this retrospective comparison as a completed prospective pilot.',
            'Agreement is not model accuracy and does not establish correctness of matching answers.',
            'No final labels, boxes, exclusions, or joint decisions have been selected automatically.',
            'lead_time is the duration recorded by Label Studio and may include pauses; manual active time was not reconstructed.',
        ],
        'statistics': stats,
        'changed_image_status_ids_since_original': changed_ids,
        'text_rows': text_rows,
        'image_rows': image_rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    template = [{'sample_id': row['sample_id'], 'dataset': row['dataset'],
                 'A_status': row['A']['status'], 'B_status': row['B']['status'],
                 'review_reasons': row['review_reasons'], 'review_priority': row['review_priority'], 'final_status': None,
                 'final_label': None, 'final_boxes': None, 'resolution_reason': '',
                 'resolved_by': [], 'resolved_at': None, 'codebook_version': None} for row in review]
    template_path = output / 'adjudication_template.json'
    if not template_path.exists():
        template_path.write_text(json.dumps({
        'instructions': 'After joint review, record actual decisions and reasons. final_status: accepted/excluded/unresolved. For accepted text final_label: not_spam/spam; for images: animal/empty. Image final_boxes use percent coordinates as in the export, with original_width and original_height. null means not decided. Do not overwrite original A/B exports.',
        'items': template,
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_readable_report(report, review, output)
    return report


REASONS_RU = {
    'label_disagreement': 'разные метки A/B', 'unresolved_status': 'needs_review или excluded',
    'unmatched_boxes': 'есть рамки без пары при IoU >= 0.50',
    'matched_iou_below_075': 'проверить границы: IoU сопоставленной пары < 0.75',
    'different_box_flags': 'разные признаки occluded/truncated',
}
for _who in ('A', 'B'):
    REASONS_RU.update({'low_confidence_' + _who: 'низкая уверенность ' + _who,
                      'missing_comment_' + _who: 'нет поясняющего комментария ' + _who,
                      'animal_without_bbox_' + _who: 'animal без рамки у ' + _who,
                      'empty_with_bbox_' + _who: 'empty с рамкой у ' + _who})


def write_readable_report(report, review, output):
    s = report['statistics']
    lines = ['СРАВНЕНИЕ РАЗМЕТКИ A/B', 'Дата: ' + report['generated_at'], '',
             'Исходные JSON не изменяются. Итоговые решения принимает человек после разбора.', '']
    for title, key in [('Тексты', 'text'), ('Наличие животных — исходный A', 'image_presence_original_A'),
                       ('Наличие животных — текущий A', 'image_presence_after_user_correction')]:
        m = s[key]
        percent = f"{100*m['agreement']:.2f}%" if m['agreement'] is not None else 'не определено'
        kappa = f"{m['cohen_kappa']:.4f}" if m['cohen_kappa'] is not None else 'не определена'
        lines += [f"{title}: {m['matching_pairs']}/{m['comparable_pairs']} совпадений ({percent}); Cohen kappa = {kappa}.",
                  f"Покрытие: {m['comparable_pairs']}/{m['total_pairs']}; небинарных пар: {m['nonbinary_pairs']}."]
    b = s['bounding_boxes']
    lines += ['', f"Рамки: A={b['boxes_A']}, B={b['boxes_B']}, сопоставлено={b['matched_pairs']}; IoU >= 0.50.",
              f"F1 сопоставления: {b['matching_f1']}; медианная IoU: {b['median_matched_iou']}.",
              f"Без пары: A={b['unmatched_A']}, B={b['unmatched_B']}. Кадры с небинарным статусом исключены из этой метрики.",
              '', f"Разбор: {s['review']['text']} сообщений и {s['review']['images']} фото.",
              f"В первую очередь: {s['review']['first_priority_text']} сообщений и {s['review']['first_priority_images']} фото.",
              '', 'Как работать: откройте ID в своих проектах, сравните ответы/рамки, сверяйтесь с кодбуком.',
              'Запишите совместный итог и причину в отдельный файл. Неясные случаи оставьте unresolved.',
              'IoU < 0.75 — подсказка для просмотра, а не автоматический брак. Пустые кадры не повышают IoU.',
              'Исходный A и текущий A различаются по статусу: ' + (', '.join(report['changed_image_status_ids_since_original']) or 'нет различий'),
              'Часть решений A по фото обсуждалась с ИИ; это нужно учитывать при описании независимости.',
              'Отдельный завершённый пилот не подтверждён. Не выдавайте этот расчёт за проведённый пилот.',
              'lead_time — время счётчика Label Studio, оно может включать паузы.',
              '', 'Файлы: comparison.json — все пары и метрики; review_list.txt — список для разбора;',
              'adjudication_template.json — пустые поля совместного решения. Скрипт не перезаписывает существующий шаблон.',
              'При смене входных экспортов используйте новую папку --output-dir, чтобы не смешивать версии.']
    (output / 'REPORT.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    rows = ['СПИСОК ДЛЯ СОВМЕСТНОГО РАЗБОРА', 'Исходные ответы не изменены; итоговые решения пока не приняты.', '']
    for row in sorted(review, key=lambda r: (r['review_priority'] != 'first', r['dataset'], r['sample_id'])):
        rows += [f"{row['sample_id']} | {row['dataset']} | очередь: {row['review_priority']}",
                 f"A: {row['A']['status']}; B: {row['B']['status']}",
                 'Проверить: ' + '; '.join(REASONS_RU.get(code, code) for code in row['review_reasons'])]
        if row['dataset'] == 'text':
            rows.append(row['text'])
        else:
            c = row['box_comparison']
            rows.append(f"Рамки A/B: {c['boxes_A']}/{c['boxes_B']}; сопоставлено: {c['matched_count']}")
        rows.append('')
    (output / 'review_list.txt').write_text('\n'.join(rows), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, key in [('text-a', 'text_A'), ('text-b', 'text_B'), ('images-a', 'images_A_current'),
                      ('images-b', 'images_B'), ('images-a-original', 'images_A_original')]:
        parser.add_argument('--' + flag, dest=key, type=Path, default=DEFAULT_INPUTS[key])
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/annotation_comparison')
    args = parser.parse_args()
    inputs = {key: getattr(args, key) for key in DEFAULT_INPUTS}
    try:
        report = build(inputs, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'Cannot compare annotations: {exc}', file=sys.stderr)
        return 1
    s = report['statistics']
    print(f"Compared {s['text']['total_pairs']} texts and {s['image_presence_original_A']['total_pairs']} images.")
    print(f"Review: {s['review']['text']} texts and {s['review']['images']} images.")
    print(f'Reports: {args.output_dir.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
