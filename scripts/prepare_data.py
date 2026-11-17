"""Download public datasets and prepare auditable, bounded research subsets.

Run from the repository root: python scripts/prepare_data.py --download-images
No models, augmentation or manual annotations are produced by this script.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import unicodedata
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
PROCESSED = ROOT / 'data/processed'
ARTIFACTS = ROOT / 'artifacts'
SEED = 42
GCP = 'https://storage.googleapis.com/public-datasets-lila/'
IMAGE_BASE = GCP + 'caltech-unzipped/cct_images/'
IMAGE_MIRROR = 'https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/'
METADATA = {
    'caltech_camera_traps.json.zip': GCP + 'caltechcameratraps/labels/caltech_camera_traps.json.zip',
    'caltech_bboxes_20200316.json': GCP + 'caltechcameratraps/labels/caltech_bboxes_20200316.json',
    'CaltechCameraTrapsSplits_v0.json': GCP + 'caltechcameratraps/CaltechCameraTrapsSplits_v0.json',
    'eccv_18_annotations.tar.gz': GCP + 'caltechcameratraps/eccv_18_annotations.tar.gz',
}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path, retries: int = 4) -> Path:
    if path.is_file() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'data-enrichment-study/0.1'}), timeout=60) as r:
                expected = r.headers.get('Content-Length')
                with partial.open('wb') as out:
                    shutil.copyfileobj(r, out)
                if expected is not None and partial.stat().st_size != int(expected):
                    raise IOError('Incomplete download')
            partial.replace(path)
            return path
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError('Unreachable')


def normalise_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text).casefold()
    text = re.sub('[\u200b-\u200d\ufeff]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def group_text_templates(clean: pd.DataFrame, threshold: float = .90) -> tuple[np.ndarray, dict]:
    """Connect near duplicates before splitting, in memory-bounded batches.

    Global text features here are ONLY used for duplicate grouping; a future model
    must fit its own vectorizer on train. Components may contain indirect matches.
    """
    features = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2,
                               max_features=30000, dtype=np.float32)
    matrix = features.fit_transform(clean['normalised_text'])
    parent = np.arange(len(clean))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = 0
    for start in range(0, len(clean), 256):
        stop = min(start + 256, len(clean))
        similarities = linear_kernel(matrix[start:stop], matrix)
        left, right = np.where(similarities >= threshold)
        left = left + start
        for a, b in zip(left, right):
            if a >= b:
                continue
            edges += 1
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        if start % 2560 == 0:
            print(f'Text similarity grouping: {stop}/{len(clean)}', flush=True)
    groups = np.array([find(i) for i in range(len(clean))])
    sizes = pd.Series(groups).value_counts()
    return groups, {'threshold': threshold, 'similar_pairs': edges,
                    'groups': len(sizes), 'nontrivial_groups': int((sizes > 1).sum()),
                    'largest_group': int(sizes.max()),
                    'method': 'Connected components of global char_wb TF-IDF cosine >= threshold; 3–5 grams, min_df=2, max_features=30000. Used only for pre-split grouping, not model features.'}


def prepare_telegram() -> dict:
    # Pin the original dataset revision on first run; keep it for later reruns.
    info_path = RAW / 'hf_dataset_info.json'
    download('https://huggingface.co/api/datasets/alt-gnome/telegram-spam', info_path)
    info = json.loads(info_path.read_text())
    revision = info['sha']
    parquet_path = RAW / 'telegram-spam.parquet'
    url = f'https://huggingface.co/datasets/alt-gnome/telegram-spam/resolve/{revision}/data/train-00000-of-00001.parquet'
    download(url, parquet_path)
    raw = pd.read_parquet(parquet_path)
    if not {'text', 'label'} <= set(raw):
        raise ValueError('Unexpected Telegram schema')
    raw.insert(0, 'source_row', np.arange(len(raw)))
    raw['sample_id'] = 'telegram_' + raw['source_row'].astype(str)
    invalid = raw['text'].isna() | ~raw['label'].isin([0, 1])
    raw['normalised_text'] = raw['text'].fillna('').map(normalise_text)
    invalid |= raw['normalised_text'].eq('')
    candidates = raw.loc[~invalid].copy()
    label_counts = candidates.groupby('normalised_text')['label'].nunique()
    conflict_keys = set(label_counts[label_counts > 1].index)
    conflict = candidates['normalised_text'].isin(conflict_keys)
    candidates.loc[conflict].to_parquet(PROCESSED / 'telegram_conflicts.parquet', index=False)
    valid = candidates.loc[~conflict].copy()
    duplicates = valid.duplicated('normalised_text', keep='first')
    valid.loc[duplicates].to_parquet(PROCESSED / 'telegram_duplicates.parquet', index=False)
    clean = valid.loc[~duplicates].copy().reset_index(drop=True)
    clean['label'] = clean['label'].astype(int)
    clean['label_name'] = clean['label'].map({0: 'not_spam', 1: 'spam'})
    clean['text_hash'] = clean['normalised_text'].map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    groups, grouping = group_text_templates(clean)
    clean['template_group'] = ['template_' + str(g) for g in groups]
    grouping['mixed_label_groups'] = int((clean.groupby('template_group')['label'].nunique() > 1).sum())
    clean['split'] = 'train'
    # 14/3/3 folds target ~70/15/15 while keeping all connected templates together.
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    for fold, (_, ids) in enumerate(splitter.split(clean, clean['label'], groups)):
        if fold < 3:
            clean.loc[ids, 'split'] = 'val'
        elif fold < 6:
            clean.loc[ids, 'split'] = 'test'
    clean['n_chars'] = clean['text'].str.len()
    clean['n_words'] = clean['text'].str.split().str.len()
    clean['has_url'] = clean['text'].str.contains(r'https?://|www\.|t\.me/', case=False, regex=True)
    clean['has_mention'] = clean['text'].str.contains(r'@[\w]+', regex=True)
    clean['digit_fraction'] = clean['text'].map(lambda s: sum(c.isdigit() for c in s) / max(len(s), 1))
    clean['source_url'] = 'https://huggingface.co/datasets/alt-gnome/telegram-spam'
    clean['source_revision'] = revision
    clean.to_parquet(PROCESSED / 'telegram.parquet', index=False)
    assert len(clean) >= 1000
    assert clean.groupby('text_hash')['split'].nunique().max() == 1
    assert clean.groupby('template_group')['split'].nunique().max() == 1
    result = {
        'raw_rows': len(raw), 'invalid_rows': int(invalid.sum()),
        'exact_duplicate_rows': int(raw.duplicated(['text', 'label']).sum()),
        'normalised_duplicate_rows_removed': int(duplicates.sum()),
        'conflicting_text_groups': len(conflict_keys), 'conflicting_rows_excluded': int(conflict.sum()),
        'prepared_rows': len(clean), 'class_counts': clean['label_name'].value_counts().to_dict(),
        'split_counts': clean['split'].value_counts().to_dict(),
        'template_grouping': grouping,
        'revision': revision, 'raw_url': url,
        'note': 'Approximately 70/15/15 StratifiedGroupKFold split after deduplication and template grouping. Near duplicates below the chosen similarity threshold may remain; no chat/user/time metadata is available.',
    }
    write_json(ARTIFACTS / 'telegram_preparation_summary.json', result)
    return result


def balanced_camera_quota(capacities: dict[str, int], total: int) -> dict[str, int]:
    """Distribute pairs fairly across cameras, within available negative sequences."""
    if total < len(capacities) or sum(capacities.values()) < total:
        raise ValueError(f'Not enough independent positive/negative pairs: need {total}, have {sum(capacities.values())}')
    quota = {c: 0 for c in capacities}
    for _ in range(total):
        available = [c for c in quota if quota[c] < capacities[c]]
        camera = min(available, key=lambda c: (quota[c], int(c)))
        quota[camera] += 1
    return quota


def prepare_caltech(total_images: int = 4000) -> dict:
    from itertools import combinations
    if total_images < 3000 or total_images % 8:
        raise ValueError('--images must be >=3000 and divisible by 8')
    for name, url in METADATA.items():
        download(url, RAW / name)
    with zipfile.ZipFile(RAW / 'caltech_camera_traps.json.zip') as z:
        labels = json.loads(z.read('caltech_images_20210113.json'))
    boxes = json.loads((RAW / 'caltech_bboxes_20200316.json').read_text())
    splits = json.loads((RAW / 'CaltechCameraTrapsSplits_v0.json').read_text())['splits']
    train_cameras = sorted(set(map(str, splits['eccv_train'])), key=int)
    held_cameras = sorted(set(map(str, splits['eccv_val'])), key=int)
    cameras = train_cameras + held_cameras
    assert len(set(cameras)) == 20
    animal_categories = {int(c['id']) for c in labels['categories']} - {30, 33}
    target_categories = [{'id': 1, 'name': 'animal'}]
    write_json(PROCESSED / 'caltech_categories.json', target_categories)
    by_image = defaultdict(set)
    for a in labels['annotations']:
        by_image[str(a['image_id'])].add(int(a['category_id']))
    by_box = defaultdict(list)
    for a in boxes['annotations']:
        if a.get('bbox') and len(a['bbox']) == 4:
            by_box[str(a['image_id'])].append(a)
    all_rows, excluded = [], defaultdict(int)
    for image in labels['images']:
        location = str(image['location'])
        if location not in cameras:
            continue
        image_id = str(image['id'])
        labs = by_image[image_id]
        if not labs:
            excluded['missing_image_label'] += 1
            continue
        animal_labels = labs & animal_categories
        if animal_labels and 30 in labs:
            excluded['empty_and_animal_label_conflict'] += 1
            continue
        source_animal_boxes = [a for a in by_box[image_id] if int(a['category_id']) in animal_categories]
        valid_boxes, covered = [], set()
        for box in source_animal_boxes:
            x, y, w, h = map(float, box['bbox'])
            if w > 0 and h > 0 and x >= -1 and y >= -1 and x + w <= image['width'] + 1 and y + h <= image['height'] + 1:
                x1, y1 = max(0., x), max(0., y)
                x2, y2 = min(float(image['width']), x+w), min(float(image['height']), y+h)
                if x2 > x1 and y2 > y1:
                    valid_boxes.append({'category_id': 1, 'bbox': [x1, y1, x2-x1, y2-y1]})
                    covered.add(int(box['category_id']))
        if animal_labels and (not animal_labels <= covered or not valid_boxes):
            excluded['animal_without_complete_valid_boxes'] += 1
            continue
        if not animal_labels and source_animal_boxes:
            excluded['negative_with_animal_bbox_conflict'] += 1
            continue
        has_animal = bool(animal_labels)
        all_rows.append({
            'image_id': image_id, 'file_name': image['file_name'], 'location': location,
            'seq_id': str(image.get('seq_id') or image_id),
            'date_captured': image.get('date_captured'), 'frame_num': image.get('frame_num'),
            'width': int(image['width']), 'height': int(image['height']),
            'has_animal': has_animal, 'n_boxes': len(valid_boxes),
            'bbox_json': json.dumps(valid_boxes),
            'source_category_ids_json': json.dumps(sorted(labs)),
            'negative_source': ('car' if 33 in labs else 'empty') if not has_animal else None,
            'rights_holder': image.get('rights_holder'),
            'official_camera_partition': 'eccv_train' if location in train_cameras else 'eccv_val',
        })
    eligible = pd.DataFrame(all_rows).sort_values('image_id').reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    eligible['random_order'] = rng.random(len(eligible))
    # One frame per sequence. Prefer a negative representative of a mixed sequence
    # because independently labeled negative triggers are the limiting resource.
    independent = eligible.sort_values(['has_animal', 'random_order']).drop_duplicates(['location', 'seq_id']).copy()
    counts = pd.crosstab(independent.location, independent.has_animal).reindex(columns=[False, True], fill_value=0)
    capacities = counts.min(axis=1).astype(int).to_dict()
    pair_targets = {'train': total_images//4, 'val': total_images//8, 'test': total_images//8}
    initial_val = set(np.random.default_rng(SEED).permutation(held_cameras).tolist()[:5])
    feasible = []
    for combo in combinations(held_cameras, 5):
        other = set(held_cameras) - set(combo)
        if sum(capacities[c] for c in combo) >= pair_targets['val'] and sum(capacities[c] for c in other) >= pair_targets['test']:
            feasible.append(combo)
    if not feasible:
        raise ValueError('Cannot form 50/50 val and test using independent held-out cameras; reduce --images')
    val_cameras = list(min(feasible, key=lambda cs: (len(set(cs) ^ initial_val), tuple(map(int, cs)))))
    test_cameras = [c for c in held_cameras if c not in val_cameras]
    camera_parts = {'train': train_cameras, 'val': val_cameras, 'test': test_cameras}
    selected_parts = []
    quota_record = {}
    for split, split_cameras in camera_parts.items():
        quota = balanced_camera_quota({c: capacities[c] for c in split_cameras}, pair_targets[split])
        quota_record[split] = quota
        # Equal positive/negative counts also hold within every camera.
        for camera, n in quota.items():
            for positive in [False, True]:
                pool = independent[(independent.location == camera) & (independent.has_animal == positive)].sort_values('image_id')
                sample = pool.iloc[rng.choice(len(pool), size=n, replace=False)].copy()
                sample['split'] = split
                selected_parts.append(sample)
    selected = pd.concat(selected_parts).drop(columns='random_order').sort_values(['location', 'image_id']).reset_index(drop=True)
    selected['image_url'] = IMAGE_BASE + selected['file_name']
    selected['local_path'] = 'data/images/caltech/' + selected['file_name']
    assert len(selected) == total_images and selected.location.nunique() == 20
    assert selected.groupby('location')['split'].nunique().max() == 1
    assert selected.seq_id.is_unique
    assert selected.groupby('split').has_animal.mean().eq(.5).all()
    assert selected.groupby('location').has_animal.mean().eq(.5).all()
    selected.to_parquet(PROCESSED / 'caltech_manifest.parquet', index=False)
    eligible.drop(columns='random_order').to_parquet(PROCESSED / 'caltech_eligible.parquet', index=False)
    selected_ids = set(selected.image_id)
    selected_coco = {
        'info': {**labels.get('info', {}), 'description': 'Modified custom 20-camera subset for single-class animal detection; 50% positive and 50% negative images. Species labels merged, non-animal boxes removed, coordinates clipped within 1px tolerance. Not the official CCT20 benchmark.', 'seed': SEED},
        'licenses': boxes.get('licenses', []), 'categories': target_categories,
        'images': [i for i in labels['images'] if str(i['id']) in selected_ids],
        'annotations': [],
    }
    label_dir = ROOT / 'data/labels/caltech'
    label_dir.mkdir(parents=True, exist_ok=True)
    # Only the selected subset has YOLO labels; remove stale generated label files.
    for old in label_dir.glob('*.txt'):
        if old.stem not in selected_ids:
            old.unlink()
    for row in selected.itertuples():
        yolo_lines = []
        for box in json.loads(row.bbox_json):
            x, y, w, h = box['bbox']
            selected_coco['annotations'].append({'id': len(selected_coco['annotations'])+1, 'image_id': row.image_id, **box, 'area': w*h, 'iscrowd': 0})
            yolo_lines.append(f'0 {(x+w/2)/row.width:.9f} {(y+h/2)/row.height:.9f} {w/row.width:.9f} {h/row.height:.9f}')
        # A negative is an empty annotation file, never a fake background box.
        (label_dir / f'{row.image_id}.txt').write_text('\n'.join(yolo_lines) + ('\n' if yolo_lines else ''))
    write_json(PROCESSED / 'caltech_annotations.json', selected_coco)
    yolo_dir = PROCESSED / 'caltech_yolo'
    yolo_dir.mkdir(exist_ok=True)
    for split in ['train', 'val', 'test']:
        ids = set(selected.loc[selected.split.eq(split), 'image_id'])
        split_coco = {**selected_coco, 'images': [i for i in selected_coco['images'] if str(i['id']) in ids],
                      'annotations': [a for a in selected_coco['annotations'] if a['image_id'] in ids]}
        write_json(PROCESSED / f'caltech_{split}.json', split_coco)
        (yolo_dir / f'{split}.txt').write_text('\n'.join(str(ROOT/p) for p in selected.loc[selected.split.eq(split), 'local_path'])+'\n')
    config = f'path: {json.dumps(str(ROOT), ensure_ascii=False)}\n'
    config += '\n'.join(f'{part}: {json.dumps(str(yolo_dir / (part+".txt")), ensure_ascii=False)}' for part in ['train', 'val', 'test'])
    config += '\nnames:\n  0: animal\n'
    (yolo_dir / 'dataset.yaml').write_text(config)
    write_json(PROCESSED / 'caltech_camera_splits.json', camera_parts)
    return {
        'task': 'single-class animal object detection',
        'full_metadata_images': len(labels['images']), 'camera_ids': sorted(cameras, key=int),
        'eligible_frames_in_20_cameras': len(eligible), 'eligible_unique_sequences': len(independent),
        'filter_exclusions': dict(excluded), 'selected_images': len(selected),
        'positive_images': int(selected.has_animal.sum()), 'negative_images': int((~selected.has_animal).sum()),
        'animal_boxes': int(selected.n_boxes.sum()), 'detection_categories': target_categories,
        'camera_counts': selected.location.value_counts().to_dict(), 'paired_quota_by_camera': quota_record,
        'split_counts': selected.split.value_counts().to_dict(), 'camera_split': camera_parts,
        'selection': 'Official 20 ECCV camera locations; one frame per sequence, preferring negative representatives of mixed sequences; exactly 50/50 animal/no-animal per camera and per split. Fair capped quotas vary between cameras because negatives are scarce. Held-out cameras chosen deterministically to make both val and test balance feasible.',
        'limitations': 'Custom balanced subset, not the official CCT20 benchmark or the natural class distribution. No-animal may contain people or cars. Original species are provenance only, never training targets. Source labels/boxes may still contain semantic errors.',
    }


def inspect_image(row: dict) -> dict:
    path = ROOT / row['local_path']
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        rgb = image.convert('RGB')
        small = rgb.resize((64, 64))
        gray = small.convert('L')
        stat = ImageStat.Stat(gray)
        a = np.asarray(small, dtype=np.int16)
        grayscale_fraction = float(((np.max(a, axis=2) - np.min(a, axis=2)) <= 5).mean())
        thumb = np.asarray(rgb.resize((9, 8)).convert('L'))
        bits = (thumb[:, 1:] > thumb[:, :-1]).ravel()
        dhash = f'{int.from_bytes(np.packbits(bits).tobytes()):016x}'
    return {
        'image_id': row['image_id'], 'status': 'ok', 'actual_width': width,
        'actual_height': height, 'size_bytes': path.stat().st_size,
        'sha256': sha256(path), 'dhash': dhash, 'brightness': stat.mean[0],
        'contrast': stat.stddev[0], 'grayscale_fraction': grayscale_fraction,
        'dimensions_match': width == row['width'] and height == row['height'],
    }


def fetch_image(row: dict) -> dict:
    path = ROOT / row['local_path']
    try:
        try:
            download(row['image_url'], path, retries=3)
        except Exception:
            download(IMAGE_MIRROR + row['file_name'], path, retries=2)
        try:
            return inspect_image(row)
        except Exception:
            # Repair a corrupt cached file once.
            path.unlink(missing_ok=True)
            download(row['image_url'], path, retries=2)
            return inspect_image(row)
    except Exception as error:
        return {'image_id': row['image_id'], 'status': 'error', 'error': str(error)}


def fetch_images(workers: int) -> dict:
    manifest = pd.read_parquet(PROCESSED / 'caltech_manifest.parquet')
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_image, row) for row in manifest.to_dict('records')]
        for f in as_completed(futures):
            results.append(f.result())
            if len(results) % 100 == 0 or len(results) == len(futures):
                print(f'Images checked: {len(results)}/{len(futures)}; errors: {sum(r["status"] != "ok" for r in results)}', flush=True)
    checks = pd.DataFrame(results).sort_values('image_id')
    checks.to_parquet(PROCESSED / 'caltech_image_checks.parquet', index=False)
    good = checks[checks['status'].eq('ok')].merge(manifest[['image_id', 'split']], on='image_id', validate='one_to_one')
    leaking_hashes = good.groupby('sha256')['split'].nunique()
    summary = {
        'valid_images': len(good), 'failed_images': int((checks['status'] != 'ok').sum()),
        'total_bytes': int(good['size_bytes'].sum()),
        'dimension_mismatches': int((~good['dimensions_match']).sum()),
        'byte_duplicate_images': int(good.duplicated('sha256').sum()),
        'byte_duplicate_groups_across_splits': int((leaking_hashes > 1).sum()),
        'dhash_duplicate_images': int(good.duplicated('dhash').sum()),
        'note': 'dHash equality is only a candidate similarity signal; static empty-camera backgrounds can collide. Raw images have not been resized or augmented.',
    }
    write_json(ARTIFACTS / 'image_download_summary.json', summary)
    if summary['failed_images'] or summary['dimension_mismatches'] or summary['byte_duplicate_groups_across_splits']:
        raise RuntimeError('Image audit failed; inspect caltech_image_checks.parquet and rerun or resolve duplicate groups')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=int, default=4000, help='Total Caltech images; >=3000, divisible by 8')
    parser.add_argument('--download-images', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--images-only', action='store_true')
    mode.add_argument('--caltech-only', action='store_true', help='Rebuild only Caltech and optionally download images')
    mode.add_argument('--text-only', action='store_true', help='Rebuild only Telegram without changing the Caltech selection')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    for p in [RAW, PROCESSED, ARTIFACTS]:
        p.mkdir(parents=True, exist_ok=True)
    if not 1 <= args.workers <= 32:
        parser.error('--workers must be between 1 and 32')
    if args.images_only:
        print(json.dumps(fetch_images(args.workers), ensure_ascii=False, indent=2))
        return
    if args.caltech_only:
        summary_path = ARTIFACTS / 'preparation_summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {'seed': SEED}
        summary['caltech'] = prepare_caltech(args.images)
        summary.pop('image_download', None)
        write_json(summary_path, summary)
        print('Caltech:', summary['caltech'], flush=True)
        if args.download_images:
            summary['image_download'] = fetch_images(args.workers)
            write_json(summary_path, summary)
        return
    print('Preparing Telegram…', flush=True)
    telegram = prepare_telegram()
    print('Telegram:', telegram, flush=True)
    if args.text_only:
        summary_path = ARTIFACTS / 'preparation_summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {'seed': SEED}
        summary['telegram'] = telegram
        summary['text_updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(summary_path, summary)
        return
    print('Preparing Caltech…', flush=True)
    caltech = prepare_caltech(args.images)
    print('Caltech:', caltech, flush=True)
    summary = {'seed': SEED, 'prepared_at_utc': datetime.now(timezone.utc).isoformat(), 'telegram': telegram, 'caltech': caltech}
    write_json(ARTIFACTS / 'preparation_summary.json', summary)
    sources = {
        'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
        'telegram': {'page': 'https://huggingface.co/datasets/alt-gnome/telegram-spam', 'license': 'CC0-1.0', 'revision': telegram['revision'], 'raw_url': telegram['raw_url']},
        'caltech': {'page': 'https://lila.science/datasets/caltech-camera-traps', 'license': 'CDLA-Permissive-1.0', 'license_url': 'https://cdla.io/permissive-1-0/', 'citation': 'Sara Beery, Grant Van Horn, Pietro Perona. Recognition in Terra Incognita. ECCV 2018.', 'metadata_urls': METADATA, 'image_base_url': IMAGE_BASE},
        'files': [{'path': str(p.relative_to(ROOT)), 'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sorted(RAW.iterdir()) if p.is_file() and not p.name.endswith('.part')],
    }
    write_json(PROCESSED / 'sources.json', sources)
    if args.download_images:
        summary['image_download'] = fetch_images(args.workers)
        write_json(ARTIFACTS / 'preparation_summary.json', summary)


if __name__ == '__main__':
    main()
