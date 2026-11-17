"""Local exploratory features; these are descriptive proxies, not model outputs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

RUSSIAN_STOP_WORDS = set('''а без более бы был была были было быть в вам вас ваш
ваша ваши ведь весь во вот все всего всех всё вы где да даже для до его ее её если
есть еще ещё же за зачем здесь и из или им их к как какая какие какой ко когда
кто ли либо мне может можно мой мы на надо нам нас наш не него нее неё нет ни
них но ну о об однако он она они оно от очень по под после при про раз с сам
себе себя сейчас со так также такой там те тем то того тоже только том тот тут
ты у уже чего чем через что чтобы эта эти это этот я меня тебе тебя ему ей
будет будут давайте потому поэтому чего которых который которую свои свой
свою если вам вас всем всего просто'''.split())
STOP_WORDS = RUSSIAN_STOP_WORDS | set(ENGLISH_STOP_WORDS)
URL_PATTERN = r'(?:https?://|www\.|t\.me/)\S+'
PROMO_WORDS = ['free', 'win', 'заработок', 'доход', 'деньги', 'бонус', 'подарок', 'удалённо']


def normalise_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text).casefold()
    return re.sub(r'\s+', ' ', re.sub('[\u200b-\u200d\ufeff]', '', text)).strip()


def tokens(text: str) -> list[str]:
    text = re.sub(URL_PATTERN, ' ', text, flags=re.I)
    text = re.sub(r'@\w+', ' ', text)
    return [w for w in re.findall(r'[а-яёa-z]{3,}', text.casefold()) if w not in STOP_WORDS]


def text_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result['n_chars'] = result.text.str.len()
    result['n_words'] = result.text.str.split().str.len()
    result['has_digit'] = result.text.map(lambda s: any(c.isdigit() for c in s))
    result['digit_fraction'] = result.text.map(lambda s: sum(c.isdigit() for c in s)/max(len(s), 1))
    result['has_url'] = result.text.str.contains(URL_PATTERN, case=False, regex=True)
    result['has_mention'] = result.text.str.contains(r'@\w+', regex=True)
    result['uppercase_fraction'] = result.text.map(lambda s: sum(c.isalpha() and c.isupper() for c in s)/max(sum(c.isalpha() for c in s), 1))
    result['n_exclamations'] = result.text.str.count(r'[!！]')
    result['has_exclamation'] = result.n_exclamations.gt(0)
    for word in PROMO_WORDS:
        result[f'word_{word}'] = result.text.str.contains(r'\b' + re.escape(word) + r'\b', case=False, regex=True)
    return result


def load_caltech_source(root: Path) -> pd.DataFrame:
    """All 140 source cameras before the annotated-subset filters."""
    with zipfile.ZipFile(root/'data/raw/caltech_camera_traps.json.zip') as z:
        source = json.loads(z.read('caltech_images_20210113.json'))
    images = pd.DataFrame(source['images']).rename(columns={'id': 'image_id'})
    images['location'] = images.location.astype(str)
    annotations = pd.DataFrame(source['annotations'])
    source_labels = annotations.groupby('image_id').category_id.agg(set)
    labels = images.image_id.map(source_labels)
    images['has_animal'] = pd.array([
        bool(labs - {30, 33}) if isinstance(labs, set) and labs else pd.NA for labs in labels
    ], dtype='boolean')
    boxes = json.loads((root/'data/raw/caltech_bboxes_20200316.json').read_text())
    bbox_counts = pd.Series([a['image_id'] for a in boxes['annotations']
                            if a.get('bbox') and int(a['category_id']) not in {30, 33}]).value_counts()
    images['source_animal_boxes'] = images.image_id.map(bbox_counts).fillna(0).astype(int)
    return images


def stratified_sample(frame: pd.DataFrame, label: str, limit: int, seed: int = 42) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame.copy()
    sizes = frame[label].value_counts().sort_index()
    quota = np.floor(sizes/len(frame)*limit).astype(int)
    remaining = limit-int(quota.sum())
    for value in (sizes/len(frame)*limit-quota).sort_values(ascending=False).index[:remaining]:
        quota.loc[value] += 1
    return pd.concat([frame[frame[label].eq(value)].sample(n=int(n), random_state=seed)
                      for value, n in quota.items()]).sort_index()


def _image_features(item: tuple[Path, str]) -> tuple[dict, np.ndarray]:
    path, image_id = item
    with Image.open(path) as image:
        image.draft('RGB', (256, 256))
        rgb = image.convert('RGB')
        quality_gray = np.asarray(ImageOps.contain(rgb, (256, 256)).convert('L'), dtype=np.float32)
        laplacian = (quality_gray[:-2, 1:-1] + quality_gray[2:, 1:-1]
                     + quality_gray[1:-1, :-2] + quality_gray[1:-1, 2:]
                     - 4*quality_gray[1:-1, 1:-1])
        small = np.asarray(rgb.resize((64, 64)), dtype=np.float32)/255
    gray = small @ np.array([.299, .587, .114], dtype=np.float32)
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    orientation = np.mod(np.arctan2(gy, gx), np.pi)
    bins = np.minimum((orientation/np.pi*9).astype(int), 8)
    # 8x8 cells, 9 orientation bins; normalize each cell histogram.
    hog = np.empty((8, 8, 9), dtype=np.float32)
    for y in range(8):
        for x in range(8):
            sy, sx = slice(y*8, (y+1)*8), slice(x*8, (x+1)*8)
            hist = np.bincount(bins[sy, sx].ravel(), weights=magnitude[sy, sx].ravel(), minlength=9)
            hog[y, x] = hist/(np.linalg.norm(hist)+1e-6)
    color = np.concatenate([np.histogram(small[:, :, c], bins=16, range=(0, 1))[0]/4096 for c in range(3)])
    appearance = np.asarray(Image.fromarray((gray*255).astype('uint8')).resize((16, 16)), dtype=np.float32).ravel()/255
    feature = np.concatenate([hog.ravel(), color, appearance]).astype(np.float32)
    stats = {'image_id': image_id, 'laplacian_variance': float(laplacian.var()),
             'dark_pixel_fraction': float((quality_gray < 30).mean()),
             'clipped_bright_fraction': float((quality_gray > 250).mean())}
    return stats, feature


def image_feature_cache(root: Path, manifest: pd.DataFrame, checks: pd.DataFrame):
    """Cache by IDs and source hashes; do not refetch or modify original JPGs."""
    ordered = manifest.sort_values('image_id')
    hashes = checks.set_index('image_id').loc[ordered.image_id, 'sha256']
    signature = hashlib.sha256(('v1\n'+'\n'.join(ordered.image_id+' '+hashes.to_numpy())).encode()).hexdigest()
    directory = root/'artifacts/eda_cache'
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory/'image_features.npz'
    table_path = directory/'image_quality.parquet'
    fingerprint = directory/'fingerprint.txt'
    if cache.exists() and table_path.exists() and fingerprint.exists() and fingerprint.read_text() == signature:
        with np.load(cache, allow_pickle=False) as stored:
            features = stored['features']
        return pd.read_parquet(table_path), features, ordered.image_id.to_numpy()
    items = [(root/row.local_path, row.image_id) for row in ordered.itertuples()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_image_features, items))
    quality = pd.DataFrame([r[0] for r in results])
    features = np.stack([r[1] for r in results])
    quality.to_parquet(table_path, index=False)
    np.savez_compressed(cache, features=features)
    fingerprint.write_text(signature)
    return quality, features, ordered.image_id.to_numpy()
