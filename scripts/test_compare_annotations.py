import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import compare_annotations as comparison


class ComparisonTests(unittest.TestCase):
    def test_kappa_and_nonbinary_coverage(self):
        labels = ['not_spam', 'spam']
        perfect = comparison.binary_metrics([('not_spam', 'not_spam'), ('spam', 'spam')], labels)
        self.assertEqual(perfect['cohen_kappa'], 1)
        opposite = comparison.binary_metrics([('not_spam', 'spam'), ('spam', 'not_spam')], labels)
        self.assertEqual(opposite['cohen_kappa'], -1)
        partial = comparison.binary_metrics([('spam', 'spam'), ('needs_review', 'spam')], labels)
        self.assertEqual(partial['coverage'], 0.5)
        self.assertEqual(partial['nonbinary_pairs'], 1)
        self.assertIsNone(partial['cohen_kappa'])
        self.assertIsNone(comparison.binary_metrics([], labels)['agreement'])

    def test_matching_prefers_cardinality_then_iou(self):
        self.assertEqual(comparison.optimal_match([[0.9, 0.8], [0.85, 0]], 2)[2], ((0, 1), (1, 0)))
        self.assertEqual(comparison.optimal_match([[0.9, 0.7], [0.65, 0.51]], 2)[2], ((0, 0), (1, 1)))
        self.assertEqual(comparison.optimal_match([[0.5]], 1)[0], 1)
        self.assertEqual(comparison.optimal_match([[0.49999]], 1)[0], 0)

    def test_flow_matches_independent_exhaustive_search(self):
        rng = random.Random(47)

        def enumerate_matchings(matrix, row=0, used=frozenset()):
            if row == len(matrix):
                return [(0, 0)]
            possibilities = enumerate_matchings(matrix, row + 1, used)
            for col, score in enumerate(matrix[row]):
                if col not in used and score >= 0.5:
                    possibilities += [(n + 1, value + score) for n, value in enumerate_matchings(matrix, row + 1, used | {col})]
            return possibilities

        for _ in range(50):
            matrix = [[rng.randrange(0, 11) / 10 for _ in range(4)] for _ in range(4)]
            expected = max(enumerate_matchings(matrix))
            count, total, pairs = comparison.optimal_match(matrix, 4)
            self.assertEqual(count, expected[0])
            self.assertAlmostEqual(total, expected[1])
            self.assertEqual(len({i for i, _ in pairs}), count)
            self.assertEqual(len({j for _, j in pairs}), count)

    def test_iou_and_empty_box_sets(self):
        box = dict(x=0, y=0, width=20, height=20, flags=[])
        shifted = dict(box, x=10)
        self.assertAlmostEqual(comparison.iou(box, shifted), 1 / 3)
        self.assertIsNone(comparison.compare_boxes([], [])['matching_f1'])
        self.assertEqual(comparison.compare_boxes([box], [])['matching_f1'], 0)
        self.assertEqual(comparison.compare_boxes([box], [box])['matching_f1'], 1)

    @staticmethod
    def task(sid, status, image=False, box=False):
        field = 'frame_status' if image else 'label'
        results = [{'id': 'choice', 'from_name': field, 'type': 'choices', 'value': {'choices': [status]}}]
        if box:
            results.append({'id': 'box', 'from_name': 'bbox', 'type': 'rectanglelabels',
                            'original_width': 100, 'original_height': 100,
                            'value': {'x': 10, 'y': 10, 'width': 20, 'height': 20, 'rotation': 0, 'rectanglelabels': ['animal']}})
        return {'data': {'sample_id': sid, ('image' if image else 'text'): ('https://example.test/' + sid if image else 'Example ' + sid)},
                'annotations': [{'was_cancelled': False, 'lead_time': 2.5, 'result': results}]}

    def test_bad_boxes_are_rejected(self):
        task = self.task('IMG001', 'animal', image=True, box=True)
        task['annotations'][0]['result'][1]['value']['width'] = -2
        with self.assertRaises(ValueError):
            comparison.unpack(task, 'images')

    def test_full_run_preserves_inputs_and_existing_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_a = [self.task('TG001', 'not_spam'), self.task('TG002', 'spam')]
            text_b = [self.task('TG001', 'spam'), self.task('TG002', 'spam')]
            original = [self.task('IMG001', 'animal', image=True), self.task('IMG002', 'animal', image=True, box=True)]
            current = copy.deepcopy(original)
            current[0]['annotations'][0]['result'][0]['value']['choices'] = ['empty']
            values = dict(text_A=text_a, text_B=text_b, images_A_original=original,
                          images_A_current=current, images_B=copy.deepcopy(current))
            inputs = {}
            for key, tasks in values.items():
                path = root / (key + '.json')
                path.write_text(json.dumps(tasks), encoding='utf-8')
                inputs[key] = path
            originals = {key: path.read_bytes() for key, path in inputs.items()}
            output = root / 'output'
            report = comparison.build(inputs, output)
            self.assertEqual(report['statistics']['image_presence_original_A']['agreement'], 0.5)
            self.assertEqual(report['statistics']['image_presence_after_user_correction']['agreement'], 1)
            self.assertEqual(report['statistics']['bounding_boxes']['matched_pairs'], 1)
            self.assertEqual(report['statistics']['review']['text'], 1)
            decision = output / 'adjudication_template.json'
            decision.write_text('{"human_decision":"preserve this"}', encoding='utf-8')
            comparison.build(inputs, output)
            self.assertEqual(json.loads(decision.read_text())['human_decision'], 'preserve this')
            for key, path in inputs.items():
                self.assertEqual(path.read_bytes(), originals[key])
            text_b[0]['data']['text'] = 'Different source message'
            inputs['text_B'].write_text(json.dumps(text_b), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'different_text_input'):
                comparison.build(inputs, root / 'bad')

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'duplicate.json'
            task = self.task('TG001', 'spam')
            path.write_text(json.dumps([task, task]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate_id'):
                comparison.load(path)


if __name__ == '__main__':
    unittest.main()
