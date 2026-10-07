from __future__ import annotations

import csv
import hashlib
import html
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from find_similar_images import SimilarPair, select_pairs, write_results


class SimilarPairSelectionTests(unittest.TestCase):
    def test_score_normalizes_each_view_before_weighting(self):
        global_vectors = np.array([[7.0, 0.0], [0.0, 2.0]])
        center_vectors = np.array([[4.0, 0.0], [3.0, 4.0]])
        original_global = global_vectors.copy()
        original_center = center_vectors.copy()
        pairs = select_pairs(global_vectors, center_vectors, threshold=0.35, top_k=1)
        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        self.assertEqual((pair.left, pair.right), (0, 1))
        self.assertAlmostEqual(pair.global_similarity, 0.0, places=6)
        self.assertAlmostEqual(pair.center_similarity, 0.6, places=6)
        self.assertAlmostEqual(pair.similarity, 0.4 * 0.0 + 0.6 * 0.6, places=6)
        np.testing.assert_array_equal(global_vectors, original_global)
        np.testing.assert_array_equal(center_vectors, original_center)

    def test_default_threshold_filters_low_similarity_pairs(self):
        vectors = np.array([[1.0, 0.0], [0.95, np.sqrt(1 - 0.95 ** 2)], [0.0, 1.0]])
        pairs = select_pairs(vectors, vectors)
        self.assertEqual([(p.left, p.right) for p in pairs], [(0, 1)])
        self.assertAlmostEqual(pairs[0].similarity, 0.95, places=6)
        self.assertEqual(select_pairs(vectors, vectors, threshold=0.99), [])

    def test_top_k_uses_union_of_both_directions_and_returns_unique_pairs(self):
        angles = np.array([0.0, 0.1, 0.3, 1.5])
        vectors = np.column_stack((np.cos(angles), np.sin(angles)))
        pairs = select_pairs(vectors, vectors, threshold=0.0, top_k=1)
        # Image 1 chooses 0; image 2 chooses 1. The 1-2 pair must survive
        # even though image 1's own nearest neighbour is image 0.
        self.assertEqual({(p.left, p.right) for p in pairs}, {(0, 1), (1, 2), (2, 3)})
        self.assertEqual(len(pairs), 3)
        self.assertTrue(all(pair.left < pair.right for pair in pairs))
        self.assertEqual([p.similarity for p in pairs],
                         sorted((p.similarity for p in pairs), reverse=True))

    def test_high_top_k_reports_all_eligible_pairs_in_descending_order(self):
        angles = np.array([0.0, 0.1, 0.3])
        vectors = np.column_stack((np.cos(angles), np.sin(angles)))
        pairs = select_pairs(vectors, vectors, threshold=0.0, top_k=10)
        self.assertEqual([(p.left, p.right) for p in pairs], [(0, 1), (1, 2), (0, 2)])

    def test_zero_vector_is_rejected_in_either_view(self):
        valid = np.array([[1.0, 0.0], [0.8, 0.6]])
        zero_row = np.array([[1.0, 0.0], [0.0, 0.0]])
        for global_vectors, center_vectors in ((zero_row, valid), (valid, zero_row)):
            with self.subTest(zero_in_global=global_vectors is zero_row):
                with self.assertRaises(ValueError):
                    select_pairs(global_vectors, center_vectors)

    def test_nonfinite_vectors_are_rejected(self):
        valid = np.array([[1.0, 0.0], [0.8, 0.6]])
        for value in (np.nan, np.inf, -np.inf):
            invalid = valid.copy()
            invalid[1, 0] = value
            for global_vectors, center_vectors in ((invalid, valid), (valid, invalid)):
                with self.subTest(value=value, invalid_global=global_vectors is invalid):
                    with self.assertRaises(ValueError):
                        select_pairs(global_vectors, center_vectors)

    def test_invalid_shapes_are_rejected(self):
        valid = np.array([[1.0, 0.0], [0.8, 0.6]])
        for invalid in (np.array([1.0, 0.0]), np.array([[1.0, 0.0]])):
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(ValueError):
                    select_pairs(valid, invalid)


class SimilarReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = []
        for index, name in enumerate(("동물 원본.png", "고양이 & '눈'.png")):
            path = self.root / name
            image = Image.new("RGB", (120, 100), (80 + index * 5, 130, 85))
            draw = ImageDraw.Draw(image)
            draw.ellipse((20, 15, 100, 95), fill=(185, 130, 75))
            draw.ellipse((42 + index, 43, 50 + index, 52), fill=(15, 20, 25))
            draw.ellipse((76, 43, 84, 52), fill=(15, 20, 25))
            image.save(path)
            self.paths.append(path)
        self.pairs = [SimilarPair(left=0, right=1, similarity=0.95,
                                  global_similarity=0.92, center_similarity=0.97)]

    def tearDown(self):
        self.temp.cleanup()

    def read_csv(self, output):
        with (output / "similar_photos.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            return reader.fieldnames, rows

    def test_report_has_csv_and_embedded_review_images_without_changing_sources(self):
        before = {path: hashlib.sha256(path.read_bytes()).digest() for path in self.paths}
        output = self.root / "report"
        write_results(self.paths, self.pairs, output)
        self.assertTrue(output.is_dir())
        headers, rows = self.read_csv(output)
        self.assertTrue(headers)
        self.assertEqual(len(rows), 1)
        values = list(rows[0].values())
        for path in self.paths:
            self.assertIn(str(path), values)
        self.assertTrue(any(value and abs(float(value) - 0.95) < 0.00001
                            for value in values if value and value.replace(".", "", 1).isdigit()))
        review = (output / "index.html").read_text(encoding="utf-8")
        self.assertIn("data:image/", review)
        for path, digest in before.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), digest)

    def test_unicode_filenames_are_visible_and_html_escaped(self):
        output = self.root / "unicode_report"
        write_results(self.paths, self.pairs, output)
        review = (output / "index.html").read_text(encoding="utf-8")
        for path in self.paths:
            self.assertIn(html.escape(path.name), review)
        self.assertNotIn("<img onerror=alert(1)>", review)
        self.assertIn("고양이", review)

    def test_hostile_filename_is_text_instead_of_an_html_element(self):
        hostile = Path('고양이 <img onerror=alert(1)>&"눈".png')
        output = self.root / "hostile_report"
        # Filename rendering must also be safe for characters not valid on Windows.
        with patch("find_similar_images._thumbnail", return_value="data:image/jpeg;base64,AA=="):
            write_results([self.paths[0], hostile], self.pairs, output)
        document = (output / "index.html").read_text(encoding="utf-8")
        self.assertIn(html.escape(hostile.name), document)
        self.assertNotIn("<img onerror=alert(1)>", document)

    def test_initial_html_filter_does_not_round_away_the_lowest_candidate(self):
        output = self.root / "rounding_report"
        pair = SimilarPair(0, 1, 0.9006, 0.90, 0.901)
        write_results(self.paths, [pair], output)
        document = (output / "index.html").read_text(encoding="utf-8")
        self.assertIn('value="0.900"', document)

    def test_empty_pairs_still_produce_csv_header_and_html(self):
        output = self.root / "empty_report"
        write_results(self.paths, [], output)
        headers, rows = self.read_csv(output)
        self.assertTrue(headers)
        self.assertEqual(rows, [])
        review = (output / "index.html").read_text(encoding="utf-8")
        self.assertIn("<html", review.lower())
        self.assertIn("<title", review.lower())

    def test_existing_output_directory_is_refused_and_preserved(self):
        output = self.root / "existing_report"
        output.mkdir()
        sentinel = output / "index.html"
        sentinel.write_text("사용자가 이미 작성한 보고서", encoding="utf-8")
        with self.assertRaises((FileExistsError, ValueError, OSError)):
            write_results(self.paths, self.pairs, output)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "사용자가 이미 작성한 보고서")
        self.assertFalse((output / "similar_photos.csv").exists())


if __name__ == "__main__":
    unittest.main()
