from __future__ import annotations

import csv
import hashlib
import html
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin

from find_duplicate_images import inspect_photo
from find_split_overlap import (SplitMatch, find_exact_overlaps, select_overlaps,
                                validate_folders, write_overlap_results)


class CrossSplitSelectionTests(unittest.TestCase):
    def test_similar_train_images_cannot_crowd_out_reference_matches(self):
        # The two train photos resemble each other more than the reference.
        # Cross-split selection must still report the reference for both.
        train = np.array([[1.0, 0.0], [0.999, np.sqrt(1 - 0.999 ** 2)]])
        references = np.array([[0.95, np.sqrt(1 - 0.95 ** 2)], [0.0, 1.0]])
        matches = select_overlaps(train, train, references, references,
                                  ["test", "val"], threshold=0.90, top_k=1)
        self.assertEqual({(m.train, m.reference) for m in matches}, {(0, 0), (1, 0)})
        self.assertTrue(all(m.reference_split == "test" for m in matches))
        self.assertTrue(all(m.match_kind == "near_duplicate_candidate" for m in matches))

    def test_per_train_top_k_preserves_test_and_val_labels(self):
        train = np.array([[1.0, 0.0], [0.0, 1.0]])
        references = np.array([[1.0, 0.0], [0.98, np.sqrt(1 - 0.98 ** 2)], [0.0, 1.0]])
        matches = select_overlaps(train, train, references, references,
                                  ["test", "val", "val"], threshold=0.97, top_k=2)
        self.assertEqual({(m.train, m.reference, m.reference_split) for m in matches},
                         {(0, 0, "test"), (0, 1, "val"), (1, 2, "val")})
        self.assertEqual(len(matches), 3)
        self.assertEqual([m.similarity for m in matches],
                         sorted((m.similarity for m in matches), reverse=True))

        only_best = select_overlaps(train, train, references, references,
                                    ["test", "val", "val"], threshold=0.97, top_k=1)
        self.assertEqual({(m.train, m.reference) for m in only_best}, {(0, 0), (1, 2)})

    def test_each_view_is_normalized_before_weighting(self):
        train_global = np.array([[8.0, 0.0]])
        train_center = np.array([[2.0, 0.0]])
        reference_global = np.array([[0.0, 4.0]])
        reference_center = np.array([[3.0, 4.0]])
        matches = select_overlaps(train_global, train_center, reference_global,
                                  reference_center, ["val"], threshold=0.35)
        self.assertEqual(len(matches), 1)
        match = matches[0]
        self.assertEqual((match.train, match.reference, match.reference_split), (0, 0, "val"))
        self.assertAlmostEqual(match.global_similarity, 0.0, places=6)
        self.assertAlmostEqual(match.center_similarity, 0.6, places=6)
        self.assertAlmostEqual(match.similarity, 0.4 * 0.0 + 0.6 * 0.6, places=6)

    def test_below_threshold_candidates_are_not_reported(self):
        train = np.array([[1.0, 0.0]])
        references = np.array([[0.8, 0.6], [0.0, 1.0]])
        self.assertEqual(select_overlaps(train, train, references, references,
                                         ["test", "val"]), [])

    def test_default_top_k_has_no_limit_on_eligible_cross_split_pairs(self):
        train = np.array([[1.0, 0.0]])
        angles = np.array([0.0, 0.02, 0.04, 0.06])
        references = np.column_stack((np.cos(angles), np.sin(angles)))
        matches = select_overlaps(train, train, references, references,
                                  ["test", "test", "val", "val"])
        self.assertEqual([(m.train, m.reference) for m in matches],
                         [(0, 0), (0, 1), (0, 2), (0, 3)])

    def test_negative_top_k_is_rejected(self):
        vectors = np.array([[1.0, 0.0]])
        with self.assertRaises(ValueError):
            select_overlaps(vectors, vectors, vectors, vectors, ["test"], top_k=-1)

    def test_reference_labels_must_be_test_or_val_and_match_vector_count(self):
        vectors = np.array([[1.0, 0.0], [0.8, 0.6]])
        for labels in (["train", "test"], ["test"], ["test", "validation"]):
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    select_overlaps(vectors, vectors, vectors, vectors, labels)

    def test_invalid_vectors_are_rejected(self):
        valid = np.array([[1.0, 0.0], [0.8, 0.6]])
        for invalid in (np.array([[0.0, 0.0], [0.8, 0.6]]),
                        np.array([[np.nan, 0.0], [0.8, 0.6]])):
            for position in range(4):
                args = [valid, valid, valid, valid]
                args[position] = invalid
                with self.subTest(position=position, invalid=invalid.tolist()):
                    with self.assertRaises(ValueError):
                        select_overlaps(*args, ["test", "val"])


class ExactCrossSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ("train", "test", "val"):
            (self.root / name).mkdir()
        self.image = Image.new("RGB", (100, 80), (90, 135, 75))
        ImageDraw.Draw(self.image).ellipse((20, 10, 85, 75), fill=(185, 120, 70))
        self.train = self.root / "train" / "동물 원본.png"
        self.image.save(self.train)

    def tearDown(self):
        self.temp.cleanup()

    def test_every_exact_file_in_test_and_val_is_reported_without_a_candidate_limit(self):
        reference_paths = []
        reference_splits = ["test", "test", "val", "val"]
        for index, split in enumerate(reference_splits):
            path = self.root / split / f"동물 복사 {index}.png"
            shutil.copyfile(self.train, path)
            reference_paths.append(path)
        all_paths = [self.train] + reference_paths
        before = {path: hashlib.sha256(path.read_bytes()).digest() for path in all_paths}
        matches = find_exact_overlaps([inspect_photo(self.train)],
                                      [inspect_photo(path) for path in reference_paths],
                                      reference_splits)
        self.assertEqual(len(matches), 4)
        self.assertEqual({(m.train, m.reference, m.reference_split) for m in matches},
                         {(0, index, split) for index, split in enumerate(reference_splits)})
        self.assertTrue(all(m.match_kind == "exact_file" for m in matches))
        self.assertTrue(all(m.similarity == 1.0 for m in matches))
        for path, digest in before.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), digest)

    def test_identical_pixels_with_changed_metadata_are_reported_as_exact_pixels(self):
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("description", "같은 표시 사진, 다른 메타데이터")
        reference = self.root / "val" / "동물 메타데이터.png"
        self.image.save(reference, pnginfo=metadata)
        train_photo, reference_photo = inspect_photo(self.train), inspect_photo(reference)
        self.assertNotEqual(train_photo.file_hash, reference_photo.file_hash)
        self.assertEqual(train_photo.pixel_hash, reference_photo.pixel_hash)
        matches = find_exact_overlaps([train_photo], [reference_photo], ["val"])
        self.assertEqual(len(matches), 1)
        self.assertEqual((matches[0].train, matches[0].reference, matches[0].reference_split),
                         (0, 0, "val"))
        self.assertEqual(matches[0].match_kind, "exact_pixels")


class SplitFolderValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train = self.root / "train split"
        self.test = self.root / "test split"
        self.val = self.root / "val split"
        for path in (self.train, self.test, self.val):
            path.mkdir()
        self.output = self.root / "새 보고서"

    def tearDown(self):
        self.temp.cleanup()

    def test_separate_inputs_are_resolved_and_output_is_not_created(self):
        folders = validate_folders(self.train, self.test, self.val, self.output)
        for name, path in (("train", self.train), ("test", self.test), ("val", self.val)):
            self.assertEqual(folders[name], path.resolve())
        self.assertFalse(self.output.exists())
        val_only = validate_folders(self.train, None, self.val, self.output)
        self.assertEqual(val_only["val"], self.val.resolve())

    def test_at_least_one_reference_split_is_required(self):
        with self.assertRaises(ValueError):
            validate_folders(self.train, None, None, self.output)

    def test_equal_and_nested_input_roots_are_rejected(self):
        nested = self.train / "reference subset"
        nested.mkdir()
        cases = ((self.train, self.train, None),
                 (self.train, nested, None),
                 (nested, self.train, None),
                 (self.train, self.test, self.test),
                 (self.train, self.test, self.test / "val subset"))
        (self.test / "val subset").mkdir()
        for train, test, val in cases:
            with self.subTest(train=train.name, test=test.name, val=val):
                with self.assertRaises(ValueError):
                    validate_folders(train, test, val, self.output)

    def test_symlink_alias_of_input_is_rejected(self):
        alias = self.root / "train alias"
        try:
            alias.symlink_to(self.train, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"directory symlinks unavailable: {error}")
        with self.assertRaises(ValueError):
            validate_folders(self.train, alias, None, self.output)

    def test_output_inside_any_input_tree_is_rejected(self):
        for parent in (self.train, self.test, self.val):
            with self.subTest(parent=parent.name):
                with self.assertRaises(ValueError):
                    validate_folders(self.train, self.test, self.val, parent / "report")

    def test_existing_output_is_rejected_without_overwriting(self):
        self.output.mkdir()
        sentinel = self.output / "index.html"
        sentinel.write_text("기존 보고서", encoding="utf-8")
        with self.assertRaises((ValueError, OSError)):
            validate_folders(self.train, self.test, self.val, self.output)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "기존 보고서")

    def test_missing_input_folder_is_rejected(self):
        with self.assertRaises((ValueError, OSError)):
            validate_folders(self.train, self.root / "missing test", None, self.output)


class CrossSplitReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train_paths = [self.make_image("train 고양이 & 털.png", (182, 131, 80))]
        self.reference_paths = [self.make_image("test 고양이 사진.png", (183, 132, 81)),
                                self.make_image("val 고양이 사진.png", (180, 130, 79))]
        self.matches = [SplitMatch(train=0, reference=0, reference_split="test",
                                   similarity=0.95, global_similarity=0.92,
                                   center_similarity=0.97),
                        SplitMatch(train=0, reference=1, reference_split="val",
                                   similarity=0.93, global_similarity=0.90,
                                   center_similarity=0.95)]

    def make_image(self, name, color):
        path = self.root / name
        image = Image.new("RGB", (120, 100), (70, 125, 90))
        draw = ImageDraw.Draw(image)
        draw.ellipse((20, 10, 100, 95), fill=color)
        draw.ellipse((42, 40, 51, 51), fill=(10, 15, 20))
        draw.ellipse((76, 40, 85, 51), fill=(10, 15, 20))
        image.save(path)
        return path

    def tearDown(self):
        self.temp.cleanup()

    def read_csv(self, output):
        with (output / "split_overlaps.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)

    def test_report_puts_train_candidates_before_reference_paths_and_keeps_all_photos(self):
        all_paths = self.train_paths + self.reference_paths
        before = {path: hashlib.sha256(path.read_bytes()).digest() for path in all_paths}
        output = self.root / "overlap report"
        write_overlap_results(self.train_paths, self.reference_paths, self.matches, output)
        headers, rows = self.read_csv(output)
        self.assertTrue(headers)
        self.assertEqual(len(rows), 2)
        for row, match in zip(rows, self.matches):
            values = list(row.values())
            train_path = str(self.train_paths[match.train])
            reference_path = str(self.reference_paths[match.reference])
            self.assertIn(train_path, values)
            self.assertIn(reference_path, values)
            self.assertLess(values.index(train_path), values.index(reference_path))
            self.assertIn(match.reference_split, values)
            self.assertIn("near_duplicate_candidate", values)
        review = (output / "index.html").read_text(encoding="utf-8")
        self.assertIn("data:image/", review)
        self.assertIn("train — 제거 검토 대상", review)
        self.assertIn("test — 유지", review)
        self.assertIn("val — 유지", review)
        with (output / "train_candidates.csv").open(encoding="utf-8-sig", newline="") as stream:
            candidates = list(csv.DictReader(stream))
        self.assertEqual([row['train_image'] for row in candidates], [str(self.train_paths[0])])
        for path in all_paths:
            self.assertIn(html.escape(path.name), review)
            self.assertTrue(path.is_file())
            self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), before[path])

    def test_no_matches_still_produces_review_and_csv_headers(self):
        output = self.root / "no overlaps"
        write_overlap_results(self.train_paths, self.reference_paths, [], output)
        headers, rows = self.read_csv(output)
        self.assertTrue(headers)
        self.assertEqual(rows, [])
        review = (output / "index.html").read_text(encoding="utf-8")
        self.assertIn("<html", review.lower())
        self.assertIn("<title", review.lower())

    def test_existing_report_is_refused_and_preserved(self):
        output = self.root / "existing overlap report"
        output.mkdir()
        sentinel = output / "index.html"
        sentinel.write_text("기존 검토 기록", encoding="utf-8")
        with self.assertRaises((ValueError, OSError)):
            write_overlap_results(self.train_paths, self.reference_paths, self.matches, output)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "기존 검토 기록")
        self.assertFalse((output / "split_overlaps.csv").exists())

    def test_partial_scan_is_visible_in_the_standalone_review(self):
        output = self.root / "partial report"
        write_overlap_results(self.train_paths, self.reference_paths, self.matches, output,
                              incomplete_count=2)
        document = (output / "index.html").read_text(encoding="utf-8")
        self.assertIn("읽기 실패 2건", document)
        self.assertIn("미완료 검사", document)


if __name__ == "__main__":
    unittest.main()
