from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, PngImagePlugin

from find_duplicate_images import compare_photos, inspect_photo, scan_directory


def animal_fixture() -> Image.Image:
    rng = np.random.default_rng(2026)
    texture = rng.integers(60, 170, (240, 320, 3), dtype=np.uint8)
    image = Image.fromarray(texture).filter(ImageFilter.GaussianBlur(3))
    draw = ImageDraw.Draw(image)
    draw.ellipse((55, 60, 245, 215), fill=(175, 111, 53))
    draw.polygon([(65, 92), (80, 24), (124, 81)], fill=(130, 76, 38))
    draw.polygon([(167, 79), (204, 30), (220, 106)], fill=(145, 86, 40))
    draw.ellipse((88, 78, 209, 175), fill=(199, 143, 77))
    draw.ellipse((103, 112, 116, 125), fill=(12, 20, 26))
    draw.ellipse((165, 107, 180, 124), fill=(12, 20, 26))
    draw.ellipse((125, 140, 157, 160), fill=(52, 29, 13))
    draw.line((198, 188, 279, 132), fill=(178, 112, 55), width=14)
    return image.filter(ImageFilter.GaussianBlur(0.6))


class DuplicateDetectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image = animal_fixture()
        self.original = self.root / "original.png"
        self.image.save(self.original)

    def tearDown(self):
        self.temp.cleanup()

    def match(self, path):
        return compare_photos(inspect_photo(self.original), inspect_photo(path))

    def test_file_copy_and_metadata_only_changes(self):
        copied = self.root / "copy.JPG"
        shutil.copyfile(self.original, copied)
        self.assertEqual(self.match(copied).kind, "exact_file")
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("description", "same displayed photo")
        changed = self.root / "metadata.png"
        self.image.save(changed, pnginfo=metadata)
        self.assertEqual(self.match(changed).kind, "exact_pixels")

    def test_all_supported_resize_filters_in_both_directions(self):
        for resampler in Image.Resampling:
            for size in ((160, 120), (640, 480)):
                with self.subTest(resampler=resampler, size=size):
                    resized = self.root / "resize.png"
                    self.image.resize(size, resampler).save(resized)
                    match = self.match(resized)
                    self.assertIsNotNone(match)
                    self.assertEqual(match.kind, "exact_resize")

    def test_high_quality_jpeg_recompression_is_only_a_candidate(self):
        jpeg = self.root / "recompressed.jpg"
        self.image.save(jpeg, quality=97, subsampling=0)
        match = self.match(jpeg)
        self.assertIsNotNone(match)
        self.assertEqual(match.kind, "resize_candidate")

    def test_proportional_thumbnail_allows_pixel_rounding(self):
        wide = self.image.resize((576, 324), Image.Resampling.BICUBIC)
        wide.save(self.original)
        thumb = wide.copy()
        thumb.thumbnail((113, 113), Image.Resampling.BICUBIC)
        self.assertEqual(thumb.size, (113, 64))
        path = self.root / "rounded_thumbnail.png"
        thumb.save(path)
        match = self.match(path)
        self.assertIsNotNone(match)
        self.assertIn(match.kind, ("exact_resize", "resize_candidate"))

    def test_different_direction_rotation_crop_and_translation(self):
        variants = {
            "mirror": ImageOps.mirror(self.image),
            "rotation": self.image.rotate(3, resample=Image.Resampling.BICUBIC),
            "crop": self.image.crop((8, 6, 312, 234)).resize(self.image.size),
            "translation": self.image.transform(self.image.size, Image.Transform.AFFINE,
                                                 (1, 0, 4, 0, 1, 0)),
        }
        for name, image in variants.items():
            with self.subTest(name=name):
                path = self.root / f"{name}.png"
                image.save(path)
                self.assertIsNone(self.match(path))

    def test_changed_animal_detail_on_same_background_is_distinct(self):
        changed = self.image.copy()
        ImageDraw.Draw(changed).rectangle((98, 106, 122, 131), fill=(199, 143, 77))
        path = self.root / "changed_eye.png"
        changed.save(path)
        self.assertIsNone(self.match(path))

    def test_color_change_is_distinct(self):
        array = np.asarray(self.image).copy()
        array[:, :, 0] = np.minimum(array[:, :, 0].astype(np.uint16) + 35, 255)
        path = self.root / "color.png"
        Image.fromarray(array).save(path)
        self.assertIsNone(self.match(path))

    def test_highest_resolution_is_keeper_and_groups_are_direct(self):
        for name, size in (("large", (640, 480)), ("small", (160, 120))):
            self.image.resize(size, Image.Resampling.LANCZOS).save(self.root / f"{name}.png")
        result = scan_directory(self.root)
        self.assertEqual(len(result.photos), 3)
        self.assertEqual(len(result.duplicates), 2)
        self.assertFalse(result.errors)
        for row in result.duplicates:
            self.assertEqual(row.keep.path.name, "large.png")
            self.assertIsNotNone(compare_photos(row.keep, row.duplicate))

    def test_tiny_resized_images_are_not_guessed(self):
        path = self.root / "tiny.png"
        self.image.resize((32, 24)).save(path)
        self.assertIsNone(self.match(path))

    def run_cli(self, directory, report):
        script = Path(__file__).resolve().parents[1] / "find_duplicate_images.py"
        return subprocess.run([sys.executable, str(script), str(directory), "-o", str(report)],
                              capture_output=True, text=True)

    def test_cli_recursive_report_and_images_remain_unchanged(self):
        nested = self.root / "class_cat"
        nested.mkdir()
        copied = nested / "copied.png"
        shutil.copyfile(self.original, copied)
        before = {p: hashlib.sha256(p.read_bytes()).digest() for p in (self.original, copied)}
        report = self.root / "report.csv"
        process = self.run_cli(self.root, report)
        self.assertEqual(process.returncode, 0, process.stderr)
        with report.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["match_kind"], "exact_file")
        for path, digest in before.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), digest)
        previous_report = report.read_bytes()
        self.assertNotEqual(self.run_cli(self.root, report).returncode, 0)
        self.assertEqual(report.read_bytes(), previous_report)

    def test_corrupt_file_and_empty_directory_return_nonzero(self):
        corrupt = self.root / "broken.jpg"
        corrupt.write_bytes(b"not an image")
        process = self.run_cli(self.root, self.root / "partial.csv")
        self.assertEqual(process.returncode, 2)
        self.assertIn("broken.jpg", process.stderr)
        empty = self.root / "empty"
        empty.mkdir()
        report = self.root / "empty.csv"
        self.assertEqual(self.run_cli(empty, report).returncode, 2)
        self.assertFalse(report.exists())


if __name__ == "__main__":
    unittest.main()
