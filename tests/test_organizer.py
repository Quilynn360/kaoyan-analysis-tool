"""tests/test_organizer.py —— SchoolClassifier + 归档功能单元测试

运行方式：
    python -m unittest tests/test_organizer.py -v
    （或） python tests/test_organizer.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# 确保能找到 src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.organizer import (
    SchoolClassifier,
    extract_year,
    safe_move,
    organize_single,
)


class TestSchoolClassifier(unittest.TestCase):
    """测试学校分类器."""

    def setUp(self):
        # 使用内置的正式映射文件
        self.classifier = SchoolClassifier()

    def test_classify_chinese_name(self):
        """包含中文关键词应命中."""
        self.assertEqual(self.classifier.classify("山东大学_2024_成绩.xlsx"), "Shandong_University")

    def test_classify_abbreviation(self):
        """包含英文缩写应命中."""
        self.assertEqual(self.classifier.classify("PKU_2023_复试名单.pdf"), "Peking_University")

    def test_classify_short_name(self):
        """包含简称应命中."""
        self.assertEqual(self.classifier.classify("哈工大_2025_数据.docx"), "Harbin_Institute_of_Technology")

    def test_classify_no_match(self):
        """无匹配返回 None."""
        self.assertIsNone(self.classifier.classify("未知名大学_2024.xlsx"))

    def test_classify_no_keyword_in_name(self):
        """文件名不含任何关键词应返回 None."""
        self.assertIsNone(self.classifier.classify("某某学院_2024.xlsx"))


class TestExtractYear(unittest.TestCase):
    """测试年份提取."""

    def test_standard_4digit(self):
        self.assertEqual(extract_year("山东大学_2024_成绩.xlsx"), "2024")

    def test_4digit_in_middle(self):
        self.assertEqual(extract_year("成绩_2023_山东大学.xlsx"), "2023")

    def test_no_year(self):
        self.assertIsNone(extract_year("山东大学_成绩.xlsx"))

    def test_historic_4digit(self):
        self.assertEqual(extract_year("清华1998_数据.pdf"), "1998")

    # ── 2 位数年份 ────────────────────────────────────────────────────

    def test_2digit_bare(self):
        """文件名中的独立 2 位数 22 应识别为 2022."""
        self.assertEqual(extract_year("北大22年推免名单.pdf"), "2022")

    def test_2digit_23(self):
        self.assertEqual(extract_year("复旦23年复试名单.png"), "2023")

    def test_2digit_24(self):
        self.assertEqual(extract_year("浙大24录取.xlsx"), "2024")

    def test_2digit_25(self):
        self.assertEqual(extract_year("南大25专业目录.pdf"), "2025")

    def test_2digit_26(self):
        self.assertEqual(extract_year("上交大26招生.xlsx"), "2026")

    def test_2digit_15_to_21(self):
        """15-21 也应正确转换."""
        self.assertEqual(extract_year("武大16年数据.pdf"), "2016")
        self.assertEqual(extract_year("中山19年.xlsx"), "2019")
        self.assertEqual(extract_year("华科21年.pdf"), "2021")

    # ── 过滤：不应误匹配 ──────────────────────────────────────────────

    def test_filter_score_50_fen(self):
        """"50分"不应被识别为年份."""
        self.assertIsNone(extract_year("复试分数线50分.pdf"))

    def test_filter_score_100_fen(self):
        self.assertIsNone(extract_year("总分100分.xlsx"))

    def test_filter_count_25_people(self):
        """"25人"不应被识别为年份."""
        self.assertIsNone(extract_year("招生计划25人.pdf"))

    def test_filter_score_75(self):
        """"英语75"仅数字且无量词，但 75 不在 15-29 范围内 → None."""
        self.assertIsNone(extract_year("英语75.xlsx"))

    def test_filter_high_score_142(self):
        """"142" 含 42 子串但前面有数字 → 不匹配."""
        self.assertIsNone(extract_year("数学142分.png"))

    def test_filter_page_number_30(self):
        """"30 页" → 30 不在 15-29 范围内."""
        self.assertIsNone(extract_year("目录30页.pdf"))

    # ── 优先级：4 位数优先 ────────────────────────────────────────────

    def test_4digit_over_2digit(self):
        """同时存在 4 位数和 2 位数时，优先返回 4 位数."""
        self.assertEqual(extract_year("山大2024复试名单22人.xlsx"), "2024")


class TestSafeMove(unittest.TestCase):
    """测试安全移动."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_move_new_file(self):
        src = self.tmp / "source.txt"
        src.write_text("hello", encoding="utf-8")
        dst_dir = self.tmp / "dest"
        result = safe_move(src, dst_dir, "target.txt")
        self.assertTrue(result.exists())
        self.assertEqual(result.read_text(), "hello")
        self.assertFalse(src.exists())

    def test_move_duplicate_content_skips(self):
        src = self.tmp / "dup_test.txt"
        src.write_text("same", encoding="utf-8")
        dst_dir = self.tmp / "dest"
        dst_dir.mkdir()
        existing = dst_dir / "dup_test.txt"
        existing.write_text("same", encoding="utf-8")

        result = safe_move(src, dst_dir, "dup_test.txt")
        # 内容相同 → 跳过，返回已有文件
        self.assertEqual(result, existing)
        self.assertTrue(src.exists())  # 源文件未被移动

    def test_move_name_conflict_renames(self):
        src = self.tmp / "conflict.txt"
        src.write_text("new content", encoding="utf-8")
        dst_dir = self.tmp / "dest"
        dst_dir.mkdir()
        existing = dst_dir / "conflict.txt"
        existing.write_text("old content", encoding="utf-8")

        result = safe_move(src, dst_dir, "conflict.txt")
        self.assertTrue(result.exists())
        self.assertNotEqual(result.name, "conflict.txt")  # 自动重命名
        self.assertFalse(src.exists())


class TestOrganizeSingle(unittest.TestCase):
    """测试单个文件归档."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raw = self.tmp / "raw"
        self.raw.mkdir()
        self.org = self.tmp / "organized"
        self.uncl = self.tmp / "unclassified"
        # 使用最小的临时映射
        self.map_file = self.tmp / "school_map.json"
        self.map_file.write_text(
            '{"Test_University": ["测试大学", "testu"]}', encoding="utf-8"
        )
        self.classifier = __import__("src.organizer", fromlist=["SchoolClassifier"]).SchoolClassifier(self.map_file)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_classified_file_moved(self):
        src = self.raw / "测试大学_2025_成绩.xlsx"
        src.write_text("dummy", encoding="utf-8")
        result = organize_single(src, self.classifier, self.org, self.uncl)
        self.assertIsNotNone(result)
        self.assertTrue(result.exists())
        self.assertIn("Test_University", str(result))
        self.assertIn("2025", str(result))

    def test_unclassified_file_moved(self):
        src = self.raw / "unknown_2024.pdf"
        src.write_text("dummy", encoding="utf-8")
        result = organize_single(src, self.classifier, self.org, self.uncl)
        self.assertIsNone(result)
        uncl_file = self.uncl / "unknown_2024.pdf"
        self.assertTrue(uncl_file.exists())

    def test_file_renamed_with_school_prefix(self):
        src = self.raw / "testu_2024.xlsx"
        src.write_text("dummy", encoding="utf-8")
        result = organize_single(src, self.classifier, self.org, self.uncl)
        self.assertIsNotNone(result)
        self.assertTrue(result.name.startswith("Test_University_"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
