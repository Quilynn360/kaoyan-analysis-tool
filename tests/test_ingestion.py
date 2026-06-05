"""tests/test_ingestion.py —— DataIngestor 多格式摄入单元测试

运行方式：
    python -m pytest tests/test_ingestion.py -v
    （或） python tests/test_ingestion.py
"""

import os
import sys
import unittest
from pathlib import Path

import pandas as pd

# 确保能找到 src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import DataIngestor, preprocess_image, STANDARD_COLUMNS


class TestNormalize(unittest.TestCase):
    """测试 _normalize 列名映射逻辑."""

    def setUp(self):
        self.ingestor = DataIngestor()

    def test_full_chinese_columns(self):
        """标准列名完全匹配."""
        raw = pd.DataFrame([
            ["张三", 350, 70, 65, 120, 95],
            ["李四", 360, 75, 68, 125, 92],
        ], columns=["姓名", "总分", "政治", "英语", "业务课一", "业务课二"])
        result = self.ingestor._normalize(raw)
        self.assertEqual(list(result.columns[:6]), STANDARD_COLUMNS)
        self.assertEqual(result.iloc[0]["姓名"], "张三")
        self.assertEqual(result.iloc[0]["总分"], 350)

    def test_english_column_aliases(self):
        """英文列名应映射到标准列."""
        raw = pd.DataFrame([
            ["张三", 350, 70, 65, 120, 95],
        ], columns=["name", "total", "politics", "english", "subject1", "subject2"])
        result = self.ingestor._normalize(raw)
        self.assertEqual(list(result.columns[:6]), STANDARD_COLUMNS)
        self.assertEqual(result.iloc[0]["姓名"], "张三")

    def test_partial_columns_remain(self):
        """只有部分标准列时，缺失列填充 NA，未映射列保留."""
        raw = pd.DataFrame([
            ["张三", 350, 70],
        ], columns=["姓名", "总分", "政治"])
        result = self.ingestor._normalize(raw)
        self.assertEqual(result.iloc[0]["姓名"], "张三")
        self.assertEqual(result.iloc[0]["总分"], 350)
        self.assertTrue(result.iloc[0]["英语"] is pd.NA or pd.isna(result.iloc[0]["英语"]))

    def test_extra_columns_preserved(self):
        """未映射的列应保留在右侧."""
        raw = pd.DataFrame([
            ["张三", 350, "男", "北京"],
        ], columns=["姓名", "总分", "性别", "籍贯"])
        result = self.ingestor._normalize(raw)
        self.assertIn("性别", result.columns)
        self.assertIn("籍贯", result.columns)
        self.assertEqual(result.iloc[0]["性别"], "男")

    def test_empty_dataframe(self):
        """空 DataFrame 返回带有标准列的空 DataFrame."""
        raw = pd.DataFrame()
        result = self.ingestor._normalize(raw)
        self.assertEqual(list(result.columns), STANDARD_COLUMNS)
        self.assertTrue(result.empty)


class TestDetectAndLoad(unittest.TestCase):
    """测试 detect_and_load 调度逻辑."""

    def setUp(self):
        self.ingestor = DataIngestor()
        self.data_dir = PROJECT_ROOT / "data"
        self.data_dir.mkdir(exist_ok=True)

    def _create_excel(self, name: str) -> Path:
        path = self.data_dir / name
        df = pd.DataFrame([
            ["张三", 350, 70, 65, 120, 95],
            ["李四", 360, 75, 68, 125, 92],
        ], columns=["姓名", "总分", "政治", "英语", "业务课一", "业务课二"])
        df.to_excel(path, index=False)
        return path

    def test_excel_load(self):
        path = self._create_excel("test_excel.xlsx")
        try:
            result = self.ingestor.load_excel(path)
            self.assertEqual(len(result), 2)
            self.assertEqual(list(result.columns[:6]), STANDARD_COLUMNS)
            self.assertEqual(result.iloc[0]["姓名"], "张三")
        finally:
            path.unlink(missing_ok=True)

    def test_detect_and_load_excel(self):
        path = self._create_excel("test_detect.xlsx")
        try:
            result = self.ingestor.detect_and_load(path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result.iloc[1]["总分"], 360)
        finally:
            path.unlink(missing_ok=True)

    def test_unsupported_format(self):
        path = self.data_dir / "test.txt"
        path.write_text("dummy", encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                self.ingestor.detect_and_load(path)
        finally:
            path.unlink(missing_ok=True)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.ingestor.detect_and_load(self.data_dir / "nonexist.xlsx")

    def test_empty_data_dir(self):
        with self.assertRaises(ValueError):
            self.ingestor.load_data_dir(self.data_dir)

    def test_load_data_dir_skip_unsupported(self):
        # 创建一个支持的文件和一个不支持的文件
        xlsx = self._create_excel("tmp_data.xlsx")
        txt = self.data_dir / "tmp_note.txt"
        txt.write_text("hello", encoding="utf-8")
        try:
            result = self.ingestor.load_data_dir(self.data_dir)
            self.assertEqual(len(result), 2)
        finally:
            xlsx.unlink(missing_ok=True)
            txt.unlink(missing_ok=True)


class TestPreprocessImage(unittest.TestCase):
    """测试 preprocess_image 函数（需要 data/ 下有样本图片）. """

    def test_image_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            preprocess_image("data/nonexist.png")

    def test_image_not_a_table(self):
        """传入一个不含表格的纯色图片应抛出 ValueError."""
        path = PROJECT_ROOT / "data" / "blank.png"
        if not path.exists():
            self.skipTest("样本文件 data/blank.png 不存在")
        with self.assertRaises(ValueError):
            preprocess_image(path)


class TestIntegration(unittest.TestCase):
    """端到端集成测试：从 data/ 加载并验证标准输出."""

    def setUp(self):
        self.ingestor = DataIngestor()
        self.data_dir = PROJECT_ROOT / "data"

    def test_load_all_from_data(self):
        """处理 data/ 下所有可用文件."""
        supported_count = sum(
            1 for f in self.data_dir.iterdir()
            if f.suffix.lower() in {
                ".xlsx", ".xls", ".pdf", ".docx", ".doc",
                ".jpg", ".jpeg", ".png", ".bmp", ".tiff",
            }
        )
        if supported_count == 0:
            self.skipTest("data/ 目录下无样本文件")

        result = self.ingestor.load_data_dir(self.data_dir)
        self.assertIsInstance(result, pd.DataFrame)
        for col in STANDARD_COLUMNS:
            self.assertIn(col, result.columns, f"缺少标准列: {col}")
        self.assertGreater(len(result), 0, "加载结果为空")


if __name__ == "__main__":
    unittest.main(verbosity=2)
