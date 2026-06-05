"""tests/test_processor.py —— 核心统计引擎单元测试

运行方式：
    python -m unittest tests/test_processor.py -v
    （或） python tests/test_processor.py
"""

import os
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.processor import (
    calculate_pearson,
    calculate_beta,
    calculate_z_score,
    dea_efficiency,
    quantile_regression,
    calculate_deviation_scores,
    monte_carlo_simulation,
    calculate_statistics,
    generate_report,
    _to_long,
)


def make_wide_df(n=20, years=None) -> pd.DataFrame:
    """生成宽格式模拟数据."""
    if years is None:
        years = [2022, 2023, 2024]
    rows = []
    for i, y in enumerate(years):
        for _ in range(n):
            rows.append({
                "姓名": f"考生_{y}_{_}",
                "总分": round(300 + np.random.randn() * 40, 1),
                "政治": round(60 + np.random.randn() * 10, 1),
                "英语": round(55 + np.random.randn() * 12, 1),
                "业务课一": round(110 + np.random.randn() * 20, 1),
                "业务课二": round(105 + np.random.randn() * 18, 1),
                "year": y,
            })
    return pd.DataFrame(rows)


import numpy as np


class TestToLong(unittest.TestCase):
    """测试宽格式→长格式转换."""

    def test_wide_to_long(self):
        wide = make_wide_df(5)
        long_df = _to_long(wide)
        self.assertIn("subject", long_df.columns)
        self.assertIn("score", long_df.columns)
        # 4 个科目 × 5*3 行 = 60
        self.assertEqual(len(long_df), 4 * len(wide))

    def test_long_passthrough(self):
        orig = pd.DataFrame({"year": [2024], "subject": ["数学"], "score": [150]})
        result = _to_long(orig)
        self.assertEqual(len(result), 1)

    def test_no_recognizable_columns(self):
        empty = pd.DataFrame({"foo": [1]})
        with self.assertRaises(ValueError):
            _to_long(empty)


class TestCalculatePearson(unittest.TestCase):
    """测试 Pearson 相关系数."""

    def test_returns_expected_keys(self):
        df = make_wide_df(50)
        result = calculate_pearson(df)
        self.assertIn("r", result)
        self.assertIn("p", result)
        self.assertIn("pairs", result)

    def test_diagonal_is_one(self):
        df = make_wide_df(30)
        result = calculate_pearson(df)
        for col in result["r"].columns:
            self.assertEqual(result["r"].loc[col, col], 1.0)


class TestCalculateBeta(unittest.TestCase):
    """测试 β 系数."""

    def test_beta_shape(self):
        df = _to_long(make_wide_df(20, years=[2022, 2023, 2024]))
        result = calculate_beta(df)
        self.assertIn("subject", result.columns)
        self.assertIn("beta", result.columns)
        self.assertGreater(len(result), 0)


class TestZScore(unittest.TestCase):
    """测试 Z-score 标准化."""

    def test_zscore_mean_zero(self):
        df = _to_long(make_wide_df(30))
        result = calculate_z_score(df)
        for subj, grp in result.groupby("subject"):
            mean_z = grp["z_score"].mean()
            self.assertAlmostEqual(mean_z, 0, places=1)


class TestDEA(unittest.TestCase):
    """测试 DEA 效率."""

    def test_efficiency_between_0_and_1(self):
        df = _to_long(make_wide_df(20))
        result = dea_efficiency(df)
        self.assertTrue((result["efficiency"] >= 0).all())
        self.assertTrue((result["efficiency"] <= 1).all())


class TestQuantileRegression(unittest.TestCase):
    """测试分位数回归."""

    def test_returns_dataframe(self):
        df = _to_long(make_wide_df(20, years=[2022, 2023, 2024]))
        result = quantile_regression(df)
        self.assertIsInstance(result, pd.DataFrame)
        if not result.empty:
            self.assertIn("quantile", result.columns)
            self.assertIn("coef_year", result.columns)


class TestDeviationScores(unittest.TestCase):
    """测试偏差分校准."""

    def test_calibrated_total_added(self):
        df = make_wide_df(10)
        result = calculate_deviation_scores(df)
        self.assertIn("calibrated_total", result.columns)
        self.assertEqual(len(result), len(df))

    def test_custom_weights(self):
        df = make_wide_df(5)
        weights = {"政治": 1.0, "英语": 1.0, "业务课一": 1.5, "业务课二": 1.2}
        result = calculate_deviation_scores(df, subject_weights=weights)
        self.assertIn("calibrated_total", result.columns)


class TestMonteCarlo(unittest.TestCase):
    """测试蒙特卡罗模拟."""

    def test_returns_summary(self):
        df = _to_long(make_wide_df(30))
        result = monte_carlo_simulation(df, years=5, n_simulations=500)
        self.assertIn("summary", result)
        if result["summary"]:
            for v in result["summary"].values():
                self.assertIn("mean_rate", v)
                self.assertIn("ci_low", v)
                self.assertIn("ci_high", v)


class TestCalculateStatistics(unittest.TestCase):
    """测试一站式统计分析."""

    def test_all_keys_present(self):
        df = make_wide_df(30)
        result = calculate_statistics(df)
        expected_keys = {"descriptive", "pearson", "beta", "z_score", "dea", "quantile_regression", "deviation"}
        self.assertEqual(set(result.keys()), expected_keys)


class TestGenerateReport(unittest.TestCase):
    """测试报告生成."""

    def test_report_generated(self):
        df = make_wide_df(20)
        tmp_out = Path(PROJECT_ROOT / "output" / "test_report.md")
        report = generate_report(df, output_path=tmp_out)
        self.assertIsInstance(report, str)
        self.assertTrue(tmp_out.exists())
        self.assertIn("考研初试数据量化分析报告", report)
        tmp_out.unlink(missing_ok=True)
        tmp_out.parent.rmdir() if tmp_out.parent.exists() else None


if __name__ == "__main__":
    unittest.main(verbosity=2)
