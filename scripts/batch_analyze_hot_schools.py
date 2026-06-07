"""批量分析热门高校并持久化分析结果为静态 JSON 文件"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

# 确保能找到 src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.processor import KaoYanAnalyzer, _parse_pdf_table
from src.report_generator import build_report_data

# ── 精选高校（按数据完整度排序） ────────────────────────────────────
HOT_SCHOOLS = [
    ("Sun_Yat_sen_University", "中山大学"),
    ("Xi_an_Jiaotong_University", "西安交通大学"),
    ("Sichuan_University", "四川大学"),
    ("Nanjing_Normal_University", "南京师范大学"),
    ("Beijing_University_of_Chemical_Technology", "北京化工大学"),
]

ORGANIZED = PROJECT_ROOT / "data" / "organized"
OUTPUT_DIR = PROJECT_ROOT / "public" / "data"


def load_school_data(school_dir: str) -> pd.DataFrame:
    """加载指定学校的所有可解析 PDF 数据."""
    base = ORGANIZED / school_dir
    frames: list[pd.DataFrame] = []
    for yr_dir in sorted(base.iterdir()):
        if not yr_dir.is_dir():
            continue
        for f in sorted(yr_dir.iterdir()):
            try:
                df = _parse_pdf_table(f)
                df["year"] = int(yr_dir.name)
                frames.append(df)
            except Exception as e:
                print(f"    [SKIP] {f.name}: {e}")
    if not frames:
        raise ValueError(f"未加载到任何数据: {school_dir}")
    return pd.concat(frames, ignore_index=True)


def analyze_school(school_dir: str, school_cn: str) -> dict:
    """对一所学校执行全量分析，返回序列化结果."""
    print(f"  [ANALYZE] {school_cn} ({school_dir})...")
    df = load_school_data(school_dir)
    print(f"    加载 {len(df)} 条记录，年份: {sorted(df['year'].unique())}")

    analyzer = KaoYanAnalyzer(school=school_cn)
    analyzer._raw = df
    analyzer._long = analyzer._to_long(df)
    analyzer._clean = df
    analyzer._clean_long = analyzer._long

    t0 = time.time()
    analyzer.compute_all()
    elapsed = time.time() - t0
    print(f"    分析完成 ({elapsed:.1f}s)")

    # 构建报告数据字典（序列化友好）
    data = build_report_data(analyzer)
    data["_meta"] = {
        "school_dir": school_dir,
        "school_cn": school_cn,
        "total_records": len(df),
        "years": sorted(int(y) for y in df["year"].unique()),
        "analysis_time_s": round(elapsed, 1),
    }
    return data


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"开始批量分析 {len(HOT_SCHOOLS)} 所高校...\n")

    for school_dir, school_cn in HOT_SCHOOLS:
        try:
            result = analyze_school(school_dir, school_cn)
            out_path = OUTPUT_DIR / f"{school_dir}_analysis.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  [SAVED] {out_path}\n")
        except Exception as e:
            print(f"  [ERROR] {school_cn}: {e}\n")

    print(f"全部完成！结果保存在 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
