#!/usr/bin/env python
"""考研初试数据量化分析系统 — 全自动入口

用法：
    python main.py
    python main.py --school Nanjing_Normal_University
    python main.py --school Fudan_University --years 2023 2024 2025 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.processor import KaoYanAnalyzer
from src.report_generator import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="考研初试数据量化分析系统")
    parser.add_argument("--school", default="Nanjing_Normal_University",
                        help="学校名称（对应 data/organized/ 下的目录名）")
    parser.add_argument("--years", type=int, nargs="*", default=None,
                        help="年份列表（默认自动检测）")
    parser.add_argument("--output", default="output/output_prompt_for_llm.md",
                        help="报告输出路径")
    parser.add_argument("--skip-filter", action="store_true",
                        help="跳过非普通计划过滤（默认过滤）")
    args = parser.parse_args()

    print(f"{'='*55}")
    print(f"  考研初试数据量化分析系统")
    print(f"{'='*55}")
    print(f"  学校: {args.school}")
    print(f"  年份: {args.years or '自动检测'}")
    print()

    # 1. 初始化分析器
    analyzer = KaoYanAnalyzer(school=args.school)

    # 2. 加载数据
    try:
        analyzer.load_data(years=args.years)
    except Exception as e:
        print(f"[ERROR] 数据加载失败: {e}")
        sys.exit(1)

    # 3. 过滤非普通计划
    if not args.skip_filter:
        try:
            analyzer.filter_regular_plans()
        except Exception as e:
            print(f"[ERROR] 过滤失败: {e}")
            sys.exit(1)

    # 4. 全量计算
    try:
        analyzer.compute_all()
    except Exception as e:
        print(f"[ERROR] 计算失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 5. 生成报告
    try:
        generate_report(analyzer, output_path=args.output)
    except Exception as e:
        print(f"[ERROR] 报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n  [OK] 全流程完成！")
    print(f"  报告文件: {args.output}")


if __name__ == "__main__":
    main()
