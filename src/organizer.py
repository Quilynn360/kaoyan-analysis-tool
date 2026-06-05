"""数据智能分类脚本：自动识别学校归属并按 学校/年份 归档文件

工作流程：
    1. 遍历 ``data/raw/`` 目录下的所有文件
    2. 加载 ``config/school_map.json`` 进行学校关键词匹配
    3. 从文件名中提取年份
    4. 移动到 ``data/organized/{学校}/{年份}/`` 并重命名
    5. 无法识别的文件移入 ``data/unclassified/``
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

# ── 路径常量 ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHOOL_MAP = PROJECT_ROOT / "config" / "school_map.json"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_ORGANIZED_DIR = PROJECT_ROOT / "data" / "organized"
DEFAULT_UNCLASSIFIED_DIR = PROJECT_ROOT / "data" / "unclassified"

# ── 年份正则 ────────────────────────────────────────────────────────────
# 4 位数年份：1900-2099（优先匹配）
YEAR_4D_PATTERN = re.compile(r"(19\d{2}|20[0-2]\d)")

# 2 位数年份：仅在无 4 位数时尝试
# 约束条件：
#   - 独立两位数（前后无数字）
#   - 后面不紧跟"分/人/名/个/位"等非年份量词（避免误判分数/人数）
#   - 数值范围 15-29（对应 2015-2029，覆盖考研数据常见区间）
YEAR_2D_PATTERN = re.compile(
    r"(?<!\d)"
    r"(1[5-9]|2[0-9])"           # 15-29
    r"(?!\d)"
    r"(?!\s*[分人名个位科门])"    # 排除分数/人数的量词后缀
)


# ═══════════════════════════════════════════════════════════════════════
# 核心类
# ═══════════════════════════════════════════════════════════════════════

class SchoolClassifier:
    """学校分类器，加载关键词映射并提供匹配方法."""

    def __init__(self, map_path: str | Path = DEFAULT_SCHOOL_MAP) -> None:
        self.map_path = Path(map_path)
        self._data: dict[str, list[str]] | None = None

    @property
    def data(self) -> dict[str, list[str]]:
        if self._data is None:
            self._data = self._load()
        return self._data

    def _load(self) -> dict[str, list[str]]:
        if not self.map_path.exists():
            raise FileNotFoundError(f"学校映射文件不存在: {self.map_path}")
        with open(self.map_path, encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)
        result: dict[str, list[str]] = {}
        for school, keywords in raw.items():
            if isinstance(keywords, str):
                keywords = [keywords]
            result[school] = [str(k).strip() for k in keywords]
        return result

    def classify(self, name: str) -> str | None:
        """根据文件名返回匹配的学校键名，匹配不到返回 None."""
        for school, keywords in self.data.items():
            for kw in keywords:
                if kw in name:
                    return school
        return None


def extract_year(name: str) -> str | None:
    """从文件名中提取年份，失败返回 None.

    优先级：
        1. 4 位数年份（如 2024、2025），直接返回
        2. 2 位数年份（如 22、23），转换为 20xx
           * 15-29 → 2015-2029
        3. 均未匹配 → None

    过滤规则：
        - 避免将分数（50、100）、人数等误判为年份
        - 2 位数：只匹配 15-29 范围，且后面不跟"分/人/名/个"等量词
    """
    # 1. 优先匹配 4 位数
    m4 = YEAR_4D_PATTERN.findall(name)
    if m4:
        year = m4[0]
        print(f"  识别到的年份：{year}")
        return year

    # 2. 回退匹配 2 位数（正则已内建量词过滤 + 范围限制）
    m2 = YEAR_2D_PATTERN.findall(name)
    if m2:
        raw = m2[0]
        year_4d = f"20{raw}"
        print(f"  识别到的年份：{raw} → {year_4d}")
        return year_4d

    print(f"  识别到的年份：无")
    return None


def safe_move(src: Path, dst_dir: Path, new_name: str) -> Path:
    """安全移动文件，检测目标是否存在以避免覆盖.

    Parameters
    ----------
    src : Path
        源文件路径.
    dst_dir : Path
        目标目录.
    new_name : str
        新文件名（不含路径）.

    Returns
    -------
    Path
        移动后文件的实际路径.

    Raises
    ------
    FileExistsError
        目标文件已存在且内容不同（同名时自动加后缀）.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / new_name
    stem = dst.stem
    suffix = dst.suffix
    counter = 1

    while dst.exists():
        # 内容完全相同 → 跳过
        if src.read_bytes() == dst.read_bytes():
            print(f"  [SKIP] 文件已存在且内容相同: {dst}")
            return dst
        dst = dst_dir / f"{stem}_dup{counter}{suffix}"
        counter += 1

    shutil.move(str(src), str(dst))
    return dst


def organize_single(
    file_path: Path,
    classifier: SchoolClassifier,
    organized_root: Path = DEFAULT_ORGANIZED_DIR,
    unclassified_root: Path = DEFAULT_UNCLASSIFIED_DIR,
) -> Path | None:
    """对单个文件进行分类、归档.

    Returns
    -------
    Path | None
        移动后的路径；无法识别时返回 None 且文件被放入 unclassified.
    """
    school = classifier.classify(file_path.name)

    if school is None:
        unclassified_root.mkdir(parents=True, exist_ok=True)
        try:
            safe_move(file_path, unclassified_root, file_path.name)
        except Exception as e:
            print(f"  [ERROR] 移动失败: {file_path} → {e}")
        return None

    year = extract_year(file_path.name) or "unknown_year"
    dst_dir = organized_root / school / year

    new_name = f"{school}_{file_path.name}"
    try:
        dst = safe_move(file_path, dst_dir, new_name)
        print(f"  [OK] {file_path.name} → {dst.relative_to(organized_root.parent)}")
        return dst
    except Exception as e:
        print(f"  [ERROR] 移动失败: {file_path} → {e}")
        return None


def organize_all(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    school_map_path: str | Path = DEFAULT_SCHOOL_MAP,
    organized_root: str | Path = DEFAULT_ORGANIZED_DIR,
    unclassified_root: str | Path = DEFAULT_UNCLASSIFIED_DIR,
) -> dict[str, int]:
    """批量归档 data/raw/ 下所有文件.

    Returns
    -------
    dict
        ``{"organized": int, "unclassified": int, "errors": int}``
    """
    raw = Path(raw_dir)
    if not raw.exists():
        raise FileNotFoundError(f"原始数据目录不存在: {raw}")

    classifier = SchoolClassifier(school_map_path)

    counts: dict[str, int] = {"organized": 0, "unclassified": 0, "errors": 0}

    files = [f for f in sorted(raw.iterdir()) if f.is_file()]
    if not files:
        print("[WARN] data/raw/ 目录为空，无文件需要分类。")
        return counts

    print(f"开始分类归档 {len(files)} 个文件 ...\n")

    for file_path in files:
        result = organize_single(file_path, classifier, Path(organized_root), Path(unclassified_root))
        if result is None:
            counts["unclassified"] += 1
        else:
            counts["organized"] += 1

    # 打印汇总
    unclassified_dir = Path(unclassified_root)
    if unclassified_dir.exists() and any(unclassified_dir.iterdir()):
        print(f"\n[WARN] 以下文件无法识别学校归属，已移入 {unclassified_dir}：")
        for f in sorted(unclassified_dir.iterdir()):
            print(f"    {f.name}")

    print(f"\n{'='*40}")
    print(f"  分类完成: organized={counts['organized']}, "
          f"unclassified={counts['unclassified']}")
    print(f"{'='*40}")

    return counts


# ═══════════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    """命令行入口。"""
    import argparse

    parser = argparse.ArgumentParser(description="考研数据智能分类归档工具")
    parser.add_argument(
        "--raw-dir", default=str(DEFAULT_RAW_DIR),
        help="原始数据目录（默认: data/raw）"
    )
    parser.add_argument(
        "--map", default=str(DEFAULT_SCHOOL_MAP),
        help="学校关键词映射 JSON（默认: config/school_map.json）"
    )
    parser.add_argument(
        "--organized-dir", default=str(DEFAULT_ORGANIZED_DIR),
        help="归档根目录（默认: data/organized）"
    )
    parser.add_argument(
        "--unclassified-dir", default=str(DEFAULT_UNCLASSIFIED_DIR),
        help="未识别文件目录（默认: data/unclassified）"
    )

    args = parser.parse_args()
    organize_all(
        raw_dir=args.raw_dir,
        school_map_path=args.map,
        organized_root=args.organized_dir,
        unclassified_root=args.unclassified_dir,
    )


if __name__ == "__main__":
    main()
