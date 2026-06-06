"""从 马原/毛概/习思想 docx 中提取结构化题库，分别输出到 data/"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from docx import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 配置：每个文件的名称、科目、输出路径 ──────────────────────────────
FILES = [
    {
        "docx": "马原题目.docx",
        "subject": "马克思主义基本原理",
        "out": "data/mayuan.json",
    },
    {
        "docx": "毛概题目.docx",
        "subject": "毛泽东思想和中国特色社会主义理论体系概论",
        "out": "data/maogai.json",
    },
    {
        "docx": "习思想题目.docx",
        "subject": "习近平新时代中国特色社会主义思想概论",
        "out": "data/xsixiang.json",
    },
]

# ── 正则 ──────────────────────────────────────────────────────────────
# 生成 第一章 ~ 第十七章（汉字数字）
_CN_NUMS = ["一","二","三","四","五","六","七","八","九","十","十一","十二","十三","十四","十五","十六","十七"]
_CH_NAMES = [f"第{n}章" for n in _CN_NUMS]
CHAPTER_PATTERN = re.compile(r"^(导论|" + "|".join(_CH_NAMES) + r"|绪论|结束语|结语|前言)")
INLINE_CHAPTER = re.compile(r"(导论|" + "|".join(_CH_NAMES) + ")")
TYPE_KEYWORDS = ["名词解释", "简答题", "论述题", "辨析题", "案例分析题", "材料分析题", "课后思考题"]
Q_NUM_PATTERN = re.compile(r"^[\s]*(\d+)[.、．\s]")
SKIP_PATTERNS = [
    re.compile(r"背诵资料"),
    re.compile(r"这部分内容简单看一下即可"),
    re.compile(r"后面的内容"),
    re.compile(r"直接写到"),
    re.compile(r"少的内容给大家放在后面"),
]


def _is_chapter(text: str) -> str | None:
    """检测是否为章节标题。返回匹配到的章节名（短名），否则返回 None."""
    t = text.strip()
    m = CHAPTER_PATTERN.match(t)
    if m:
        return m.group(1)
    # 只匹配行首附近的章节标记（如 '23版《新思想》第五章——名解'），
    # 排除 '第四章少的内容给大家放在后面啦' 这种误匹配
    m = INLINE_CHAPTER.search(t)
    if m:
        start = m.start()
        matched = m.group(1)
        end = m.end()
        if start <= 20:
            prefix = t[:start]
            suffix = t[end:end+3]  # 章节名后的 3 个字符
            # 章节名前应是开头、书名号、或「版」
            valid_prefix = not prefix or any(prefix.endswith(c) for c in ("《", "》", "版", " "))
            # 章节名后应有 ——、）、/、或行尾（排除「少的内容给大家」等中文字句）
            valid_suffix = not suffix or any(suffix.startswith(c) for c in ("—", "）", ")", "/", "名", "简", "论", "课", "＋", "+", " "))
            if valid_prefix and valid_suffix:
                return matched
    return None


def _is_type(text: str) -> str | None:
    t = text.strip()
    for kw in TYPE_KEYWORDS:
        if kw in t:
            return kw
    return None


def _is_question_start(text: str) -> bool:
    return bool(Q_NUM_PATTERN.match(text.strip()))


def _strip_qnum(text: str) -> str:
    return Q_NUM_PATTERN.sub("", text, count=1).strip()


def _should_skip(text: str) -> bool:
    return any(p.search(text) for p in SKIP_PATTERNS)


def parse_docx(docx_path: Path, subject: str) -> list[dict]:
    doc = Document(docx_path)
    entries: list[dict] = []
    current_chapter = "导论"
    current_type = "名词解释"
    current_question = ""
    current_answer_parts: list[str] = []
    collecting_answer = False

    for para in doc.paragraphs:
        raw = para.text.strip()
        if not raw or _should_skip(raw):
            continue

        # 1. 章节检测
        chapter_detected = _is_chapter(raw)
        if chapter_detected:
            if current_question and collecting_answer:
                _save_entry(entries, subject, current_chapter, current_type,
                            current_question, current_answer_parts)
                current_question = ""
                current_answer_parts = []
                collecting_answer = False
            current_chapter = chapter_detected
            continue

        # 2. 题型检测
        detected_type = _is_type(raw)
        if detected_type:
            if current_question and collecting_answer:
                _save_entry(entries, subject, current_chapter, current_type,
                            current_question, current_answer_parts)
                current_question = ""
                current_answer_parts = []
                collecting_answer = False
            current_type = detected_type
            continue

        # 3. 题目起始行
        if _is_question_start(raw):
            if current_question and collecting_answer:
                _save_entry(entries, subject, current_chapter, current_type,
                            current_question, current_answer_parts)
            current_question = _strip_qnum(raw)
            current_answer_parts = []
            collecting_answer = True
            continue

        # 4. 答案内容
        if collecting_answer:
            current_answer_parts.append(raw)

    # 最后一题
    if current_question and collecting_answer:
        _save_entry(entries, subject, current_chapter, current_type,
                    current_question, current_answer_parts)

    return entries


def _save_entry(
    entries: list[dict],
    subject: str,
    chapter: str,
    qtype: str,
    question: str,
    answer_parts: list[str],
) -> None:
    answer_text = "\n".join(answer_parts).strip()
    entries.append({
        "subject": subject,
        "chapter": chapter,
        "type": qtype,
        "question": question,
        "answer": answer_text,
    })


def main() -> None:
    for cfg in FILES:
        docx_path = PROJECT_ROOT / cfg["docx"]
        if not docx_path.exists():
            print(f"[SKIP] {docx_path} 不存在")
            continue
        data = parse_docx(docx_path, cfg["subject"])
        out_path = PROJECT_ROOT / cfg["out"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 统计输出
        print(f"[OK] {cfg['docx']} → {cfg['out']}  ({len(data)} 条)")
        stats: dict[str, int] = {}
        for d in data:
            key = f"{d['chapter']} / {d['type']}"
            stats[key] = stats.get(key, 0) + 1
        for k, c in sorted(stats.items()):
            print(f"      {k}: {c} 题")

    print("\n[OK] 全部处理完成")


if __name__ == "__main__":
    main()
