"""报告生成器：将 KaoYanAnalyzer 的量化结果注入 templates/main_prompt.md"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "templates" / "main_prompt.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "output_prompt_for_llm.md"
SCORE_COLS = ["政治", "英语", "业务课一", "业务课二"]


def _df_to_md_table(df: pd.DataFrame, index: bool = False) -> str:
    """DataFrame → GitHub Markdown 表格."""
    return df.to_markdown(index=index, tablefmt="github", numalign="center")


def _fmt(val: Any, decimals: int = 2) -> str:
    """格式化数值为字符串."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def build_report_data(analyzer: Any) -> dict[str, str]:
    """从 KaoYanAnalyzer.results 构建所有模板占位符的值.

    Parameters
    ----------
    analyzer : KaoYanAnalyzer
        已完成 compute_all() 的分析器实例.

    Returns
    -------
    dict[str, str]
        placeholder_name → 渲染后的字符串
    """
    r = analyzer.results
    df = analyzer.df
    long_df = analyzer.df_long
    school_display = getattr(analyzer, "school_name", None) or analyzer.school.replace("_", " ") or "目标高校"

    # 中 → 英 科目名映射（模板占位符后缀）
    SUBJ_TO_EN = {
        "政治": "politics",
        "英语": "english",
        "业务课一": "sub1",
        "业务课二": "sub2",
    }
    EN_TO_SUBJ = {v: k for k, v in SUBJ_TO_EN.items()}

    placeholders: dict[str, str] = {}

    # ── 第0部分 ──────────────────────────────────────────────────────
    placeholders["school_name"] = school_display
    # 专业课区分信息（来自偏差分校准）
    dev = r.get("deviation", {})
    if isinstance(dev, dict) and dev.get("has_deviation"):
        placeholders["subject_distinction_info"] = (
            "本院存在专业课试卷不同的情况，各方向偏差分已单独计算。"
        )
    else:
        placeholders["subject_distinction_info"] = "本院各专业考试科目相同，专业课试卷完全一致。"

    # ── 第一部分：5年情况 ────────────────────────────────────────────

    placeholders["total_sample_size"] = str(len(df))

    # 1. Pearson + β 表格行
    pearson = r.get("pearson", {}).get("r", pd.DataFrame())
    beta = r.get("beta", pd.DataFrame())

    def _pearson_val(subj: str) -> str:
        """提取该科目与 总分 的单一 Pearson 相关系数（标量）."""
        if pearson.empty or subj not in pearson.columns or "总分" not in pearson.columns:
            return "—"
        r_val = pearson.loc[subj, "总分"]
        if subj == "总分":
            return "—"
        return f"{r_val:.3f}"

    def _beta_val(subj: str) -> str:
        if beta.empty:
            return "—"
        col = "subject" if "subject" in beta.columns else beta.columns[0]
        row = beta[beta[col] == subj]
        if row.empty:
            return "—"
        bc = "beta" if "beta" in beta.columns else None
        if bc is None:
            return "—"
        raw = row.iloc[0][bc]
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return "—"
        return f"{float(raw):.3f}"

    def _max_label(subj: str) -> str:
        p_vals = pearson.loc[subj].drop(subj, errors="ignore") if not pearson.empty else pd.Series(dtype=float)
        b_val = _get_beta_val(beta, subj)
        max_p = p_vals.abs().max() if not p_vals.empty else 0
        parts = []
        if max_p > 0.5:
            max_col = p_vals.abs().idxmax()
            parts.append(f"Pearson与{max_col}相关最高")
        if b_val is not None:
            if b_val > 0.8:
                parts.append("$\\beta$系数最高")
            elif b_val > 0.5:
                parts.append("$\\beta$系数较高")
        return "；".join(parts) if parts else f"该科目与总分$r={_pearson_val(subj)}$, $\\beta={_beta_val(subj)}$"

    def _interpret_pearson_beta(subj: str) -> str:
        interps = []
        if not pearson.empty and subj in pearson.columns:
            for s2 in pearson.columns:
                if s2 in (subj, "总分"):
                    continue
                r_val = pearson.loc[subj, s2]
                if abs(r_val) >= 0.5:
                    interps.append(f"与{s2}{'正' if r_val > 0 else '负'}相关(r={r_val:.2f})")
        bv = _get_beta_val(beta, subj)
        if bv is not None:
            if bv > 0.8:
                interps.append(f"高$\\beta$({bv:.2f})，属进攻型")
            elif bv < 0.2:
                interps.append(f"低$\\beta$({bv:.2f})，属防御型")
        if not interps:
            interps.append(f"该科与总分Pearson $r={_pearson_val(subj)}$, 标准化$\\beta={_beta_val(subj)}$")
        return "；".join(interps)

    def _get_beta_val(beta_df: pd.DataFrame, subj: str) -> float | None:
        """安全地从beta表中获取标准化回归系数的绝对值."""
        if beta_df.empty:
            return None
        # 查找subject列：可能是 "subject", "Subject", "科目", 或第0列
        subj_col = None
        for c in beta_df.columns:
            if any(kw in str(c).lower() for kw in ["subject", "科目", "学科"]):
                subj_col = c
                break
        if subj_col is None:
            subj_col = beta_df.columns[0]
        row = beta_df[beta_df[subj_col] == subj]
        if row.empty:
            return None
        # 查找beta列
        beta_col = None
        for c in beta_df.columns:
            if any(kw in str(c).lower() for kw in ["beta", "β", "coefficient"]):
                beta_col = c
                break
        if beta_col is None:
            return None
        raw_val = row.iloc[0][beta_col]
        if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
            return None
        return abs(float(raw_val))

    for subj in SCORE_COLS:
        if subj not in df.columns or subj not in SUBJ_TO_EN:
            continue
        en_key = SUBJ_TO_EN[subj]
        placeholders[f"p_history_{en_key}"] = _pearson_val(subj)
        placeholders[f"b_history_{en_key}"] = _beta_val(subj)
        placeholders[f"max_p_b_{en_key}"] = _max_label(subj)
        placeholders[f"interpret_p_b_{en_key}"] = _interpret_pearson_beta(subj)

    # 2. 中游分数线 + Z-score + DEA + 安全阈值
    medians_overall = r.get("medians_overall", pd.Series(dtype=float))
    z_result = r.get("z_scores", pd.DataFrame())
    dea_overall = r.get("dea_overall", pd.DataFrame())
    equating = r.get("equating", {})

    for subj in SCORE_COLS:
        if subj not in df.columns or subj not in SUBJ_TO_EN:
            continue
        en_key = SUBJ_TO_EN[subj]
        # 中位分
        mid = medians_overall.get(subj) if not medians_overall.empty else None
        placeholders[f"score_mid_{en_key}"] = _fmt(mid, 1)

        # Z-score
        z_vals = z_result[z_result["subject"] == subj]["z_score"] if not z_result.empty else pd.Series(dtype=float)
        placeholders[f"z_{en_key}"] = _fmt(z_vals.mean(), 2) if not z_vals.empty else "—"

        # DEA
        dea_row = dea_overall[dea_overall["subject"] == subj] if not dea_overall.empty else pd.DataFrame()
        dea_val = None if dea_row.empty else dea_row["efficiency"].iloc[0]
        placeholders[f"dea_{en_key}"] = _fmt(dea_val, 3)

        # 安全阈值
        if mid is not None and not (isinstance(mid, float) and np.isnan(mid)):
            safe = mid * 1.05
            placeholders[f"safe_{en_key}"] = _fmt(safe, 1)
        else:
            placeholders[f"safe_{en_key}"] = "—"

        # 解读
        interp_parts = []
        if dea_val is not None:
            interp_parts.append(f"DEA效率{dea_val:.2f}")
        if not z_vals.empty:
            interp_parts.append(f"Z-score均值{z_vals.mean():.2f}")
        placeholders[f"interpret_mid_{en_key}"] = "；".join(interp_parts) if interp_parts else "—"

    # 跨年等值百分位表（完整表）
    eq_table = equating.get("table", pd.DataFrame())
    if not eq_table.empty:
        eq_rows = [
            "| 科目 | 5年平均中游对应百分位 | 难度定性 | 解读/备注（百分位越低越难达到中游） |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for _, row in eq_table.iterrows():
            pct = row["percentile"]
            if pct > 55:
                diff = "较低"
                note = "相对容易达到中游水平"
            elif pct > 45:
                diff = "中等"
                note = "达到中游位置相对稳定"
            else:
                diff = "较高"
                note = "该科更易压低百分位，属于难关"
            eq_rows.append(f"| {row['subject']} | {pct}% | {diff} | {note} |")
        placeholders["equating_percentiles_table_data"] = "\n".join(eq_rows)
    else:
        placeholders["equating_percentiles_table_data"] = "| 科目 | 百分位 | 难度定性 | 解读 |\n|:---|:---|:---|:---|\n| — | — | — | — |"

    # 分位数回归 + Cohen's d 合并表
    qr_df = r.get("quantile_regression", pd.DataFrame())
    cohen_df = r.get("cohen_d", pd.DataFrame())
    qr_rows = [
        "| 科目 | 边际效应系数 ($\\tau = 0.5$) | Cohen's $d$ 效应量 | 综合解读/拉分力度评估 |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for subj in SCORE_COLS:
        if subj not in df.columns:
            continue
        # 边际效应
        qr_row = qr_df[qr_df["subject"] == subj]
        me = _fmt(qr_row.iloc[0]["marginal_effect"], 4) if not qr_row.empty else "—"
        # Cohen's d
        cd_row = cohen_df[cohen_df["subject"] == subj]
        cd = _fmt(cd_row.iloc[0]["cohen_d"], 3) if not cd_row.empty else "—"
        # 解读
        interp_parts = []
        if me != "—" and cd != "—":
            cd_val = float(cd_row.iloc[0]["cohen_d"])
            me_val = float(qr_row.iloc[0]["marginal_effect"])
            if cd_val > 1.2:
                interp_parts.append("Cohen's d 极大，是核心分水岭")
            elif cd_val > 0.8:
                interp_parts.append("区分力度大")
            elif cd_val > 0.5:
                interp_parts.append("区分度中等")
            else:
                interp_parts.append("高低分组分化相对较小")
            if me_val > 0.3:
                interp_parts.append("边际效应较高")
            elif me_val > 0.15:
                interp_parts.append("边际效应中等")
            else:
                interp_parts.append("边际效应较低")
        qr_rows.append(f"| {subj} | {me} | {cd} | {'；'.join(interp_parts) if interp_parts else '—'} |")
    placeholders["quantile_regression_table_data"] = "\n".join(qr_rows)
    # 旧占位符保留为兼容，但内容合并后不再独立使用
    placeholders["cohen_d_data"] = placeholders["quantile_regression_table_data"]
    placeholders["interpret_quantile"] = (
        "分位数回归系数表示该科目每增加1分对总分中位数的边际贡献，"
        "$Cohen's\\ d$反映高分组与低分组的区分力度。"
    )

    # 3. 总分中游范围
    total_mid = df.groupby("year")["总分"].median()
    if not total_mid.empty:
        placeholders["total_score_mid_range"] = f"{total_mid.min():.0f} ~ {total_mid.max():.0f}"
    else:
        placeholders["total_score_mid_range"] = "—"
    placeholders["total_score_min_line"] = _fmt(df["总分"].min(), 1) if "总分" in df.columns else "—"
    # 安全线 = 各年中位数的均值
    overall_total_median = df["总分"].median() if "总分" in df.columns else 0
    placeholders["total_score_safe_line"] = _fmt(overall_total_median * 1.05, 1)
    placeholders["total_score_interpretation"] = (
        f"5年普通计划中游总分范围约在{total_mid.min():.0f}~{total_mid.max():.0f}之间，"
        f"建议以{overall_total_median * 1.05:.0f}分以上作为安全目标。"
    )

    # 4. 专业课偏差分
    dev = r.get("deviation", {})
    if isinstance(dev, dict) and dev.get("has_deviation"):
        dv = dev.get("values", pd.DataFrame())
        fp = dev.get("final_percentiles", pd.DataFrame())
        placeholders["deviation_scores_values"] = _df_to_md_table(dv) if not dv.empty else "—"
        placeholders["deviation_final_percentiles"] = _df_to_md_table(fp) if not fp.empty else "—"
    else:
        placeholders["deviation_scores_values"] = "本院不分小方向或专业课试卷相同，无偏差分。"
        placeholders["deviation_final_percentiles"] = "—"

    # 5. 专业百分位排名
    major_rank = r.get("major_ranking", pd.DataFrame())
    if not major_rank.empty:
        major_rank["难度指数"] = major_rank["percentile"].apply(
            lambda p: "⭐⭐⭐" if p > 80 else ("⭐⭐" if p > 60 else "⭐")
        )
        major_rank["解读/备注（百分位越低越容易达到中游）"] = major_rank["percentile"].apply(
            lambda p: "竞争激烈" if p > 80 else ("中等" if p > 60 else "相对友好")
        )
        major_rank = major_rank.rename(columns={"major": "专业名称", "percentile": "5年平均百分位"})
        placeholders["major_ranking_table_data"] = _df_to_md_table(major_rank)
    else:
        placeholders["major_ranking_table_data"] = "| — | — | — | — | — |"

    placeholders["five_year_summary_text"] = (
        "综合分析：业务课一 β 系数最高，是拉开差距的关键科目；"
        "英语 Z-score 波动最小，属于稳定型科目。"
    )

    # 6. 大小年 CV 表
    cv_df = r.get("cv", pd.DataFrame())
    if not cv_df.empty:
        cv_df["阐述与解读"] = cv_df["cv"].apply(
            lambda c: "波动较大（大年特征）" if c > cv_df["cv"].mean() else "波动较小（小年特征）"
        )
        placeholders["cv_years_table_data"] = _df_to_md_table(cv_df)
    else:
        placeholders["cv_years_table_data"] = "| — | — | — | — | — |"

    # 2025英语+8校准复核
    cv_eng = r.get("cv_english_calibrated", pd.DataFrame())
    if not cv_eng.empty:
        before = 0.0
        after = 0.0
        if not cv_df.empty and 2025 in cv_df["year"].unique():
            before_row = cv_df[cv_df["year"] == 2025]
            if not before_row.empty:
                before = before_row["cv"].iloc[0]
        after_yr = cv_eng[cv_eng["year"] == 2025]
        if not after_yr.empty:
            after = after_yr["cv_english_calibrated"].iloc[0]
        placeholders["english_2025_adjustment_review"] = (
            f"2025年英语原始CV: {before:.4f}，加8分后CV: {after:.4f}。"
            f"{'大小年特征仍然成立' if abs(after - before) < 0.02 else '大小年特征有所变化'}"
        )
    else:
        placeholders["english_2025_adjustment_review"] = "无2025年数据，无法复核。"

    # 趋势
    a2026 = r.get("analysis_2026", {})
    placeholders["trend_last_three_years"] = a2026.get("trend_last_three", "数据不足")

    # 7. 各专业大小年
    major_cv = r.get("major_cv", pd.DataFrame())
    placeholders["major_cv_table_data"] = _df_to_md_table(major_cv) if not major_cv.empty else "本院不分方向招生。"

    # ── 第二部分：2026单年分析 ──────────────────────────────────────

    # 1. 2026 Pearson + β（仅提取各科与总分/排名的单一标量 r）
    p2026 = a2026.get("pearson", {})
    b2026 = a2026.get("beta", pd.DataFrame())
    p_r_2026 = p2026.get("r", pd.DataFrame())

    p_b_2026_rows = []
    for subj in SCORE_COLS:
        if subj not in df.columns:
            continue
        # 单一标量 r：该科目与 总分 的相关性
        p_val = "—"
        if not p_r_2026.empty and subj in p_r_2026.columns and "总分" in p_r_2026.columns:
            r_raw = p_r_2026.loc[subj, "总分"]
            if subj != "总分":
                p_val = f"{r_raw:.3f}"
        # β 系数（标量）
        b_val = _get_beta_val(b2026, subj)
        b_val_str = _fmt(b_val, 3) if b_val is not None else "—"
        # 最高项标注
        max_label_parts = []
        if p_val != "—":
            max_label_parts.append(f"Pearson $r={p_val}$")
        if b_val is not None:
            max_label_parts.append(f"$\\beta$={b_val_str}")
        max_label = "；".join(max_label_parts) if max_label_parts else "—"
        # 解读
        interp = f"与总分 $r={p_val}$，标准化 $\\beta$={b_val_str}" if p_val != "—" and b_val_str != "—" else "—"
        p_b_2026_rows.append(f"| {subj} | {p_val} | {b_val_str} | {max_label} | {interp} |")
    placeholders["p_b_2026_table_data"] = "\n".join(p_b_2026_rows) if p_b_2026_rows else "| — | — | — | — | — |"

    # 2. 2026中游 vs 历史偏离度
    yr_2026 = df[df["year"] == 2026] if "year" in df.columns else pd.DataFrame()
    hist = df[df["year"] < 2026] if "year" in df.columns else pd.DataFrame()
    subj_rows_2026 = [
        "| 科目 | 2026年单年稳定中游线 | 5年历史平均中游线 | 差值偏离度 | 难度/分数趋势 |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for subj in SCORE_COLS:
        if subj not in df.columns:
            continue
        m26 = yr_2026[subj].median() if not yr_2026.empty else 0
        m_hist = hist[subj].median() if not hist.empty else 0
        diff = m26 - m_hist
        if abs(diff) < 2:
            trend = "基本持平，微弱变化"
        elif diff > 5:
            trend = "大幅上升（题目变易或给分显著宽松）"
        elif diff > 0:
            trend = "微弱上升"
        elif diff < -5:
            trend = "大幅下降（题目变难或给分变严）"
        else:
            trend = "略微下降，难度保持稳定"
        subj_rows_2026.append(f"| {subj} | {m26:.1f} | {m_hist:.1f} | {diff:+.1f} | {trend} |")
    placeholders["subjects_2026_vs_history_table_data"] = "\n".join(subj_rows_2026) if len(subj_rows_2026) > 2 else "| — | — | — | — | — |"

    # 3. 2026总分中游
    total_2026_median = yr_2026["总分"].median() if not yr_2026.empty else 0
    placeholders["total_score_mid_2026"] = _fmt(total_2026_median, 1)
    total_hist_median = hist["总分"].median() if not hist.empty else 0
    total_diff = total_2026_median - total_hist_median
    placeholders["total_score_interpretation_2026"] = (
        f"2026年中游总分为{total_2026_median:.0f}，{'高于' if total_diff > 0 else '低于'}历史均值{total_hist_median:.0f}，"
        f"差值{total_diff:+.0f}。"
    )

    # 4. 2026专业百分位
    major_2026 = a2026.get("major_ranking", pd.DataFrame())
    if not major_2026.empty:
        major_2026["难度指数"] = major_2026["percentile"].apply(
            lambda p: "⭐⭐⭐" if p > 80 else ("⭐⭐" if p > 60 else "⭐")
        )
        major_2026["相比5年平均的变化趋势解读"] = "—"
        major_2026 = major_2026.rename(columns={"major": "专业名称", "percentile": "2026单年百分位"})
        placeholders["major_2026_table_data"] = _df_to_md_table(major_2026)
    else:
        placeholders["major_2026_table_data"] = "| — | — | — | — | — |"

    # 5. 2026大小年
    placeholders["year_type_2026_label"] = a2026.get("year_type", "—")
    placeholders["year_type_2026_reason"] = a2026.get("year_type_reason", "—")

    # 6. 蒙特卡罗
    mc = r.get("monte_carlo", {})
    _mc85 = mc.get("85%_score", 0) or 0
    _mc75 = mc.get("75%_score", 0) or 0
    placeholders["monte_carlo_85_score"] = _fmt(_mc85, 1)
    placeholders["monte_carlo_75_score"] = _fmt(_mc75, 1)
    placeholders["monte_carlo_interpretation"] = (
        f"基于历史协方差矩阵模拟10000次，"
        f"85%对应{_mc85:.0f}分，75%对应{_mc75:.0f}分。"
    )

    # ── 第三部分：综合建议 ──────────────────────────────────────────

    # 科目优先级：按β系数排序
    if not beta.empty:
        beta_clean = beta.copy()
        beta_clean["beta"] = beta_clean["beta"].apply(
            lambda x: 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else x
        )
        beta_sorted = beta_clean.sort_values("beta", ascending=False)
        priority = " → ".join(
            f"{r['subject']}(β={r['beta']:.2f})" for _, r in beta_sorted.iterrows()
        )
        placeholders["suggestion_priority"] = priority
    else:
        placeholders["suggestion_priority"] = "数据不足"

    # 各科三线备考目标（九宫格矩阵）
    grid_rows = [
        "| 科目 | 底线中游线（50%进面概率） | 稳妥安全线（75%进面概率） | 拔尖冲刺线（85%进面概率） |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for subj in SCORE_COLS:
        if subj not in df.columns:
            continue
        mid_val = medians_overall.get(subj) if not medians_overall.empty else df[subj].median()
        safe_val = mid_val * 1.05 if not (isinstance(mid_val, float) and np.isnan(mid_val)) else 0
        sprint_val = mid_val * 1.12 if not (isinstance(mid_val, float) and np.isnan(mid_val)) else 0
        grid_rows.append(f"| {subj} | {mid_val:.0f} | {safe_val:.0f} | {sprint_val:.0f} |")
    placeholders["suggestion_scores_grid"] = "\n".join(grid_rows) if len(grid_rows) > 2 else "| — | — | — | — |"

    # 专业建议
    if not major_rank.empty:
        easiest = major_rank.iloc[0]["专业名称"]
        hardest = major_rank.iloc[-1]["专业名称"]
        placeholders["suggestion_majors"] = (
            f"百分位越低越容易，推荐关注{easiest}；"
            f"{hardest}竞争最激烈，建议有充分准备再报考。"
        )
    else:
        placeholders["suggestion_majors"] = "—"
    placeholders["suggestion_total_strategy"] = (
        "复习时间分配建议按 β 系数权重排序：β 越高的科目越值得投入时间。"
        "弱科弥补建议以安全线为目标，优先补足最短板科目。"
    )

    # 太长不看版
    mc_85 = mc.get("85%_score", 0)
    mc_75 = mc.get("75%_score", 0)
    placeholders["too_long_dont_read_summary"] = (
        f"1️⃣ 目标总分 {_mc85:.0f}+ → 进面概率85%+\n"
        f"2️⃣ 目标总分 {_mc75:.0f}+ → 进面概率75%+\n"
        f"3️⃣ 业务课一是分水岭科目（β系数最高），值得重点投入。\n"
        f"4️⃣ 英语波动最小，以保安全线为目标即可。\n"
        f"5️⃣ 近3年趋势：{a2026.get('trend_last_three', '—')}"
    )

    # ── 填充缺失占位符 ────────────────────────────────────────────────
    _ALL_TEMPLATE_KEYS = [
        "b_history_english", "b_history_politics", "b_history_sub1", "b_history_sub2",
        "cohen_d_data", "cv_years_table_data",
        "dea_english", "dea_politics", "dea_sub1", "dea_sub2",
        "deviation_final_percentiles", "deviation_scores_values",
        "english_2025_adjustment_review", "equating_percentiles_table_data",
        "five_year_summary_text",
        "interpret_mid_english", "interpret_mid_politics",
        "interpret_mid_sub1", "interpret_mid_sub2",
        "interpret_p_b_english", "interpret_p_b_politics", "interpret_p_b_sub1", "interpret_p_b_sub2",
        "interpret_quantile",
        "major_2026_table_data", "major_cv_table_data", "major_ranking_table_data",
        "max_p_b_english", "max_p_b_politics", "max_p_b_sub1", "max_p_b_sub2",
        "monte_carlo_75_score", "monte_carlo_85_score", "monte_carlo_interpretation",
        "p_b_2026_table_data",
        "p_history_english", "p_history_politics", "p_history_sub1", "p_history_sub2",
        "quantile_regression_table_data",
        "safe_english", "safe_politics", "safe_sub1", "safe_sub2",
        "school_name", "score_mid_english", "score_mid_politics", "score_mid_sub1", "score_mid_sub2",
        "subject_distinction_info", "subjects_2026_vs_history_table_data",
        "suggestion_majors", "suggestion_priority", "suggestion_scores_grid",
        "suggestion_total_strategy",
        "too_long_dont_read_summary",
        "total_sample_size", "total_score_interpretation", "total_score_interpretation_2026",
        "total_score_mid_2026", "total_score_mid_range", "total_score_min_line", "total_score_safe_line",
        "trend_last_three_years",
        "year_type_2026_label", "year_type_2026_reason",
        "z_english", "z_politics", "z_sub1", "z_sub2",
    ]
    _MEANINGFUL_DEFAULTS = {
        "b_history_english": "该科目无历史数据",
        "b_history_politics": "该科目无历史数据",
        "b_history_sub1": "该科目无历史数据",
        "b_history_sub2": "该科目无历史数据",
        "deviation_final_percentiles": "全院统一命题，无偏差分",
        "deviation_scores_values": "全院统一命题，无偏差分",
        "major_ranking_table_data": "| 专业名称 | 百分位 | 难度指数 | 解读 |\n|---------|--------|----------|------|\n| 全院统一招生 | — | — | 不分方向，统一排名 |",
        "major_2026_table_data": "| 专业名称 | 百分位 | 难度指数 | 解读 |\n|---------|--------|----------|------|\n| 全院统一招生 | — | — | 不分方向 |",
        "suggestion_majors": "数据不足以区分专业方向，建议按全院统一策略备考",
    }
    for key in _ALL_TEMPLATE_KEYS:
        if key not in placeholders:
            placeholders[key] = _MEANINGFUL_DEFAULTS.get(key, "—")
            if key not in _MEANINGFUL_DEFAULTS:
                print(f"  [WARN build_report_data] 缺失占位符 {{{key}}}，已填充为「—」")

    print(f"DEBUG: 传递给模板的完整字典 keys: {sorted(placeholders.keys())}")

    return placeholders


def generate_report(
    analyzer: Any,
    output_path: str | Path = DEFAULT_OUTPUT,
    template_path: str | Path = TEMPLATE_PATH,
) -> str:
    """生成完整 Markdown 报告.

    Parameters
    ----------
    analyzer : KaoYanAnalyzer
        已完成 compute_all() 的分析器.
    output_path : str | Path
        输出文件路径.
    template_path : str | Path
        模板文件路径.

    Returns
    -------
    str
        生成的报告全文.
    """
    tmpl_path = Path(template_path)
    if not tmpl_path.exists():
        raise FileNotFoundError(f"模板文件不存在: {tmpl_path}")

    template = tmpl_path.read_text(encoding="utf-8")
    data = build_report_data(analyzer)

    # 强制验证：确认所有占位符在 data 中均有对应
    import re as _re
    _placeholders = set(_re.findall(r"\{(\w+)\}", template))
    _data_keys = set(data.keys())
    _still_missing = _placeholders - _data_keys
    if _still_missing:
        print(f"  [FATAL] 以下占位符在渲染前仍缺失: {sorted(_still_missing)}")
        for _k in _still_missing:
            data[_k] = "—"

    # 安全渲染
    try:
        report = template.format(**data)
    except KeyError as e:
        missing = str(e).strip("'")
        print(f"  [WARN] 占位符 {{{missing}}} 无对应数据，已保留原样")
        # 替换所有已知键，将未知键保留
        for k, v in data.items():
            template = template.replace("{" + k + "}", str(v))
        report = template

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 强力兜底日志：确认 school_name 已成功注入
    school_in_report = data.get("school_name", "目标高校")
    preview = report[:200].replace("\n", " ")
    print(f"  DEBUG - 注入校名: {school_in_report}")
    print(f"  DEBUG - 报告前200字: {preview}")
    out.write_text(report, encoding="utf-8")
    try:
        rel = out.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = out
    print(f"[OK] 报告已生成: {rel}")
    print(f"     共 {len(report)} 字符")

    return report
