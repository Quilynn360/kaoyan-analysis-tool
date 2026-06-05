"""KaoYanAnalyzer —— 考研初试数据量化分析核心引擎

用法：
    from src.processor import KaoYanAnalyzer
    analyzer = KaoYanAnalyzer(school="Nanjing_Normal_University")
    analyzer.load_data(years=[2023, 2024, 2025, 2026])
    analyzer.filter_regular_plans()
    analyzer.compute_all()
    analyzer.print_summary()
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ═══════════════════════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ORGANIZED_DIR = PROJECT_ROOT / "data" / "organized"
SCORE_COLS = ["政治", "英语", "业务课一", "业务课二"]
SUBJECT_COLS = SCORE_COLS + ["总分"]

NON_REGULAR_KEYWORDS = [
    "退役大学生士兵", "士兵计划", "大学生士兵",
    "少数民族骨干计划", "少骨", "少干",
    "少数民族", "少民", "少民计划",
    "强军计划", "援藏计划", "单独考试",
]


# ═══════════════════════════════════════════════════════════════════════════
#  方向探测引擎
# ═══════════════════════════════════════════════════════════════════════════

def detect_specialization_structure(df: pd.DataFrame) -> tuple[bool, list[str], str | None]:
    """自动探测数据是否按研究方向分组招生.

    检测逻辑（按优先级）：
        1. 是否存在 "研究方向"/"专业方向" 等明确列 → 直接使用
        2. 否则扫描原始数据顺序中的总分"跳变点"：
           在原始数据顺序中，如果总分呈现 "下降→跳升→下降→跳升" 的循环，
           则每个跳升点就是方向边界（说明数据按方向分组排列）。

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple[bool, list[str], str | None]
        (has_directions, direction_names, explicit_column_name)
    """
    # ── 1. 显式列检测 ────────────────────────────────────────────────
    for col in df.columns:
        lower = col.lower().replace(" ", "")
        if any(kw in lower for kw in ["方向", "specialization", "细分", "小方向"]):
            uniq = df[col].dropna().unique()
            if len(uniq) >= 2:
                print(f"  [DIRECTION] 检测到显式方向列「{col}」→ {len(uniq)} 个方向")
                return True, sorted(str(v) for v in uniq), col

    # ── 2. 基于总分顺序的隐式探测 ────────────────────────────────────
    if "总分" not in df.columns:
        return False, [], None

    # 按年份分组检测：方向划分应在每年内一致
    year_col = "year" if "year" in df.columns else None
    all_yr_results: list[tuple[list[int], list[int]]] = []  # [(jumps, group_sizes), ...]

    if year_col is not None:
        year_groups = [g for _, g in df.groupby(year_col) if len(g) >= 8]
    else:
        year_groups = [df] if len(df) >= 8 else []

    for grp in year_groups:
        scores = grp["总分"].values
        score_range = float(scores.max() - scores.min())
        threshold = max(15, score_range * 0.08)  # 严格阈值：8% or 15分
        jumps: list[int] = []
        for i in range(1, len(scores)):
            if scores[i] - scores[i - 1] > threshold:
                jumps.append(i)
        if not jumps:
            continue
        # 每组大小
        groups = [jumps[0]] + [jumps[k] - jumps[k - 1] for k in range(1, len(jumps))]
        groups.append(len(scores) - jumps[-1])
        # 严格验证
        dec = sum(1 for i in range(1, len(scores)) if scores[i] <= scores[i - 1])
        if dec / (len(scores) - 1) < 0.7:  # 至少 70% 是递减
            continue
        if any(s < 4 for s in groups):  # 每组至少4人
            continue
        if len(jumps) < 1:  # 至少1个跳变(2个方向)
            continue
        # 组大小平衡检查：最大组 / 最小组 < 4
        if max(groups) / min(groups) >= 4:
            continue
        all_yr_results.append((jumps, groups))

    if len(all_yr_results) < 2:  # 至少2年表现一致才判定
        return False, [], None

    # 跨年一致性检查：每年检测到的方向数应该相同
    n_dirs_list = [len(j) + 1 for j, _ in all_yr_results]
    if len(set(n_dirs_list)) > 1:
        return False, [], None

    n_dirs = n_dirs_list[0]
    if n_dirs < 2:
        return False, [], None

    # 取最后一年（最新）的跳变作为最终分组方案
    final_jumps, final_groups = all_yr_results[-1]
    names = [chr(65 + i) for i in range(n_dirs)]
    print(f"  [DIRECTION] 检测到按方向招生，共 {n_dirs} 个方向: {names}")
    print(f"           各组人数: {final_groups}")
    return True, names, None


def _find_score_jumps(scores: np.ndarray) -> list[int]:
    """在原始分数序列中查找「跳变点」：score[i] 明显高于 score[i-1].

    用于隐式方向探测的分组边界识别.

    Parameters
    ----------
    scores : np.ndarray
        原始顺序（未排序）的总分数组.

    Returns
    -------
    list[int]
        跳变点的索引位置列表.
    """
    n = len(scores)
    if n < 8:
        return []
    score_range = float(scores.max() - scores.min())
    threshold = max(15, score_range * 0.08)
    jumps: list[int] = []
    for i in range(1, n):
        if scores[i] - scores[i - 1] > threshold:
            jumps.append(i)
    return jumps


# ═══════════════════════════════════════════════════════════════════════════
#  KaoYanAnalyzer
# ═══════════════════════════════════════════════════════════════════════════

class KaoYanAnalyzer:
    """考研初试数据量化分析器.

    负责从 ``data/organized/{school}/`` 加载数据、
    过滤非普通计划考生、计算全部核心指标。
    """

    def __init__(
        self,
        school: str = "",
        organized_root: str | Path = DEFAULT_ORGANIZED_DIR,
    ) -> None:
        self.school = school
        self.organized_root = Path(organized_root)
        self._raw: pd.DataFrame | None = None
        self._long: pd.DataFrame | None = None
        self._clean: pd.DataFrame | None = None
        self._clean_long: pd.DataFrame | None = None
        self.results: dict[str, Any] = {}
        self._direction_info: tuple[bool, list[str], str | None] = (False, [], None)

    # ═══════════════════════════════════════════════════════════════════════
    #  1. 数据加载与过滤
    # ═══════════════════════════════════════════════════════════════════════

    def load_data(self, years: list[int] | None = None) -> pd.DataFrame:
        """从 ``data/organized/{school}/{year}/`` 加载指定年份数据."""
        school_dir = self.organized_root / self.school
        if not school_dir.exists():
            raise FileNotFoundError(f"学校目录不存在: {school_dir}")

        if years is None:
            years = sorted(
                int(d.name) for d in school_dir.iterdir()
                if d.is_dir() and d.name.isdigit()
            )

        frames: list[pd.DataFrame] = []
        for y in years:
            yr_dir = school_dir / str(y)
            if not yr_dir.exists():
                print(f"  [WARN] 年份 {y} 目录不存在，跳过")
                continue
            for f in sorted(yr_dir.iterdir()):
                if not f.is_file():
                    continue
                try:
                    df = self._load_single(f)
                    df["year"] = y
                    frames.append(df)
                except Exception as e:
                    print(f"  [WARN] 跳过 {f.name}: {e}")

        if not frames:
            raise ValueError(f"未加载到任何数据（school={self.school}, years={years}）")

        self._raw = pd.concat(frames, ignore_index=True)
        self._raw = self._repair_columns(self._raw)
        self._long = self._to_long(self._raw)
        print(f"  [DATA] 加载 {len(self._raw)} 条记录，列: {list(self._raw.columns)}")
        return self._raw

    @staticmethod
    def _repair_columns(df: pd.DataFrame) -> pd.DataFrame:
        """修复常见列名问题，确保与标准 schema 对齐."""
        df = df.copy()
        rename = {}
        for col in df.columns:
            lower = str(col).replace(" ", "").lower()
            if any(kw in lower for kw in ["学院"]):
                rename[col] = "学院"
            elif any(kw in lower for kw in ["专业", "方向"]):
                rename[col] = "专业"
            elif any(kw in lower for kw in ["备注", "专项", "计划"]):
                rename[col] = "备注"
            # 修复因编码导致的重名列
        df = df.rename(columns=rename)
        # 删除完全重复的列
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    @staticmethod
    def _load_single(file_path: Path) -> pd.DataFrame:
        """加载单个文件: PDF 使用直接解析, 其他格式走 ingestion."""
        if file_path.suffix.lower() == ".pdf":
            return _parse_pdf_table(file_path)
        try:
            from src.ingestion import DataIngestor
            return DataIngestor().detect_and_load(file_path)
        except Exception as e:
            raise ValueError(f"无法解析文件 {file_path.name}: {e}") from e

    def filter_regular_plans(self, remark_cols: list[str] | None = None) -> pd.DataFrame:
        """剔除所有非普通计划专项考生的数据."""
        if self._raw is None:
            raise ValueError("请先调用 load_data()")

        if remark_cols is None:
            remark_cols = [
                c for c in self._raw.columns
                if any(kw in c for kw in ["备注", "专项", "计划", "类别", "note", "remark"])
            ]

        mask = pd.Series(True, index=self._raw.index)
        if remark_cols:
            for col in remark_cols:
                if col not in self._raw.columns:
                    continue
                col_str = self._raw[col].astype(str).fillna("")
                for kw in NON_REGULAR_KEYWORDS:
                    mask &= ~col_str.str.contains(kw, case=False, na=False)

        dropped = (~mask).sum()
        if dropped:
            print(f"  [FILTER] 剔除 {dropped} 条非普通计划考生记录")

        self._clean = self._raw[mask].copy().reset_index(drop=True)
        self._clean_long = self._to_long(self._clean)
        return self._clean

    # ═══════════════════════════════════════════════════════════════════════
    #  2. 工具方法
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _to_long(df: pd.DataFrame) -> pd.DataFrame:
        """宽格式 → 长格式."""
        if "subject" in df.columns and "score" in df.columns:
            return df.copy()
        id_vars = [c for c in ["姓名", "year"] if c in df.columns]
        available = [c for c in SCORE_COLS if c in df.columns]
        if not available:
            raise ValueError("无法识别数据格式")
        long_df = df.melt(id_vars=id_vars, value_vars=available, var_name="subject", value_name="score")
        if "year" not in long_df.columns:
            long_df["year"] = pd.NA
        return long_df.dropna(subset=["score"])

    @property
    def df(self) -> pd.DataFrame:
        return self._clean if self._clean is not None else self._raw

    @property
    def df_long(self) -> pd.DataFrame:
        return self._clean_long if self._clean_long is not None else self._long

    # ═══════════════════════════════════════════════════════════════════════
    #  3. Pearson 相关系数 + 标准化 β 系数
    # ═══════════════════════════════════════════════════════════════════════

    def compute_pearson(self) -> dict:
        """Pearson 相关系数矩阵（含各科与总分的相关性）."""
        df = self.df
        # 包含总分
        available = [c for c in SUBJECT_COLS if c in df.columns]
        n = len(available)
        r_mat = pd.DataFrame(np.eye(n), index=available, columns=available)
        p_mat = pd.DataFrame(np.zeros((n, n)), index=available, columns=available)
        pairs: list[tuple[str, str, float, float]] = []

        for i in range(n):
            for j in range(i + 1, n):
                valid = df[[available[i], available[j]]].dropna()
                if len(valid) < 3:
                    continue
                r, p = sp_stats.pearsonr(valid[available[i]], valid[available[j]])
                r_mat.iloc[i, j] = r_mat.iloc[j, i] = round(r, 4)
                p_mat.iloc[i, j] = p_mat.iloc[j, i] = round(p, 4)
                pairs.append((available[i], available[j], round(r, 4), round(p, 4)))

        result = {"r": r_mat, "p": p_mat, "pairs": pairs}
        self.results["pearson"] = result
        return result

    def compute_beta(self) -> pd.DataFrame:
        """标准化回归 β 系数（因变量 = 当年普通计划内绝对排名，statsmodels OLS on Z-scores）.

        对每个学生计算其在当年全部考生中的总分排名（rank 1 = 最高分），
        然后对各科分数（Z-score 化）与排名（Z-score 化）做回归，
        回归系数即为标准化 β，反映该科每提升 1σ 对排名提升的贡献。
        """
        import statsmodels.api as sm

        df = self.df
        if "总分" not in df.columns or "year" not in df.columns:
            self.results["beta"] = pd.DataFrame()
            return pd.DataFrame()

        # 计算当年绝对排名（1 = 最高分）
        temp = df[["year", "总分"] + [c for c in SCORE_COLS if c in df.columns]].copy()
        temp["rank"] = temp.groupby("year")["总分"].rank(ascending=False, method="min")

        rows: list[dict[str, Any]] = []
        for subj in SCORE_COLS:
            if subj not in temp.columns:
                continue
            valid = temp[[subj, "rank"]].dropna()
            if len(valid) < 10:
                rows.append({"subject": subj, "beta": None, "p_value": None, "r_squared": None})
                continue
            x = sp_stats.zscore(valid[subj].values, ddof=1).reshape(-1, 1)
            y = sp_stats.zscore(valid["rank"].values, ddof=1)
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            rows.append({
                "subject": subj,
                "beta": round(float(model.params[1]), 4),
                "p_value": round(float(model.pvalues[1]), 4),
                "r_squared": round(float(model.rsquared), 4),
            })
        result = pd.DataFrame(rows)
        self.results["beta"] = result
        return result

    # ═══════════════════════════════════════════════════════════════════════
    #  4. Z-score · 中位数 · DEA · 跨年等值百分位
    # ═══════════════════════════════════════════════════════════════════════

    def compute_z_scores(self) -> pd.DataFrame:
        """按科目计算 Z-score 标准化."""
        df_long = self.df_long
        result = df_long.copy()
        result["z_score"] = result.groupby("subject")["score"].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0.0
        )
        self.results["z_scores"] = result
        return result

    def compute_medians(self) -> pd.DataFrame:
        """各科历年及总体的中位数分数."""
        df = self.df
        long_df = self.df_long
        # 各科5年总体中位数
        overall = long_df.groupby("subject")["score"].median().round(1)
        overall.name = "median_5yr"
        # 各科历年
        yearly = long_df.groupby(["year", "subject"])["score"].median().round(1).reset_index()
        self.results["medians_overall"] = overall
        self.results["medians_yearly"] = yearly
        return yearly

    def compute_dea(self) -> pd.DataFrame:
        """DEA 边际贡献率（产出 / 前沿最大值）."""
        long_df = self.df_long
        eff = long_df.groupby(["year", "subject"])["score"].mean().reset_index(name="avg_score")
        frontier = eff.groupby("year")["avg_score"].transform("max")
        eff["efficiency"] = (eff["avg_score"] / frontier).round(4)
        # 5年平均DEA
        overall_dea = eff.groupby("subject")["efficiency"].mean().round(4).reset_index()
        self.results["dea"] = eff
        self.results["dea_overall"] = overall_dea
        return eff

    def compute_equating_percentiles(self) -> dict:
        """跨年等百分位等值：各科5年平均中游对应的百分位."""
        long_df = self.df_long
        rows: list[dict[str, Any]] = []
        for subj, grp in long_df.groupby("subject"):
            scores = grp["score"].dropna().values
            if len(scores) < 2:
                continue
            median_score = np.median(scores)
            pct = sp_stats.percentileofscore(scores, median_score, kind="mean")
            rows.append({"subject": subj, "median_score": round(median_score, 1), "percentile": round(pct, 1)})
        result_df = pd.DataFrame(rows)
        hardest = result_df.loc[result_df["percentile"].idxmin(), "subject"] if not result_df.empty else "—"
        result = {"table": result_df, "hardest_subject": hardest}
        self.results["equating"] = result
        return result

    # ═══════════════════════════════════════════════════════════════════════
    #  5. 分位数回归（tau=0.5）+ Cohen's d
    # ═══════════════════════════════════════════════════════════════════════

    def compute_quantile_regression(self) -> pd.DataFrame:
        """中位数分位数回归（$\tau=0.5$），模型 score ~ total."""
        import statsmodels.formula.api as smf

        df = self.df
        available = [c for c in SCORE_COLS if c in df.columns]
        rows: list[dict[str, Any]] = []
        for subj in available:
            valid = df[[subj, "总分"]].dropna()
            if len(valid) < 5:
                continue
            try:
                model = smf.quantreg(f"{subj} ~ 总分", valid)
                res = model.fit(q=0.5)
                rows.append({
                    "subject": subj,
                    "marginal_effect": round(float(res.params["总分"]), 4),
                    "p_value": round(float(res.pvalues["总分"]), 4),
                    "intercept": round(float(res.params["Intercept"]), 4),
                })
            except Exception:
                continue
        result = pd.DataFrame(rows)
        self.results["quantile_regression"] = result
        return result

    def compute_cohen_d(self) -> pd.DataFrame:
        """Cohen's d 效应量：高分半 vs 低分半（按总分分组）."""
        df = self.df
        available = [c for c in SCORE_COLS if c in df.columns and c in df.columns]
        median_total = df["总分"].median()
        high = df[df["总分"] >= median_total]
        low = df[df["总分"] < median_total]
        rows: list[dict[str, Any]] = []
        for subj in available:
            h = high[subj].dropna().values
            l = low[subj].dropna().values
            if len(h) < 2 or len(l) < 2:
                continue
            n1, n2 = len(h), len(l)
            s1, s2 = np.var(h, ddof=1), np.var(l, ddof=1)
            pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
            d = (np.mean(h) - np.mean(l)) / pooled if pooled > 0 else 0
            rows.append({"subject": subj, "cohen_d": round(float(d), 4)})
        result = pd.DataFrame(rows)
        self.results["cohen_d"] = result
        return result

    # ═══════════════════════════════════════════════════════════════════════
    #  6. 变异系数（CV）与大小年分析
    # ═══════════════════════════════════════════════════════════════════════

    def compute_cv(self) -> pd.DataFrame:
        """计算历年变异系数（CV = std/mean）分析大小年."""
        df = self.df
        rows: list[dict[str, Any]] = []
        for y in sorted(df["year"].unique()):
            yr = df[df["year"] == y]
            total = yr["总分"].dropna()
            if len(total) < 2:
                continue
            median = round(float(total.median()), 1)
            mean = round(float(total.mean()), 1)
            cv = round(float(total.std(ddof=0) / total.mean()), 4)
            rows.append({"year": int(y), "median": median, "mean": mean, "cv": cv})
        result = pd.DataFrame(rows)
        self.results["cv"] = result

        # 2025英语+8校准复核
        if 2025 in df["year"].values:
            self._english_2025_calibration(df)

        return result

    def _english_2025_calibration(self, df: pd.DataFrame) -> None:
        """2025年英语+8分后的CV变动复核."""
        df_2025 = df[df["year"] == 2025].copy()
        df_2025["英语_校准"] = df_2025["英语"] + 8
        df_other = df[df["year"] != 2025].copy()
        df_other["英语_校准"] = df_other["英语"]

        combined = pd.concat([df_2025, df_other], ignore_index=True)
        rows: list[dict[str, Any]] = []
        for y in sorted(combined["year"].unique()):
            yr = combined[combined["year"] == y]
            eng = yr["英语_校准"].dropna()
            if len(eng) < 2:
                continue
            cv_orig = round(float(eng.std(ddof=0) / eng.mean()), 4)
            rows.append({"year": int(y), "cv_english_calibrated": cv_orig})
        self.results["cv_english_calibrated"] = pd.DataFrame(rows)

    def compute_major_cv(self) -> pd.DataFrame:
        """各专业方向变异系数（若有方向区分列）. """
        df = self.df
        # 尝试查找专业/方向列
        major_col = None
        for col in df.columns:
            if any(kw in col for kw in ["专业", "方向", "major"]):
                major_col = col
                break
        if major_col is None:
            self.results["major_cv"] = pd.DataFrame()
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for (y, major), grp in df.groupby(["year", major_col]):
            total = grp["总分"].dropna()
            if len(total) < 2:
                continue
            cv = round(float(total.std(ddof=0) / total.mean()), 4)
            rows.append({"year": int(y), "major": major, "cv": cv})
        result = pd.DataFrame(rows)
        self.results["major_cv"] = result
        return result

    # ═══════════════════════════════════════════════════════════════════════
    #  7. 专业课偏差分校准
    # ═══════════════════════════════════════════════════════════════════════

    def compute_deviation_scores(self) -> dict:
        """专业课试卷差异偏差分分析.

        若院内分小方向且专业课试卷不同（通过专业/方向列判断），
        分别计算各方向平均分与中位分，计算偏差分修正值。
        """
        df = self.df
        # 方向列：先找显式列，否则用探测结果
        has_dir, dir_names, col_name = self._direction_info
        # 去掉 _override_ 前缀
        actual_col = col_name.replace("_override_", "", 1) if col_name and "_override_" in col_name else col_name
        major_col = actual_col  # 显式列
        use_detected = False

        if major_col is None:
            # 尝试查找显式专业/方向列
            for col in df.columns:
                if any(kw in col for kw in ["专业", "方向", "major"]):
                    major_col = col
                    break

        if major_col is None or major_col not in df.columns or (major_col in df.columns and df[major_col].nunique() < 2):
            if has_dir:
                use_detected = True
                major_col = "_detected_dir"
                df = df.copy()
                # 平分方向标签（按原始顺序切分）
                n = len(df)
                dir_names_sorted = sorted(dir_names) if dir_names else [chr(65+i) for i in range(2)]
                chunk = max(1, n // len(dir_names_sorted))
                labels = []
                for i in range(n):
                    idx = min(i // chunk, len(dir_names_sorted) - 1)
                    labels.append(dir_names_sorted[idx])
                df[major_col] = labels
            else:
                result = {
                    "has_deviation": False,
                    "message": "不分小方向或专业课试卷完全相同，无需偏差分校准",
                    "values": None,
                    "final_percentiles": None,
                }
                self.results["deviation"] = result
                return result

        # 计算各方向各科偏差
        deviations: list[dict[str, Any]] = []
        overall_median = df["总分"].median()
        for major, grp in df.groupby(major_col):
            delta = float(grp["总分"].median() - overall_median)
            deviations.append({"major": major, "deviation": round(delta, 1)})

        # 校准后百分位
        final_rows: list[dict[str, Any]] = []
        all_totals = df["总分"].dropna()
        for major, grp in df.groupby(major_col):
            calibrated = grp["总分"] - (grp["总分"].median() - overall_median)
            pct = sp_stats.percentileofscore(all_totals, calibrated.median(), kind="mean")
            # NaN/Inf 保护
            if np.isnan(pct) or np.isinf(pct):
                pct = sp_stats.percentileofscore(all_totals, grp["总分"].median(), kind="mean")
            final_rows.append({"major": major, "calibrated_percentile": round(float(pct), 1)})

        result = {
            "has_deviation": True,
            "values": pd.DataFrame(deviations),
            "final_percentiles": pd.DataFrame(final_rows),
        }
        self.results["deviation"] = result
        return result

    # ═══════════════════════════════════════════════════════════════════════
    #  8. 蒙特卡罗模拟（协方差矩阵法）
    # ═══════════════════════════════════════════════════════════════════════

    def compute_monte_carlo(self, n_simulations: int = 10000) -> dict:
        """基于历史均值与协方差矩阵，模拟2027年进面概率.

        Returns
        -------
        dict :
            {"80%_score": ..., "50%_score": ..., "20%_score": ...,
             "all_scores": ndarray, "total_scores": ndarray}
        """
        df = self.df
        available = [c for c in SCORE_COLS if c in df.columns]
        data = df[available].dropna()
        if len(data) < 5:
            return {"error": "数据不足"}
        mean_vec = data.mean().values
        cov_mat = data.cov().values

        # 稳健处理：确保协方差矩阵正定
        try:
            samples = np.random.multivariate_normal(mean_vec, cov_mat, size=n_simulations)
        except np.linalg.LinAlgError:
            cov_mat += np.eye(len(mean_vec)) * 1e-6
            samples = np.random.multivariate_normal(mean_vec, cov_mat, size=n_simulations)

        total_scores = samples.sum(axis=1)
        result = {
            "90%_score": round(float(np.percentile(total_scores, 90)), 1),
            "85%_score": round(float(np.percentile(total_scores, 85)), 1),
            "75%_score": round(float(np.percentile(total_scores, 75)), 1),
            "80%_score": round(float(np.percentile(total_scores, 80)), 1),
            "50%_score": round(float(np.percentile(total_scores, 50)), 1),
            "all_scores": samples,
            "total_scores": total_scores,
        }
        self.results["monte_carlo"] = result
        return result

    # ═══════════════════════════════════════════════════════════════════════
    #  9. 各专业进面中位百分位排名
    # ═══════════════════════════════════════════════════════════════════════

    def compute_major_ranking(self, year: int | None = None) -> pd.DataFrame:
        """各专业进面中位成绩在总分分布中的百分位排名."""
        df = self.df
        has_dir, dir_names, col_name = self._direction_info
        actual_col = col_name.replace("_override_", "", 1) if col_name and "_override_" in col_name else col_name
        major_col = actual_col

        if major_col is None:
            for col in df.columns:
                if any(kw in col for kw in ["专业", "方向", "major"]):
                    major_col = col
                    break

        if major_col is None or major_col not in df.columns or (major_col in df.columns and df[major_col].nunique() < 2):
            if has_dir:
                major_col = "_detected_dir"
                df = df.copy()
                n = len(df)
                dir_names_sorted = sorted(dir_names) if dir_names else [chr(65+i) for i in range(2)]
                chunk = max(1, n // len(dir_names_sorted))
                labels = []
                for i in range(n):
                    idx = min(i // chunk, len(dir_names_sorted) - 1)
                    labels.append(dir_names_sorted[idx])
                df[major_col] = labels
            else:
                self.results["major_ranking"] = pd.DataFrame()
            return pd.DataFrame()

        if year is not None:
            df = df[df["year"] == year]

        rows: list[dict[str, Any]] = []
        all_totals = df["总分"].dropna()
        for major, grp in df.groupby(major_col):
            median_total = grp["总分"].median()
            pct = sp_stats.percentileofscore(all_totals, median_total, kind="mean")
            # NaN/Inf 保护：回退到该专业中位分在全体中的排名百分位
            if np.isnan(pct) or np.isinf(pct):
                pct = float(sp_stats.percentileofscore(all_totals, grp["总分"].rank(pct=True).median(), kind="mean"))
            rows.append({"major": major, "percentile": round(float(pct), 1), "median_total": round(float(median_total), 1)})
        result = pd.DataFrame(rows).sort_values("percentile")
        self.results["major_ranking"] = result
        return result

    # ═══════════════════════════════════════════════════════════════════════
    #  10. 2026 单年分析
    # ═══════════════════════════════════════════════════════════════════════

    def compute_2026_analysis(self) -> dict:
        """2026年单独分析：Pearson + β + 中游 + 百分位."""
        df_2026 = self.df[self.df["year"] == 2026].copy()
        if df_2026.empty:
            self.results["analysis_2026"] = {"error": "无2026年数据"}
            return self.results["analysis_2026"]

        # 保存原始结果，临时替换df进行单年计算
        _saved_results = dict(self.results)
        _orig_clean, _orig_long = self._clean, self._clean_long
        self.results = {}
        self._clean = df_2026
        self._clean_long = self._to_long(df_2026)

        result = {}
        try:
            result["pearson"] = self.compute_pearson()
        except Exception:
            result["pearson"] = {}
        try:
            # 单年数据可能不足，beta可能为空
            result["beta"] = self.compute_beta()
        except Exception:
            result["beta"] = pd.DataFrame()
        try:
            result["medians"] = self.compute_medians()
        except Exception:
            result["medians"] = pd.DataFrame()
        try:
            result["major_ranking"] = self.compute_major_ranking(year=2026)
        except Exception:
            result["major_ranking"] = pd.DataFrame()

        # 恢复
        self._clean, self._clean_long = _orig_clean, _orig_long
        self.results = _saved_results

        # 年份类型定性
        cv_data = self.results.get("cv")
        if cv_data is not None and not cv_data.empty:
            cv_2026 = cv_data[cv_data["year"] == 2026]
            if not cv_2026.empty:
                cv_val = cv_2026["cv"].values[0]
                all_cv = cv_data["cv"].mean()
                result["year_type"] = "大年" if cv_val > all_cv else "小年"
                result["year_type_reason"] = (
                    f"2026年变异系数为{cv_val}，{'高于' if cv_val > all_cv else '低于'}5年均值{all_cv:.4f}"
                )

            # 趋势判断
            recent = cv_data[cv_data["year"] >= 2023]
            if len(recent) >= 2:
                trend_coef = np.polyfit(recent["year"], recent["cv"], 1)[0]
                result["trend_last_three"] = "逐年上升（竞争加剧）" if trend_coef > 0 else "逐年下降（竞争缓和）"

        self.results["analysis_2026"] = result
        return result

    # ═══════════════════════════════════════════════════════════════════════
    #  11. 一站式计算
    # ═══════════════════════════════════════════════════════════════════════

    def compute_all(self) -> dict:
        """执行全部核心分析。"""
        print(f"\n  [KaoYanAnalyzer] 开始全量分析 — {self.school}")
        print(f"  样本量: {len(self.df)} 人, 年份: {sorted(self.df['year'].unique())}")

        # 方向探测（如果有用户覆盖则跳过自动探测）
        has_dir, dir_names, col_name = self._direction_info
        if not (has_dir and col_name is not None and col_name.startswith("_override_")):
            self._direction_info = detect_specialization_structure(self.df)
            has_dir, dir_names, col_name = self._direction_info
        self.results["direction_info"] = {
            "has_directions": has_dir,
            "names": dir_names,
            "column": col_name,
        }
        if has_dir:
            print(f"  [DIRECTION] 后续偏差分校准/专业排名将按 {len(dir_names)} 个方向分块计算")
        else:
            print("  [DIRECTION] 未检测到方向划分，按全院统一计算")

        self.compute_pearson()
        print("  [OK] Pearson相关系数")
        self.compute_beta()
        print("  [OK] Beta系数")
        self.compute_z_scores()
        print("  [OK] Z-score")
        self.compute_medians()
        print("  [OK] 中位数")
        self.compute_dea()
        print("  [OK] DEA效率")
        self.compute_equating_percentiles()
        print("  [OK] 跨年等值百分位")
        self.compute_quantile_regression()
        print("  [OK] 分位数回归")
        self.compute_cohen_d()
        print("  [OK] Cohen's d")
        self.compute_cv()
        print("  [OK] 变异系数(CV)")
        self.compute_major_cv()
        print("  [OK] 专业方向CV")
        self.compute_deviation_scores()
        print("  [OK] 偏差分校准")
        self.compute_monte_carlo()
        print("  [OK] 蒙特卡罗模拟")
        self.compute_major_ranking()
        print("  [OK] 专业百分位排名")
        self.compute_2026_analysis()
        print("  [OK] 2026单年分析")

        print("  [完成] 全部指标已存入 self.results\n")
        return self.results

    def print_summary(self) -> None:
        """打印全部计算结果摘要."""
        print(f"\n{'='*50}")
        print(f"  KaoYanAnalyzer 分析摘要 — {self.school}")
        print(f"{'='*50}")
        if self._raw is not None:
            print(f"  原始记录: {len(self._raw)}")
        if self._clean is not None:
            print(f"  过滤后记录: {len(self._clean)}")
        print()
        for key, val in self.results.items():
            print(f"  [{key}]")
            if isinstance(val, pd.DataFrame):
                print(val.to_string(index=False))
            elif isinstance(val, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, pd.DataFrame):
                        print(f"    {k2}:")
                        print(v2.to_string(index=False))
                    else:
                        print(f"    {k2}: {v2}")
            print()


# ═══════════════════════════════════════════════════════════════════════════
#  PDF 手动解析兜底
# ═══════════════════════════════════════════════════════════════════════════

def _looks_like_header_row(row: list[str]) -> bool:
    """判断一行是否为表头.

    规则：
    1. 第一格不是纯数字（排除序号类数据行）
    2. 第一格包含表头特征词（如"序"、"姓"、"政治"、"外语"等）
    3. 或者整行文本中连续出现至少 3 个表头特征词
    """
    if not row:
        return False
    first = row[0].replace("\n", "").strip()
    header_kw = ["序号", "姓名", "政治", "外语", "英语",
                 "业务课", "总分", "总成绩", "初试总", "备注",
                 "考生编号", "专业名称", "学院名称"]

    # 规则1+2: 第一格不是数字且包含特征词
    if not first.replace(".", "").isdigit():
        if any(kw in first for kw in ["序", "姓", "政治", "外语", "英语", "业务", "总", "初", "备", "编"]):
            return True

    # 规则3: 整行中至少出现 3 个表头关键词
    text = " ".join(str(c) for c in row if c).replace("\n", "")
    hits = sum(1 for kw in header_kw if kw in text)
    return hits >= 3


def _infer_columns(header: list[str]) -> dict[int, str]:
    """根据表头行推断各列的索引 → 标准列名映射."""
    col_map: dict[int, str] = {}
    for idx, cell in enumerate(header):
        cell_lower = cell.lower().replace(" ", "").replace("\n", "")
        if any(kw in cell_lower for kw in ["姓名", "考生", "名字"]):
            col_map[idx] = "姓名"
        elif any(kw in cell_lower for kw in ["政治", "思想"]):
            col_map[idx] = "政治"
        elif any(kw in cell_lower for kw in ["外语", "英语", "外"]):
            col_map[idx] = "英语"
        elif any(kw in cell_lower for kw in ["业务课1", "业务一", "专一", "课一", "业务1"]):
            col_map[idx] = "业务课一"
        elif any(kw in cell_lower for kw in ["业务课2", "业务二", "专二", "课二", "业务2"]):
            col_map[idx] = "业务课二"
        elif any(kw in cell_lower for kw in ["总分", "总成绩", "合计", "初试总", "考试总分"]):
            col_map[idx] = "总分"
        # 其他列（序号、学院、专业等）跳过，不映射
    return col_map


def _parse_pdf_table(file_path: Path) -> pd.DataFrame:
    """使用 pdfplumber 提取 PDF 中的表格并返回宽格式 DataFrame.

    自动识别表头行，支持多页表格合并。
    """
    import pdfplumber

    all_rows: list[list[str]] = []
    col_map: dict[int, str] = {}
    header_found = False

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    clean = [str(c).strip().replace("\n", "") if c else "" for c in row]
                    if not any(clean):
                        continue
                    if not header_found and _looks_like_header_row(clean):
                        col_map = _infer_columns(clean)
                        header_found = True
                        continue
                    # 如果是表头但前面已经有过表头了 → skip (多页重复表头)
                    if header_found and _looks_like_header_row(clean):
                        continue
                    all_rows.append(clean)

    if not all_rows:
        raise ValueError(f"PDF 中未提取到有效表格: {file_path}")

    # 构建 DataFrame：只提取映射过的列
    if col_map:
        rows_dict: list[dict[str, str]] = []
        for row in all_rows:
            record: dict[str, str] = {}
            for idx, col_name in col_map.items():
                record[col_name] = row[idx] if idx < len(row) else ""
            rows_dict.append(record)
        df = pd.DataFrame(rows_dict)
        # 强制转换数字列
        num_cols = [c for c in df.columns if c in ("总分", "政治", "英语", "业务课一", "业务课二")]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    else:
        return pd.DataFrame(all_rows)
