"""考研初试数据量化分析系统 — Streamlit 前端界面

用法：
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 确保能找到 src/
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import DataIngestor
from src.organizer import extract_year
from src.processor import KaoYanAnalyzer, _parse_pdf_table
from src.report_generator import build_report_data

# ── 中文字体配置 ──────────────────────────────────────────────────────
import platform as _platform
if _platform.system() == "Windows":
    plt.rcParams["font.sans-serif"] = ["SimHei"]
elif _platform.system() == "Darwin":
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
else:
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


st.set_page_config(
    page_title="考研初试数据量化分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════
#  会话状态初始化
# ═══════════════════════════════════════════════════════════════════════

# ── 学校结构补丁路径 ─────────────────────────────────────────────────
PATCH_PATH = PROJECT_ROOT / "config" / "school_patch.json"

def _load_school_patch() -> dict:
    """读取持久化的学校结构补丁."""
    if PATCH_PATH.exists():
        import json
        return json.loads(PATCH_PATH.read_text(encoding="utf-8"))
    return {}

def _save_school_patch(school: str, mode: str, col: str | None) -> None:
    """保存学校结构补丁到 config/school_patch.json."""
    import json
    patches = _load_school_patch()
    patches[school] = {"mode": mode, "column": col}
    PATCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    PATCH_PATH.write_text(json.dumps(patches, ensure_ascii=False, indent=2), encoding="utf-8")


def _init_state() -> None:
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "df" not in st.session_state:
        st.session_state.df = None
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = None
    if "report_md" not in st.session_state:
        st.session_state.report_md = ""
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "struct_mode" not in st.session_state:
        st.session_state.struct_mode = "全院统一"
    if "struct_col" not in st.session_state:
        st.session_state.struct_col = None
    if "school_name" not in st.session_state:
        st.session_state.school_name = "目标高校"

_init_state()


# ═══════════════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data
def _load_dataframe(uploaded_files: list) -> pd.DataFrame:
    """将上传的文件加载为 DataFrame，并从文件名中提取年份."""
    frames: list[pd.DataFrame] = []
    for uf in uploaded_files:
        suffix = Path(uf.name).suffix.lower()
        year = extract_year(uf.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getbuffer())
            tmp_path = tmp.name
        try:
            if suffix in (".pdf",):
                df = _parse_pdf_table(Path(tmp_path))
            else:
                ingestor = DataIngestor()
                df = ingestor.detect_and_load(tmp_path)
            if year is not None:
                df["year"] = int(year)
            elif "year" not in df.columns:
                df["year"] = pd.NA
            else:
                df["year"] = pd.to_numeric(df["year"], errors="coerce")
            frames.append(df)
        except Exception as e:
            st.warning(f"跳过 {uf.name}: {e}")
        finally:
            os.unlink(tmp_path)
    if not frames:
        raise ValueError("没有成功加载任何文件")
    combined = pd.concat(frames, ignore_index=True)
    if "year" in combined.columns:
        combined["year"] = pd.to_numeric(combined["year"], errors="coerce")
    return combined


def _run_analysis(df: pd.DataFrame,
                  struct_mode: str | None = None,
                  struct_col: str | None = None,
                  school_name: str = "目标高校") -> KaoYanAnalyzer:
    """运行全量分析.

    Parameters
    ----------
    df : pd.DataFrame
    struct_mode : str | None
        "全院统一" 或 "按研究方向"
    struct_col : str | None
        方向列名（仅 struct_mode="按研究方向" 时有效）
    """
    analyzer = KaoYanAnalyzer(school=school_name)
    analyzer.school_name = school_name
    # 注入方向覆盖（_override_ 前缀告诉 compute_all 跳过自动探测）
    if struct_mode == "按研究方向" and struct_col:
        found_col = struct_col if struct_col in df.columns else None
        if found_col:
            uniq = df[found_col].dropna().unique()
            analyzer._direction_info = (True, sorted(str(v) for v in uniq), f"_override_{found_col}")
        else:
            analyzer._direction_info = (True, [struct_col], f"_override_{struct_col}")
    else:
        analyzer._direction_info = (False, [], None)

    analyzer._raw = df
    analyzer._long = analyzer._to_long(df)
    try:
        analyzer.filter_regular_plans()
    except Exception:
        analyzer._clean = df
        analyzer._clean_long = analyzer._long
    analyzer.compute_all()
    return analyzer


def _plot_trend_chart(analyzer: KaoYanAnalyzer) -> go.Figure:
    """绘制 5 年各科均分趋势线（Plotly）."""
    long_df = analyzer.df_long
    yearly = long_df.groupby(["year", "subject"])["score"].mean().reset_index()
    fig = px.line(
        yearly,
        x="year",
        y="score",
        color="subject",
        markers=True,
        title="各科历年平均分趋势",
        labels={"year": "年份", "score": "平均分", "subject": "科目"},
    )
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _plot_heatmap(analyzer: KaoYanAnalyzer) -> matplotlib.figure.Figure:
    """绘制科目相关性热力图（Matplotlib）. """
    pearson = analyzer.results.get("pearson", {}).get("r", None)
    if pearson is None or pearson.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "数据不足", ha="center", va="center", fontsize=14)
        return fig

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pearson.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(pearson.columns)))
    ax.set_yticks(range(len(pearson.index)))
    ax.set_xticklabels(pearson.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(pearson.index, fontsize=9)

    for i in range(len(pearson.index)):
        for j in range(len(pearson.columns)):
            val = pearson.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if abs(val) > 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("各科 Pearson 相关系数", fontsize=12)
    fig.tight_layout()
    return fig


def _plot_cv_chart(analyzer: KaoYanAnalyzer) -> go.Figure:
    """绘制大小年变异系数柱状图."""
    cv_df = analyzer.results.get("cv")
    if cv_df is None or cv_df.empty:
        return go.Figure()
    fig = px.bar(
        cv_df, x="year", y="cv",
        color="cv", color_continuous_scale="Blues",
        labels={"year": "年份", "cv": "变异系数 (CV)"},
        title="历年变异系数（大小年）",
    )
    fig.add_hline(y=cv_df["cv"].mean(), line_dash="dash", line_color="red",
                  annotation_text=f"均值 {cv_df['cv'].mean():.4f}")
    fig.update_layout(showlegend=False)
    return fig


def _generate_final_report(analyzer: KaoYanAnalyzer, school_name: str = "目标高校") -> str:
    """生成最终分析帖 Markdown."""
    data = build_report_data(analyzer)
    data["school_name"] = data.get("school_name", school_name)
    tmpl_path = PROJECT_ROOT / "templates" / "main_prompt.md"
    template = tmpl_path.read_text(encoding="utf-8")
    # 强制验证：确认所有占位符在 data 中均有对应
    import re as _re
    _phs = set(_re.findall(r"\{(\w+)\}", template))
    _still_missing = _phs - set(data.keys())
    if _still_missing:
        print(f"  [FATAL] 以下占位符在渲染前仍缺失: {sorted(_still_missing)}")
        for _k in _still_missing:
            data[_k] = "—"
    try:
        report = template.format(**data)
    except KeyError as e:
        missing = str(e).strip("'")
        print(f"  [WARN] 占位符 {{{missing}}} 无对应数据，已保留原样")
        for k, v in data.items():
            template = template.replace("{" + k + "}", str(v))
        report = template
    # 强力兜底日志
    school_in_report = data.get("school_name", "目标高校")
    preview = report[:200].replace("\n", " ")
    print(f"  DEBUG - 注入校名: {school_in_report}")
    print(f"  DEBUG - 报告前200字: {preview}")
    return report


# ═══════════════════════════════════════════════════════════════════════
#  UI — 侧边栏
# ═══════════════════════════════════════════════════════════════════════

st.sidebar.title("📊 考研量化分析")
st.sidebar.markdown("---")
st.sidebar.header("1. 上传数据")
uploaded_files = st.sidebar.file_uploader(
    "支持格式：xlsx / pdf / docx / jpg / png",
    type=["xlsx", "xls", "pdf", "docx", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="file_uploader",
)

if uploaded_files and len(uploaded_files) > 0:
    st.session_state.uploaded_files = uploaded_files

st.sidebar.markdown("---")
st.sidebar.header("2. 分析控制")

analyze_btn = st.sidebar.button(
    "🚀 运行全量分析",
    type="primary",
    use_container_width=True,
    disabled=len(st.session_state.uploaded_files) == 0,
)

reprocess_btn = st.sidebar.button(
    "🔄 重新生成报告",
    use_container_width=True,
    disabled=not st.session_state.data_loaded,
)

# ── 数据结构修正面板 ──────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("3. 数据结构修正")
with st.sidebar.expander("招生模式与方向列", expanded=False):
    _saved_mode = st.session_state.struct_mode
    _mode = st.radio("招生模式", ["全院统一", "按研究方向"],
                     index=0 if _saved_mode == "全院统一" else 1,
                     key="struct_radio")
    st.session_state.struct_mode = _mode

    _col = st.session_state.struct_col
    _col_options = ["自动检测", "报考专业", "专业", "报考方向", "方向", "研究方向", "手动指定"]
    _col_idx = _col_options.index(_col) if _col in _col_options else 0
    _sel_col = st.selectbox("方向所在列", _col_options, index=_col_idx,
                            disabled=(_mode == "全院统一"), key="struct_select")
    if _sel_col == "手动指定":
        _custom = st.text_input("输入自定义列名", value=_col if _col and _col not in _col_options else "",
                                key="struct_custom")
        st.session_state.struct_col = _custom
    elif _sel_col == "自动检测":
        st.session_state.struct_col = None
    else:
        st.session_state.struct_col = _sel_col

    reload_btn = st.button("🔄 重载数据", type="primary", use_container_width=True,
                           disabled=not st.session_state.data_loaded)

# ── 底部状态 ──────────────────────────────────────────────────────────
# ── 学校确认输入框 ──────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**📢 当前分析目标高校**")
_school_input = st.sidebar.text_input(
    "校名（可手动修正）",
    value=st.session_state.school_name,
    label_visibility="collapsed",
)
st.session_state.school_name = _school_input
st.sidebar.caption(f"已上传：{len(st.session_state.uploaded_files)} 个文件")


# ═══════════════════════════════════════════════════════════════════════
#  UI — 主区域
# ═══════════════════════════════════════════════════════════════════════

st.title("📈 考研初试数据量化分析系统")
st.markdown(
    f"当前学校：**{st.session_state.school_name}**  "
    f"｜已上传 **{len(st.session_state.uploaded_files)}** 个文件  "
    f"｜{'✅ 数据已加载' if st.session_state.data_loaded else '⬆️ 请上传数据'}"
)

# ── 数据处理与分析的触发逻辑 ──────────────────────────────────────────

if analyze_btn and st.session_state.uploaded_files:
    with st.spinner("正在加载并分析数据..."):
        try:
            df = _load_dataframe(st.session_state.uploaded_files)
            st.session_state.df = df
            mode = st.session_state.struct_mode
            scol = st.session_state.struct_col
            sn = st.session_state.school_name
            analyzer = _run_analysis(df, struct_mode=mode, struct_col=scol, school_name=sn)
            st.session_state.analyzer = analyzer
            st.session_state.report_md = _generate_final_report(analyzer, st.session_state.school_name)
            st.session_state.data_loaded = True
            st.success(f"分析完成！共 {len(df)} 条有效记录。")
            st.rerun()
        except Exception as e:
            st.error(f"分析失败: {e}")
            import traceback
            st.exception(e)

elif reprocess_btn and st.session_state.data_loaded:
    with st.spinner("重新生成报告中..."):
        try:
            st.session_state.report_md = _generate_final_report(st.session_state.analyzer, st.session_state.school_name)
            st.success("报告已重新生成。")
            st.rerun()
        except Exception as e:
            st.error(f"报告生成失败: {e}")

# ── 重载数据（数据结构修正触发） ──────────────────────────────────────

if reload_btn and st.session_state.df is not None:
    st.cache_data.clear()
    with st.spinner("按修正后的结构重新分析..."):
        try:
            mode = st.session_state.struct_mode
            scol = st.session_state.struct_col
            analyzer = _run_analysis(st.session_state.df,
                                     struct_mode=mode, struct_col=scol)
            st.session_state.analyzer = analyzer
            st.session_state.report_md = _generate_final_report(analyzer, st.session_state.school_name)
            st.session_state.data_loaded = True
            # 持久化
            _save_school_patch(st.session_state.school_name, mode, scol)
            st.success(f"重载完成（{mode}）！")
            st.rerun()
        except Exception as e:
            st.error(f"重载失败: {e}")
            import traceback
            st.exception(e)

# ── 数据预览 ──────────────────────────────────────────────────────────

if st.session_state.data_loaded and st.session_state.df is not None:
    df = st.session_state.df
    analyzer = st.session_state.analyzer

    with st.expander("📋 数据预览", expanded=False):
        st.dataframe(df.head(100), width="stretch")
        st.caption(f"共 {len(df)} 行 × {len(df.columns)} 列")

    # ── 图表区域 ──────────────────────────────────────────────────────
    st.subheader("📉 可视化分析")

    col1, col2 = st.columns(2)

    with col1:
        trend_fig = _plot_trend_chart(analyzer)
        st.plotly_chart(trend_fig, use_container_width=True)

    with col2:
        heat_fig = _plot_heatmap(analyzer)
        st.pyplot(heat_fig, use_container_width=True)

    cv_fig = _plot_cv_chart(analyzer)
    if cv_fig.data:
        st.plotly_chart(cv_fig, use_container_width=True)

    # ── 指标卡片 ──────────────────────────────────────────────────────
    st.subheader("📊 核心指标速览")

    beta = analyzer.results.get("beta", pd.DataFrame())
    dea = analyzer.results.get("dea_overall", pd.DataFrame())
    mc = analyzer.results.get("monte_carlo", {})
    cv_df = analyzer.results.get("cv", pd.DataFrame())

    # 蒙特卡罗进面线（75%/85%/90%），低于5年最低分则不显示
    _mc_75 = mc.get("75%_score")
    _mc_85 = mc.get("85%_score")
    _mc_90 = mc.get("90%_score")
    _min_5yr = float(df["总分"].mean() - 2 * df["总分"].std()) if "总分" in df.columns else 0

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        beta_max = beta.loc[beta["beta"].idxmax()] if not beta.empty and beta["beta"].notna().any() else None
        st.metric("β 最高科目", beta_max["subject"] if beta_max is not None else "—",
                  beta_max["beta"] if beta_max is not None else "")
    with kpi_cols[1]:
        dea_max = dea.loc[dea["efficiency"].idxmax()] if not dea.empty else None
        st.metric("DEA 前沿科目", dea_max["subject"] if dea_max is not None else "—",
                  f"{dea_max['efficiency']:.2f}" if dea_max is not None else "")
    with kpi_cols[2]:
        mc_show = []
        for pct, val in [("75%", _mc_75), ("85%", _mc_85), ("90%", _mc_90)]:
            if val is not None and val >= _min_5yr:
                mc_show.append(f"{pct}:{val:.0f}")
        st.metric("MC 进面线", " | ".join(mc_show) if mc_show else "—")
    with kpi_cols[3]:
        years_count = df["year"].nunique() if "year" in df.columns else "—"
        st.metric("覆盖年份", str(years_count), f"{len(df)} 条记录")

    # ── 结果列表 ──────────────────────────────────────────────────────
    with st.expander("📄 详细统计结果"):

        # —— 各科与总分的相关性 + 各科之间的相关性 ——
        pearson = analyzer.results.get("pearson", {})
        r_mat = pearson.get("r")
        if r_mat is not None and not r_mat.empty:
            st.markdown("**① 各科与总分的 Pearson 相关系数**")
            if "总分" in r_mat.columns:
                total_corr = r_mat["总分"].drop("总分", errors="ignore").sort_values(ascending=False)
                st.dataframe(total_corr.to_frame("与总分的相关系数"), width="stretch")
            st.markdown("**② 各科之间的 Pearson 相关系数矩阵**")
            no_total = r_mat.drop(columns=["总分"], errors="ignore").drop(index=["总分"], errors="ignore")
            if not no_total.empty:
                st.dataframe(no_total, width="stretch")

        # —— 偏差分校准（无数据则隐藏） ——
        deviation = analyzer.results.get("deviation", {})
        if isinstance(deviation, dict) and deviation.get("has_deviation"):
            st.markdown("**③ 专业课偏差分校准**")
            dv_vals = deviation.get("values")
            if dv_vals is not None and not dv_vals.empty:
                st.dataframe(dv_vals, width="stretch")
        elif isinstance(deviation, dict) and not deviation.get("has_deviation", True):
            pass  # 隐藏

        # —— 专业百分位排名（无数据则隐藏） ——
        major_rank = analyzer.results.get("major_ranking", pd.DataFrame())
        if not major_rank.empty:
            st.markdown("**④ 各专业进面中位百分位排名**")
            st.dataframe(major_rank, width="stretch")

        # —— 其余通用指标 ——
        for key in ["beta", "dea", "quantile_regression", "cohen_d", "cv"]:
            val = analyzer.results.get(key)
            if val is None:
                continue
            st.markdown(f"**⑤ {key}**")
            if isinstance(val, pd.DataFrame):
                st.dataframe(val, width="stretch")
            elif isinstance(val, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, pd.DataFrame):
                        st.markdown(f"&nbsp;&nbsp;*{k2}*")
                        st.dataframe(v2, width="stretch")
                    else:
                        st.write(f"&nbsp;&nbsp;{k2}: {v2}")

        # —— 蒙特卡罗 ——
        mc = analyzer.results.get("monte_carlo", {})
        if mc and "error" not in mc:
            st.markdown("**⑥ 蒙特卡罗模拟 — 进面概率 vs 分数线**")
            _min5 = float(df["总分"].mean() - 2 * df["总分"].std()) if "总分" in df.columns else 0
            mc_rows = []
            for pct, key in [("75%", "75%_score"), ("85%", "85%_score"), ("90%", "90%_score")]:
                val = mc.get(key)
                if val is not None and val >= _min5:
                    mc_rows.append({"进面概率": pct, "建议分数线": round(val, 1)})
            if mc_rows:
                st.dataframe(pd.DataFrame(mc_rows), width="stretch")
            else:
                st.info("所有模拟分数线均低于历史最低值，不予显示。")

    # ── 报告生成 ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📝 生成的分析帖（LLM Prompt）")

    if st.session_state.report_md:
        st.markdown(
            "下方文本已用真实统计数据填充 `templates/main_prompt.md` 模板，"
            "可直接复制后交给大语言模型生成分析帖。"
        )

        tab1, tab2 = st.tabs(["渲染预览", "Markdown 源码"])

        with tab1:
            st.markdown(st.session_state.report_md)

        with tab2:
            st.code(st.session_state.report_md, language="markdown", line_numbers=True)

        st.info("💡 选中下方 Markdown 源码区域，使用 `Ctrl+A` → `Ctrl+C` 一键复制。")

        st.download_button(
            "⬇️ 下载 Markdown 文件",
            data=st.session_state.report_md,
            file_name=f"{st.session_state.school_name}_analysis_prompt.md".replace(" ", "_"),
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        st.info("请先上传数据并点击「运行全量分析」。")

# ── 空状态提示 ────────────────────────────────────────────────────────

else:
    st.info("👈 请在左侧边栏上传数据文件，然后点击「运行全量分析」。")
    st.markdown("""
    **支持的文件格式：**
    - Excel（.xlsx / .xls）
    - PDF（.pdf）— 包含表格的扫描件或电子版
    - Word（.docx）
    - 图片（.jpg / .png）— 含清晰表格的截图

    **分析流程：**
    1. 上传数据 → 2. 多格式解析 → 3. 全量统计分析 → 4. 生成 LLM Prompt
    """)
