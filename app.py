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
import re as _re
import json as _json
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
from src.ai_advisor import generate_ai_insights_from_markdown

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
    page_title="马理论考研初试助力系统",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
#  全局 CSS — CSS 变量双模式体系
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    :root {
        --bg-page: var(--background-color);
        --card-bg: var(--secondary-background-color);
        --text-primary: var(--text-color);
        --border-light: rgba(128, 128, 128, 0.2);
        --table-header-bg: rgba(128, 128, 128, 0.05);
        --sidebar-card-bg: rgba(25, 118, 210, 0.05);
        --marquee-bg: rgba(25, 118, 210, 0.08);
        --accent-blue: #1976D2;
        --accent-hover: #1565C0;
    }

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: var(--bg-page) !important;
    }
    [data-testid="stVerticalBlock"] > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    div.stTabContent, div[data-testid="stTabs"], .stTabs [data-baseweb="tab-panel"] {
        background-color: transparent !important;
    }

    a { color: var(--accent-blue) !important; }
    a:hover { color: var(--accent-hover) !important; text-decoration: underline; }

    /* ── 门户卡片 ── */
    .portal-card {
        background: var(--card-bg);
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        border: 1px solid var(--border-light);
        color: var(--text-primary);
    }
    .portal-card * { color: var(--text-primary); }
    .portal-card h3, .portal-card h4 {
        border-left: 4px solid var(--accent-blue);
        padding-left: 10px;
        margin-top: 0;
        font-size: 1.05rem;
    }

    /* ── 侧边栏向导卡片 ── */
    .sidebar-card {
        background: var(--sidebar-card-bg);
        border-radius: 4px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.6rem;
        border: 1px solid var(--border-light);
        color: var(--text-primary);
    }
    .sidebar-card * { color: var(--text-primary); }

    /* ── 滚动激励栏 ── */
    .marquee-bar {
        background: var(--marquee-bg);
        border-left: 4px solid var(--accent-blue);
        padding: 0.35rem 0;
        margin-bottom: 0.8rem;
        border-radius: 3px;
        overflow: hidden;
        color: var(--text-primary);
    }
    .marquee-track {
        display: inline-block;
        white-space: nowrap;
        animation: marquee 40s linear infinite;
        color: var(--accent-blue);
        font-size: 0.9rem;
    }
    @keyframes marquee {
        0%   { transform: translateX(100vw); }
        100% { transform: translateX(-100%); }
    }

    /* ── 表格 ── */
    .stMarkdown table, .stDataFrame table {
        border-collapse: collapse;
        width: 100%;
        border: 1px solid var(--border-light) !important;
        font-size: 0.88rem;
        color: var(--text-primary) !important;
        background: transparent !important;
    }
    .stMarkdown th {
        background-color: var(--table-header-bg) !important;
        border-bottom: 2px solid var(--accent-blue) !important;
        padding: 6px 10px !important;
        text-align: center;
        color: var(--text-primary) !important;
    }
    .stMarkdown td {
        padding: 5px 10px !important;
        border-bottom: 1px solid var(--border-light) !important;
        text-align: center;
        color: var(--text-primary) !important;
    }
    .stMarkdown tr:hover td { background-color: rgba(25,118,210,0.06) !important; }

    /* ── metric 卡片 ── */
    div[data-testid="metric-container"] {
        background: var(--card-bg);
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        padding: 0.5rem 0.8rem;
        color: var(--text-primary);
    }

    /* ── 标题降级 ── */
    h2, .stMarkdown h2 {
        font-size: 1.3rem;
        border-bottom: 2px solid var(--accent-blue);
        padding-bottom: 4px;
        margin: 1.2rem 0 0.6rem;
        color: var(--text-primary);
    }
    h3, .stMarkdown h3 {
        font-size: 1.1rem;
        margin: 1rem 0 0.4rem;
        color: var(--text-primary);
    }
    .stMarkdown p, .stMarkdown li, .stMarkdown span { color: var(--text-primary); }
    .stMarkdown { color: var(--text-primary); }
</style>
""", unsafe_allow_html=True)


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
#  UI — 侧边栏（向导 + 数据面板）
# ═══════════════════════════════════════════════════════════════════════

st.sidebar.markdown("<h2 style='font-size:1.2rem;margin-bottom:0.5rem;'>📂 择校向导</h2>", unsafe_allow_html=True)

st.sidebar.info(
    "**操作流程**\n\n"
    "第一步 👉 确认学校名称\n\n"
    "第二步 👉 确认是否按方向招生\n\n"
    "第三步 👉 上传复试名单文件\n\n"
    "第四步 👉 点击「运行全量分析」"
)

with st.sidebar:
    # ── 步骤 A：高校名称 ──
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("**① 目标高校**")
    _school_input = st.text_input(
        "校名", value=st.session_state.school_name,
        label_visibility="collapsed", key="wizard_school",
    )
    st.session_state.school_name = _school_input
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 步骤 B：是否分方向 ──
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("**② 招生模式**")
    _saved_mode = st.session_state.struct_mode
    _mode = st.radio(
        "招生模式", ["全院统一", "按研究方向"],
        index=0 if _saved_mode == "全院统一" else 1,
        key="wizard_mode", label_visibility="collapsed",
    )
    st.session_state.struct_mode = _mode
    if _mode == "按研究方向":
        st.markdown("**方向列名**")
        _col_options = ["自动检测", "报考专业", "专业", "报考方向", "方向", "研究方向", "手动指定"]
        _col_idx = _col_options.index(st.session_state.struct_col) if st.session_state.struct_col in _col_options else 0
        _sel_col = st.selectbox("选择方向所在列", _col_options, index=_col_idx, key="wizard_col")
        if _sel_col == "手动指定":
            _custom = st.text_input("输入自定义列名", value=st.session_state.struct_col or "", key="wizard_custom")
            st.session_state.struct_col = _custom
        elif _sel_col == "自动检测":
            st.session_state.struct_col = None
        else:
            st.session_state.struct_col = _sel_col
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 步骤 C：上传文件 ──
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("**③ 上传数据**")
    uploaded_files = st.file_uploader(
        "支持 xlsx / pdf / docx / jpg / png",
        type=["xlsx", "xls", "pdf", "docx", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="file_uploader",
    )
    if uploaded_files and len(uploaded_files) > 0:
        st.session_state.uploaded_files = uploaded_files
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 步骤 D：运行 ──
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("**④ 执行分析**")
    analyze_btn = st.button("🚀 运行全量分析", type="primary", use_container_width=True,
                            disabled=len(st.session_state.uploaded_files) == 0)
    reprocess_btn = st.button("🔄 重新生成报告", use_container_width=True,
                              disabled=not st.session_state.data_loaded)
    reload_btn = st.button("🔄 重载数据（按修正方向）", use_container_width=True,
                           disabled=not st.session_state.data_loaded)
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(f"已上传 {len(st.session_state.uploaded_files)} 个文件")


# ═══════════════════════════════════════════════════════════════════════
#  UI — 数据处理与分析的触发逻辑（保留原位，供按钮回调）
# ═══════════════════════════════════════════════════════════════════════

if analyze_btn and st.session_state.uploaded_files:
    # 切换学校时清空旧 AI 缓存，防止数据错位
    st.session_state.pop("ai_insights_cache", None)
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

if reload_btn and st.session_state.df is not None:
    st.cache_data.clear()
    with st.spinner("按修正后的结构重新分析..."):
        try:
            mode = st.session_state.struct_mode
            scol = st.session_state.struct_col
            analyzer = _run_analysis(st.session_state.df, struct_mode=mode, struct_col=scol)
            st.session_state.analyzer = analyzer
            st.session_state.report_md = _generate_final_report(analyzer, st.session_state.school_name)
            st.session_state.data_loaded = True
            _save_school_patch(st.session_state.school_name, mode, scol)
            st.success(f"重载完成（{mode}）！")
            st.rerun()
        except Exception as e:
            st.error(f"重载失败: {e}")
            import traceback
            st.exception(e)

# ═══════════════════════════════════════════════════════════════════════
#  UI — 主区域（门户布局）
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
#  门户网站横幅大标题 (Banner)
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="
    background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-hover) 100%);
    padding: 1.8rem 2rem; 
    border-radius: 8px; 
    margin-bottom: 2rem; 
    text-align: center; 
    box-shadow: 0 4px 12px rgba(25, 118, 210, 0.2);
">
    <h1 style="
        color: #FFFFFF !important; 
        margin: 0; 
        font-size: 2.2rem; 
        font-weight: 800; 
        letter-spacing: 2px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        border: none;
    ">
        🎓 马理论考研全景量化分析 & 背诵库门户
    </h1>
    <p style="
        color: rgba(255,255,255,0.9) !important; 
        margin: 0.8rem 0 0 0; 
        font-size: 1.05rem; 
        letter-spacing: 1px;
    ">
        穿透数据迷雾 · 直击进面真相 | 纯普通计划独立样本空间智库
    </p>
</div>
""", unsafe_allow_html=True)

_tabs = st.tabs(["🎯 目标高校分析", "📕 马理论全景背诵库", "📈 全国马理论大盘", "📚 历史真题库（敬请期待）"])

with _tabs[0]:
    # ── 考研激励滚动栏 ──
    try:
        import random
        _all_lines = [l.strip() for l in open(PROJECT_ROOT / "data" / "motivation.txt", encoding="utf-8") if l.strip()]
        _picked = random.sample(_all_lines, min(5, len(_all_lines)))
        _marquee_text = "  ✦  ".join(_picked)
    except Exception:
        _marquee_text = "考研是一场修行，坚持到最后的人，运气都不会太差。"
    st.markdown(f"""
    <div class="marquee-bar">
        <div class="marquee-track">{_marquee_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # 顶部状态栏（含择校锁定信息）
    _status_color = "#52c41a" if st.session_state.data_loaded else "#faad14"
    _status_text = "分析已完成" if st.session_state.data_loaded else "待分析"
    _direction_label = st.session_state.struct_mode
    if _direction_label == "按研究方向" and st.session_state.struct_col:
        _direction_label += f"（{st.session_state.struct_col}）"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;flex-wrap:wrap;">
        <span style="background:#1976D2;color:#fff;padding:2px 12px;border-radius:10px;font-size:0.8rem;white-space:nowrap;">✓ 锁定</span>
        <span style="font-size:0.95rem;"><strong>🏫 {st.session_state.school_name}</strong></span>
        <span style="font-size:0.85rem;color:var(--text-color,#555);">📂 {_direction_label}</span>
        <span style="font-size:0.85rem;color:var(--text-color,#555);">📄 {len(st.session_state.uploaded_files)} 个文件</span>
        <span style="font-size:0.85rem;color:{_status_color};">● {_status_text}</span>
    </div>
    """, unsafe_allow_html=True)

    # 三栏布局
    left_col, mid_col, right_col = st.columns([1, 2.2, 1])

    with left_col:
        # ── 高校信息卡片 ──
        st.markdown('<div class="portal-card">', unsafe_allow_html=True)
        st.markdown("##### 🏫 高校信息")
        st.markdown(f"**校名**：{st.session_state.school_name}")
        if st.session_state.data_loaded:
            df = st.session_state.df
            yr = df["year"].nunique() if "year" in df.columns else 0
            st.markdown(f"**覆盖年份**：{yr} 年")
            st.markdown(f"**样本量**：{len(df)} 人")
        else:
            st.markdown("**状态**：待上传数据")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── 量化术语科普 ──
        st.markdown('<div class="portal-card">', unsafe_allow_html=True)
        st.markdown("##### 📖 量化术语科普")
        st.markdown("**Z-score**：$z = \\frac{x-\\mu}{\\sigma}$，衡量该科分数偏离平均的程度（>2 为异常高分，<−2 为异常低分）")
        st.markdown("**DEA 效率**：1.0 表示该科是当年最高均分科目，0.8 表示比最高低 20%")
        st.markdown("**蒙特卡罗模拟**：基于历史均值和协方差矩阵，生成 10000 次 2027 年模拟成绩，统计进面概率")
        st.markdown("**Cohen's d**：高分半与低分半之间的标准化均值差，>0.8 为区分度大")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── 快捷链接 ──
        st.markdown('<div class="portal-card">', unsafe_allow_html=True)
        st.markdown("##### 🔗 快捷资源")
        st.markdown("- [研招网](https://yz.chsi.com.cn)")
        st.markdown("- [学信网](https://www.chsi.com.cn)")
        st.markdown("- [各校招生简章汇总](https://yz.chsi.com.cn/zsml/zyfx_search.jsp)")
        st.markdown('</div>', unsafe_allow_html=True)

    with mid_col:
        if st.session_state.data_loaded and st.session_state.df is not None:
            df = st.session_state.df
            analyzer = st.session_state.analyzer

            # ── 图表 + KPI ──
            st.markdown('<div class="portal-card">', unsafe_allow_html=True)
            st.markdown("##### 📊 核心指标速览")
            beta = analyzer.results.get("beta", pd.DataFrame())
            dea = analyzer.results.get("dea_overall", pd.DataFrame())
            mc = analyzer.results.get("monte_carlo", {})
            _mc_75 = mc.get("75%_score")
            _mc_85 = mc.get("85%_score")
            _mc_90 = mc.get("90%_score")
            _min_5yr = float(df["总分"].mean() - 2 * df["总分"].std()) if "总分" in df.columns else 0
            kcols = st.columns(4)
            with kcols[0]:
                bm = None
                if not beta.empty and beta["beta"].notna().any():
                    try:
                        bm = beta.loc[beta["beta"].idxmax()]
                    except Exception:
                        bm = None
                _subj_b = "—"
                _beta_val = ""
                if bm is not None and not (hasattr(bm, "empty") and bm.empty):
                    _subj_b = str(bm.get("subject", "—")) if hasattr(bm, "get") else "—"
                    _beta_val = f"{float(bm['beta']):.2f}" if "beta" in bm.index else ""
                st.metric("β 最高", _subj_b, _beta_val)
            with kcols[1]:
                dm = None
                if not dea.empty:
                    try:
                        dm = dea.loc[dea["efficiency"].idxmax()]
                    except Exception:
                        dm = None
                _subj_d = "—"
                _eff_val = ""
                if dm is not None and not (hasattr(dm, "empty") and dm.empty):
                    _subj_d = str(dm.get("subject", "—")) if hasattr(dm, "get") else "—"
                    _eff_val = f"{float(dm['efficiency']):.2f}" if "efficiency" in dm.index else ""
                st.metric("DEA 前沿", _subj_d, _eff_val)
            with kcols[2]:
                mc_show = []
                for pct, val in [("75%", _mc_75), ("85%", _mc_85)]:
                    if val is not None and val >= _min_5yr:
                        mc_show.append(f"{pct}:{val:.0f}")
                st.metric("MC 进面线", " | ".join(mc_show) if mc_show else "—")
            with kcols[3]:
                yc = df["year"].nunique() if "year" in df.columns else "—"
                st.metric("覆盖年份", str(yc), f"{len(df)} 条")
            st.markdown('</div>', unsafe_allow_html=True)

            # ── 可视化图表 ──
            st.markdown('<div class="portal-card">', unsafe_allow_html=True)
            st.markdown("##### 📉 趋势 & 相关性")
            gcol1, gcol2 = st.columns(2)
            with gcol1:
                st.plotly_chart(_plot_trend_chart(analyzer), use_container_width=True)
            with gcol2:
                st.pyplot(_plot_heatmap(analyzer), use_container_width=True)
            cv_fig = _plot_cv_chart(analyzer)
            if cv_fig.data:
                st.plotly_chart(cv_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── 数据预览 ──
            with st.expander("📋 原始数据预览", expanded=False):
                st.dataframe(df.head(100), width="stretch")
                st.caption(f"共 {len(df)} 行 × {len(df.columns)} 列")

            # ── 详细统计结果 ──
            with st.expander("📄 详细统计结果"):
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
                deviation = analyzer.results.get("deviation", {})
                if isinstance(deviation, dict) and deviation.get("has_deviation"):
                    st.markdown("**③ 专业课偏差分校准**")
                    dv_vals = deviation.get("values")
                    if dv_vals is not None and not dv_vals.empty:
                        st.dataframe(dv_vals, width="stretch")
                major_rank = analyzer.results.get("major_ranking", pd.DataFrame())
                if not major_rank.empty:
                    st.markdown("**④ 各专业进面中位百分位排名**")
                    st.dataframe(major_rank, width="stretch")
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
                mc2 = analyzer.results.get("monte_carlo", {})
                if mc2 and "error" not in mc2:
                    st.markdown("**⑥ 蒙特卡罗模拟 — 进面概率 vs 分数线**")
                    _min5 = float(df["总分"].mean() - 2 * df["总分"].std()) if "总分" in df.columns else 0
                    mc_rows = []
                    for pct, key in [("75%", "75%_score"), ("85%", "85%_score"), ("90%", "90%_score")]:
                        val = mc2.get(key)
                        if val is not None and val >= _min5:
                            mc_rows.append({"进面概率": pct, "建议分数线": round(val, 1)})
                    if mc_rows:
                        st.dataframe(pd.DataFrame(mc_rows), width="stretch")
                    else:
                        st.info("所有模拟分数线均低于历史最低值，不予显示。")

        else:
            st.markdown('<div class="portal-card">', unsafe_allow_html=True)
            st.info("👈 请在左侧面板上传数据文件，然后点击「运行全量分析」。")
            st.markdown("""
            **支持的文件格式：**
            - Excel（.xlsx / .xls）
            - PDF（.pdf）— 包含表格的扫描件或电子版
            - Word（.docx）
            - 图片（.jpg / .png）— 含清晰表格的截图
            """)
            # ── 小红书引流卡片 ──
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#fff1f0 0%,#fff2e8 100%);
                        border:1px solid #ffa39e;padding:1.2rem;border-radius:8px;margin-top:1rem;margin-bottom:1.5rem;
                        box-shadow:0 2px 8px rgba(255,77,79,0.05);">
                <p style="margin:0;font-size:1rem;color:#cf1322;font-weight:bold;display:flex;align-items:center;">
                    <span style="font-size:1.3rem;margin-right:8px;">📌</span>
                    找不到院校数据？部分热门院校考研量化分析已在小红书主页同步更新，无需自行上传！
                </p>
                <p style="margin:0.5rem 0 0 0;font-size:0.9rem;color:#434343;line-height:1.5;">
                    由于各高校导出的原始档案格式多变，自行上传可能因字段不匹配导致系统分析失败。建议优先前往主页直接检索已跑通的现成量化成果。
                </p>
                <div style="margin-top:0.8rem;">
                    <a href="https://www.xiaohongshu.com/user/profile/65e7cfe2000000000500b6ed" target="_blank"
                       style="display:inline-block;background:#ff4d4f;color:white;text-decoration:none;
                              padding:0.5rem 1.2rem;border-radius:20px;font-weight:bold;font-size:0.9rem;
                              box-shadow:0 2px 6px rgba(255,77,79,0.3);transition:all 0.3s;">
                        👉 点击直接前往我的小红书主页查询
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        # ── 招生数据速递 ──
        st.markdown('<div class="portal-card">', unsafe_allow_html=True)
        st.markdown("##### 📈 招生数据速递")
        if st.session_state.data_loaded:
            df = st.session_state.df
            st.markdown(f"**样本量**：{len(df)} 人")
            if "year" in df.columns:
                yrs = sorted(df["year"].unique())
                st.markdown(f"**年份**：{'、'.join(str(int(y)) for y in yrs)}")
            if "总分" in df.columns:
                st.markdown(f"**总分中位数**：{df['总分'].median():.0f}")
                st.markdown(f"**总分范围**：{df['总分'].min():.0f} ~ {df['总分'].max():.0f}")
        else:
            st.markdown("待上传数据后展示")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── 热点专题 ──
        st.markdown('<div class="portal-card">', unsafe_allow_html=True)
        st.markdown("##### 🔥 考研热词")
        st.markdown("`大小年` `等百分位等值` `协方差矩阵` `分位数回归` `DEA效率` `Cohen d` `MC模拟`")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 报告生成（全宽，AI 解读前置） ──
    if st.session_state.data_loaded and st.session_state.report_md:

        # ── ① 小红书引流卡片 ──
        st.markdown(f"""
        <div style="background:#fff1f0;border:1px solid #ffa39e;padding:1rem;border-radius:8px;margin-bottom:1rem;">
            <p style="margin:0;font-size:0.95rem;color:#cf1322;font-weight:bold;">
                📌 找不到数据？部分热门院校考研量化分析已在小红书主页同步更新，无需自行上传！
                <a href="https://www.xiaohongshu.com/user/profile/65e7cfe2000000000500b6ed" target="_blank" style="color:#1890ff;text-decoration:underline;margin-left:5px;">
                    👉 点击直接前往我的小红书主页查询
                </a>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── ② AI 考研导师专属解读面板 ──
        st.markdown("---")
        st.markdown("### 🤖 DeepSeek AI 考研导师专属解读面板")
        if st.button("✨ 一键唤醒 DeepSeek：深度解析目标院校", type="primary", use_container_width=True):
            if "report_md" in st.session_state and st.session_state.report_md:
                with st.spinner("🔮 DeepSeek 正在用大模型进行多维度量化推演与话术组织，请稍候..."):
                    ai_insights = generate_ai_insights_from_markdown(st.session_state.report_md)
                    st.session_state.ai_insights_cache = ai_insights
            else:
                st.warning("请先在上方成功生成基础院校量化报告。")
        if "ai_insights_cache" in st.session_state and st.session_state.ai_insights_cache:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(24,144,255,0.04) 0%,rgba(114,46,209,0.04) 100%);
                        padding:1.2rem;border-left:5px solid #722ed1;border-radius:6px;margin-top:1rem;margin-bottom:1rem;">
                <h4 style="margin-top:0;margin-bottom:0.5rem;color:#722ed1;font-weight:bold;">🔮 DeepSeek 大咖内参视角解读成功</h4>
                <p style="margin:0;font-size:0.9rem;color:var(--text-secondary);">基于全景量化指标与 DEA 前沿效率，已为您自动生成高价值备考指导。</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(st.session_state.ai_insights_cache)
            st.download_button(
                label="📥 下载这份 DeepSeek 专属报考内参",
                data=st.session_state.ai_insights_cache,
                file_name=f"{st.session_state.get('school_name','目标院校')}_AI报考内参.md",
                mime="text/markdown",
                key="btn_download_ai_report",
            )

        # ── ③ 基础量化分析帖（后置） ──
        st.markdown("---")
        st.markdown('<div class="portal-card">', unsafe_allow_html=True)
        st.markdown("##### 📝 基础量化分析帖（LLM Prompt）")
        st.markdown("下方文本已用真实统计数据填充模板，可复制后交给大语言模型生成分析帖。")
        rtab1, rtab2 = st.tabs(["渲染预览", "Markdown 源码"])
        with rtab1:
            st.markdown(st.session_state.report_md)
        with rtab2:
            st.code(st.session_state.report_md, language="markdown", line_numbers=True)
        st.download_button("⬇️ 下载 Markdown 文件",
                           data=st.session_state.report_md,
                           file_name=f"{st.session_state.school_name}_analysis_prompt.md".replace(" ", "_"),
                           mime="text/markdown", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  Tab 1 — 📕 马理论全景背诵库
# ═══════════════════════════════════════════════════════════════════════
with _tabs[1]:
    _CN = ["全选","导论","第一章","第二章","第三章","第四章","第五章","第六章","第七章",
           "第八章","第九章","第十章","第十一章","第十二章","第十三章","第十四章","第十五章","第十六章","第十七章"]
    _COURSES_AND_CHAPTERS = {
        "马克思主义基本原理": [(c, c) for c in _CN[:9]],
        "毛泽东思想和中国特色社会主义理论体系概论": [(c, c) for c in _CN[:10]],
        "习近平新时代中国特色社会主义思想概论": [(c, c) for c in _CN],
    }

    def _fmt_answer(text: str) -> str:
        if not text:
            return ""
        text = _re.sub(r'(第一|第二|第三|第四|第五|首先|其次|最后|一是|二是|三是|四是|五是)', r'<br/><br/><b>\1</b>', text)
        text = _re.sub(r'([\(\（][1-9][\)\）])', r'<br/><br/><b>\1</b>', text)
        return text

    _SUBJECT_FILES = {
        "马克思主义基本原理": "data/mayuan.json",
        "毛泽东思想和中国特色社会主义理论体系概论": "data/maogai.json",
        "习近平新时代中国特色社会主义思想概论": "data/xsixiang.json",
    }
    import json, random
    _all_qs = []
    for _subj, _fp in _SUBJECT_FILES.items():
        _p = Path(__file__).resolve().parent / _fp
        if _p.exists():
            _all_qs.extend(json.loads(_p.read_text(encoding="utf-8")))
    _df_qs = pd.DataFrame(_all_qs)
    if _df_qs.empty:
        st.markdown('<div class="portal-card">', unsafe_allow_html=True)
        st.error("题库文件 data/*.json 不存在，请先运行 scripts/parse_mayuan.py")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # ── 三列联动筛选 ──
        st.markdown('<div class="portal-card">', unsafe_allow_html=True)
        st.markdown("##### 🔍 检索筛选")
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            _subjects = list(_COURSES_AND_CHAPTERS.keys())
            _sel_subj = st.selectbox("科目", _subjects, key="recite_subj")
        with fcol2:
            _chapter_tuples = _COURSES_AND_CHAPTERS.get(_sel_subj, [("全选", "全选")])
            _chapter_display = [t[0] for t in _chapter_tuples]
            _sel_ch_display = st.selectbox("章节", _chapter_display, key="recite_ch")
            _ch_to_data = dict(_chapter_tuples)
            _sel_ch = _ch_to_data.get(_sel_ch_display, "全选")
        with fcol3:
            _types = ["全部"] + sorted(_df_qs["type"].unique())
            _sel_type = st.selectbox("题型", _types, key="recite_type")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── 关键词搜索 ──
        _keyword = st.text_input("🔍 输入核心关键词（如：唯物辩证法、实践是检验真理的唯一标准）", key="recite_kw")

        # ── 层层过滤（科目 → 章节 → 题型 → 关键词） ──
        _df_current = _df_qs[_df_qs["subject"] == _sel_subj]
        if _sel_ch != "全选":
            _df_current = _df_current[_df_current["chapter"] == _sel_ch]
        if _sel_type != "全部":
            _df_current = _df_current[_df_current["type"] == _sel_type]
        if _keyword:
            _df_current = _df_current[
                _df_current["question"].str.contains(_keyword, case=False, na=False) |
                _df_current["answer"].str.contains(_keyword, case=False, na=False)
            ]

        st.caption(f"共 {len(_df_current)} 条题目")

        # ── 🎲 盲盒（使用 session_state 持久化抽题结果） ──
        _pool_blind = _df_current if not _df_current.empty else _df_qs[_df_qs["subject"] == _sel_subj]
        st.markdown('<div class="portal-card" style="padding:0.6rem 1rem;border-left:4px solid #FF9800;">', unsafe_allow_html=True)
        if st.button("🎲 盲抽今日背诵大题", key="btn_random_box", use_container_width=True):
            if not _pool_blind.empty:
                st.session_state.current_blind_question = _pool_blind.sample(n=1).iloc[0].to_dict()
            else:
                st.warning("当前题库暂无满足条件的数据可供盲抽。")
        if "current_blind_question" in st.session_state and st.session_state.current_blind_question:
            _bq = st.session_state.current_blind_question
            st.markdown(f"""<div style="padding:0.6rem 0;"><h4 style="margin:0 0 4px 0;color:#FF9800;">🎲 背诵盲盒：📕 [{_bq['subject']}] · [{_bq['chapter']}] · [{_bq['type']}]</h4><p style="font-weight:bold;font-size:1.05rem;margin:6px 0;">💡 题目：{_bq['question']}</p></div>""", unsafe_allow_html=True)
            if st.checkbox("👁️ 开启盲盒：显示参考答案要点", key="toggle_box_answer"):
                st.markdown(f'<div style="padding:1rem;border-left:3px solid #FF9800;background:rgba(128,128,128,0.03);margin-top:0.3rem;line-height:1.7;color:var(--text-primary);">{_fmt_answer(str(_bq["answer"]))}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── 题目卡片列表 ──
        if _df_current.empty:
            st.info("当前筛选条件下没有题目，请调整筛选条件。")
        for _idx, (_, _row) in enumerate(_df_current.iterrows()):
            _q, _a, _ch, _tp = str(_row["question"]), str(_row["answer"]), str(_row["chapter"]), str(_row["type"])
            if _keyword:
                _esc = _re.escape(_keyword)
                _q_hl = _re.sub(f'({_esc})', r'<span style="background:rgba(25,118,210,0.15);font-weight:bold;">\1</span>', _q, flags=_re.IGNORECASE)
                _a_hl = _re.sub(f'({_esc})', r'<span style="background:rgba(25,118,210,0.15);font-weight:bold;">\1</span>', _a, flags=_re.IGNORECASE)
            else:
                _q_hl, _a_hl = _q, _a
            st.markdown(f"""<div class="portal-card" style="padding:0.8rem 1.2rem;"><div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;flex-wrap:wrap;"><span style="font-size:0.7rem;background:var(--accent-blue);color:#fff;padding:1px 8px;border-radius:8px;">{str(_row.get('subject',''))[:6]}..</span><span style="font-size:0.7rem;color:var(--text-secondary);">|</span><span style="font-size:0.7rem;background:rgba(25,118,210,0.1);color:var(--accent-blue);padding:1px 8px;border-radius:8px;">{_ch}</span><span style="font-size:0.7rem;color:var(--text-secondary);">·</span><span style="font-size:0.7rem;color:var(--text-secondary);">{_tp}</span></div><div style="font-size:0.95rem;line-height:1.5;color:var(--text-primary);">💡 {_q_hl}</div></div>""", unsafe_allow_html=True)
            if st.checkbox("👁️ 显示参考答案", key=f"reveal_main_{_ch}_{_tp}_{_idx}"):
                st.markdown(f'<div style="padding:10px 14px;border-radius:4px;background:var(--sidebar-card-bg);color:var(--text-primary);font-size:0.9rem;line-height:1.7;margin-bottom:0.6rem;">{_fmt_answer(_a_hl)}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  Tab 2 — 📈 全国马理论大盘
# ═══════════════════════════════════════════════════════════════════════
with _tabs[2]:
    _PUBLIC_DIR = Path(__file__).resolve().parent / "public" / "data"
    _HOT_SCHOOLS = [
        ("Sun_Yat_sen_University", "中山大学", "524条 · 5年覆盖"),
        ("Xi_an_Jiaotong_University", "西安交通大学", "399条 · 5年覆盖"),
        ("Sichuan_University", "四川大学", "428条 · 4年覆盖"),
        ("Nanjing_Normal_University", "南京师范大学", "251条 · 4年覆盖"),
        ("Beijing_University_of_Chemical_Technology", "北京化工大学", "192条 · 5年覆盖"),
    ]

    # ── 如果已选中某校，显示详情页 ──
    if "selected_school_cache" in st.session_state and st.session_state.selected_school_cache is not None:
        _sc = st.session_state.selected_school_cache
        if st.button("← 返回院校列表", key="back_to_school_list"):
            st.session_state.selected_school_cache = None
            st.rerun()
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;flex-wrap:wrap;">
            <span style="background:#52c41a;color:#fff;padding:2px 10px;border-radius:10px;font-size:0.75rem;">已解锁</span>
            <span style="font-size:1.1rem;font-weight:bold;">🏫 {_sc['_meta']['school_cn']}</span>
            <span style="font-size:0.85rem;color:var(--text-secondary);">📄 {_sc['_meta']['total_records']} 条记录</span>
            <span style="font-size:0.85rem;color:var(--text-secondary);">📅 {', '.join(str(y) for y in _sc['_meta']['years'])}</span>
        </div>
        """, unsafe_allow_html=True)
        # 报告内容预览
        st.markdown("##### 📊 5年核心趋势概览")
        _trend_str = _sc.get("trend_last_three_years", "数据不足")
        _mc_str = _sc.get("monte_carlo_interpretation", "—")
        st.markdown(f"**趋势**：{_trend_str}")
        st.markdown(f"**蒙特卡罗模拟**：{_mc_str}")
        st.markdown("---")
        st.markdown("##### 📝 完整量化分析报告")
        _report_template = (Path(__file__).resolve().parent / "templates" / "main_prompt.md").read_text(encoding="utf-8")
        try:
            _rendered = _report_template.format(**_sc)
        except KeyError:
            _rendered = "模板渲染失败，部分数据缺失。"
        rtab_a, rtab_b = st.tabs(["渲染预览", "Markdown 源码"])
        with rtab_a:
            st.markdown(_rendered)
        with rtab_b:
            st.code(_rendered, language="markdown", line_numbers=True)
    else:
        # ── 院校选择网格 ──
        st.markdown('<div class="portal-card">', unsafe_allow_html=True)
        st.markdown("##### 📈 全国马理论大盘 — 精选高校直达")
        st.markdown('<p style="font-size:0.9rem;color:var(--text-secondary);">以下高校数据完整、分析结果已预先缓存，点击即可查看量化分析报告。</p>', unsafe_allow_html=True)
        _cols = st.columns(5)
        for _ci, (_key, _name, _desc) in enumerate(_HOT_SCHOOLS):
            _json_path = _PUBLIC_DIR / f"{_key}_analysis.json"
            _has_cache = _json_path.exists()
            with _cols[_ci]:
                st.markdown(f"""
                <div class="portal-card" style="padding:0.8rem;text-align:center;margin-bottom:0.4rem;">
                    <div style="font-size:1.6rem;margin-bottom:4px;">🏫</div>
                    <div style="font-weight:bold;font-size:0.9rem;">{_name}</div>
                    <div style="font-size:0.7rem;color:var(--text-secondary);">{_desc}</div>
                    <div style="font-size:0.7rem;margin-top:5px;">
                        <span style="background:{'#52c41a' if _has_cache else '#d9d9d9'};color:#fff;padding:1px 10px;border-radius:8px;">
                            {'✅ 已解锁' if _has_cache else '⏳ 待加入'}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if _has_cache:
                    if st.button(f"查看报告", key=f"enter_school_{_ci}", use_container_width=True):
                        _data = _json.loads(_json_path.read_text(encoding="utf-8"))
                        st.session_state.selected_school_cache = _data
                        st.rerun()
                else:
                    st.button(f"申请分析", key=f"req_school_{_ci}", disabled=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with _tabs[3]:
    st.markdown('<div class="portal-card">', unsafe_allow_html=True)
    st.markdown("##### 📚 历史真题库")
    st.markdown("该功能正在开发中。后续将收录各校历年真题、解析与刷题功能。")
    st.markdown('</div>', unsafe_allow_html=True)
