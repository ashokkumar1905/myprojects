"""
DataForge Pro — Industrial Data Cleaning Tool
Streamlit UI with full pipeline, analytics, and export.
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
import time
import os

from cleaner_engine import run_full_pipeline, detect_column_types
from visualizer import (
    plot_missing_heatmap,
    plot_dtype_distribution,
    plot_before_after_missing,
    plot_numeric_distributions,
    plot_correlation_heatmap,
    plot_ml_readiness_gauge,
    df_to_csv_bytes,
    df_to_excel_bytes,
    df_to_json_bytes,
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="DataForge Pro",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg: #0D1117;
    --panel: #161B22;
    --border: #30363D;
    --accent1: #00D4FF;
    --accent2: #7C3AED;
    --accent3: #10B981;
    --accent4: #F59E0B;
    --danger: #EF4444;
    --text: #E6EDF3;
    --muted: #8B949E;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* HERO HEADER */
.hero-header {
    background: linear-gradient(135deg, #0D1117 0%, #161B22 50%, #0D1117 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent2), var(--accent1), var(--accent3));
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    font-family: 'Syne', sans-serif;
    background: linear-gradient(135deg, #00D4FF, #7C3AED);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}

/* METRIC CARDS */
.metric-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
.metric-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    flex: 1;
    min-width: 130px;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
}
.metric-card.blue::after { background: var(--accent1); }
.metric-card.purple::after { background: var(--accent2); }
.metric-card.green::after { background: var(--accent3); }
.metric-card.orange::after { background: var(--accent4); }
.metric-card.red::after { background: var(--danger); }

.metric-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; font-family: 'JetBrains Mono', monospace; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: var(--text); margin-top: 0.2rem; }
.metric-delta { font-size: 0.75rem; margin-top: 0.15rem; font-family: 'JetBrains Mono', monospace; }
.delta-good { color: var(--accent3); }
.delta-bad { color: var(--danger); }
.delta-neutral { color: var(--muted); }

/* SECTION HEADER */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.8rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--accent1);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace;
}

/* STEP BADGE */
.step-badge {
    display: inline-flex;
    align-items: center;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    margin: 0.25rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text);
}
.step-badge .icon { margin-right: 0.4rem; }

/* SCORE RING */
.score-ring-wrap { text-align: center; }

/* CHECKLIST */
.checklist-item {
    padding: 0.4rem 0.8rem;
    margin: 0.25rem 0;
    background: var(--panel);
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    border-left: 3px solid var(--border);
}

/* TABLE STYLING */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, var(--accent2), var(--accent1));
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* DOWNLOAD BUTTONS */
.stDownloadButton > button {
    background: var(--panel);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 8px;
    width: 100%;
    font-family: 'JetBrains Mono', monospace;
}
.stDownloadButton > button:hover { border-color: var(--accent1); color: var(--accent1); }

/* TAB STYLING */
.stTabs [data-baseweb="tab-list"] {
    background: var(--panel);
    border-radius: 10px;
    border: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}
.stTabs [aria-selected="true"] {
    background: var(--bg);
    color: var(--accent1) !important;
}

/* UPLOAD ZONE */
.stFileUploader {
    border: 2px dashed var(--border);
    border-radius: 12px;
    background: var(--panel);
    padding: 1rem;
}

/* PROGRESS BAR */
.stProgress > div > div { background: linear-gradient(90deg, var(--accent2), var(--accent1)); }

/* EXPANDER */
.streamlit-expanderHeader {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}

/* SUCCESS/ERROR ALERTS */
.stSuccess { background: rgba(16, 185, 129, 0.1); border-left: 4px solid var(--accent3); border-radius: 6px; }
.stError { background: rgba(239, 68, 68, 0.1); border-left: 4px solid var(--danger); border-radius: 6px; }
.stInfo { background: rgba(0, 212, 255, 0.08); border-left: 4px solid var(--accent1); border-radius: 6px; }
.stWarning { background: rgba(245, 158, 11, 0.1); border-left: 4px solid var(--accent4); border-radius: 6px; }

/* SELECT/SLIDER */
.stSelectbox > div, .stSlider { font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def load_file(uploaded_file) -> pd.DataFrame | None:
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            # Try detecting encoding
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    content = uploaded_file.read().decode(enc)
                    uploaded_file.seek(0)
                    return pd.read_csv(StringIO(content))
                except Exception:
                    uploaded_file.seek(0)
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        elif name.endswith(".json"):
            return pd.read_json(uploaded_file)
        elif name.endswith(".tsv"):
            return pd.read_csv(uploaded_file, sep="\t")
        elif name.endswith(".parquet"):
            return pd.read_parquet(uploaded_file)
    except Exception as e:
        st.error(f"Could not load file: {e}")
    return None


def render_metric_card(label, value, delta=None, color="blue"):
    color_map = {
        "blue": "#00D4FF", "purple": "#7C3AED",
        "green": "#10B981", "orange": "#F59E0B", "red": "#EF4444"
    }
    accent = color_map.get(color, "#00D4FF")
    delta_html = ""
    if delta:
        if any(x in delta for x in ["↓", "resolved", "removed", "saved"]):
            delta_color = "#10B981"
        elif any(x in delta for x in ["↑", "remain"]):
            delta_color = "#EF4444"
        else:
            delta_color = "#8B949E"
        delta_html = f'<div style="font-size:0.72rem;font-family:JetBrains Mono,monospace;margin-top:0.15rem;color:{delta_color}">{delta}</div>'
    st.markdown(f"""
    <div style="background:#161B22;border:1px solid #30363D;border-radius:12px;
                padding:1rem 1.4rem;position:relative;overflow:hidden;
                border-bottom:3px solid {accent};">
        <div style="font-size:0.7rem;color:#8B949E;text-transform:uppercase;
                    letter-spacing:0.08em;font-family:JetBrains Mono,monospace">{label}</div>
        <div style="font-size:1.6rem;font-weight:700;color:#E6EDF3;margin-top:0.2rem">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def section_header(icon, title):
    st.markdown(f"""
    <div class="section-header">
        <span style="font-size:1.2rem">{icon}</span>
        <span class="section-title">{title}</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR: CONFIG
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
        <div style="font-size:2.2rem">⚙️</div>
        <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:1.1rem; color:#00D4FF;">DataForge Pro</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#8B949E;">v2.0 · Industrial Cleaner</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔧 Cleaning Pipeline")

    fix_names = st.toggle("Standardize Column Names", value=True)
    remove_dups = st.toggle("Remove Duplicates", value=True)
    fix_types = st.toggle("Auto-Fix Data Types", value=True)

    st.markdown("### 🩹 Missing Value Strategy")
    missing_strategy = st.selectbox(
        "Imputation Method",
        options=["smart", "mean", "median", "mode", "knn", "drop_rows", "none"],
        index=0,
        help="smart = median for >10% missing, mean otherwise; knn = K-Nearest Neighbors"
    )

    st.markdown("### 🎯 Outlier Handling")
    outlier_action = st.selectbox(
        "Outlier Action",
        options=["flag", "cap", "remove", "none"],
        index=0,
        help="flag = add indicator column; cap = clip to bounds; remove = delete rows"
    )
    outlier_method = st.selectbox(
        "Detection Method",
        options=["iqr", "zscore"],
        index=0,
        help="IQR = robust, recommended; Z-score = assumes normality"
    )

    st.markdown("### 🤖 ML Readiness")
    encode_cats = st.toggle("Encode Categoricals for ML", value=False)
    if encode_cats:
        encode_method = st.selectbox("Encoding Method", ["label", "onehot"], index=0)
    else:
        encode_method = "label"

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#8B949E; text-align:center;">
    Supports: CSV · Excel · JSON · TSV · Parquet<br>
    Max recommended: 500MB
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">⚙️ DataForge Pro</h1>
    <p class="hero-sub">Industrial-Grade Data Cleaning & ML Readiness Scorer</p>
    <p style="color:#8B949E; font-size:0.8rem; margin:0.5rem 0 0; font-family:'JetBrains Mono',monospace;">
        Upload any tabular dataset → Auto-clean → Visual Report → Download
    </p>
</div>
""", unsafe_allow_html=True)

# Upload zone
section_header("📁", "Upload Dataset")
uploaded_file = st.file_uploader(
    "Drop your dataset here",
    type=["csv", "xlsx", "xls", "json", "tsv", "parquet"],
    help="Supports CSV, Excel, JSON, TSV, Parquet"
)

if uploaded_file is None:
    st.markdown("""
    <div style="background:#161B22; border:1px solid #30363D; border-radius:12px; padding:2rem; text-align:center; margin-top:1rem;">
        <div style="font-size:3rem; margin-bottom:1rem;">📊</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.1rem; color:#E6EDF3; font-weight:600;">No Dataset Loaded</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:#8B949E; margin-top:0.5rem;">
            Upload a file above to begin cleaning.<br>Works with sales data, ML datasets, HR data, financial records, and more.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
#  LOAD & PREVIEW
# ─────────────────────────────────────────────

df_original = load_file(uploaded_file)
if df_original is None:
    st.stop()

section_header("🔍", "Original Dataset Preview")

r, c = df_original.shape
missing_total = df_original.isnull().sum().sum()
dup_total = df_original.duplicated().sum()
mem_mb = df_original.memory_usage(deep=True).sum() / 1024 / 1024
_m1, _m2, _m3, _m4, _m5 = st.columns(5)

with _m1: render_metric_card("Rows", f"{r:,}", color="blue")
with _m2: render_metric_card("Columns", f"{c:,}", color="purple")
with _m3: render_metric_card("Missing Cells", f"{missing_total:,}", f"{missing_total/max(r*c,1)*100:.1f}% of data", "orange" if missing_total > 0 else "green")
with _m4: render_metric_card("Duplicates", f"{dup_total:,}", color="red" if dup_total > 0 else "green")
with _m5: render_metric_card("Memory", f"{mem_mb:.1f} MB", color="blue")
with st.expander("👁️ Preview Raw Data (first 100 rows)", expanded=False):
    st.dataframe(df_original.head(100), use_container_width=True)

with st.expander("🧬 Column Type Detection", expanded=False):
    col_info = detect_column_types(df_original)
    type_df = pd.DataFrame([
        {
            "Column": col,
            "Original Dtype": info["original_dtype"],
            "Semantic Type": info["semantic"],
            "Unique Values": info["n_unique"],
            "Missing": info["n_missing"],
            "Missing %": f"{info['n_missing']/max(r,1)*100:.1f}%"
        }
        for col, info in col_info.items()
    ])
    st.dataframe(type_df, use_container_width=True)


# ─────────────────────────────────────────────
#  RUN PIPELINE
# ─────────────────────────────────────────────

section_header("🚀", "Run Cleaning Pipeline")

col_run, col_info_text = st.columns([1, 3])
with col_run:
    run_btn = st.button("⚡ Clean Dataset", use_container_width=True)

with col_info_text:
    steps_active = []
    if fix_names: steps_active.append("🏷️ Names")
    if remove_dups: steps_active.append("🗑️ Dups")
    if fix_types: steps_active.append("🔧 Types")
    steps_active.append(f"🩹 Missing({missing_strategy})")
    if outlier_action != "none": steps_active.append(f"🎯 Outliers({outlier_action})")
    if encode_cats: steps_active.append(f"🤖 Encode({encode_method})")
    
    badges = " ".join([f'<span class="step-badge"><span class="icon"></span>{s}</span>' for s in steps_active])
    st.markdown(f'<div style="margin-top:0.4rem">{badges}</div>', unsafe_allow_html=True)

if not run_btn:
    st.info("👆 Configure settings in the sidebar, then click **Clean Dataset** to begin.")
    st.stop()


# ─────────────────────────────────────────────
#  PIPELINE EXECUTION
# ─────────────────────────────────────────────

progress = st.progress(0, text="Initializing pipeline...")
time.sleep(0.2)

config = {
    "fix_names": fix_names,
    "remove_dups": remove_dups,
    "fix_types": fix_types,
    "missing_strategy": missing_strategy,
    "outlier_method": outlier_method,
    "outlier_action": outlier_action,
    "encode": encode_cats,
    "encode_method": encode_method,
}

progress.progress(10, text="Running cleaning engine...")
with st.spinner("Processing..."):
    results = run_full_pipeline(df_original, config)

progress.progress(60, text="Generating visualizations...")
df_cleaned = results["df_cleaned"]
progress.progress(85, text="Computing ML readiness score...")
time.sleep(0.2)
progress.progress(100, text="Done!")
time.sleep(0.3)
progress.empty()

st.success(f"✅ Pipeline complete! {r:,} → {df_cleaned.shape[0]:,} rows | {c} → {df_cleaned.shape[1]} columns")


# ─────────────────────────────────────────────
#  RESULTS TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Summary Report",
    "📈 Visualizations",
    "🤖 ML Readiness",
    "🔍 Data Preview",
    "💾 Export"
])


# ── TAB 1: Summary ────────────────────────────

with tab1:
    section_header("📋", "Cleaning Summary")
    
    r2, c2 = df_cleaned.shape
    missing_after = df_cleaned.isnull().sum().sum()
    dups_after = df_cleaned.duplicated().sum()
    mem_after = df_cleaned.memory_usage(deep=True).sum() / 1024 / 1024

    rows_removed = r - r2
    missing_fixed = missing_total - missing_after

    _t1, _t2, _t3, _t4, _t5 = st.columns(5)
    with _t1: render_metric_card("Rows After", f"{r2:,}", f"↓ {rows_removed:,} removed" if rows_removed > 0 else "✓ unchanged", "green")
    with _t2: render_metric_card("Columns After", f"{c2:,}", f"+{c2-c} added (flags)" if c2 > c else "✓ unchanged", "blue")
    with _t3: render_metric_card("Missing Fixed", f"{missing_fixed:,}", f"{missing_fixed/max(missing_total,1)*100:.0f}% resolved", "green")
    with _t4: render_metric_card("Dups Removed", f"{results['steps'].get('duplicates',{}).get('removed',0):,}", color="orange")
    with _t5: render_metric_card("Memory", f"{mem_after:.1f} MB", f"↓ {mem_mb-mem_after:.1f} MB saved" if mem_after < mem_mb else "unchanged", "blue")  
    # Steps taken
    section_header("🔬", "Steps Executed")
    steps = results["steps"]
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        if "column_names" in steps:
            changed = steps["column_names"]["changed"]
            if changed:
                st.markdown(f"**🏷️ Column Names Fixed:** {len(changed)}")
                name_df = pd.DataFrame(changed, columns=["Original", "Cleaned"])
                st.dataframe(name_df, use_container_width=True, height=150)
            else:
                st.markdown("**🏷️ Column Names:** Already clean ✓")

        if "duplicates" in steps:
            n = steps["duplicates"]["removed"]
            st.markdown(f"**🗑️ Duplicates Removed:** {n:,}")

        if "type_fixing" in steps:
            tf = steps["type_fixing"]
            st.markdown(f"**🔧 Numeric Strings Fixed:** {len(tf['numeric_cols_fixed'])}")
            if tf["numeric_cols_fixed"]:
                st.code(", ".join(tf["numeric_cols_fixed"]))
            st.markdown(f"**📅 Datetime Columns Parsed:** {len(tf['datetime_cols_fixed'])}")
            if tf["datetime_cols_fixed"]:
                st.code(", ".join(tf["datetime_cols_fixed"]))

    with col_s2:
        if results["missing_report"]:
            st.markdown(f"**🩹 Missing Value Imputation:**")
            miss_summary = pd.DataFrame([
                {"Column": col, "Was Missing": v["missing"], "% Missing": f"{v['pct']}%", "Action": v["action"]}
                for col, v in results["missing_report"].items()
            ])
            st.dataframe(miss_summary, use_container_width=True, height=200)

        if results["outlier_report"]:
            st.markdown(f"**🎯 Outlier Detection:**")
            out_summary = pd.DataFrame([
                {"Column": col, "Outliers": v["n_outliers"], "Lower": v["lower"], "Upper": v["upper"], "Action": v["action"]}
                for col, v in results["outlier_report"].items()
            ])
            st.dataframe(out_summary, use_container_width=True, height=200)


# ── TAB 2: Visualizations ─────────────────────

with tab2:
    section_header("📈", "Visual Analysis")

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("**Missing Value Heatmap (Original)**")
        img = plot_missing_heatmap(df_original)
        st.image(img, use_container_width=True)

    with viz_col2:
        st.markdown("**Column Type Distribution**")
        img = plot_dtype_distribution(df_cleaned)
        st.image(img, use_container_width=True)

    st.markdown("**Before vs After — Missing Values**")
    img = plot_before_after_missing(df_original, df_cleaned)
    st.image(img, use_container_width=True)

    st.markdown("**Numeric Column Distributions (Cleaned)**")
    img = plot_numeric_distributions(df_cleaned)
    st.image(img, use_container_width=True)

    st.markdown("**Feature Correlation Matrix**")
    img = plot_correlation_heatmap(df_cleaned)
    st.image(img, use_container_width=True)


# ── TAB 3: ML Readiness ───────────────────────

with tab3:
    section_header("🤖", "ML Readiness Report")
    ml = results["ml_readiness"]
    score = ml["score"]

    gauge_col, check_col = st.columns([1, 2])

    with gauge_col:
        gauge_img = plot_ml_readiness_gauge(score)
        st.image(gauge_img, use_container_width=True)

    with check_col:
        st.markdown("**Quality Checklist:**")
        for icon, item in ml["checklist"]:
            color = "#10B981" if "✅" in icon else ("#F59E0B" if "⚠️" in icon else ("#00D4FF" if "ℹ️" in icon else "#EF4444"))
            st.markdown(f"""
            <div class="checklist-item" style="border-left-color:{color}">
                {icon} {item}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    section_header("📐", "Encoding Recommendations")

    col_info_final = detect_column_types(df_cleaned)
    enc_recs = []
    for col, info in col_info_final.items():
        if info["semantic"] == "categorical":
            n = info["n_unique"]
            rec = "Label Encoding" if n > 10 else "One-Hot Encoding"
            enc_recs.append({"Column": col, "Unique Values": n, "Recommendation": rec})
        elif info["semantic"] == "text":
            enc_recs.append({"Column": col, "Unique Values": info["n_unique"], "Recommendation": "TF-IDF / Embeddings"})

    if enc_recs:
        st.dataframe(pd.DataFrame(enc_recs), use_container_width=True)
    else:
        st.success("✅ All columns are already numerically encoded and ML-ready!")

    # Feature stats
    section_header("📊", "Feature Statistics (Cleaned)")
    st.dataframe(df_cleaned.describe(include="all").round(4), use_container_width=True)


# ── TAB 4: Data Preview ───────────────────────

with tab4:
    section_header("🔍", "Cleaned Data Preview")
    
    search = st.text_input("🔎 Filter by column value (format: column=value)", placeholder="e.g. status=active")
    
    display_df = df_cleaned.copy()
    if search and "=" in search:
        try:
            col_f, val_f = search.split("=", 1)
            col_f = col_f.strip()
            val_f = val_f.strip()
            if col_f in display_df.columns:
                display_df = display_df[display_df[col_f].astype(str).str.contains(val_f, case=False, na=False)]
        except Exception:
            pass

    st.markdown(f"Showing **{len(display_df):,}** of **{len(df_cleaned):,}** rows")
    st.dataframe(display_df.head(500), use_container_width=True)

    # Side-by-side comparison
    with st.expander("🔄 Side-by-Side: Original vs Cleaned"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original (first 50 rows)**")
            st.dataframe(df_original.head(50), use_container_width=True)
        with c2:
            st.markdown("**Cleaned (first 50 rows)**")
            st.dataframe(df_cleaned.head(50), use_container_width=True)


# ── TAB 5: Export ─────────────────────────────

with tab5:
    section_header("💾", "Export Cleaned Dataset")
    
    fname = uploaded_file.name.rsplit(".", 1)[0] + "_cleaned"

    export_col1, export_col2, export_col3 = st.columns(3)

    with export_col1:
        st.markdown("**📄 CSV Format**")
        st.markdown("<small style='color:#8B949E'>Best for: Most ML frameworks, pandas, R</small>", unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download CSV",
            data=df_to_csv_bytes(df_cleaned),
            file_name=f"{fname}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with export_col2:
        st.markdown("**📊 Excel Format**")
        st.markdown("<small style='color:#8B949E'>Best for: Business reporting, sharing</small>", unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download Excel",
            data=df_to_excel_bytes(df_cleaned),
            file_name=f"{fname}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with export_col3:
        st.markdown("**🔵 JSON Format**")
        st.markdown("<small style='color:#8B949E'>Best for: APIs, web apps, NoSQL DBs</small>", unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download JSON",
            data=df_to_json_bytes(df_cleaned),
            file_name=f"{fname}.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("---")
    section_header("📋", "Cleaning Report (Text)")
    
    report_lines = [
        "=" * 60,
        "  DATAFORGE PRO — CLEANING REPORT",
        "=" * 60,
        f"Original Shape  : {results['original_shape'][0]:,} rows × {results['original_shape'][1]} cols",
        f"Cleaned Shape   : {results['cleaned_shape'][0]:,} rows × {results['cleaned_shape'][1]} cols",
        f"ML Score        : {results['ml_readiness']['score']}/100",
        "",
        "STEPS TAKEN:",
    ]
    for step, detail in results["steps"].items():
        report_lines.append(f"  ✓ {step}: {detail}")
    
    report_lines += ["", "MISSING VALUE HANDLING:"]
    for col, v in results["missing_report"].items():
        report_lines.append(f"  • {col}: {v['missing']} missing ({v['pct']}%) → {v['action']}")
    
    report_lines += ["", "OUTLIER DETECTION:"]
    for col, v in results["outlier_report"].items():
        report_lines.append(f"  • {col}: {v['n_outliers']} outliers → {v['action']}")
    
    report_lines += ["", "ML READINESS CHECKLIST:"]
    for icon, item in results["ml_readiness"]["checklist"]:
        report_lines.append(f"  {icon} {item}")
    
    report_text = "\n".join(report_lines)
    
    st.download_button(
        "⬇️ Download Text Report",
        data=report_text.encode("utf-8"),
        file_name=f"{fname}_report.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    with st.expander("👁️ Preview Report"):
        st.code(report_text, language="text")


# ─── Footer ───────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; margin-top:3rem; padding:1rem; border-top:1px solid #30363D;">
    <span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#8B949E;">
        DataForge Pro · Industrial Data Cleaning · Built with Python + Streamlit
    </span>
</div>
""", unsafe_allow_html=True)
