"""
Visualization and PDF Report Generator for Data Cleaning Tool
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from io import BytesIO
import base64

# ─── Color palette ───────────────────────────────────────────────
PALETTE = {
    "bg": "#0D1117",
    "panel": "#161B22",
    "accent1": "#00D4FF",
    "accent2": "#7C3AED",
    "accent3": "#10B981",
    "accent4": "#F59E0B",
    "danger": "#EF4444",
    "text": "#E6EDF3",
    "muted": "#8B949E",
}

def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": PALETTE["panel"],
        "axes.edgecolor": "#30363D",
        "axes.labelcolor": PALETTE["text"],
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "text.color": PALETTE["text"],
        "grid.color": "#21262D",
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "font.family": "monospace",
    })


def fig_to_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
#  CHART 1: Missing Value Heatmap
# ─────────────────────────────────────────────

def plot_missing_heatmap(df: pd.DataFrame) -> bytes:
    _apply_dark_style()
    missing = df.isnull()
    n_cols = len(df.columns)

    if missing.sum().sum() == 0:
        fig, ax = plt.subplots(figsize=(8, 3), facecolor=PALETTE["bg"])
        ax.text(0.5, 0.5, "✓ No Missing Values Found", ha="center", va="center",
                fontsize=18, color=PALETTE["accent3"], transform=ax.transAxes)
        ax.axis("off")
        return fig_to_bytes(fig)

    fig_h = max(4, min(n_cols * 0.4, 14))
    fig, ax = plt.subplots(figsize=(14, fig_h), facecolor=PALETTE["bg"])
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "missing", [PALETTE["panel"], PALETTE["danger"]]
    )
    
    sample = missing if len(df) <= 500 else missing.sample(500, random_state=42)
    sns.heatmap(
        sample.T, cmap=cmap, cbar=False,
        linewidths=0, ax=ax, yticklabels=True
    )
    ax.set_title("Missing Value Distribution", fontsize=14, color=PALETTE["accent1"], pad=12, fontweight="bold")
    ax.set_xlabel("Row Index (sample)", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    return fig_to_bytes(fig)


# ─────────────────────────────────────────────
#  CHART 2: Data Type Distribution
# ─────────────────────────────────────────────

def plot_dtype_distribution(df: pd.DataFrame) -> bytes:
    _apply_dark_style()
    dtype_counts = {}
    for col in df.columns:
        d = str(df[col].dtype)
        if "int" in d or "float" in d:
            dtype_counts["Numeric"] = dtype_counts.get("Numeric", 0) + 1
        elif "datetime" in d:
            dtype_counts["DateTime"] = dtype_counts.get("DateTime", 0) + 1
        elif d == "bool":
            dtype_counts["Boolean"] = dtype_counts.get("Boolean", 0) + 1
        else:
            dtype_counts["Text/Cat"] = dtype_counts.get("Text/Cat", 0) + 1

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["bg"])
    colors = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"], PALETTE["accent4"]]
    wedges, texts, autotexts = ax.pie(
        dtype_counts.values(), labels=dtype_counts.keys(),
        colors=colors[:len(dtype_counts)],
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(edgecolor=PALETTE["bg"], linewidth=2)
    )
    for text in texts:
        text.set_color(PALETTE["text"])
        text.set_fontsize(11)
    for at in autotexts:
        at.set_color(PALETTE["bg"])
        at.set_fontweight("bold")
    ax.set_title("Column Type Distribution", fontsize=14, color=PALETTE["accent1"], pad=12, fontweight="bold")
    plt.tight_layout()
    return fig_to_bytes(fig)


# ─────────────────────────────────────────────
#  CHART 3: Before/After Missing Values Bar
# ─────────────────────────────────────────────

def plot_before_after_missing(df_before: pd.DataFrame, df_after: pd.DataFrame) -> bytes:
    _apply_dark_style()
    before_missing = df_before.isnull().sum()
    after_missing = df_after.isnull().sum()
    
    cols_with_missing = before_missing[before_missing > 0].index.tolist()
    if not cols_with_missing:
        fig, ax = plt.subplots(figsize=(8, 3), facecolor=PALETTE["bg"])
        ax.text(0.5, 0.5, "✓ No Missing Values in Original Dataset", ha="center", va="center",
                fontsize=14, color=PALETTE["accent3"], transform=ax.transAxes)
        ax.axis("off")
        return fig_to_bytes(fig)

    x = np.arange(len(cols_with_missing))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(cols_with_missing) * 1.2), 5), facecolor=PALETTE["bg"])
    
    bars1 = ax.bar(x - w/2, [before_missing[c] for c in cols_with_missing], w,
                   label="Before", color=PALETTE["danger"], alpha=0.85, edgecolor=PALETTE["bg"])
    bars2 = ax.bar(x + w/2, [after_missing.get(c, 0) for c in cols_with_missing], w,
                   label="After", color=PALETTE["accent3"], alpha=0.85, edgecolor=PALETTE["bg"])

    ax.set_xticks(x)
    ax.set_xticklabels(cols_with_missing, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Missing Count")
    ax.set_title("Missing Values: Before vs After Cleaning", fontsize=14, color=PALETTE["accent1"], pad=12, fontweight="bold")
    ax.legend(facecolor=PALETTE["panel"], edgecolor="#30363D")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig_to_bytes(fig)


# ─────────────────────────────────────────────
#  CHART 4: Numeric Distribution Grid
# ─────────────────────────────────────────────

def plot_numeric_distributions(df: pd.DataFrame, max_cols: int = 9) -> bytes:
    _apply_dark_style()
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 5][:max_cols]
    
    if not num_cols:
        fig, ax = plt.subplots(figsize=(8, 3), facecolor=PALETTE["bg"])
        ax.text(0.5, 0.5, "No Numeric Columns to Display", ha="center", va="center",
                fontsize=14, color=PALETTE["muted"], transform=ax.transAxes)
        ax.axis("off")
        return fig_to_bytes(fig)

    n = len(num_cols)
    cols_grid = min(3, n)
    rows_grid = (n + cols_grid - 1) // cols_grid
    
    fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(cols_grid * 4, rows_grid * 3.5), facecolor=PALETTE["bg"])
    if n == 1:
        axes = np.array([[axes]])
    elif rows_grid == 1:
        axes = axes.reshape(1, -1)
    elif cols_grid == 1:
        axes = axes.reshape(-1, 1)

    accent_cycle = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"], PALETTE["accent4"]]
    
    for idx, col in enumerate(num_cols):
        r, c = divmod(idx, cols_grid)
        ax = axes[r][c]
        color = accent_cycle[idx % len(accent_cycle)]
        data = df[col].dropna()
        ax.hist(data, bins=30, color=color, alpha=0.75, edgecolor=PALETTE["bg"], linewidth=0.5)
        ax.axvline(data.mean(), color="white", linestyle="--", linewidth=1, alpha=0.6, label=f"μ={data.mean():.2f}")
        ax.set_title(col[:22], fontsize=9, color=PALETTE["text"], pad=6)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("")
        ax.legend(fontsize=7, facecolor=PALETTE["panel"])

    # Hide unused axes
    for idx in range(n, rows_grid * cols_grid):
        r, c = divmod(idx, cols_grid)
        axes[r][c].axis("off")

    fig.suptitle("Numeric Column Distributions (Cleaned)", fontsize=14, color=PALETTE["accent1"], y=1.01, fontweight="bold")
    plt.tight_layout()
    return fig_to_bytes(fig)


# ─────────────────────────────────────────────
#  CHART 5: Correlation Heatmap
# ─────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> bytes:
    _apply_dark_style()
    num_df = df.select_dtypes(include=[np.number])
    
    if num_df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(8, 3), facecolor=PALETTE["bg"])
        ax.text(0.5, 0.5, "Need ≥2 numeric columns for correlation", ha="center", va="center",
                fontsize=14, color=PALETTE["muted"], transform=ax.transAxes)
        ax.axis("off")
        return fig_to_bytes(fig)

    corr = num_df.corr()
    n = len(corr)
    fig_size = max(6, min(n * 0.7, 16))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85), facecolor=PALETTE["bg"])
    
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
        square=True, linewidths=0.5, linecolor="#21262D",
        annot=n <= 15, fmt=".2f", annot_kws={"size": 8},
        ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, color=PALETTE["accent1"], pad=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout()
    return fig_to_bytes(fig)


# ─────────────────────────────────────────────
#  CHART 6: ML Readiness Gauge
# ─────────────────────────────────────────────

def plot_ml_readiness_gauge(score: int) -> bytes:
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(polar=False), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.axis("off")

    # Gauge arc
    theta = np.linspace(np.pi, 0, 300)
    r_outer, r_inner = 1.0, 0.6

    # Background arc
    ax.plot(np.cos(theta), np.sin(theta), lw=30, color="#21262D", solid_capstyle="butt", transform=ax.transData)

    # Score arc
    score_angle = np.pi - (score / 100) * np.pi
    theta_score = np.linspace(np.pi, score_angle, 300)
    if score >= 75:
        arc_color = PALETTE["accent3"]
    elif score >= 50:
        arc_color = PALETTE["accent4"]
    else:
        arc_color = PALETTE["danger"]

    ax.plot(np.cos(theta_score), np.sin(theta_score), lw=30, color=arc_color, solid_capstyle="butt")

    # Score text
    ax.text(0, 0.1, f"{score}", ha="center", va="center", fontsize=52, fontweight="bold",
            color=arc_color, transform=ax.transData)
    ax.text(0, -0.2, "ML READINESS SCORE", ha="center", va="center", fontsize=9,
            color=PALETTE["muted"], transform=ax.transData)

    label = "Excellent" if score >= 80 else ("Good" if score >= 60 else ("Fair" if score >= 40 else "Needs Work"))
    ax.text(0, -0.42, label, ha="center", va="center", fontsize=14, fontweight="bold",
            color=arc_color, transform=ax.transData)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.6, 1.2)
    fig.suptitle("Dataset ML Readiness", fontsize=13, color=PALETTE["accent1"], y=0.98, fontweight="bold")
    plt.tight_layout()
    return fig_to_bytes(fig)


# ─────────────────────────────────────────────
#  EXPORT HELPERS
# ─────────────────────────────────────────────

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned_Data")
    buf.seek(0)
    return buf.read()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def df_to_json_bytes(df: pd.DataFrame) -> bytes:
    return df.to_json(orient="records", indent=2).encode("utf-8")
