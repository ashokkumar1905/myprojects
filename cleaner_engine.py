"""
Industrial-Grade Data Cleaning Engine
Handles any tabular dataset with smart detection and cleaning strategies.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  TYPE DETECTION
# ─────────────────────────────────────────────

def detect_column_types(df: pd.DataFrame) -> dict:
    """Detect semantic types for each column beyond basic dtype."""
    col_info = {}
    for col in df.columns:
        series = df[col].dropna()
        dtype = str(df[col].dtype) if hasattr(df[col], 'dtype') else "object"
        n_unique = int(df[col].nunique()); n_missing = int(df[col].isna().sum()); info = {"original_dtype": dtype, "semantic": "unknown", "n_unique": n_unique, "n_missing": n_missing}

        if len(series) == 0:
            info["semantic"] = "empty"
        elif dtype in ["int64", "float64", "int32", "float32"]:
            if info["n_unique"] <= 20 and info["n_unique"] / len(df) < 0.05:
                info["semantic"] = "categorical_numeric"
            else:
                info["semantic"] = "numeric"
        elif dtype == "object":
            # Try date
            try:
                pd.to_datetime(series.head(50), infer_datetime_format=True)
                info["semantic"] = "datetime"
            except Exception:
                pass
            if info["semantic"] == "unknown":
                # Try numeric string
                try:
                    series.astype(float)
                    info["semantic"] = "numeric_string"
                except Exception:
                    pass
            if info["semantic"] == "unknown":
                if info["n_unique"] / max(len(df), 1) < 0.1 or info["n_unique"] <= 30:
                    info["semantic"] = "categorical"
                else:
                    info["semantic"] = "text"
        elif "datetime" in dtype:
            info["semantic"] = "datetime"
        elif dtype == "bool":
            info["semantic"] = "boolean"
        else:
            info["semantic"] = "other"

        col_info[col] = info
    return col_info


# ─────────────────────────────────────────────
#  CLEANING STRATEGIES
# ─────────────────────────────────────────────

def standardize_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Normalize column names: lowercase, strip spaces, replace separators."""
    original = list(df.columns)
    new_cols = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-\.]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    # Resolve duplicate column names by appending _2, _3 etc.
    seen = {}
    deduped = []
    for name in new_cols:
        if name in seen:
            seen[name] += 1
            deduped.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 1
            deduped.append(name)
    df.columns = deduped
    changed = [(o, n) for o, n in zip(original, df.columns) if o != n]
    return df, changed


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    return df.reset_index(drop=True), before - len(df)


def fix_numeric_strings(df: pd.DataFrame, col_info: dict) -> tuple[pd.DataFrame, list]:
    """Convert numeric-string columns to actual numeric."""
    fixed = []
    for col, info in col_info.items():
        if info["semantic"] == "numeric_string":
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[,\$\%£€]", "", regex=True), errors="coerce")
                fixed.append(col)
                col_info[col]["semantic"] = "numeric"
                col_info[col]["original_dtype"] = "float64"
            except Exception:
                pass
    return df, fixed


def fix_datetime_columns(df: pd.DataFrame, col_info: dict) -> tuple[pd.DataFrame, list]:
    """Parse datetime columns."""
    fixed = []
    for col, info in col_info.items():
        if info["semantic"] == "datetime" and "datetime" not in info["original_dtype"]:
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                fixed.append(col)
            except Exception:
                pass
    return df, fixed


def handle_missing_values(df: pd.DataFrame, col_info: dict, strategy: str = "smart") -> tuple[pd.DataFrame, dict]:
    """
    strategy: 'smart' | 'mean' | 'median' | 'mode' | 'knn' | 'drop_rows' | 'none'
    """
    report = {}
    numeric_cols = [c for c, i in col_info.items() if i["semantic"] in ("numeric", "numeric_string", "categorical_numeric") and c in df.columns]
    cat_cols = [c for c, i in col_info.items() if i["semantic"] in ("categorical", "text") and c in df.columns]

    for col in df.columns:
        missing = df[col].isna().sum()
        if missing == 0:
            continue
        pct = missing / len(df)
        report[col] = {"missing": int(missing), "pct": round(pct * 100, 2), "action": "none"}

        if pct > 0.6:
            # More than 60% missing — flag, don't blindly impute
            report[col]["action"] = "flagged_high_missing"
            continue

        if strategy == "drop_rows":
            df = df.dropna(subset=[col])
            report[col]["action"] = "dropped_rows"

        elif strategy in ("mean", "smart") and col in numeric_cols:
            fill = df[col].mean() if strategy == "mean" else (
                df[col].median() if pct > 0.1 else df[col].mean()
            )
            df[col] = df[col].fillna(fill)
            report[col]["action"] = f"filled_mean={round(float(fill), 4)}"

        elif strategy == "median" and col in numeric_cols:
            fill = df[col].median()
            df[col] = df[col].fillna(fill)
            report[col]["action"] = f"filled_median={round(float(fill), 4)}"

        elif col in cat_cols or col in [c for c, i in col_info.items() if i["semantic"] == "categorical_numeric"]:
            fill = df[col].mode()
            if len(fill) > 0:
                df[col] = df[col].fillna(fill[0])
                report[col]["action"] = f"filled_mode={fill[0]}"

        elif strategy == "knn" and col in numeric_cols:
            try:
                imputer = KNNImputer(n_neighbors=5)
                df[[col]] = imputer.fit_transform(df[[col]])
                report[col]["action"] = "filled_knn"
            except Exception:
                df[col] = df[col].fillna(df[col].median())
                report[col]["action"] = "filled_median_fallback"

    return df.reset_index(drop=True), report


def detect_and_handle_outliers(df: pd.DataFrame, col_info: dict, method: str = "iqr", action: str = "flag") -> tuple[pd.DataFrame, dict]:
    """
    method: 'iqr' | 'zscore'
    action: 'flag' | 'cap' | 'remove'
    """
    report = {}
    numeric_cols = [c for c, i in col_info.items() if i["semantic"] == "numeric" and c in df.columns]

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue

        if method == "iqr":
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        else:
            z = np.abs(stats.zscore(series))
            lower = series[z < 3].min()
            upper = series[z < 3].max()

        mask = (df[col] < lower) | (df[col] > upper)
        n_out = mask.sum()

        if n_out == 0:
            continue

        report[col] = {"n_outliers": int(n_out), "lower": round(float(lower), 4), "upper": round(float(upper), 4), "action": action}

        if action == "cap":
            df[col] = df[col].clip(lower=lower, upper=upper)
        elif action == "remove":
            df = df[~mask]
        elif action == "flag":
            df[f"{col}_outlier_flag"] = mask.astype(int)

    return df.reset_index(drop=True), report


def encode_categoricals(df: pd.DataFrame, col_info: dict, method: str = "label") -> tuple[pd.DataFrame, dict]:
    """Encode categorical columns for ML readiness."""
    report = {}
    for col, info in col_info.items():
        if col not in df.columns:
            continue
        if info["semantic"] == "categorical":
            n_unique = info["n_unique"]
            if method == "label" or n_unique > 10:
                df[col] = df[col].astype("category").cat.codes
                report[col] = {"method": "label_encoding", "n_categories": n_unique}
            else:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                report[col] = {"method": "one_hot", "n_categories": n_unique}
    return df, report


# ─────────────────────────────────────────────
#  ML READINESS SCORE
# ─────────────────────────────────────────────

def compute_ml_readiness(df_original: pd.DataFrame, df_cleaned: pd.DataFrame, col_info: dict, missing_report: dict, outlier_report: dict) -> dict:
    """Score the cleaned dataset on ML readiness (0–100)."""
    score = 100
    checklist = []

    # Missing values
    remaining_missing = df_cleaned.isna().sum().sum()
    if remaining_missing == 0:
        checklist.append(("✅", "No missing values"))
    else:
        pct = remaining_missing / df_cleaned.size * 100
        score -= min(30, int(pct * 3))
        checklist.append(("⚠️", f"{remaining_missing} missing values remain ({pct:.1f}%)"))

    # Duplicates
    dups = df_cleaned.duplicated().sum()
    if dups == 0:
        checklist.append(("✅", "No duplicate rows"))
    else:
        score -= 10
        checklist.append(("❌", f"{dups} duplicate rows remain"))

    # High-cardinality object columns
    obj_cols = [c for c in df_cleaned.columns if df_cleaned[c].dtype == "object"]
    if not obj_cols:
        checklist.append(("✅", "No unencoded string columns"))
    else:
        score -= len(obj_cols) * 5
        checklist.append(("⚠️", f"{len(obj_cols)} string columns not yet encoded: {obj_cols[:3]}"))

    # Outliers
    if not outlier_report:
        checklist.append(("✅", "No significant outliers detected"))
    else:
        score -= min(15, len(outlier_report) * 3)
        checklist.append(("⚠️", f"Outliers found in {len(outlier_report)} columns"))

    # Dataset size
    n_rows, n_cols = df_cleaned.shape
    if n_rows >= 100:
        checklist.append(("✅", f"Adequate dataset size ({n_rows} rows)"))
    else:
        score -= 10
        checklist.append(("⚠️", f"Small dataset ({n_rows} rows) — may underfit models"))

    # Column name cleanliness
    bad_names = [c for c in df_cleaned.columns if c != c.lower() or " " in c]
    if not bad_names:
        checklist.append(("✅", "Column names are clean and consistent"))
    else:
        score -= 5
        checklist.append(("⚠️", f"{len(bad_names)} columns with messy names"))

    # Numeric ratio
    numeric_ratio = sum(1 for c in df_cleaned.columns if pd.api.types.is_numeric_dtype(df_cleaned[c])) / max(len(df_cleaned.columns), 1)
    if numeric_ratio >= 0.7:
        checklist.append(("✅", f"High numeric column ratio ({numeric_ratio:.0%})"))
    else:
        checklist.append(("ℹ️", f"Numeric ratio: {numeric_ratio:.0%} — consider encoding more columns"))

    score = max(0, min(100, score))
    return {"score": score, "checklist": checklist}


# ─────────────────────────────────────────────
#  MASTER PIPELINE
# ─────────────────────────────────────────────

def run_full_pipeline(df: pd.DataFrame, config: dict) -> dict:
    """
    config keys:
        fix_names: bool
        remove_dups: bool
        fix_types: bool
        missing_strategy: str
        outlier_method: str
        outlier_action: str
        encode: bool
        encode_method: str
    """
    results = {
        "original_shape": df.shape,
        "steps": {},
        "col_info": {},
        "missing_report": {},
        "outlier_report": {},
        "encode_report": {},
        "ml_readiness": {},
        "df_cleaned": None,
    }
    df = df.copy()
    original = df.copy()

    # Detect types
    col_info = detect_column_types(df)
    results["col_info"] = col_info

    # Step 1: Column names
    if config.get("fix_names", True):
        df, changed = standardize_column_names(df)
        results["steps"]["column_names"] = {"changed": changed}
        # Rebuild col_info with new names
        col_info = detect_column_types(df)

    # Step 2: Remove duplicates
    if config.get("remove_dups", True):
        df, n_removed = remove_duplicates(df)
        results["steps"]["duplicates"] = {"removed": n_removed}

    # Step 3: Fix types
    if config.get("fix_types", True):
        df, num_fixed = fix_numeric_strings(df, col_info)
        df, dt_fixed = fix_datetime_columns(df, col_info)
        results["steps"]["type_fixing"] = {"numeric_cols_fixed": num_fixed, "datetime_cols_fixed": dt_fixed}

    # Step 4: Missing values
    strategy = config.get("missing_strategy", "smart")
    df, missing_report = handle_missing_values(df, col_info, strategy=strategy)
    results["missing_report"] = missing_report
    results["steps"]["missing_values"] = {"strategy": strategy, "cols_imputed": len(missing_report)}

    # Step 5: Outliers
    o_method = config.get("outlier_method", "iqr")
    o_action = config.get("outlier_action", "flag")
    if o_action != "none":
        df, outlier_report = detect_and_handle_outliers(df, col_info, method=o_method, action=o_action)
        results["outlier_report"] = outlier_report
        results["steps"]["outliers"] = {"method": o_method, "action": o_action, "cols_affected": len(outlier_report)}

    # Step 6: Encoding
    if config.get("encode", False):
        df, encode_report = encode_categoricals(df, col_info, method=config.get("encode_method", "label"))
        results["encode_report"] = encode_report
        results["steps"]["encoding"] = {"cols_encoded": len(encode_report)}

    # ML Readiness
    results["ml_readiness"] = compute_ml_readiness(original, df, col_info, missing_report, results.get("outlier_report", {}))
    results["cleaned_shape"] = df.shape
    results["df_cleaned"] = df

    return results
