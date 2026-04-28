<div align="center">

# ⚙️ Datacleaner

### Industrial-Grade Data Cleaning & ML Readiness Tool

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![pandas](https://img.shields.io/badge/pandas-2.1%2B-150458?style=flat-square&logo=pandas)](https://pandas.pydata.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Upload any messy dataset → Auto-clean → Visual Report → Download clean data**

[🚀 Live Demo](#) · [📖 Documentation](#how-it-works) · [🐛 Report Bug](../../issues) · [✨ Request Feature](../../issues)

</div>

---

## 📸 Preview

> Dark-themed dashboard with real-time metric cards, 6 chart types, ML readiness gauge, and multi-format export.

```
Upload CSV/Excel/JSON  →  Configure Pipeline  →  Clean  →  Download + Report
```

---

## ✨ Features

### 🔧 Cleaning Pipeline
| Feature | What It Does |
|---|---|
| **Column Name Standardizer** | Strips spaces, fixes hyphens/dots, lowercases, resolves duplicates |
| **Duplicate Remover** | Exact row deduplication with count report |
| **Smart Type Detection** | Converts `"$1,200"` → float, `"2023-01-01"` → datetime automatically |
| **Smart Imputation** | Mean / Median / Mode / KNN per column based on missing % |
| **Outlier Detection** | IQR or Z-score; choose to flag, cap, or remove outliers |
| **Categorical Encoding** | Label or One-Hot encoding for ML readiness |

### 📊 Visual Report (6 Charts)
- 🔥 Missing Value Heatmap
- 🍩 Column Type Distribution Pie
- 📊 Before vs After Missing Values Bar Chart
- 📈 Numeric Column Distribution Grid
- 🧮 Feature Correlation Heatmap
- 🎯 ML Readiness Gauge (0–100 Score)

### 💾 Export Options
- **CSV** — for pandas, R, ML frameworks
- **Excel (.xlsx)** — for business reporting
- **JSON** — for APIs and web apps
- **Text Report** — full cleaning summary log

---

## 🚀 Quick Start

### Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/ashokkumar1905/myprojects.git
cd dataforge-pro

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Generate a messy test dataset
python generate_sample.py

# 4. Launch the app
streamlit run app.py
```

Open your browser at **https://myprojects-wu3zgeabbz3d7upfmqqlce.streamlit.app/**

---

## 📁 Project Structure

```
dataforge-pro/
│
├── app.py                 # ← Streamlit UI  (run this)
├── cleaner_engine.py      # ← Core cleaning pipeline logic
├── visualizer.py          # ← Charts, plots & export helpers
│
├── generate_sample.py     # Generate messy test datasets
├── sample_messy_sales.csv # Ready-to-use messy dataset
├── requirements.txt       # All dependencies
└── README.md
```

### How It Works

```
app.py
  ├── imports cleaner_engine.py   →  runs the cleaning pipeline
  └── imports visualizer.py       →  generates all charts & exports
```

`app.py` is the only file you run. Everything else loads automatically.

---

## 📦 Supported File Formats

| Format | Extension | Notes |
|---|---|---|
| CSV | `.csv` | Auto-detects encoding (UTF-8, Latin-1, CP1252) |
| Excel | `.xlsx` `.xls` | Multi-sheet support |
| JSON | `.json` | Records-oriented |
| TSV | `.tsv` | Tab-separated |
| Parquet | `.parquet` | Columnar format |

---

## 🤖 ML Readiness Score

After cleaning, the tool scores your dataset **0–100** based on:

| Check | Points |
|---|---|
| ✅ No missing values | +30 |
| ✅ No duplicate rows | +10 |
| ✅ All columns numerically encoded | +15 |
| ✅ Outliers handled | +15 |
| ✅ Adequate dataset size (≥100 rows) | +10 |
| ✅ Clean column names | +5 |
| ✅ High numeric ratio | +15 |

**Score guide:** `80–100` Excellent · `60–79` Good · `40–59` Fair · `<40` Needs Work

---

## 🧪 Test With Sample Data

The included `sample_messy_sales.csv` is intentionally broken — great for testing:

| Problem | Details |
|---|---|
| Bad column names | Trailing spaces, hyphens, mixed case |
| String numbers | `"$5,200.00"`, `"12.5%"` |
| Unparsed dates | Stored as plain strings |
| Missing values | ~8% per column |
| Duplicate rows | 20 injected duplicates |
| Outliers | 10 extreme values injected |

```bash
python generate_sample.py   # regenerate anytime
streamlit run app.py        # upload it in the UI
```

---

## 🏭 Industrial Notes

- Handles datasets up to **~500MB** with pandas
- For **>1GB** files: swap pandas for Polars or Dask (roadmap)
- **KNN Imputation** via scikit-learn — more accurate than simple mean/median
- **IQR method** recommended for outliers — robust against non-normal distributions
- Duplicate column names after cleaning are auto-resolved (`sale_amount` → `sale_amount_2`)

---

## 🗺️ Roadmap

- [x] Core cleaning pipeline
- [x] 6-chart visual report
- [x] ML Readiness Score
- [x] Multi-format export
- [x] Streamlit Cloud deploy
- [ ] AI-powered column suggestions (Claude API)
- [ ] Full EDA profiling report per column
- [ ] Fuzzy duplicate detection
- [ ] Time-series gap detection
- [ ] Polars backend for large files
- [ ] FastAPI REST wrapper

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Data | pandas, NumPy |
| ML / Imputation | scikit-learn |
| Statistics | SciPy |
| Visualization | Matplotlib, Seaborn |
| Export | openpyxl, built-in json/csv |

---

## 💡 Career Value

This project demonstrates skills across 4 domains:

- **Data Engineering** — Pipeline design, type inference, imputation strategies
- **ML Engineering** — Feature encoding, readiness scoring, outlier analysis
- **Software Engineering** — Modular 3-file architecture, error handling
- **Data Visualization** — Multi-chart reporting, before/after analysis

> Comparable commercial tools: **Trifacta** (acquired by Google for ~$1.8B), **Alteryx**, **DataPrep.ai**

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

Built with 🖤 using Python + Streamlit

⭐ **Star this repo if it helped you!**

</div>
