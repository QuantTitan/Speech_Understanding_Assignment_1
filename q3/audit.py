"""
audit.py — Ethical Audit of Mozilla Common Voice Dataset
=========================================================
Performs a "Sound Check" audit identifying:
  • Documentation Debt (missing metadata fields)
  • Representation bias (gender, age, dialect/accent)

Outputs:
  • Printed statistics + CSV report
  • audit_plots.pdf  (saved to q3/ directory)

Usage:
    python audit.py [--sample 5000] [--output audit_plots.pdf]
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_common_voice(sample_size: int = 5000, seed: int = 42):
    """
    Load Mozilla Common Voice (en) via HuggingFace datasets.
    Falls back to a synthetic surrogate if HF download is unavailable.
    """
    try:
        from datasets import load_dataset
        print("[INFO] Loading Common Voice (en) — this may take a moment …")
        ds = load_dataset(
            "mozilla-foundation/common_voice_11_0",
            "en",
            split="train",
            trust_remote_code=True,
        )
        # Convert to pandas for easier analysis
        cols = ["sentence", "gender", "age", "accent", "locale", "client_id"]
        df = ds.to_pandas()[cols].copy()
        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size, random_state=seed).reset_index(drop=True)
        print(f"[INFO] Loaded {len(df):,} rows from Common Voice.")
        return df, "Mozilla Common Voice 11.0 (en)"

    except Exception as e:
        print(f"[WARN] HuggingFace download failed ({e}). Using synthetic dataset.")
        return _synthetic_cv_surrogate(sample_size, seed), "Synthetic CV Surrogate"


def _synthetic_cv_surrogate(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a statistically realistic *surrogate* of Common Voice metadata
    that mirrors known demographic skews documented in the literature.
    """
    rng = np.random.default_rng(seed)

    # Gender distribution: heavily male-skewed (~70 % male, 15 % female, rest unknown/other)
    gender_probs = [0.70, 0.15, 0.10, 0.05]
    gender_labels = ["male", "female", "other", ""]   # "" = missing
    gender = rng.choice(gender_labels, size=n, p=gender_probs)

    # Age distribution: over-represented 20–29 age group
    age_probs = [0.02, 0.08, 0.28, 0.22, 0.18, 0.10, 0.05, 0.07]
    age_labels = ["teens", "twenties", "thirties", "fourties",
                  "fifties", "sixties", "seventies", ""]
    age = rng.choice(age_labels, size=n, p=age_probs)

    # Accent: heavily American English
    accent_probs = [0.55, 0.12, 0.08, 0.06, 0.04, 0.04, 0.03, 0.03, 0.02, 0.03]
    accent_labels = ["us", "england", "australia", "canada", "indian",
                     "ireland", "scotland", "wales", "hongkong", ""]
    accent = rng.choice(accent_labels, size=n, p=accent_probs)

    # Sentence length (words) — proxy for documentation quality
    sent_len = rng.integers(3, 25, size=n)
    sentences = [f"Sample sentence of length {l}." for l in sent_len]
    client_ids = [f"client_{i:05d}" for i in rng.integers(0, n // 2, size=n)]

    return pd.DataFrame({
        "sentence": sentences,
        "gender": gender,
        "age": age,
        "accent": accent,
        "locale": "en",
        "client_id": client_ids,
        "sent_len": sent_len,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DOCUMENTATION DEBT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

MISSING_TOKENS = {"", "nan", "none", "n/a", "unknown", None}

def is_missing(val) -> bool:
    if val is None:
        return True
    return str(val).strip().lower() in MISSING_TOKENS


def documentation_debt_report(df: pd.DataFrame) -> dict:
    """
    Quantify missing / undocumented metadata across key fields.
    Returns a dict: field → {count, pct, severity}.
    """
    report = {}
    metadata_fields = ["gender", "age", "accent", "locale", "client_id"]
    for field in metadata_fields:
        if field not in df.columns:
            continue
        missing_count = df[field].apply(is_missing).sum()
        pct = 100 * missing_count / len(df)
        severity = "CRITICAL" if pct > 30 else "HIGH" if pct > 15 else "MEDIUM" if pct > 5 else "LOW"
        report[field] = {"count": int(missing_count), "pct": round(pct, 2), "severity": severity}

    print("\n" + "═" * 60)
    print("  DOCUMENTATION DEBT REPORT")
    print("═" * 60)
    print(f"  {'Field':<15} {'Missing':>8} {'%':>7}  Severity")
    print("─" * 60)
    for f, v in report.items():
        print(f"  {f:<15} {v['count']:>8,} {v['pct']:>6.1f}%  [{v['severity']}]")
    print("═" * 60)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 3.  REPRESENTATION BIAS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(series: pd.Series) -> pd.Series:
    """Replace blank/missing with 'unknown' and lowercase."""
    return series.apply(lambda v: "unknown" if is_missing(v) else str(v).strip().lower())


def gender_bias(df: pd.DataFrame) -> pd.Series:
    counts = _normalize(df["gender"]).value_counts()
    print(f"\n[Gender Distribution]\n{counts.to_string()}")
    return counts


def age_bias(df: pd.DataFrame) -> pd.Series:
    order = ["teens", "twenties", "thirties", "fourties",
             "fifties", "sixties", "seventies", "eighties", "nineties", "unknown"]
    counts = _normalize(df["age"]).value_counts()
    counts = counts.reindex([a for a in order if a in counts.index], fill_value=0)
    print(f"\n[Age Distribution]\n{counts.to_string()}")
    return counts


def accent_bias(df: pd.DataFrame) -> pd.Series:
    counts = _normalize(df["accent"]).value_counts()
    print(f"\n[Accent Distribution (top 15)]\n{counts.head(15).to_string()}")
    return counts


def speaker_count_per_demographic(df: pd.DataFrame) -> pd.DataFrame:
    """Unique speakers per (gender × age) cell — reveals intersectional under-representation."""
    df2 = df.copy()
    df2["gender_n"] = _normalize(df2["gender"])
    df2["age_n"] = _normalize(df2["age"])
    pivot = df2.groupby(["gender_n", "age_n"])["client_id"].nunique().unstack(fill_value=0)
    return pivot


def gini_coefficient(counts: pd.Series) -> float:
    """Gini coefficient of distribution (0 = perfect equality, 1 = total concentration)."""
    arr = np.sort(counts.values.astype(float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.dot(idx, arr) - (n + 1) * arr.sum()) / (n * arr.sum()))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "male": "#4C72B0", "female": "#DD8452", "other": "#55A868",
    "unknown": "#C7C7C7",
}

def _bar_color(labels, palette):
    return [palette.get(str(l).lower(), "#8172B2") for l in labels]


def make_audit_plots(df: pd.DataFrame, debt: dict, dataset_name: str,
                     output_path: str = "audit_plots.pdf"):
    gender_counts = gender_bias(df)
    age_counts = age_bias(df)
    accent_counts = accent_bias(df)
    pivot = speaker_count_per_demographic(df)

    with PdfPages(output_path) as pdf:
        # ── Page 1: Overview dashboard ────────────────────────────────────────
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f"Ethical Audit — {dataset_name}\nRepresentation Bias & Documentation Debt",
                     fontsize=14, fontweight="bold", y=0.98)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        # 1a. Gender pie
        ax1 = fig.add_subplot(gs[0, 0])
        labels_g = gender_counts.index.tolist()
        colors_g = _bar_color(labels_g, PALETTE)
        ax1.pie(gender_counts.values, labels=labels_g, colors=colors_g,
                autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
        ax1.set_title("Gender Distribution", fontweight="bold")

        # 1b. Age bar
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.barh(age_counts.index[::-1], age_counts.values[::-1], color="#4C72B0", edgecolor="white")
        ax2.set_xlabel("Count")
        ax2.set_title("Age Distribution", fontweight="bold")
        ax2.tick_params(labelsize=8)

        # 1c. Accent top-10 bar
        ax3 = fig.add_subplot(gs[0, 2])
        top10 = accent_counts.head(10)
        ax3.barh(top10.index[::-1], top10.values[::-1], color="#DD8452", edgecolor="white")
        ax3.set_xlabel("Count")
        ax3.set_title("Top-10 Accents / Dialects", fontweight="bold")
        ax3.tick_params(labelsize=7)

        # 1d. Documentation Debt bar
        ax4 = fig.add_subplot(gs[1, 0:2])
        fields = list(debt.keys())
        pcts = [debt[f]["pct"] for f in fields]
        colors_d = ["#d62728" if p > 30 else "#ff7f0e" if p > 15 else "#2ca02c" for p in pcts]
        bars = ax4.bar(fields, pcts, color=colors_d, edgecolor="white")
        ax4.set_ylabel("% Missing")
        ax4.set_title("Documentation Debt — Missing Metadata (%)", fontweight="bold")
        ax4.set_ylim(0, max(pcts + [5]) * 1.2)
        for bar, pct in zip(bars, pcts):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
        ax4.axhline(30, color="red", linestyle="--", linewidth=1, label="Critical threshold (30%)")
        ax4.axhline(15, color="orange", linestyle="--", linewidth=1, label="High threshold (15%)")
        ax4.legend(fontsize=7)

        # 1e. Gini summary text box
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis("off")
        g_gini = gini_coefficient(gender_counts)
        a_gini = gini_coefficient(age_counts)
        ac_gini = gini_coefficient(accent_counts)
        text = (
            "Bias Severity Index\n"
            "─────────────────────\n"
            f"Gender   Gini: {g_gini:.3f}\n"
            f"Age      Gini: {a_gini:.3f}\n"
            f"Accent   Gini: {ac_gini:.3f}\n\n"
            "Gini Scale:\n"
            "  0.0–0.2 : Low bias\n"
            "  0.2–0.4 : Moderate\n"
            "  0.4–0.6 : High\n"
            "  0.6–1.0 : Severe\n\n"
            f"Total samples: {len(df):,}"
        )
        ax5.text(0.05, 0.95, text, transform=ax5.transAxes, fontsize=9,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 2: Intersectional heatmap ────────────────────────────────────
        if pivot.shape[0] > 1 and pivot.shape[1] > 1:
            fig2, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=9)
            plt.colorbar(im, ax=ax, label="Unique Speaker Count")
            ax.set_title("Intersectional Representation: Gender × Age (Unique Speakers)",
                         fontweight="bold", pad=15)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    ax.text(j, i, str(val), ha="center", va="center",
                            fontsize=7, color="black" if val < pivot.values.max() * 0.6 else "white")
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

        # ── Page 3: Recommendations ───────────────────────────────────────────
        fig3 = plt.figure(figsize=(12, 8))
        fig3.patch.set_facecolor("#f8f9fa")
        ax_r = fig3.add_subplot(111)
        ax_r.axis("off")
        recs = [
            ("1. Gender Imbalance",
             f"  Male speakers comprise ~{gender_counts.get('male', 0)/len(df)*100:.0f}% of data.\n"
             "  → Actively recruit female & non-binary contributors.\n"
             "  → Apply gender-balanced sampling during model training."),
            ("2. Age Under-representation",
             "  Older speakers (60+) and teens are severely under-represented.\n"
             "  → Partner with schools and senior centres.\n"
             "  → Apply age-stratified data augmentation (pitch/tempo shift)."),
            ("3. Accent/Dialect Concentration",
             f"  American English accounts for ~{accent_counts.get('us', 0)/len(df)*100:.0f}% of data.\n"
             "  → Expand collection to AAVE, Scottish, Indian accents.\n"
             "  → Use dialect-aware language model fine-tuning."),
            ("4. Documentation Debt",
             "  Missing demographic metadata prevents auditing & debiasing.\n"
             "  → Make demographic fields mandatory (with opt-out for privacy).\n"
             "  → Retroactively label using automated speaker profiling."),
            ("5. Intersectional Gaps",
             "  Certain (gender × age) cells have near-zero representation.\n"
             "  → Compute fairness metrics per demographic subgroup.\n"
             "  → Use Fairness Loss (see train_fair.py) to close gaps."),
        ]
        y = 0.95
        ax_r.text(0.5, y, "Audit Recommendations", transform=ax_r.transAxes,
                  fontsize=14, fontweight="bold", ha="center", va="top")
        y -= 0.06
        for title, body in recs:
            ax_r.text(0.04, y, title, transform=ax_r.transAxes,
                      fontsize=11, fontweight="bold", color="#2c3e50", va="top")
            y -= 0.05
            ax_r.text(0.06, y, body, transform=ax_r.transAxes,
                      fontsize=9, color="#555555", va="top", wrap=True)
            y -= 0.12
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

    print(f"\n[INFO] Audit plots saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame, debt: dict, out_dir: str = "."):
    # Demographic counts
    rows = []
    for field in ["gender", "age", "accent"]:
        if field not in df.columns:
            continue
        for val, cnt in _normalize(df[field]).value_counts().items():
            rows.append({"field": field, "value": val, "count": cnt,
                         "pct": round(100 * cnt / len(df), 2)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "audit_demographics.csv"), index=False)

    # Debt summary
    debt_rows = [{"field": f, **v} for f, v in debt.items()]
    pd.DataFrame(debt_rows).to_csv(os.path.join(out_dir, "audit_debt.csv"), index=False)
    print("[INFO] CSV reports written to audit_demographics.csv and audit_debt.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Common Voice Ethical Audit")
    parser.add_argument("--sample", type=int, default=5000, help="Rows to analyse")
    parser.add_argument("--output", default="audit_plots.pdf", help="Output PDF path")
    parser.add_argument("--csv_dir", default=".", help="Directory for CSV reports")
    args = parser.parse_args()

    df, dataset_name = load_common_voice(args.sample)

    print("\n[INFO] Running documentation debt analysis …")
    debt = documentation_debt_report(df)

    print("\n[INFO] Running representation bias analysis …")
    _ = gender_bias(df)
    _ = age_bias(df)
    _ = accent_bias(df)

    print("\n[INFO] Generating plots …")
    make_audit_plots(df, debt, dataset_name, output_path=args.output)
    export_csv(df, debt, out_dir=args.csv_dir)

    # Gini summary
    print("\n[Bias Severity — Gini Coefficients]")
    for field, label in [("gender", "Gender"), ("age", "Age"), ("accent", "Accent")]:
        if field in df.columns:
            g = gini_coefficient(_normalize(df[field]).value_counts())
            severity = "LOW" if g < 0.2 else "MODERATE" if g < 0.4 else "HIGH" if g < 0.6 else "SEVERE"
            print(f"  {label:<10} Gini={g:.4f}  [{severity}]")

    print("\n[DONE] Audit complete.\n")


if __name__ == "__main__":
    main()
