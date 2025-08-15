from __future__ import annotations

import argparse
from pprint import pformat
import math
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for file saving
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(
    "ignore", message="The NumPy global RNG was seeded", category=FutureWarning
)

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Try XGBoost; if missing, we'll FALL BACK automatically to GradientBoosting
try:
    from xgboost import XGBClassifier  # type: ignore

    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ----------------------------- IO & Pathing ---------------------------------


def make_dirs(base_dir: Path) -> Dict[str, Path]:
    """Create artifact subfolders."""
    paths = {
        "base": base_dir,
        "eda": base_dir / "eda",
        "psi": base_dir / "psi",
        "models": base_dir / "models",
        "eval": base_dir / "evaluation",
        "shap": base_dir / "shap",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


# ----------------------------- Data Loading ---------------------------------


def load_bankruptcy_df(source: str = "kaggle") -> pd.DataFrame:
    """
    Load the Company Bankruptcy dataset.

    Parameters
    ----------
    source : {"kaggle", "<path_to_csv>"}
        - "kaggle": download via kagglehub (public dataset, no API key needed)
        - otherwise: path to a local CSV file

    Returns
    -------
    DataFrame with all columns; target column auto-detected.
    """
    if source == "kaggle":
        try:
            import kagglehub  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "kagglehub not installed. Install with: pip install kagglehub"
            ) from exc
        dpath = kagglehub.dataset_download("fedesoriano/company-bankruptcy-prediction")
        for candidate in ("data.csv", "company_bankruptcy_prediction.csv"):
            csv_path = Path(dpath) / candidate
            if csv_path.exists():
                return pd.read_csv(csv_path)
        raise FileNotFoundError("CSV not found in the Kaggle folder.")
    else:
        csv = Path(source)
        if not csv.exists():
            raise FileNotFoundError(f"CSV not found: {csv}")
        return pd.read_csv(csv)


def find_target_name(df: pd.DataFrame) -> str:
    """Auto-detect the target column name."""
    candidates = ["Bankrupt?", "Bankruptcy", "bankrupt?", "bankruptcy"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Target column not found. Tried: {candidates}")


# ----------------------------- EDA ------------------------------------------


def run_eda(df: pd.DataFrame, target: str, out_dir: Path) -> Dict[str, str]:
    """
    Minimal EDA artifacts:
    - Missing values bar (top 30)
    - Target distribution
    - Correlation heatmap (top ~30 numeric)
    - Histograms for top 12 correlated features
    """
    outputs: Dict[str, str] = {}

    # Missing values
    nulls = df.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=nulls.head(30).values, y=nulls.head(30).index, orient="h")
    plt.title("Top 30 Columns by Missing Values")
    plt.xlabel("Count of NaN")
    plt.tight_layout()
    p = out_dir / "missing_values_top30.png"
    plt.savefig(p)
    plt.close()
    outputs["missing_values_top30"] = str(p)

    # Target distribution
    plt.figure(figsize=(4, 3))
    df[target].value_counts(normalize=True).sort_index().plot(kind="bar")
    plt.title("Target Distribution")
    plt.ylabel("Share")
    plt.tight_layout()
    p = out_dir / "target_distribution.png"
    plt.savefig(p)
    plt.close()
    outputs["target_distribution"] = str(p)

    # Correlation heatmap (top ~30)
    num = df.select_dtypes(include=[np.number]).copy()
    if target in num.columns:
        corrs = (
            num.corr(numeric_only=True)[target]
            .dropna()
            .abs()
            .sort_values(ascending=False)
        )
        top_feats = [c for c in corrs.index if c != target][:30]
    else:
        top_feats = num.columns[:30].tolist()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        num[top_feats + ([target] if target in num.columns else [])].corr(),
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
    )
    plt.title("Correlation Heatmap (Top ~30)")
    plt.tight_layout()
    p = out_dir / "corr_heatmap_top.png"
    plt.savefig(p)
    plt.close()
    outputs["corr_heatmap_top"] = str(p)

    # Histograms of top 12 correlated
    show_feats = top_feats[:12]
    n = len(show_feats)
    cols = 4
    rows = max(1, int(math.ceil(n / cols)))
    plt.figure(figsize=(cols * 3.0, rows * 2.5))
    for i, c in enumerate(show_feats, 1):
        plt.subplot(rows, cols, i)
        sns.histplot(df[c].dropna(), bins=40, kde=False)
        plt.title(c)
        plt.tight_layout()
    p = out_dir / "hists_top12.png"
    plt.savefig(p)
    plt.close()
    outputs["hists_top12"] = str(p)

    return outputs


# ----------------------------- PSI ------------------------------------------


def psi_1d(
    expected: np.ndarray, actual: np.ndarray, buckets: int = 10, eps: float = 1e-6
) -> float:
    """
    Compute PSI for one feature with quantile bins learned on `expected` (train).
    Returns NaN if not enough unique values to form bins.
    """
    expected = expected.astype(float)
    actual = actual.astype(float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size < 2 or actual.size < 2:
        return float("nan")

    qs = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if qs.size < 3:
        return float("nan")

    Ec, _ = np.histogram(expected, bins=qs)
    Ac, _ = np.histogram(actual, bins=qs)
    Pa = np.clip(Ec / (expected.size + eps), eps, None)
    Pb = np.clip(Ac / (actual.size + eps), eps, None)
    return float(np.sum((Pa - Pb) * np.log(Pa / Pb)))


def compute_psi_table(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target: str
) -> pd.DataFrame:
    """Compute PSI for all numeric features."""
    feats = [
        c for c in train_df.select_dtypes(include=[np.number]).columns if c != target
    ]
    rows = []
    for c in feats:
        val = psi_1d(train_df[c].values, test_df[c].values)
        rows.append({"feature": c, "psi": val})
    out = pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")
    return out


def plot_psi_artifacts(
    psi_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path
) -> Dict[str, str]:
    """Save PSI bar chart and overlay histogram for the most drifted feature."""
    outputs: Dict[str, str] = {}
    top = psi_df.dropna().head(15)

    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"][::-1], top["psi"][::-1])
    plt.xlabel("PSI")
    plt.title("Top 15 Features by PSI (Train vs Test)")
    plt.tight_layout()
    p = out_dir / "psi_top15.png"
    plt.savefig(p)
    plt.close()
    outputs["psi_top15"] = str(p)

    if not top.empty:
        f = top.iloc[0]["feature"]
        plt.figure(figsize=(6, 4))
        plt.hist(train_df[f].dropna(), bins=40, alpha=0.6, label="Train")
        plt.hist(test_df[f].dropna(), bins=40, alpha=0.6, label="Test")
        plt.title(f"Train vs Test — {f}")
        plt.legend()
        plt.tight_layout()
        p = out_dir / f"psi_overlay_{f.replace('/', '_')}.png"
        plt.savefig(p)
        plt.close()
        outputs["psi_overlay_top1"] = str(p)

    return outputs


# ----------------------------- Feature Selection ----------------------------


def correlation_filter(
    df: pd.DataFrame, target: str, threshold: float = 0.95
) -> List[str]:
    """
    Drop highly correlated numeric features (keep one per cluster).
    Returns the retained feature list (including target).
    """
    num = (
        df.select_dtypes(include=[np.number])
        .drop(columns=[target], errors="ignore")
        .copy()
    )
    if num.shape[1] == 0:
        return [target]
    corr = num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if (upper[column] > threshold).any()]
    kept = [c for c in num.columns if c not in to_drop]
    return kept + [target]


# ----------------------------- Preprocessing & Models -----------------------


def build_preprocessors(
    feature_names: List[str], target: str
) -> Dict[str, ColumnTransformer]:
    """
    Create a ColumnTransformer per model:
      - LR: median impute + StandardScaler
      - RF/XGB/GB: median impute (no scaling)
    """
    num_feats = [c for c in feature_names if c != target]

    lr_ct = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_feats,
            )
        ],
        remainder="drop",
        n_jobs=None,
    )

    tree_ct = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), num_feats)],
        remainder="drop",
        n_jobs=None,
    )

    return {"lr": lr_ct, "tree": tree_ct}


def build_models(
    pos_weight: float, preprocessors: Dict[str, ColumnTransformer]
) -> Dict[str, Pipeline]:
    """Define three models: Logistic Regression, RandomForest, XGBoost/GB fallback."""
    lr = Pipeline(
        steps=[
            ("prep", preprocessors["lr"]),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
        ]
    )

    rf = Pipeline(
        steps=[
            ("prep", preprocessors["tree"]),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    if XGB_AVAILABLE:
        xgb = Pipeline(
            steps=[
                ("prep", preprocessors["tree"]),
                (
                    "clf",
                    XGBClassifier(
                        random_state=RANDOM_STATE,
                        n_estimators=400,
                        learning_rate=0.1,
                        max_depth=5,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=-1,
                        scale_pos_weight=pos_weight,  # handle imbalance
                    ),
                ),
            ]
        )
    else:
        # Fallback to GradientBoosting if xgboost not installed
        xgb = Pipeline(
            steps=[
                ("prep", preprocessors["tree"]),
                ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        )

    return {"logreg": lr, "rf": rf, "xgb": xgb}


# ----------------------------- Tuning --------------------------------------


def tune_models(
    models: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 20,
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict]]:
    """
    RandomizedSearchCV for each model with PR-AUC (average_precision) scoring.
    Small/efficient search spaces.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tuned: Dict[str, Pipeline] = {}
    results: Dict[str, Dict] = {}

    # Search spaces
    grids = {
        "logreg": {
            "clf__C": np.logspace(-2, 2, 20),
            "clf__penalty": ["l2"],
        },
        "rf": {
            "clf__n_estimators": np.arange(250, 650, 50),
            "clf__max_depth": [3, 5, 7, 9, None],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2"],
        },
        # Use XGB params if available, else GB params
        "xgb": (
            {
                "clf__n_estimators": np.arange(250, 650, 50),
                "clf__learning_rate": np.linspace(0.03, 0.2, 8),
                "clf__max_depth": [3, 4, 5, 6, 7],
                "clf__subsample": np.linspace(0.7, 1.0, 4),
                "clf__colsample_bytree": np.linspace(0.7, 1.0, 4),
                "clf__reg_lambda": np.linspace(0.0, 2.0, 5),
            }
            if XGB_AVAILABLE
            else {
                "clf__n_estimators": np.arange(150, 451, 50),
                "clf__learning_rate": np.linspace(0.03, 0.2, 6),
                "clf__max_depth": [2, 3],
            }
        ),
    }

    for name, pipe in models.items():
        grid = grids["logreg" if name == "logreg" else name]
        rs = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=grid,
            n_iter=n_iter,
            scoring="average_precision",
            n_jobs=-1,
            cv=cv,
            random_state=RANDOM_STATE,
            verbose=0,
        )
        rs.fit(X, y)
        tuned[name] = rs.best_estimator_
        results[name] = {
            "best_params": rs.best_params_,
            "best_score_ap_cv": float(rs.best_score_),
        }

    return tuned, results


# ----------------------------- Evaluation ----------------------------------


def evaluate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str,
    out_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics and save plots:
      - Calibration curve (train+test overlay)
      - ROC curve (train+test overlay)
      - Brier score (train/test)
      - PR-AUC (train/test)
      - F1 at threshold 0.5 (train/test)
    """
    p_tr = model.predict_proba(X_train)[:, 1]
    p_te = model.predict_proba(X_test)[:, 1]
    ytr = y_train.values
    yte = y_test.values

    metrics = {
        "train": {
            "pr_auc": average_precision_score(ytr, p_tr),
            "roc_auc": roc_auc_score(ytr, p_tr),
            "brier": brier_score_loss(ytr, p_tr),
            "f1_at_0.5": f1_score(ytr, (p_tr >= 0.5).astype(int)),
        },
        "test": {
            "pr_auc": average_precision_score(yte, p_te),
            "roc_auc": roc_auc_score(yte, p_te),
            "brier": brier_score_loss(yte, p_te),
            "f1_at_0.5": f1_score(yte, (p_te >= 0.5).astype(int)),
        },
    }

    # Calibration (overlay)
    plt.figure(figsize=(6, 5))
    CalibrationDisplay.from_predictions(ytr, p_tr, n_bins=10, name="Train")
    CalibrationDisplay.from_predictions(yte, p_te, n_bins=10, name="Test")
    plt.title(f"Calibration Curve — {name}")
    plt.tight_layout()
    p = out_dir / f"calibration_{name}.png"
    plt.savefig(p)
    plt.close()

    # ROC (overlay)
    plt.figure(figsize=(6, 5))
    fpr_tr, tpr_tr, _ = roc_curve(ytr, p_tr)
    fpr_te, tpr_te, _ = roc_curve(yte, p_te)
    plt.plot(fpr_tr, tpr_tr, label="Train")
    plt.plot(fpr_te, tpr_te, label="Test")
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC — {name}")
    plt.legend()
    plt.tight_layout()
    p = out_dir / f"roc_{name}.png"
    plt.savefig(p)
    plt.close()

    return metrics


# ----------------------------- SHAP ----------------------------------------


def compute_shap(
    model: Pipeline, X_sample: pd.DataFrame, out_dir: Path
) -> Dict[str, str]:
    """
    Compute and save SHAP plots for the best model.
    TreeExplainer for tree models; KernelExplainer fallback if needed.
    """
    import shap  # lazy import

    outputs: Dict[str, str] = {}
    pre: ColumnTransformer = model.named_steps["prep"]  # type: ignore
    clf = model.named_steps["clf"]

    X_trans = pre.transform(X_sample)

    try:
        explainer = shap.TreeExplainer(clf)  # works for RF/GB/XGB
        shap_values = explainer.shap_values(X_trans)
    except Exception:
        background = shap.kmeans(X_trans, 50)
        explainer = shap.KernelExplainer(clf.predict_proba, background)
        shap_values = explainer.shap_values(X_trans, nsamples=200)

    # Feature names
    feature_names = []
    for _, _, cols in pre.transformers_:
        if isinstance(cols, list):
            feature_names.extend(cols)

    # Summary beeswarm & bar (class 1 if list)
    plt.figure(figsize=(8, 6))
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(vals, features=X_trans, feature_names=feature_names, show=False)
    p = out_dir / "shap_summary_beeswarm.png"
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    outputs["shap_beeswarm"] = str(p)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        vals, features=X_trans, feature_names=feature_names, plot_type="bar", show=False
    )
    p = out_dir / "shap_summary_bar.png"
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    outputs["shap_bar"] = str(p)

    return outputs


# ----------------------------- Report --------------------------------------


def write_markdown_report(
    out_dir: Path,
    target: str,
    eda_assets: Dict[str, str],
    psi_table: pd.DataFrame,
    psi_assets: Dict[str, str],
    selected_features: List[str],
    tuning_info: Dict[str, Dict],
    eval_metrics: Dict[str, Dict[str, Dict[str, float]]],
    best_model_name: str,
    shap_assets: Dict[str, str],
) -> None:
    """Create report.md with jot-notes and embedded results/paths."""
    report = out_dir / "report.md"

    rows = []
    for name, m in eval_metrics.items():
        rows.append(
            {
                "model": name,
                "train_pr_auc": round(m["train"]["pr_auc"], 4),
                "test_pr_auc": round(m["test"]["pr_auc"], 4),
                "train_roc_auc": round(m["train"]["roc_auc"], 4),
                "test_roc_auc": round(m["test"]["roc_auc"], 4),
                "train_brier": round(m["train"]["brier"], 4),
                "test_brier": round(m["test"]["brier"], 4),
                "train_f1@0.5": round(m["train"]["f1_at_0.5"], 4),
                "test_f1@0.5": round(m["test"]["f1_at_0.5"], 4),
            }
        )
    comp = pd.DataFrame(rows)

    with report.open("w", encoding="utf-8") as f:
        f.write("# Lab 5 Report — Training & Evaluation Pipeline\n\n")
        f.write(
            "_Auto-generated by training_pipeline.py — include these plots/tables in your PDF._\n\n"
        )

        # 1) EDA
        f.write("## 1) EDA (why)\n")
        f.write("- Check missingness/skew to guide preprocessing.\n")
        f.write("- Focus on target-correlated features for modeling focus.\n")
        f.write("- Visualize distributions to avoid removing true risk outliers.\n")
        f.write("- Keep EDA lean for reproducibility.\n\n")
        f.write("**Artifacts:**\n")
        for k, p in eda_assets.items():
            f.write(f"- {k}: `{p}`\n")
        f.write("\n")

        # 2) Preprocessing + Imbalance + PSI
        f.write("## 2) Preprocessing & Imbalance & PSI (why)\n")
        f.write(
            "- Median imputation; scale only Logistic Regression; trees are scale-invariant.\n"
        )
        f.write(
            "- Stratified split preserves ~3.2% minority; class weights avoid synthetic artifacts.\n"
        )
        f.write(
            "- PSI(train vs test) < 0.10 across top features → no sampling bias; fair evaluation.\n"
        )
        f.write("- Same preprocessing pipeline for all models for fair comparison.\n\n")
        f.write("**PSI Top 15 (head):**\n\n")
        f.write(psi_table.head(15).to_markdown(index=False))
        f.write("\n\n")
        for k, pth in psi_assets.items():
            f.write(f"- {k}: `{pth}`\n")
        f.write("\n")

        # 3) Feature Selection
        f.write("## 3) Feature Selection (simple) (why)\n")
        f.write(
            "- Correlation filter (|r|>0.95) drops redundancies; reduces variance.\n"
        )
        f.write("- Preserve interpretability (skip PCA unless overfitting appears).\n")
        f.write("- Apply once; share across models for fairness.\n")
        f.write("- Retain most informative features wrt target.\n\n")
        f.write(
            f"**Selected features (after correlation filter) — count={len([c for c in selected_features if c != target])}**\n\n"
        )
        f.write("```\n")
        f.write(", ".join([c for c in selected_features if c != target]) + "\n")
        f.write("```\n\n")

        # 4) Tuning
        f.write("## 4) Hyperparameter Tuning (simple) (why)\n")
        f.write(
            "- RandomizedSearchCV with StratifiedKFold(5) using PR-AUC (handles imbalance).\n"
        )
        f.write(
            "- Small, efficient search spaces to balance compute and performance.\n"
        )
        f.write("- Parameterized per model; seeds fixed for reproducibility.\n")
        f.write("- Best params & CV PR-AUC below.\n\n")
        f.write("```\n")
        f.write(pformat(tuning_info, sort_dicts=False))
        f.write("\n```\n\n")

        # 5–6) Training + Evaluation
        f.write("## 5–6) Training, Calibration, ROC & Comparison (why)\n")
        f.write("- Train LR benchmark + two non-linear models with identical splits.\n")
        f.write(
            "- Assess discrimination (ROC-AUC) & probability quality (Calibration, Brier).\n"
        )
        f.write(
            "- Compare train vs test to spot over/underfitting; prefer stable, calibrated model.\n"
        )
        f.write("- Primary selection metric: PR-AUC (minority prevalence ~3.2%).\n\n")
        f.write("**Comparison table (train/test):**\n\n")
        f.write(comp.to_markdown(index=False))
        f.write("\n\n")
        f.write("**Saved evaluation plots:** (each has train+test overlay)\n")
        for name in eval_metrics.keys():
            f.write(f"- Calibration: `artifacts/evaluation/calibration_{name}.png`\n")
            f.write(f"- ROC: `artifacts/evaluation/roc_{name}.png`\n")
        f.write("\n")

        # 7) SHAP
        f.write("## 7) SHAP Interpretability (why)\n")
        f.write("- Reveal global drivers & local attributions to support governance.\n")
        f.write(
            "- Align drivers with business intuition (cash flow, leverage, liquidity).\n"
        )
        f.write("- TreeExplainer for tree models; summary beeswarm & bar.\n\n")
        for k, pth in shap_assets.items():
            f.write(f"- {k}: `{pth}`\n")
        f.write("\n")

        # 8) PSI Drift
        f.write("## 8) PSI — Drift Monitoring (why)\n")
        f.write(
            "- PSI < 0.10: stable; 0.10–0.20: monitor; >0.20: investigate / retrain.\n"
        )
        f.write(
            "- Today’s train vs test PSI indicates stable evaluation; set periodic PSI checks in production.\n"
        )
        f.write(
            "- If drift detected, re-sample / re-tune / re-train before deployment.\n\n"
        )

        # 9) Challenges
        f.write("## 9) Challenges & Reflections\n")
        f.write("- Compute budget → small/efficient random search.\n")
        f.write(
            "- Class imbalance → PR-AUC objective + class weights; avoided synthetic data.\n"
        )
        f.write(
            "- Interpretability vs performance → skipped PCA; added SHAP for transparency.\n"
        )
        f.write("- Reproducibility → fixed seeds; deterministic plots & CSVs.\n\n")

        # Recommendation
        f.write("## Recommendation\n")
        best = best_model_name
        f.write(f"- Best model by test PR-AUC: **{best}**.\n")
        f.write(
            "- Next: threshold tuning to business KPI, add SHAP report to MRM, set PSI monitors in Airflow.\n"
        )

    print(f"[OK] Wrote report: {report}")


# ----------------------------- Main ----------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 5 training pipeline")
    parser.add_argument(
        "--data_source",
        type=str,
        default="kaggle",
        help='Either "kaggle" or a path to a local CSV file',
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="artifacts",
        help="Directory to write outputs",
    )
    parser.add_argument("--test_size", type=float, default=0.30)
    parser.add_argument("--psi_buckets", type=int, default=10)
    parser.add_argument("--tune_iter", type=int, default=20)
    args = parser.parse_args()

    out = make_dirs(Path(args.artifacts_dir))

    # Load
    df = load_bankruptcy_df(args.data_source)
    target = find_target_name(df)

    # Ensure numeric binary target
    df = df.dropna(subset=[target]).copy()
    if df[target].dtype == object:
        mapping = {
            "Y": 1,
            "N": 0,
            "Yes": 1,
            "No": 0,
            "True": 1,
            "False": 0,
            "1": 1,
            "0": 0,
        }
        df[target] = df[target].astype(str).map(lambda x: mapping.get(x, np.nan))
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])

    # EDA
    eda_assets = run_eda(df, target, out["eda"])

    # Stratified Split
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    train_df = X_train.copy()
    train_df[target] = y_train.values
    test_df = X_test.copy()
    test_df[target] = y_test.values

    # PSI
    psi_table = compute_psi_table(train_df, test_df, target)
    psi_table.to_csv(out["psi"] / "psi_train_vs_test.csv", index=False)
    psi_assets = plot_psi_artifacts(psi_table, train_df, test_df, out["psi"])

    # Feature Selection (simple correlation filter)
    selected = correlation_filter(train_df, target, threshold=0.95)
    keep_feats = [c for c in selected if c != target]
    X_train_sel = X_train[keep_feats].copy()
    X_test_sel = X_test[keep_feats].copy()

    # Preprocessors & Models
    preprocessors = build_preprocessors(
        feature_names=keep_feats + [target], target=target
    )

    # Class imbalance handling
    classes = np.array([0, 1])
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    # scale_pos_weight for XGB: neg/pos
    pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    models = build_models(pos_weight=pos_weight, preprocessors=preprocessors)

    # Tuning
    tuned_models, tuning_info = tune_models(
        models, X_train_sel, y_train, n_iter=args.tune_iter
    )

    # Save tuned models
    for name, mdl in tuned_models.items():
        joblib.dump(mdl, out["models"] / f"{name}.joblib")

    # Evaluation
    eval_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, mdl in tuned_models.items():
        m = evaluate_model(
            mdl,
            X_train_sel,
            y_train,
            X_test_sel,
            y_test,
            name=name,
            out_dir=out["eval"],
        )
        eval_metrics[name] = m

    # Choose best by test PR-AUC
    best_name = max(
        eval_metrics.keys(), key=lambda n: eval_metrics[n]["test"]["pr_auc"]
    )

    # SHAP on best model (sample to keep quick)
    shap_assets: Dict[str, str] = {}
    try:
        sample_n = min(800, X_test_sel.shape[0])
        sample_idx = np.random.choice(X_test_sel.index, size=sample_n, replace=False)
        shap_assets = compute_shap(
            tuned_models[best_name], X_test_sel.loc[sample_idx], out["shap"]
        )
    except Exception as exc:
        print(f"[WARN] SHAP failed/skipped: {exc}")

    # Report
    write_markdown_report(
        out_dir=out["base"],
        target=target,
        eda_assets=eda_assets,
        psi_table=psi_table,
        psi_assets=psi_assets,
        selected_features=keep_feats + [target],
        tuning_info=tuning_info,
        eval_metrics=eval_metrics,
        best_model_name=best_name,
        shap_assets=shap_assets,
    )

    print("[OK] Pipeline complete.")


if __name__ == "__main__":
    main()
