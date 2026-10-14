"""Train and evaluate question-type classifiers on the synthetic dataset.

Two evaluation protocols:
  1. random   : stratified 80/20 split (templates are shared between train/test,
                so this is optimistic)
  2. heldout  : whole templates are held out for testing (~25% per class), so the
                model must generalise to unseen question phrasings.
"""
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

SEED = 42
LABELS = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]


def word_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True, min_df=1)


def char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, sublinear_tf=True)


def models():
    return {
        "Majority baseline": Pipeline([("clf", DummyClassifier(strategy="most_frequent")),]) ,
        "TF-IDF + Naive Bayes": Pipeline([("tf", word_tfidf()), ("clf", MultinomialNB(alpha=0.1))]),
        "TF-IDF + Logistic Regression": Pipeline([("tf", word_tfidf()),
                                                  ("clf", LogisticRegression(max_iter=2000, C=10))]),
        "TF-IDF + Linear SVM": Pipeline([("tf", word_tfidf()), ("clf", LinearSVC(C=1.0))]),
        "Word+Char TF-IDF + Linear SVM": Pipeline([
            ("tf", FeatureUnion([("w", word_tfidf()), ("c", char_tfidf())])),
            ("clf", LinearSVC(C=1.0))]),
    }


def fit_predict(model, Xtr, ytr, Xte):
    if model.steps[0][0] == "clf":  # dummy needs 2D-like input only
        model.fit(np.zeros((len(Xtr), 1)), ytr)
        return model.predict(np.zeros((len(Xte), 1)))
    model.fit(Xtr, ytr)
    return model.predict(Xte)


def heldout_split(df, frac=0.25, seed=SEED):
    rng = random.Random(seed)
    test_templates = set()
    for lab, g in df.groupby("label"):
        tids = sorted(g.template_id.unique())
        rng.shuffle(tids)
        k = max(1, round(len(tids) * frac))
        test_templates.update(tids[:k])
    te = df[df.template_id.isin(test_templates)]
    tr = df[~df.template_id.isin(test_templates)]
    return tr, te, sorted(test_templates)


def evaluate(df, tr, te, tag, outdir, save_best=True):
    res, preds = {}, {}
    for name, m in models().items():
        p = fit_predict(m, tr.question.values, tr.label.values, te.question.values)
        res[name] = {"accuracy": accuracy_score(te.label, p),
                     "macro_f1": f1_score(te.label, p, average="macro", labels=LABELS)}
        preds[name] = p
    best = max((k for k in res if k != "Majority baseline"), key=lambda k: res[k]["macro_f1"])
    p = preds[best]
    rep = classification_report(te.label, p, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(te.label, p, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{best}\n({tag} split)", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / f"confusion_{tag}.png", dpi=160)
    plt.close(fig)
    err = te.assign(pred=p)
    err = err[err.label != err.pred][["question", "label", "pred", "template_id"]]
    err.to_csv(outdir / f"errors_{tag}.csv", index=False)
    return res, best, rep, cm.tolist(), len(err)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    df = pd.read_csv(root / "data" / "synthetic_questions.csv")
    out = root / "results"
    out.mkdir(exist_ok=True)

    summary = {"n_rows": len(df), "class_counts": df.label.value_counts().to_dict(),
               "n_templates": int(df.template_id.nunique())}

    # --- Protocol 1: random stratified split
    tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=SEED)
    r1, best1, rep1, cm1, nerr1 = evaluate(df, tr, te, "random", out)
    summary["random"] = {"train": len(tr), "test": len(te), "results": r1, "best": best1,
                         "per_class": rep1, "confusion": cm1, "n_errors": nerr1}

    # --- 5-fold CV on random protocol for the best model (mean +- std)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv = cross_val_score(models()[best1], df.question.values, df.label.values, cv=skf, scoring="f1_macro")
    summary["random"]["cv_macro_f1_mean"] = float(cv.mean())
    summary["random"]["cv_macro_f1_std"] = float(cv.std())

    # --- Protocol 2: held-out templates
    tr2, te2, tt = heldout_split(df)
    r2, best2, rep2, cm2, nerr2 = evaluate(df, tr2, te2, "heldout", out)
    summary["heldout"] = {"train": len(tr2), "test": len(te2), "test_templates": tt, "results": r2,
                          "best": best2, "per_class": rep2, "confusion": cm2, "n_errors": nerr2}

    # --- bar chart comparing both protocols
    names = [k for k in r1]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(x - 0.2, [r1[n]["macro_f1"] for n in names], 0.4, label="Random split", color="#2a6f97")
    ax.bar(x + 0.2, [r2[n]["macro_f1"] for n in names], 0.4, label="Held-out templates", color="#e07a5f")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" + ", "\n+ ") for n in names], fontsize=7)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=160)
    plt.close(fig)

    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("n_rows", "class_counts", "n_templates")}, indent=2))
    for tag in ("random", "heldout"):
        print(f"\n== {tag} (train={summary[tag]['train']}, test={summary[tag]['test']}) best={summary[tag]['best']}")
        for k, v in summary[tag]["results"].items():
            print(f"  {k:34s} acc={v['accuracy']:.3f}  macroF1={v['macro_f1']:.3f}")
        print("  errors:", summary[tag]["n_errors"])
    print(f"\n5-fold CV macro-F1 ({best1}): {cv.mean():.3f} +- {cv.std():.3f}")


if __name__ == "__main__":
    main()
