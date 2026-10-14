# Question Type Classification (NLP Course Project)

Classifies natural-language questions into six coarse TREC-style types:
`ABBR` (abbreviation), `DESC` (description), `ENTY` (entity), `HUM` (human), `LOC` (location), `NUM` (numeric).

> **Dataset note:** the data is **synthetic** (template-generated, 4,200 questions, 48 templates).
> The real TREC question-classification data could not be downloaded in the environment this was built in.
> Results are therefore illustrative of the method, not comparable to published TREC numbers.
> To use real data, replace `data/synthetic_questions.csv` with a CSV that has `question,label,template_id`
> columns (for TREC, `template_id` can be any constant and the held-out evaluation should be skipped).

## Layout
```
src/generate_dataset.py   template-based synthetic data generator (seeded)
src/train_eval.py         training + evaluation (two protocols)
data/synthetic_questions.csv
results/                  metrics.json, confusion matrices, error CSVs, comparison chart
report/                   project report
requirements.txt
```

## Run
```bash
pip install -r requirements.txt
python src/generate_dataset.py data/synthetic_questions.csv
python src/train_eval.py .
```

## Models
Majority baseline, TF-IDF (word 1-2 grams) with Naive Bayes / Logistic Regression / Linear SVM,
and Word+Char TF-IDF with Linear SVM.

## Evaluation protocols
1. **Random split** (stratified 80/20): optimistic, because templates are shared by train and test.
2. **Held-out templates**: 12 of the 48 templates are held out entirely for testing, which measures
   generalisation to unseen phrasings.

## Results (macro-F1)
| Model | Random split | Held-out templates |
|---|---|---|
| Majority baseline | 0.048 | 0.043 |
| TF-IDF + Naive Bayes | 1.000 | 0.556 |
| TF-IDF + Logistic Regression | 1.000 | 0.612 |
| TF-IDF + Linear SVM | 1.000 | 0.612 |
| Word+Char TF-IDF + Linear SVM | 1.000 | 0.646 |

The perfect random-split scores come from template leakage, not real skill; the held-out column is the meaningful one.
