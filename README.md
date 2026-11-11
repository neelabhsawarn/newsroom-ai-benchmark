# Newsroom AI Benchmark: Topic Tagging Evaluation

**Portfolio project | AI product discovery, experimentation and evaluation | Python**

## The product question

Can automated tagging reduce editorial metadata work without sending too many articles to the wrong content queue? Compare a transparent **keyword baseline**, **TF–IDF + Multinomial Naive Bayes**, and **TF–IDF + Logistic Regression** on identical news headlines. The runner also supports optional **OpenAI model IDs** for an LLM-versus-baseline experiment (requires your own API key; **not executed in the supplied report**).

**Labels:** World / Sports / Business / Sci/Tech. One primary label per item. See [label policy](data/labeling_guide.md).

## What has actually been run

- **100 original synthetic test headlines**, balanced 25 per label, separate from 100 original synthetic training headlines. These are demonstration data, **not sourced from a newsroom, not independently labelled and not a valid estimate of production accuracy**.
- **Three approaches; three passes over the same 100 examples**. All generated valid categories and produced identical outputs across runs (these offline approaches are deterministic).
- Metrics computed programmatically: overall accuracy, macro precision/recall/F1, inference latency (median / p95), valid-output coverage, repeat consistency, first-run confusion counts and example mistakes.
- Local approaches incur **no external API fees**; hosting and compute cost were **not measured**.

| Approach | Accuracy (synthetic) | Macro F1 | Valid output | Repeat consistency |
|---|---:|---:|---:|---:|
| Keyword rules | 86% | 0.868 | 100% | 100% |
| TF–IDF + Naive Bayes | 89% | 0.889 | 100% | 100% |
| TF–IDF + Logistic Regression | 89% | 0.889 | 100% | 100% |

See [generated report](results/demo/run_report.md) and [raw metrics](results/demo/metrics.csv). Running the benchmark generates per-item predictions, confusion counts and error analysis locally. Observed shortcomings include sports/business headlines that lack training vocabulary and false matches from generic words. Identical repeat predictions for deterministic classifiers are expected, **not evidence of LLM consistency**.

## Reproduce the demonstration

Requires Python 3.10+; recommended: create a virtual environment.

```bash
python -m pip install -r requirements.txt
python src/create_demo_data.py
python src/benchmark.py
python -m pytest -q
```

First create the synthetic datasets using `python src/create_demo_data.py` (they are generated rather than checked in).

## Run the publishable next-stage benchmark on real AG News labels

The [AG News](https://huggingface.co/datasets/sh0416/ag_news) dataset provides published news titles/descriptions and four categories. The source has **120,000 train examples** and **7,600 test examples**. The data was compiled around 2004–2005, so it is **not a modern content-domain or production-readiness benchmark**. It is described for research and other non-commercial use; comply with its terms and attribution. The script downloads the original `train.csv`/`test.csv` files from the open GitHub mirror and verifies their SHA-256 hashes before sampling.

```bash
python src/prepare_ag_news.py --train-per-class 500 --eval-per-class 20 --seed 2026
python src/benchmark.py --train data/train_agnews.csv --eval data/eval_agnews.csv --out results/agnews --runs 3
```

This uses a **2,000-item independent training sample** and **80 test examples** (20 per class), sampled deterministically from the original separate train/test splits. It evaluates **titles only** by default, consistent with the content-tagging use case. Add `--include-description` to prepare both text fields; keep settings the same across models. **This real-data run could not be completed in the packaged environment because external dataset downloading was unavailable. No real-data metrics are asserted.**

## Compare real LLMs, once you choose API models

The code supports any model IDs accessible to your OpenAI account through the Responses API. Install dependencies, configure `OPENAI_API_KEY` in your **local environment** (never commit or share it), and run, for example:

```bash
# Replace MODEL_A and MODEL_B with valid model identifiers you can access.
python src/benchmark.py --train data/train_agnews.csv --eval data/eval_agnews.csv \
  --models 'keyword,openai:MODEL_A,openai:MODEL_B' --runs 3 --out results/openai \
  --rates data/api_rates.csv
```

A sample `data/api_rates_example.csv` shows the expected header (`model,input_usd_per_million,output_usd_per_million`). Add real prices valid on the date of the run to `data/api_rates.csv` to calculate per-model **estimated inference API cost**, using measured input/output token totals. Without supplied rates, API cost reports **unknown**, not zero. LLM runs can incur charges. For a cheap smoke test, generate a smaller test subset first. The sample does not use actual published content until you run the AG News preparation step.

**OpenAI integration is implemented but not exercised in the packaged test run** (no key was supplied). Output format failures, API errors and timeouts count as coverage failures; accuracy counts them as incorrect. Latency and costs depend on the network/model/prices at evaluation time.

## Blind human review and agreement

A [24-item review form](data/human_review_blank.csv) has blank `human_label` cells. Give it to a second annotator, ideally with the [label policy](data/labeling_guide.md), not the answer key. For the **real-data** run, generate a new review sheet:

```bash
python src/human_agreement.py --eval data/eval_agnews.csv --sheet data/human_review_agnews.csv --n 24
# After the independent reviewer fills human_label with World/Sports/Business/Sci/Tech:
python src/human_agreement.py --eval data/eval_agnews.csv --sheet data/human_review_agnews.csv --score
```

The script computes raw agreement and Cohen's kappa **against published reference labels**. Do not call this inter-annotator agreement unless those reference labels were themselves independently annotated. **No human review was completed**, so no agreement score is reported.

## Product decisions, limitations and next steps

A model with slightly better average accuracy is not automatically a better editorial product. Compare **specific confusion pairs**, coverage, latency, costs, review effort, data freshness, and editorial harm before deciding to pilot. A sensible pilot gates low-confidence and ambiguous cases for human review and sets a minimum class-specific quality threshold after stakeholders define the cost of misrouting. The two local ML systems do not expose comparable calibrated confidence in the current interface; that gate is **not implemented**. If an LLM is added, test output-format stability, repeat-run variance and token cost explicitly.

**Portfolio claim you can truthfully make today:** designed and implemented a reproducible three-approach news-tagging benchmark with 100 synthetic evaluation samples, multi-metric scorecards, failure analysis, latency and human-review workflow. **Do not claim published-data model scores, independent human validation or LLM testing until those runs are done.**

### Data attribution

Real-data workflow: Zhang, X., Zhao, J., and LeCun, Y. (2015), *Character-level Convolutional Networks for Text Classification*; [AG News source description](https://huggingface.co/datasets/sh0416/ag_news); [mirror referenced by downloader](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv). Synthetic examples: authored for this demonstration, not quotations or verified reports of news events.

## Continuous integration and CV entry

A GitHub Actions workflow in [`.github/workflows/python-tests.yml`](.github/workflows/python-tests.yml) is configured to install dependencies and run the six tests on a push or pull request. The workflow will run after the project is uploaded to GitHub; no GitHub Actions run has yet been performed.

A conservative, [CV-ready project description](CV_PROJECT_ENTRY.md) is included. Preserve the *synthetic-data* qualification until real-data results are available.