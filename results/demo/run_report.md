# Newsroom AI tagging benchmark — run report

- Evaluation: `eval_demo.csv`; training: `train_demo.csv`
- Data provenance: **original synthetic demo examples**
- Labels: World / Sports / Business / Sci/Tech
- Same test examples used for each approach; ML training set is separate from evaluation set.
- Accuracy and macro recall count missing or invalid outputs as misses. Macro precision/F1 use all attempted examples.
- Repeat consistency: share of test items with exactly the same output on every run.
- Latency is local wall-clock inference latency per item and is hardware/network dependent.
- API cost is estimated only when per-token pricing is configured; offline models have no API fee.

| Approach | Accuracy | Macro F1 | Coverage | Consistency | Median latency | p95 latency | API cost |
|---|---:|---:|---:|---:|---:|---:|---|
| keyword | 86.0% | 0.868 | 100.0% | 100.0% | 0.00 ms | 0.01 ms | $0.00000 |
| nb | 89.0% | 0.889 | 100.0% | 100.0% | 0.22 ms | 0.35 ms | $0.00000 |
| logreg | 89.0% | 0.889 | 100.0% | 100.0% | 0.22 ms | 0.36 ms | $0.00000 |

## Observed failure examples

### keyword
- `SYN-EV-029`: true **Sports**, predicted **World** — Runner sets personal best at indoor championships.
- `SYN-EV-044`: true **Sports**, predicted **World** — Baseball manager reviews season performance.
- `SYN-EV-048`: true **Sports**, predicted **World** — Sailing team reaches next stage of regatta.
- `SYN-EV-050`: true **Sports**, predicted **World** — Formula driver takes pole position in qualifying.
- `SYN-EV-064`: true **Business**, predicted **World** — Commodity traders react to supply constraints.
### nb
- `SYN-EV-021`: true **World**, predicted **Sports** — Parliament passes changes to citizenship rules.
- `SYN-EV-028`: true **Sports**, predicted **Sci/Tech** — Cricket bowlers dominate opening day of test.
- `SYN-EV-037`: true **Sports**, predicted **Sci/Tech** — Motorsport team upgrades car before grand prix.
- `SYN-EV-042`: true **Sports**, predicted **Business** — Boxing challenger trains ahead of title fight.
- `SYN-EV-053`: true **Business**, predicted **World** — Merger talks continue between two insurance firms.
### logreg
- `SYN-EV-021`: true **World**, predicted **Sports** — Parliament passes changes to citizenship rules.
- `SYN-EV-028`: true **Sports**, predicted **Sci/Tech** — Cricket bowlers dominate opening day of test.
- `SYN-EV-037`: true **Sports**, predicted **Sci/Tech** — Motorsport team upgrades car before grand prix.
- `SYN-EV-042`: true **Sports**, predicted **Business** — Boxing challenger trains ahead of title fight.
- `SYN-EV-053`: true **Business**, predicted **World** — Merger talks continue between two insurance firms.

## Business interpretation

Accuracy alone is insufficient for publication workflows: measure class-specific errors, invalid output coverage, and human escalation needs. Use the confusion matrix and prediction CSV to inspect mistakes.
For a production pilot, define thresholds per high-risk class, adjudicate ambiguous labels, re-run on recent editorial data, and monitor drift.
A result from a tiny or synthetic sample is **not** a claim of production accuracy, model superiority, or human annotation agreement.

## Human review

Give `data/human_review_blank.csv` to a second annotator. They should fill `human_label` without seeing the original label where practical. Run `python src/human_agreement.py` after labels are provided. Agreement has not been measured unless a populated report exists.