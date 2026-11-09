The UFC-MPE is a machine learning pipeline designed to predict the winner of UFC matchups with high precision. Unlike traditional models that rely solely on physical attributes (Reach, Age, Weight) and raw skill statistics, this project introduces a novel feature engineering layer designed to quantify fighter resilience ("Heart") and durability ("Chin").

By combining XGBoost with Optuna hyperparameter tuning and rigorous SMOTE balancing, the model achieves a ~76% F1-score on unseen test data, significantly outperforming the betting market baseline (~65-70%).

The pipeline is built for robustness and prevention of data leakage.

Model: XGBoost Classifier (Gradient Boosting).

Optimization: Optuna (Bayesian Optimization) for hyperparameter tuning.

Imbalance Handling: RandomUnderSampling (initial) + SMOTE (Synthetic Minority Over-sampling Technique) inside Cross-Validation.

Validation Strategy: Stratified K-Fold Cross-Validation (5 Splits).

Pipeline Flow

Ingestion: Raw UFC stats loaded and cleaned.

Feature Engineering: Vectorized calculation of "Heart" metrics and physical differentials (Reach Diff, Age Diff).

Segmentation: Dataset split into cohorts (Women, Tiny, Big Guy) for specialized modeling.

Optimization: Optuna searches for optimal parameters while preventing leakage by applying SMOTE only to training folds.

Evaluation: Final testing on a hold-out set (20% of data).


## Data sources & promotions

The pipeline no longer needs a local CSV: it pulls data from APIs and caches every response under `data/cache/` (gzip), so only the first run is slow.

| Promotion | Key (`--promotions`) | Sports | Source |
|---|---|---|---|
| UFC (+ Dana White's Contender Series) | `ufc` | MMA | Official UFC API (`d29dxerjsp82wz.cloudfront.net/api/v3`): events, fight cards, results, fighter bios, per-fight stats |
| GLORY | `glory` | Kickboxing | Wikipedia "`<year> in Glory`" results pages (2012–) via the MediaWiki API, + Wikidata (DOB, height) |
| ONE Championship | `one` (alias `onefc`) | MMA, Muay Thai, Kickboxing, Grappling | Wikipedia "`<year> in ONE Championship`" pages (2011–), + Wikidata |

GLORY's and ONE's own websites have no public API, so their results come from Wikipedia. More promotions with yearly Wikipedia results pages can be added with one entry in `fightdata/promotions.py`.

## Usage

```bash
pip install -r requirements.txt
python predictions.py train --promotions ufc glory one      # fetch + train (one model set per promotion & sport)
python predictions.py predict --promotions ufc --days 14     # win probabilities for upcoming bouts
python predictions.py fetch --out bouts.csv                  # just the bouts + features table
python predictions.py train --csv large_dataset.csv          # original CSV workflow
```

Useful flags: `--group-by sport` (pool e.g. GLORY + ONE kickboxing), `--since 2015`, `--trials 30`, `--no-ufc-stats` (skip ~9k per-fight stat requests), `--ufc-event-range 1200:1350`, `--offline` (cache only). Wikimedia asks API clients to identify themselves: set `FIGHTDATA_USER_AGENT="my-app/1.0 (you@example.com)"`.

## Features (no leakage)

All features are computed from each fighter's **earlier** bouts only (strictly earlier dates), then fed to the model as red-minus-blue differences:

* record (all promotions): fights, win rate, streak, KO/sub win rates, finish rate, **Chin** (1 − KO/TKO loss rate), **Heart** (win rate in fights past round 1), layoff
* physical: age at fight time (from DOB), height, reach
* UFC stats: career sig. strike accuracy & defense, strikes landed/absorbed per minute, takedown accuracy & defense, submission attempts, reversals, knockdowns

Wikipedia lists the winner first, so corners for GLORY/ONE are assigned by a deterministic coin flip that does not depend on the result. The UFC API's `Record`/`Age` fields are current values and are deliberately not used.
