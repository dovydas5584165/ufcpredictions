# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# pip install -r requirements.txt
#
# Usage:
#   python predictions.py train   --promotions ufc glory one   # fetch data via APIs, train per promotion/sport
#   python predictions.py predict --promotions ufc             # win probabilities for upcoming bouts
#   python predictions.py fetch   --out bouts.csv              # just download + build features
#   python predictions.py train   --csv large_dataset.csv      # original Kaggle-CSV workflow

import argparse
import json
import logging
import re
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score, classification_report
from imblearn.pipeline import Pipeline as ImbPipeline  # Critical for using SMOTE in pipelines
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from fightdata import PROMOTIONS, build_features, candidate_features, load_bouts
from fightdata.promotions import default_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RANDOM_STATE = 42
TARGET_COL = 'winner'
MIN_SEGMENT_SIZE = 50
MIN_FEATURE_COVERAGE = 0.2   # drop features missing in >80% of a segment's rows
LEGACY_FEATURES = [
    'reach_diff', 'age_diff', 'rev_diff',
    'sig_str_acc_diff', 'td_acc_diff', 'sub_attempt_diff'
]
COHORTS = {
    'Women': "gender == 'Women'",
    'Tiny': "r_weight < 80 & b_weight < 80",
    'Big Guy': "r_weight >= 80 & b_weight >= 80",
}

# ==========================================
# 2. DATA PROCESSING FUNCTIONS
# ==========================================

def load_api_dataset(args) -> pd.DataFrame:
    """Fetch bouts (UFC API, Wikipedia for GLORY/ONE; cached on disk) and build pre-fight features."""
    client = default_client(cache_dir=args.cache_dir, offline=args.offline)
    bouts = load_bouts(args.promotions, client=client, ufc_stats=not args.no_ufc_stats,
                       ufc_event_range=args.ufc_event_range)
    logging.info(f"Loaded {len(bouts)} bouts; engineering leak-free pre-fight features...")
    return build_features(bouts)


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """Original workflow: a Kaggle-style CSV with r_/b_ columns and a Red/Blue `winner`."""
    logging.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath).dropna(subset=[TARGET_COL])
    return df.assign(
        age_diff=df['r_age'] - df['b_age'],
        reach_diff=df['r_reach'] - df['b_reach'],
        rev_diff=df['r_rev'] - df['b_rev'],
        sig_str_acc_diff=df['r_sig_str_acc'] - df['b_sig_str_acc'],
        td_acc_diff=df['r_td_acc'] - df['b_td_acc'],
        sub_attempt_diff=df['r_sub_att'] - df['b_sub_att'],
        target=df[TARGET_COL].map({'Red': 1, 'Blue': 0}),
    )


def segment_data(df: pd.DataFrame, group_by: str = 'promotion') -> dict:
    """
    Splits data into segments: one group per promotion+sport (or per sport),
    each split into cohorts (Women, Tiny < 80 kg, Big Guy >= 80 kg).
    Returns {segment name: (sport, frame)}.
    """
    if 'promotion' not in df:  # legacy CSV
        return {name: ('mma', df.query(q)) for name, q in COHORTS.items()}
    keys = ['promotion', 'sport'] if group_by == 'promotion' else ['sport']
    segments = {}
    for values, group in df.groupby(keys, sort=True):
        prefix = '_'.join(values if isinstance(values, tuple) else (values,))
        for cohort, query in COHORTS.items():
            segments[f"{prefix}_{cohort}"] = (group['sport'].iloc[0], group.query(query))
    return segments


def select_features(df: pd.DataFrame, candidates: list) -> list:
    """Keep candidates with enough coverage and some variance in this segment."""
    keep = []
    for col in candidates:
        if col in df:
            values = pd.to_numeric(df[col], errors='coerce')
            if values.notna().mean() >= MIN_FEATURE_COVERAGE and values.nunique() > 1:
                keep.append(col)
    return keep


def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Random undersampling of the majority class (index-based, works with any column types)."""
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    rus.fit_resample(np.arange(len(df)).reshape(-1, 1), df['target'].astype(int))
    return df.iloc[np.sort(rus.sample_indices_)]

# ==========================================
# 3. OPTIMIZATION & TRAINING ENGINE
# ==========================================

def objective(trial, X, y):
    """Optuna objective: CV F1 with SMOTE applied only to training folds (no leakage)."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss',
        'n_jobs': 1,
    }
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(**params)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        pipeline.fit(X[train_idx], y[train_idx])
        scores.append(f1_score(y[val_idx], pipeline.predict(X[val_idx])))
    return np.mean(scores)


def _slug(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')


def train_segment_model(group_name, df_group, candidates, n_trials, out_dir: Path):
    """Impute -> Optuna search -> final model -> save artifacts -> evaluate on 20% hold-out."""
    if len(df_group) < MIN_SEGMENT_SIZE:
        logging.warning(f"Skipping group '{group_name}': Insufficient data points ({len(df_group)}).")
        return None
    features = select_features(df_group, candidates)
    if not features:
        logging.warning(f"Skipping group '{group_name}': no usable features.")
        return None
    df_group = balance_classes(df_group)
    if len(df_group) < MIN_SEGMENT_SIZE:
        logging.warning(f"Skipping group '{group_name}': too few rows after balancing ({len(df_group)}).")
        return None

    logging.info(f"--- Starting Process for Group: {group_name} ({len(df_group)} rows, {len(features)} features) ---")
    X = df_group[features].to_numpy(dtype=float)
    y = df_group['target'].to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    imputer = KNNImputer(n_neighbors=5, keep_empty_features=True)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials, n_jobs=-1)
    best_params = study.best_params
    logging.info(f"[{group_name}] Best Parameters found: {best_params}")

    final_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(**best_params, random_state=RANDOM_STATE, eval_metric='logloss')),
    ])
    final_pipeline.fit(X_train, y_train)

    y_pred = final_pipeline.predict(X_test)
    score = f1_score(y_test, y_pred)
    accuracy = float((y_pred == y_test).mean())
    print(f"\n=== Classification Report for {group_name} ===")
    print(f"Test F1 Score: {score:.4f}")
    print(classification_report(y_test, y_pred))
    print("=" * 40 + "\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(group_name)
    joblib.dump(final_pipeline, out_dir / f'model_pipeline_{slug}.pkl')
    joblib.dump(imputer, out_dir / f'imputer_{slug}.pkl')
    meta = {'segment': group_name, 'features': features, 'rows': len(df_group),
            'test_f1': score, 'test_accuracy': accuracy, 'best_params': best_params}
    (out_dir / f'features_{slug}.json').write_text(json.dumps(meta, indent=2))
    logging.info(f"[{group_name}] Models saved to {out_dir}.")
    return meta

# ==========================================
# 4. PREDICTION OF UPCOMING BOUTS
# ==========================================

def _segment_for(bout, group_by):
    prefix = f"{bout['promotion']}_{bout['sport']}" if group_by == 'promotion' else bout['sport']
    rw, bw = bout['r_weight'], bout['b_weight']
    cohorts = []
    if bout['gender'] == 'Women':
        cohorts.append('Women')
    if pd.notna(rw) and pd.notna(bw):
        cohorts.append('Tiny' if (rw + bw) / 2 < 80 else 'Big Guy')
    return [f"{prefix}_{c}" for c in cohorts]


def predict_upcoming(df, models_dir: Path, group_by='promotion', days=None):
    today = pd.Timestamp.today().normalize()
    upcoming = df[(df['status'] == 'upcoming') & (df['event_date'] >= today - pd.Timedelta(days=1))]
    if days:
        upcoming = upcoming[upcoming['event_date'] <= today + pd.Timedelta(days=days)]
    cache, rows = {}, []
    for _, bout in upcoming.iterrows():
        for segment in _segment_for(bout, group_by):
            slug = _slug(segment)
            if slug not in cache:
                meta_path = models_dir / f'features_{slug}.json'
                cache[slug] = None if not meta_path.exists() else (
                    json.loads(meta_path.read_text()),
                    joblib.load(models_dir / f'imputer_{slug}.pkl'),
                    joblib.load(models_dir / f'model_pipeline_{slug}.pkl'))
            if cache[slug] is None:
                continue
            meta, imputer, model = cache[slug]
            x = imputer.transform(bout[meta['features']].to_numpy(dtype=float).reshape(1, -1))
            p_red = float(model.predict_proba(x)[0, 1])
            rows.append({'date': bout['event_date'].date(), 'event': bout['event_name'],
                         'weight_class': bout['weight_class'], 'red': bout['r_name'], 'blue': bout['b_name'],
                         'p_red': round(p_red, 3), 'pick': bout['r_name'] if p_red >= 0.5 else bout['b_name'],
                         'model': segment})
            break
    result = pd.DataFrame(rows)
    logging.info(f"{len(upcoming)} upcoming bouts, {len(result)} with a trained model.")
    return result

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def _event_range(text):
    first, _, last = text.partition(':')
    return int(first or 1), (int(last) if last else None)


def build_parser():
    p = argparse.ArgumentParser(description='Fight outcome models for UFC, GLORY, ONE Championship.')
    p.add_argument('command', nargs='?', default='train', choices=['train', 'predict', 'fetch'])
    p.add_argument('--promotions', nargs='+', default=['ufc', 'glory', 'one'],
                   help=f"any of: {', '.join(PROMOTIONS)} (aliases like 'onefc' work)")
    p.add_argument('--group-by', choices=['promotion', 'sport'], default='promotion',
                   help="train one model set per promotion+sport, or pool promotions per sport")
    p.add_argument('--since', type=int, help='only train on bouts from this year on (history still used)')
    p.add_argument('--trials', type=int, default=30, help='Optuna trials per segment')
    p.add_argument('--models-dir', type=Path, default=Path('models'))
    p.add_argument('--cache-dir', default=None, help='HTTP cache (default data/cache)')
    p.add_argument('--offline', action='store_true', help='use cached API responses only')
    p.add_argument('--no-ufc-stats', action='store_true', help='skip per-fight UFC stats (much faster first run)')
    p.add_argument('--ufc-event-range', type=_event_range, metavar='FIRST:LAST', help='UFC API event ID range')
    p.add_argument('--days', type=int, help='predict: only bouts within this many days')
    p.add_argument('--out', type=Path, help='fetch/predict: write a CSV here')
    p.add_argument('--csv', help='train on a local Kaggle-style CSV instead of the APIs (original workflow)')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.csv:
        df = load_and_preprocess_data(args.csv)
        candidates_for = lambda sport: LEGACY_FEATURES
    else:
        df = load_api_dataset(args)
        candidates_for = candidate_features

    if args.command == 'fetch':
        out = args.out or Path('bouts_features.csv')
        df.to_csv(out, index=False)
        logging.info(f"Wrote {len(df)} rows to {out}")
        return
    if args.command == 'predict':
        result = predict_upcoming(df, args.models_dir, args.group_by, args.days)
        print(result.to_string(index=False) if not result.empty else "No upcoming bouts with a trained model.")
        if args.out is not None:
            result.to_csv(args.out, index=False)
        return

    train = df[df['target'].notna()]
    if args.since and 'event_date' in train:
        train = train[train['event_date'].dt.year >= args.since]
    summary = []
    for group_name, (sport, df_segment) in segment_data(train, args.group_by).items():
        meta = train_segment_model(group_name, df_segment, candidates_for(sport), args.trials, args.models_dir)
        if meta:
            summary.append({k: meta[k] for k in ('segment', 'rows', 'test_f1', 'test_accuracy')})
    if summary:
        print(pd.DataFrame(summary).to_string(index=False))
    logging.info("All training tasks completed successfully.")


if __name__ == "__main__":
    main()
