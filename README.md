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

