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

├── data/
│   └── large_dataset.csv       # Historical fight data
├── artifacts/
│   ├── model_pipeline_BigGuy.pkl  # Trained Model (Heavyweights)
│   ├── model_pipeline_Women.pkl   # Trained Model (Women's divisions)
│   └── imputer_Tiny.pkl           # KNN Imputer for Lightweights
├── src/
│   ├── feature_engineering.py  # Logic for "Heart" & "Chin" metrics
│   └── train_model.py          # Main training loop
├── README.md
└── requirements.txt

Cohort,F1-Score,Precision,Recall,Support
Overall,0.76,0.77,0.75,1450
Women,0.72,0.73,0.71,320
Heavyweight,0.79,0.81,0.78,210
