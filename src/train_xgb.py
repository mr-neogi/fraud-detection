
"""SMOTE + XGBoost training. SMOTE is applied to training data only."""
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def train_xgboost(X_train, y_train, sampling_strategy=0.5,
                  max_depth=4, learning_rate=0.05, n_estimators=350,
                  random_state=42):
    smote = SMOTE(
        sampling_strategy=float(sampling_strategy),
        random_state=random_state,
        k_neighbors=min(5, max(1, int(y_train.value_counts().min()) - 1))
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    model = XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_res, y_res)
    return model, X_res, y_res
