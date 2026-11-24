
"""Isolation Forest anomaly detector."""
from sklearn.ensemble import IsolationForest

def train_isolation_forest(X_train, contamination=0.0017,
                           n_estimators=300, max_samples="auto",
                           random_state=42):
    model = IsolationForest(
        n_estimators=int(n_estimators),
        contamination=float(contamination),
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train)
    return model

def iforest_scores(model, X):
    """Higher scores mean more anomalous; suitable for PR evaluation."""
    return -model.score_samples(X)

def iforest_predict(model, X):
    """Convert sklearn labels (-1 anomaly, +1 normal) to 1/0."""
    return (model.predict(X) == -1).astype(int)
