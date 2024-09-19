# src/models/model.py

from xgboost import XGBClassifier

def create_xgboost_model():
    """
    Create an XGBoost classifier.
    """
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    return model
