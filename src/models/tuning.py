"""Hyperparameter search spaces + RandomizedSearchCV wrapper. All tuning
happens via cross-validation on the training split only — the held-out test
set is never touched here."""

from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

PARAM_DISTRIBUTIONS = {
    "Logistic Regression": {
        "model__C": uniform(0.01, 10),
        "model__class_weight": [None, "balanced"],
    },
    "Random Forest": {
        "model__n_estimators": randint(100, 400),
        "model__max_depth": randint(3, 25),
        "model__min_samples_leaf": randint(1, 6),
        "model__class_weight": [None, "balanced"],
    },
    "Gradient Boosting": {
        "model__n_estimators": randint(50, 300),
        "model__max_depth": randint(2, 6),
        "model__learning_rate": uniform(0.02, 0.3),
    },
    "HistGradientBoosting": {
        "model__max_iter": randint(50, 300),
        "model__max_depth": randint(2, 12),
        "model__learning_rate": uniform(0.02, 0.3),
    },
    "SVM": {
        "model__estimator__C": uniform(0.1, 20),
        "model__estimator__gamma": ["scale", "auto"],
        "model__estimator__class_weight": [None, "balanced"],
    },
}


def tune(pipeline, model_name: str, X_train, y_train, n_iter: int = 25, random_state: int = 42):
    distributions = PARAM_DISTRIBUTIONS.get(model_name)
    if not distributions:
        pipeline.fit(X_train, y_train)
        return pipeline, {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=distributions,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_
