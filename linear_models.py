import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучение линейной регрессии пакетным градиентным спуском (MSE)."""
    X, y, X_test = np.asarray(data.X_train), np.asarray(data.y_train), np.asarray(data.X_test)
    w, b = np.zeros(X.shape[1]), 0.0
    lr, epochs, history = 0.01, 1500, []

    for _ in range(epochs):
        err = (X @ w + b) - y
        history.append(float(np.mean(err ** 2)))
        w -= lr * (2 / len(X)) * (X.T @ err)
        b -= lr * (2 / len(X)) * np.sum(err)

    return RegressionResult(w, float(b), history, X_test @ w + b)

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучение логистической регрессии пакетным градиентным спуском (BCE)."""
    X, y, X_test = np.asarray(data.X_train), np.asarray(data.y_train), np.asarray(data.X_test)
    w, b = np.zeros(X.shape[1]), 0.0
    lr, epochs, history = 0.1, 1500, []

    for _ in range(epochs):
        p = _sigmoid(X @ w + b)
        p_c = np.clip(p, 1e-15, 1 - 1e-15)
        history.append(-float(np.mean(y * np.log(p_c) + (1 - y) * np.log(1 - p_c))))
        w -= lr * (1 / len(X)) * (X.T @ (p - y))
        b -= lr * (1 / len(X)) * np.sum(p - y)

    prob = _sigmoid(X_test @ w + b)
    return ClassificationResult(w, float(b), history, prob, (prob >= 0.5).astype(int))

def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравнение собственной модели, Logistic Regression, Decision Tree и KNN."""
    y_test = np.asarray(data.y_test)
    own = train_logistic_classifier(data)

    models = {
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=7),
    }

    metrics = {
        "own_logistic": _calc_metrics(y_test, own.predictions, own.probabilities)
    }

    for name, model in models.items():
        model.fit(data.X_train, data.y_train)
        metrics[name] = _calc_metrics(
            y_test,
            model.predict(data.X_test),
            model.predict_proba(data.X_test)[:, 1],
        )

    return BaselineComparisonResult(metrics=metrics)