"""Run real scikit-learn classification tasks on its bundled breast-cancer dataset."""
import os
import time
import threading
import psutil
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

DATASET_NAME = "Breast Cancer Wisconsin (Diagnostic) — scikit-learn bundled dataset"
DATASET_SOURCE = "UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic), dataset 17; distributed via scikit-learn load_breast_cancer()."
OPERATION_LABELS = {
    "logistic_regression": "Binary Classification (Logistic Regression)",
    "random_forest": "Binary Classification (Random Forest)",
    "decision_tree": "Binary Classification (Decision Tree)",
}


def ensure_dataset(db, models):
    row = db.query(models.Dataset).filter(models.Dataset.name == DATASET_NAME).first()
    if not row:
        row = models.Dataset(name=DATASET_NAME, size_mb=0, n_features=30, n_classes=2)
        db.add(row)
        db.flush()
    return row


def run_classification(operation: str):
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=42, stratify=dataset.target
    )
    if operation == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    elif operation == "decision_tree":
        model = DecisionTreeClassifier(random_state=42, max_depth=5)
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))

    process = psutil.Process(os.getpid())
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 * 1024)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_finished = time.perf_counter()
    predictions = model.predict(X_test)
    finished = time.perf_counter()
    mem_after = process.memory_info().rss / (1024 * 1024)
    cpu_after = psutil.cpu_percent(interval=None)
    return {
        "duration": max(finished - started, 0.000001),
        "fit_time": max(fit_finished - started, 0.0),
        "predict_time": max(finished - fit_finished, 0.0),
        "accuracy": accuracy_score(y_test, predictions) * 100,
        "cpu": max(cpu_before, cpu_after),
        "ram_mb": max(mem_before, mem_after),
    }
