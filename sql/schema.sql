-- Reference schema for Adaptive ML Workload Orchestrator Phase 2.
-- The application creates these tables through SQLAlchemy on first startup.
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    size_mb REAL DEFAULT 0,
    n_features INTEGER DEFAULT 30,
    n_classes INTEGER DEFAULT 2
);
CREATE TABLE IF NOT EXISTS tasks (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT NOT NULL,
    task_type TEXT NOT NULL DEFAULT 'Classification',
    dataset_id INTEGER REFERENCES datasets(dataset_id),
    priority INTEGER NOT NULL DEFAULT 5,
    subtype TEXT NOT NULL DEFAULT 'logistic_regression',
    data_fraction REAL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'pending',
    algorithm_used TEXT DEFAULT 'FCFS',
    submit_time TIMESTAMP,
    start_time TIMESTAMP,
    end_time TIMESTAMP
);
CREATE TABLE IF NOT EXISTS resource_requirements (
    req_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER REFERENCES tasks(task_id) ON DELETE CASCADE,
    baseline_cpu REAL,
    baseline_ram_mb REAL,
    baseline_duration_sec REAL,
    predicted_cpu REAL,
    predicted_ram_mb REAL,
    predicted_duration_sec REAL
);
CREATE TABLE IF NOT EXISTS execution_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER REFERENCES tasks(task_id),
    actual_cpu REAL,
    actual_ram_mb REAL,
    actual_duration_sec REAL,
    wait_time_sec REAL,
    accuracy REAL,
    fit_time_sec REAL,
    predict_time_sec REAL,
    algorithm_used TEXT,
    is_anomaly BOOLEAN DEFAULT 0,
    execution_time TIMESTAMP
);
CREATE TABLE IF NOT EXISTS scheduling_decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER REFERENCES tasks(task_id),
    decision TEXT,
    reason TEXT,
    timestamp TIMESTAMP
);
CREATE TABLE IF NOT EXISTS resource_locks (
    lock_id INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_name TEXT NOT NULL,
    held_by_task_id INTEGER REFERENCES tasks(task_id),
    requested_by_task_id INTEGER REFERENCES tasks(task_id)
);
