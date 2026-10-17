# Adaptive ML Workload Orchestrator — Phase 2

A small student-friendly PBL project demonstrating **Operating Systems CPU scheduling and deadlock detection** together with **DBMS persistence using SQLite**. The dashboard is served by FastAPI and plain HTML/CSS/JavaScript.

## Features
- Submit a real ML job with a name, ML operation, and priority.
- Run pending jobs sequentially using FCFS, SJF, or Priority Scheduling.
- Execute Logistic Regression, Decision Tree, or Random Forest on a real bundled classification dataset.
- Save submitted jobs and measured execution results to SQLite (`backend/scheduler.db`).
- Keep the active queue separate from execution history.
- Compare scheduling algorithms using actual measured durations saved in the database.
- Demonstrate a safe wait-for graph and a circular-wait deadlock. This is a small teaching scenario, not a detector for all operating-system processes.

## Dataset source
The model jobs use scikit-learn's `load_breast_cancer()` dataset, the Breast Cancer Wisconsin (Diagnostic) dataset (569 samples, 30 features), originally distributed through the UCI Machine Learning Repository, dataset 17: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic . The application displays the source in the UI. The model is trained and evaluated during execution; duration and accuracy are measured, not generated as sample results.

## Run on Windows / VS Code
Open a terminal in the project root (`ml_orchestrator_team_split`) and run:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Then open http://127.0.0.1:8000 . API documentation is at http://127.0.0.1:8000/docs .

If running from inside the `backend` folder instead, use `python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload`.

## Where data is stored
SQLite database file: `backend/scheduler.db`. Submitted tasks and execution history persist after the app closes. The **Reset DB** button clears task/history records only; it keeps dataset metadata. Back up the database file before resetting if you want to keep old results.

## Scheduling notes
- FCFS orders jobs by submission/ID.
- SJF orders pending jobs by their measured duration when available; for never-run jobs it uses a small transparent operation-based estimate only for queue ordering. This estimate is not displayed as an actual result.
- Priority Scheduling uses priority 10 as highest.
- The comparison uses the same completed jobs and their measured durations, treating their arrival times as zero for a like-for-like algorithm comparison. Waiting time is the sum of prior burst durations; turnaround time is completion time when all arrivals are zero. The final makespan is equal for these non-preemptive algorithms with no idle time, so the Fastest option breaks that tie using the sum of individual completion times.
- This is a single-worker educational prototype, not a production scheduler. The real model training runs on the local machine.

## Main files
- `backend/main.py`: API, queue processing, execution-history writes, scheduling comparison.
- `backend/models.py`: SQLite tables for tasks, dataset metadata, execution history, decisions, and resource records.
- `backend/database.py`: SQLite connection and table initialization.
- `backend/ml_runner.py`: real scikit-learn model execution and measured metrics.
- `backend/person3_algorithms.py`: FCFS, SJF, Priority and scheduling formulas.
- `backend/person4_simulator.py`: small programmatic wait-for-cycle detection for deadlock teaching scenarios.
- `frontend/index.html`: six-page dashboard and client-side interactions.
- `sql/schema.sql`: reference schema from the team project.
