"""Adaptive ML Workload Orchestrator — simple Phase 2 OS + DBMS demo."""

from datetime import datetime
from pathlib import Path
import asyncio
import math
from typing import Literal

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import desc

try:
    from . import models, ml_runner
    from .database import SessionLocal, get_db, init_db
    from . import person3_algorithms as scheduling
    from . import person4_simulator as deadlocks
except ImportError:
    import models
    from database import SessionLocal, get_db, init_db
    import ml_runner
    import person3_algorithms as scheduling
    import person4_simulator as deadlocks


# Initialize the database tables.
init_db()

app = FastAPI(title="Adaptive ML Workload Orchestrator — Phase 2")

QUEUE_LOCK = asyncio.Lock()
QUEUE_RUNNING = False
CURRENT_ALGORITHM = "FCFS"


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class TaskInput(BaseModel):
    task_name: str = Field(min_length=1, max_length=100)
    operation: Literal[
        "logistic_regression",
        "random_forest",
        "decision_tree"
    ] = "logistic_regression"
    priority: int = Field(default=5, ge=1, le=10)


class QueueInput(BaseModel):
    algorithm: Literal["FCFS", "SJF", "Priority"] = "FCFS"


# --------------------------------------------------
# TASK RESPONSE FORMAT
# --------------------------------------------------

def task_dict(task, history=None):
    turnaround = None

    if task.end_time and task.submit_time:
        turnaround = max(
            (task.end_time - task.submit_time).total_seconds(),
            0
        )

    if history and history.wait_time_sec is not None:
        wait_time = history.wait_time_sec
    elif task.start_time and task.submit_time:
        wait_time = max(
            (task.start_time - task.submit_time).total_seconds(),
            0
        )
    else:
        wait_time = None

    return {
        "task_id": task.task_id,
        "task_name": task.task_name,
        "operation": task.subtype,
        "operation_label": ml_runner.OPERATION_LABELS.get(
            task.subtype,
            task.subtype
        ),
        "priority": task.priority,
        "status": task.status,
        "algorithm_used": task.algorithm_used,
        "submit_time": (
            task.submit_time.isoformat()
            if task.submit_time else None
        ),
        "start_time": (
            task.start_time.isoformat()
            if task.start_time else None
        ),
        "end_time": (
            task.end_time.isoformat()
            if task.end_time else None
        ),
        "duration_sec": (
            round(history.actual_duration_sec, 4)
            if history and history.actual_duration_sec is not None
            else None
        ),
        "wait_time_sec": (
            round(wait_time, 4)
            if wait_time is not None else None
        ),
        "turnaround_time_sec": (
            round(turnaround, 4)
            if turnaround is not None else None
        ),
        "accuracy": (
            round(history.accuracy, 2)
            if history and history.accuracy is not None
            else None
        ),
        "completed_at": (
            task.end_time.strftime("%Y-%m-%d %H:%M:%S")
            if task.end_time else None
        ),
        "error": None,
    }


# --------------------------------------------------
# FRONTEND
# --------------------------------------------------

@app.get("/")
def index():
    path = (
        Path(__file__).resolve().parent.parent
        / "frontend"
        / "index.html"
    )
    return FileResponse(path)


# --------------------------------------------------
# GENERAL INFORMATION
# --------------------------------------------------

@app.get("/api/info")
def info():
    return {
        "dataset_name": ml_runner.DATASET_NAME,
        "dataset_source": ml_runner.DATASET_SOURCE,
        "operations": ml_runner.OPERATION_LABELS,
        "worker_running": QUEUE_RUNNING,
        "algorithm": CURRENT_ALGORITHM,
    }


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

@app.get("/api/dashboard")
def dashboard(db: Session = Depends(get_db)):
    tasks = db.query(models.Task).all()

    histories = {
        history.task_id: history
        for history in db.query(models.ExecutionHistory).all()
    }

    statuses = {
        status: sum(1 for task in tasks if task.status == status)
        for status in ["pending", "running", "completed", "failed"]
    }

    waits = [
        history.wait_time_sec
        for history in histories.values()
        if history.wait_time_sec is not None
    ]

    recent = sorted(
        tasks,
        key=lambda task: task.submit_time or datetime.min,
        reverse=True
    )[:8]

    return {
        "total": len(tasks),
        **statuses,
        "avg_wait": (
            round(sum(waits) / len(waits), 3)
            if waits else 0
        ),
        "recent": [
            task_dict(task, histories.get(task.task_id))
            for task in recent
        ],
        "worker_running": QUEUE_RUNNING,
    }


# --------------------------------------------------
# ACTIVE TASK QUEUE
# --------------------------------------------------

@app.get("/api/tasks/active")
def active_tasks(db: Session = Depends(get_db)):
    tasks = (
        db.query(models.Task)
        .filter(models.Task.status.in_(["pending", "running"]))
        .order_by(models.Task.task_id.asc())
        .all()
    )

    return [task_dict(task) for task in tasks]


# --------------------------------------------------
# EXECUTION HISTORY
# --------------------------------------------------

@app.get("/api/tasks/history")
def execution_history(db: Session = Depends(get_db)):
    tasks = (
        db.query(models.Task)
        .filter(models.Task.status.in_(["completed", "failed"]))
        .order_by(
            desc(models.Task.end_time),
            desc(models.Task.task_id)
        )
        .all()
    )

    histories = {
        history.task_id: history
        for history in db.query(models.ExecutionHistory).all()
    }

    return [
        task_dict(task, histories.get(task.task_id))
        for task in tasks
    ]


# --------------------------------------------------
# SUBMIT A TASK
# --------------------------------------------------

@app.post("/api/tasks")
def submit_task(
    payload: TaskInput,
    db: Session = Depends(get_db)
):
    name = payload.task_name.strip()

    if not name:
        raise HTTPException(
            status_code=400,
            detail="Task name cannot be empty."
        )

    dataset = ml_runner.ensure_dataset(db, models)

    task = models.Task(
        task_name=name,
        task_type="Classification",
        dataset_id=dataset.dataset_id,
        priority=payload.priority,
        subtype=payload.operation,
        data_fraction=1.0,
        status="pending",
        algorithm_used=CURRENT_ALGORITHM,
        submit_time=datetime.utcnow(),
    )

    db.add(task)
    db.commit()
    db.refresh(task)

    return {
        "message": "Task submitted to SQLite queue.",
        "task": task_dict(task),
        "dataset": ml_runner.DATASET_NAME,
        "dataset_source": ml_runner.DATASET_SOURCE,
    }


# --------------------------------------------------
# EXECUTE ONE ML TASK
# --------------------------------------------------

async def execute_task(task_id: int, algorithm: str):
    db = SessionLocal()

    try:
        task = (
            db.query(models.Task)
            .filter(models.Task.task_id == task_id)
            .first()
        )

        if not task or task.status != "pending":
            return

        task.status = "running"
        task.start_time = datetime.utcnow()
        task.algorithm_used = algorithm

        submitted_at = task.submit_time or task.start_time
        operation = task.subtype

        db.add(
            models.SchedulingDecision(
                task_id=task.task_id,
                decision="started",
                reason=f"{algorithm} scheduling",
            )
        )

        db.commit()

    finally:
        db.close()

    try:
        # Run model training without blocking FastAPI's event loop.
        result = await asyncio.to_thread(
            ml_runner.run_classification,
            operation
        )

        finished_at = datetime.utcnow()

        db = SessionLocal()

        try:
            task = (
                db.query(models.Task)
                .filter(models.Task.task_id == task_id)
                .first()
            )

            if not task:
                return

            task.status = "completed"
            task.end_time = finished_at

            wait_time = max(
                (task.start_time - submitted_at).total_seconds(),
                0
            ) if task.start_time else 0

            db.add(
                models.ExecutionHistory(
                    task_id=task.task_id,
                    actual_cpu=result["cpu"],
                    actual_ram_mb=result["ram_mb"],
                    actual_duration_sec=result["duration"],
                    wait_time_sec=wait_time,
                    accuracy=result["accuracy"],
                    fit_time_sec=result["fit_time"],
                    predict_time_sec=result["predict_time"],
                    algorithm_used=algorithm,
                    is_anomaly=False,
                    execution_time=finished_at,
                )
            )

            db.add(
                models.SchedulingDecision(
                    task_id=task.task_id,
                    decision="completed",
                    reason=(
                        "Measured model training and evaluation finished"
                    ),
                )
            )

            db.commit()

        finally:
            db.close()

    except Exception as exc:
        db = SessionLocal()

        try:
            task = (
                db.query(models.Task)
                .filter(models.Task.task_id == task_id)
                .first()
            )

            if task:
                task.status = "failed"
                task.end_time = datetime.utcnow()

                db.add(
                    models.SchedulingDecision(
                        task_id=task.task_id,
                        decision="failed",
                        reason=str(exc)[:250],
                    )
                )

                db.commit()

        finally:
            db.close()


# --------------------------------------------------
# QUEUE PROCESSING AND SCHEDULING
# --------------------------------------------------

async def process_queue(algorithm: str):
    global QUEUE_RUNNING, CURRENT_ALGORITHM

    async with QUEUE_LOCK:
        QUEUE_RUNNING = True
        CURRENT_ALGORITHM = algorithm

        try:
            while True:
                db = SessionLocal()

                try:
                    pending = (
                        db.query(models.Task)
                        .filter(models.Task.status == "pending")
                        .all()
                    )

                    if not pending:
                        break

                    # Retrieve historical execution times for completed jobs.
                    completed_rows = (
                        db.query(
                            models.Task.subtype,
                            models.ExecutionHistory.actual_duration_sec,
                        )
                        .join(
                            models.ExecutionHistory,
                            models.ExecutionHistory.task_id
                            == models.Task.task_id,
                        )
                        .filter(
                            models.Task.status == "completed",
                            models.ExecutionHistory.actual_duration_sec > 0,
                        )
                        .all()
                    )

                    # Group execution durations by ML operation.
                    durations_by_operation = {}

                    for operation, duration in completed_rows:
                        durations_by_operation.setdefault(
                            operation, []
                        ).append(float(duration))

                    # Calculate the average measured duration for each operation.
                    average_duration = {
                        operation: sum(durations) / len(durations)
                        for operation, durations
                        in durations_by_operation.items()
                        if durations
                    }

                    # Estimates used only when no history exists for an operation.
                    fallback_estimates = {
                        "logistic_regression": 0.15,
                        "decision_tree": 0.08,
                        "random_forest": 0.25,
                    }

                    estimated = {
                        task.task_id: average_duration.get(
                            task.subtype,
                            fallback_estimates.get(task.subtype, 0.15),
                        )
                        for task in pending
                    }

                    if algorithm == "SJF":
                        ordered = scheduling.order_sjf(
                            pending,
                            estimated
                        )

                    elif algorithm == "Priority":
                        ordered = scheduling.order_priority(pending)

                    else:
                        ordered = scheduling.order_fcfs(pending)

                    next_task_id = ordered[0].task_id

                finally:
                    db.close()

                await execute_task(next_task_id, algorithm)

        finally:
            QUEUE_RUNNING = False


# --------------------------------------------------
# START THE QUEUE WORKER
# --------------------------------------------------

@app.post("/api/queue/run")
async def run_queue(payload: QueueInput):
    global QUEUE_RUNNING

    # Reject another request if the worker is already active.
    if QUEUE_RUNNING:
        return {
            "message": "The worker is already processing the queue.",
            "worker_running": True,
        }

    # Check whether any tasks are waiting.
    db = SessionLocal()

    try:
        count = (
            db.query(models.Task)
            .filter(models.Task.status == "pending")
            .count()
        )

    finally:
        db.close()

    if count == 0:
        raise HTTPException(
            status_code=400,
            detail="There are no pending tasks. Submit a task first."
        )

    # Set the flag before scheduling the background coroutine.
    # This closes the gap in which a second request could start another worker.
    QUEUE_RUNNING = True

    try:
        asyncio.create_task(process_queue(payload.algorithm))

    except Exception:
        QUEUE_RUNNING = False
        raise

    return {
        "message": f"Worker started using {payload.algorithm}.",
        "worker_running": True,
    }


# --------------------------------------------------
# CREATE SAMPLE TASKS
# --------------------------------------------------

@app.post("/api/sample-tasks")
def sample_tasks(db: Session = Depends(get_db)):
    dataset = ml_runner.ensure_dataset(db, models)

    samples = [
        ("Demo - Logistic Regression", "logistic_regression", 8),
        ("Demo - Decision Tree", "decision_tree", 5),
        ("Demo - Random Forest", "random_forest", 3),
    ]

    created = []

    for name, operation, priority in samples:
        task = models.Task(
            task_name=name,
            task_type="Classification",
            dataset_id=dataset.dataset_id,
            priority=priority,
            subtype=operation,
            data_fraction=1.0,
            status="pending",
            algorithm_used=CURRENT_ALGORITHM,
            submit_time=datetime.utcnow(),
        )

        db.add(task)
        db.flush()
        created.append(task.task_id)

    db.commit()

    return {
        "message": (
            "Three demonstration jobs were added. They use real model "
            "training on the documented dataset; timings and accuracy "
            "will be measured during execution."
        ),
        "task_ids": created,
    }


# --------------------------------------------------
# SCHEDULING ALGORITHM COMPARISON
# --------------------------------------------------

@app.get("/api/scheduling-comparison")
def scheduling_comparison(db: Session = Depends(get_db)):
    rows = (
        db.query(models.Task, models.ExecutionHistory)
        .join(
            models.ExecutionHistory,
            models.ExecutionHistory.task_id == models.Task.task_id,
        )
        .filter(models.Task.status == "completed")
        .all()
    )

    # Use actual measured ML runtimes as burst times.
    items = [
        {
            "task_id": task.task_id,
            "priority": int(task.priority or 1),
            "burst": max(
                float(history.actual_duration_sec or 0),
                0.000001,
            ),
        }
        for task, history in rows
    ]

    if not items:
        return {
            "ready": False,
            "message": (
                "Run at least one real task first. Comparison uses "
                "measured execution durations from SQLite."
            ),
            "task_count": 0,
            "results": [],
        }

    results = scheduling.compare_workload(items)

    # Best overall: lowest average wait, then lowest average turnaround.
    best = min(
        results,
        key=lambda row: (
            row["average_waiting_time"],
            row["average_turnaround_time"],
            row["algorithm"],
        ),
    )["algorithm"]

    # With all arrival times set to zero and no idle time, makespan
    # is the sum of burst times for every non-preemptive algorithm.
    fastest_time = min(
        row["total_completion_time"] for row in results
    )

    fastest_candidates = [
        row for row in results
        if math.isclose(
            row["total_completion_time"],
            fastest_time,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ]

    # When makespan ties, minimize the sum of individual completion times.
    fastest = min(
        fastest_candidates,
        key=lambda row: (
            row["sum_of_completion_times"],
            row["algorithm"],
        ),
    )["algorithm"]

    lowest_wait = min(
        results,
        key=lambda row: (
            row["average_waiting_time"],
            row["algorithm"],
        ),
    )["algorithm"]

    # Priority 7–10 is considered high priority.
    high_priority_results = [
        row for row in results
        if row["high_priority_average_wait"] is not None
    ]

    if high_priority_results:
        priority_best = min(
            high_priority_results,
            key=lambda row: (
                row["high_priority_average_wait"],
                row["algorithm"],
            ),
        )["algorithm"]
    else:
        priority_best = "Priority"

    return {
        "ready": True,
        "task_count": len(items),
        "arrival_assumption": (
            "All tasks are treated as arriving at time 0 for a fair, "
            "like-for-like algorithm comparison."
        ),
        "burst_source": (
            "Measured execution durations stored in SQLite; "
            "no comparison values are hardcoded."
        ),
        "results": results,
        "recommendations": {
            "best_overall": best,
            "fastest": fastest,
            "lowest_wait": lowest_wait,
            "priority_handling": priority_best,
        },
    }


# --------------------------------------------------
# DEADLOCK DEMONSTRATION
# --------------------------------------------------

@app.post("/api/deadlock/{scenario}")
def deadlock_demo(scenario: str):
    if scenario not in ("safe", "deadlock"):
        raise HTTPException(
            status_code=400,
            detail="Scenario must be 'safe' or 'deadlock'.",
        )

    return deadlocks.deadlock_scenario(scenario)


# --------------------------------------------------
# RESET DATABASE RECORDS
# --------------------------------------------------

@app.post("/api/reset")
def reset_database(db: Session = Depends(get_db)):
    if QUEUE_RUNNING:
        raise HTTPException(
            status_code=409,
            detail="Wait for the worker to finish before resetting the database.",
        )

    db.query(models.SchedulingDecision).delete()
    db.query(models.ResourceLock).delete()
    db.query(models.ExecutionHistory).delete()
    db.query(models.ResourceRequirement).delete()
    db.query(models.Task).delete()

    db.commit()

    return {
        "message": (
            "Task and execution records were cleared. "
            "Dataset metadata was kept."
        )
    }


# --------------------------------------------------
# COMPATIBILITY ROUTES
# --------------------------------------------------

@app.get("/deadlock_check/")
def legacy_deadlock_check():
    return deadlocks.deadlock_scenario("safe")


@app.get("/current_policy/")
def legacy_policy():
    return {
        "algorithm": CURRENT_ALGORITHM,
        "reason": "Selected in the Task Queue & Worker page.",
    }