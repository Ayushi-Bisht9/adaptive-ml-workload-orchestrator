from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Dataset(Base):
    __tablename__ = "datasets"
    dataset_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    size_mb = Column(Float, default=0)
    n_features = Column(Integer, default=30)
    n_classes = Column(Integer, default=2)

class Task(Base):
    __tablename__ = "tasks"
    task_id = Column(Integer, primary_key=True, index=True)
    task_name = Column(String, nullable=False)
    task_type = Column(String, nullable=False, default="Classification")
    dataset_id = Column(Integer, ForeignKey("datasets.dataset_id"))
    priority = Column(Integer, default=5)
    subtype = Column(String, default="logistic_regression")
    data_fraction = Column(Float, default=1.0)
    status = Column(String, default="pending", index=True)
    algorithm_used = Column(String, default="FCFS")
    submit_time = Column(DateTime, default=datetime.utcnow)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)

class ResourceRequirement(Base):
    __tablename__ = "resource_requirements"
    req_id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.task_id", ondelete="CASCADE"))
    baseline_cpu = Column(Float, default=0)
    baseline_ram_mb = Column(Float, default=0)
    baseline_duration_sec = Column(Float, default=0)
    predicted_cpu = Column(Float, default=0)
    predicted_ram_mb = Column(Float, default=0)
    predicted_duration_sec = Column(Float, default=0)

class ExecutionHistory(Base):
    __tablename__ = "execution_history"
    history_id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.task_id"))
    actual_cpu = Column(Float, default=0)
    actual_ram_mb = Column(Float, default=0)
    actual_duration_sec = Column(Float, default=0)
    wait_time_sec = Column(Float, default=0)
    accuracy = Column(Float, nullable=True)
    fit_time_sec = Column(Float, nullable=True)
    predict_time_sec = Column(Float, nullable=True)
    algorithm_used = Column(String, default="FCFS")
    is_anomaly = Column(Boolean, default=False)
    execution_time = Column(DateTime, default=datetime.utcnow)

class SchedulingDecision(Base):
    __tablename__ = "scheduling_decisions"
    decision_id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.task_id"))
    decision = Column(String)
    reason = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ResourceLock(Base):
    __tablename__ = "resource_locks"
    lock_id = Column(Integer, primary_key=True, index=True)
    resource_name = Column(String, nullable=False)
    held_by_task_id = Column(Integer, ForeignKey("tasks.task_id"), nullable=True)
    requested_by_task_id = Column(Integer, ForeignKey("tasks.task_id"), nullable=True)
