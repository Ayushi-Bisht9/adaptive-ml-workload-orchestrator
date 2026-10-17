"""Simple non-preemptive CPU scheduling algorithms used by the dashboard."""
from typing import Iterable


def order_fcfs(tasks):
    return sorted(tasks, key=lambda t: (t.submit_time or 0, t.task_id))


def order_sjf(tasks, burst_times=None):
    burst_times = burst_times or {}
    return sorted(tasks, key=lambda t: (burst_times.get(t.task_id, 1.0), t.task_id))


def order_priority(tasks):
    # Larger priority number means more important; ties use arrival/ID order.
    return sorted(tasks, key=lambda t: (-int(t.priority or 1), t.submit_time, t.task_id))


def compare_workload(items):
    """items: dicts with task_id, priority, burst. All comparison arrivals are 0."""
    algorithms = {
        "FCFS": sorted(items, key=lambda x: x["task_id"]),
        "SJF": sorted(items, key=lambda x: (x["burst"], x["task_id"])),
        "Priority": sorted(items, key=lambda x: (-x["priority"], x["task_id"])),
    }
    results = []
    for name, ordered in algorithms.items():
        elapsed = 0.0
        waits, turns, completions = [], [], []
        high_priority_waits = []
        for item in ordered:
            wait = elapsed
            elapsed += item["burst"]
            turnaround = elapsed  # every task has the same arrival time: zero
            waits.append(wait)
            turns.append(turnaround)
            completions.append(elapsed)
            if item["priority"] >= 7:
                high_priority_waits.append(wait)
        results.append({
            "algorithm": name,
            "average_waiting_time": round(sum(waits) / len(waits), 4) if waits else 0,
            "average_turnaround_time": round(sum(turns) / len(turns), 4) if turns else 0,
            "total_completion_time": round(max(completions), 4) if completions else 0,
            "sum_of_completion_times": round(sum(completions), 4) if completions else 0,
            "high_priority_average_wait": round(sum(high_priority_waits) / len(high_priority_waits), 4) if high_priority_waits else None,
            "order": [x["task_id"] for x in ordered],
        })
    return results
