from backend.person3_algorithms import compare_workload
from backend.person4_simulator import deadlock_scenario, detect_cycle


def test_safe_scenario_has_no_cycle():
    result = deadlock_scenario("safe")
    assert result["deadlock_detected"] is False
    assert result["cycle"] is None


def test_deadlock_scenario_detects_cycle():
    result = deadlock_scenario("deadlock")
    assert result["deadlock_detected"] is True
    assert result["cycle"][0] == result["cycle"][-1]


def test_cycle_detector_handles_acyclic_graph():
    assert detect_cycle({"Task 1": ["Task 2"], "Task 2": []}) is None


def test_scheduling_comparison_uses_all_algorithms_and_formulas():
    items = [
        {"task_id": 1, "priority": 2, "burst": 5.0},
        {"task_id": 2, "priority": 9, "burst": 2.0},
        {"task_id": 3, "priority": 5, "burst": 1.0},
    ]
    result = {row["algorithm"]: row for row in compare_workload(items)}
    assert set(result) == {"FCFS", "SJF", "Priority"}
    assert result["FCFS"]["average_waiting_time"] == round((0 + 5 + 7) / 3, 4)
    assert result["SJF"]["average_waiting_time"] == round((0 + 1 + 3) / 3, 4)
    assert result["Priority"]["order"] == [2, 3, 1]
    assert len({r["total_completion_time"] for r in result.values()}) == 1
